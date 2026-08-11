#!/usr/bin/env python3
"""Run HA/NA full-length experiments in the isolated experiment directory.

English: Execute and summarize the HA-only and combined HA/NA control modes.
中文：执行并汇总 HA-only 及 HA/NA 组合对照模式。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = EXP_ROOT.parents[1]


def env_path(name: str, default: Path | None = None) -> Path | None:
    configured = os.environ.get(name)
    if configured:
        return Path(configured).expanduser().resolve()
    return default


def configured_workspace() -> Path:
    """Return only an explicitly configured source workspace.

    English: Do not discover a sibling workspace implicitly.
    中文：不自动发现同级工作区。
    """
    return env_path("FUTUREFLU_WORKSPACE_ROOT", PACKAGE_ROOT / "external" / "workspace")


WORKSPACE = configured_workspace()
PRE_ROOT = env_path("FUTUREFLU_SOURCE_ROOT", PACKAGE_ROOT / "external" / "inputs")
REF_ROOT = env_path(
    "FUTUREFLU_REFERENCE_ROOT",
    WORKSPACE / "reference" / "predictions",
)
REF_LINEAR = REF_ROOT / "linear"
REF_RISK = REF_ROOT / "risk_components"
REF_STEP2_SCRIPT = env_path(
    "FUTUREFLU_STEP2_SCRIPT",
    REF_LINEAR / "predict_mutations_linear.py",
)
MUTATION_COMPONENTS_SCRIPT = env_path(
    "FUTUREFLU_MUTATION_COMPONENTS_SCRIPT",
    REF_RISK / "mutation_components" / "analyze_mutation_components.py",
)

MODES = ("ha_full", "ha_full_na_full", "ha1_na_full")
BASELINE_MODE = "baseline_ha1"
SUBTYPES = ("H1N1", "H3N2", "Victoria")
HEMISPHERES = ("North", "South")
YEARS = tuple(range(2013, 2025))
COMBOS = ("E", "G", "D", "E+G", "E+D", "G+D", "E+G+D")
CLADE_ACC_CSV = "clade_accuracy/clade_component_acc.csv"
CLADE_MAX_CSV = "clade_accuracy/clade_component_max.csv"
BASE_COLUMNS = [
    "accession number",
    "name",
    "clade",
    "collection_date",
    "submission_date",
    "season",
]

def mode_root(mode: str) -> Path:
    return EXP_ROOT / "runs" / mode


def data_out(mode: str) -> Path:
    return mode_root(mode) / "data"


def linear_out(mode: str) -> Path:
    return mode_root(mode) / "predictions" / "linear" / "results"


def risk_out(mode: str) -> Path:
    return mode_root(mode) / "predictions" / "risk_components"


def sequence_csv_path(mode: str, subtype: str) -> Path:
    return EXP_ROOT / "data" / "sequences" / mode / f"{subtype}_sequence_20250131.csv"


def ensure_mode_dirs(mode: str) -> None:
    for path in [
        data_out(mode) / "futureflu_rank",
        data_out(mode) / "clade_counts",
        linear_out(mode),
        risk_out(mode) / "mutation_components",
        risk_out(mode) / "antigenic_novelty",
        risk_out(mode) / "clade_accuracy",
        risk_out(mode) / "component_combinations",
        mode_root(mode) / "scripts",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_pre_config(subtype: str) -> dict:
    cfg_name = {
        "H1N1": "h1n1_pre2024.json",
        "H3N2": "h3n2_pre2024.json",
        "Victoria": "victoria_pre2024.json",
    }[subtype]
    return json.loads((PRE_ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))


def pre_path(rel: str) -> Path:
    return (PRE_ROOT / rel).resolve()


def numeric_x_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if re.fullmatch(r"X\d+", str(col))]
    return sorted(cols, key=lambda col: int(col[1:]))


def site_prevalence_all_sites(seq: pd.DataFrame, predict_season: int, hemisphere: str) -> pd.DataFrame:
    years = sorted([y for y in seq["season"].unique() if y > 2009 and y < predict_season])
    amino_acids = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
        "-",
    ]

    seq = seq.copy()
    seq["submission_date"] = seq["submission_date"].fillna(seq["collection_date"])
    seq["collection_date"] = pd.to_datetime(seq["collection_date"])
    seq["submission_date"] = pd.to_datetime(seq["submission_date"])
    site_columns = numeric_x_columns(seq)
    for col in site_columns:
        seq[col] = seq[col].astype(str).str.upper()

    annual_data = {}
    prevalence_data = []

    def consensus(data: pd.DataFrame) -> dict[str, str]:
        out = {}
        for col in site_columns:
            valid = data[col][data[col] != "X"]
            if len(valid) > 0:
                out[col] = valid.mode().iloc[0]
        return out

    def significance(new_aa_current, new_aa_prev, total_current, total_prev) -> float:
        from scipy import stats

        observed = np.array(
            [
                [new_aa_current, total_current - new_aa_current],
                [new_aa_prev, total_prev - new_aa_prev],
            ]
        )
        if np.any(observed == 0):
            return 0.0
        return stats.chi2_contingency(observed)[1]

    for year in years:
        if hemisphere == "North":
            start = f"{year}-09-01"
            end = f"{year + 1}-02-01"
            submission_end = f"{predict_season}-02-01"
            prev_start = f"{year - 1}-09-01"
            prev_end = f"{year}-02-01"
        else:
            start = f"{year}-02-01"
            end = f"{year}-09-01"
            submission_end = f"{predict_season - 1}-09-01"
            prev_start = f"{year - 1}-02-01"
            prev_end = f"{year - 1}-09-01"

        season_data = seq[
            (seq["collection_date"] >= start)
            & (seq["collection_date"] < end)
            & (seq["submission_date"] < submission_end)
        ]

        year_freq_data = {}
        year_counts_data = {}
        for site_col in site_columns:
            valid = season_data[site_col][season_data[site_col] != "X"]
            freq = valid.value_counts(normalize=True)
            counts = valid.value_counts()
            for aa in amino_acids:
                year_freq_data[f"{site_col}{aa}"] = freq.get(aa, 0.0)
            year_counts_data[site_col] = counts
        prevalence_data.append(pd.Series(year_freq_data, name=year))

        annual_data[year] = {
            "freqs": year_freq_data,
            "counts": year_counts_data,
            "dominant_mutations": [],
        }
        if year == years[0]:
            continue

        prev_data = seq[
            (seq["collection_date"] >= prev_start)
            & (seq["collection_date"] < prev_end)
            & (seq["submission_date"] < submission_end)
        ]
        current_consensus = consensus(season_data)
        prev_consensus = consensus(prev_data)
        prev_counts_data = {}
        for site_col in site_columns:
            valid = prev_data[site_col][prev_data[site_col] != "X"]
            prev_counts_data[site_col] = valid.value_counts()

        identified = []
        for site_col in site_columns:
            if (
                site_col in current_consensus
                and site_col in prev_consensus
                and current_consensus[site_col] != prev_consensus[site_col]
            ):
                new_aa = current_consensus[site_col]
                current_counts = year_counts_data[site_col]
                prev_counts = prev_counts_data[site_col]
                p_value = significance(
                    current_counts.get(new_aa, 0),
                    prev_counts.get(new_aa, 0),
                    sum(current_counts),
                    sum(prev_counts),
                )
                if p_value < 0.05:
                    identified.append(f"{site_col[1:]}{new_aa}")
        annual_data[year]["dominant_mutations"] = sorted(identified)

    prevalence = pd.concat(prevalence_data, axis=1).T
    prevalence["dominant_mutation"] = [
        ", ".join(annual_data[y]["dominant_mutations"]) for y in years
    ]
    mutation_columns = [col for col in prevalence.columns if col != "dominant_mutation"]
    return prevalence[mutation_columns + ["dominant_mutation"]].reset_index().rename(
        columns={"index": "season"}
    )


def run_linear(mode: str, subtypes: Sequence[str], resume: bool) -> None:
    ensure_mode_dirs(mode)
    step2 = load_module(
        f"reference_step2_{mode}",
        REF_STEP2_SCRIPT,
    )
    theta_range = np.arange(0.1, 0.5 + 0.1 / 2, 0.1)

    for subtype in subtypes:
        seq_path = sequence_csv_path(mode, subtype)
        print(f"[linear:{mode}] loading {seq_path}", flush=True)
        seq_df = pd.read_csv(seq_path)
        for hemisphere in HEMISPHERES:
            epi_path = pre_path(f"data/positivity/{subtype}_positive_rate_{hemisphere.lower()}.csv")
            epi_df = pd.read_csv(epi_path)
            for year in YEARS:
                out_dir = linear_out(mode) / f"{subtype}_{hemisphere}" / str(year)
                out_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{subtype}_{hemisphere}_{year}"
                paths = [
                    out_dir / f"{prefix}_prevalence.csv",
                    out_dir / f"{prefix}_gmeasure.csv",
                    out_dir / f"{prefix}_mutations.csv",
                    out_dir / f"{prefix}_distribution.csv",
                ]
                if resume and all(path.exists() for path in paths):
                    continue
                print(f"[linear:{mode}] {subtype} {hemisphere} {year}", flush=True)
                log_print = step2.setup_logging(str(out_dir))
                log_print(f"ha_na_experiment mode: {mode}")
                log_print(f"input seq_file: {seq_path}")
                prev_data = site_prevalence_all_sites(seq_df, year, hemisphere)
                prev_data.to_csv(paths[0], index=False)
                gmeasure_data = step2.gmeasure(prev_data, theta_range)
                gmeasure_data.to_csv(paths[1], index=False)
                best_theta, best_r2, years_used = step2.fit_regression(
                    gmeasure_data, epi_df, log_print
                )
                log_print(f"best theta: {best_theta}, R2: {best_r2}, years used: {years_used}")
                if best_theta is None:
                    pd.DataFrame(
                        columns=[
                            "predict_season",
                            "risk_mutation",
                            "previous_prevalence",
                            "predicted_prevalence",
                            "delta",
                            "model",
                        ]
                    ).to_csv(paths[2], index=False)
                    pd.DataFrame(columns=["risk_mutation_group", "count", "model"]).to_csv(
                        paths[3], index=False
                    )
                    continue
                mutations = step2.predict_mutations_multi_model(year, best_theta, prev_data, None)
                mutations.to_csv(paths[2], index=False)
                distribution = step2.analyze_risk_mutations(seq_df, mutations, year, hemisphere)
                distribution.to_csv(paths[3], index=False)


def copy_antigenic_inputs(mode: str) -> None:
    out_dir = risk_out(mode) / "antigenic_novelty"
    out_dir.mkdir(parents=True, exist_ok=True)
    for subtype in SUBTYPES:
        src = WORKSPACE / f"strain_antigenic_novelty_{subtype}.csv"
        if not src.exists():
            src = REF_RISK / "antigenic_novelty" / f"strain_antigenic_novelty_{subtype}.csv"
        dst = out_dir / f"strain_antigenic_novelty_{subtype}.csv"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())


def run_component(mode: str, subtypes: Sequence[str]) -> None:
    ensure_mode_dirs(mode)
    copy_antigenic_inputs(mode)
    if not MUTATION_COMPONENTS_SCRIPT.exists():
        raise FileNotFoundError(
            "mutation-components script not found: "
            f"{MUTATION_COMPONENTS_SCRIPT}. Set FUTUREFLU_MUTATION_COMPONENTS_SCRIPT."
        )
    comp = load_module(
        f"reference_component_{mode}",
        MUTATION_COMPONENTS_SCRIPT,
    )
    frames: List[pd.DataFrame] = []

    for subtype in subtypes:
        cfg = read_pre_config(subtype)
        eve_prefix = cfg["evescape_prefix"]
        evescape_dir = pre_path(cfg["evescape_dir"])
        sequence_df = pd.read_csv(sequence_csv_path(mode, subtype)).rename(
            columns={"accession number": "accession_number"}
        )
        sequence_df["collection_date"] = pd.to_datetime(sequence_df["collection_date"])
        sequence_df["submission_date"] = pd.to_datetime(sequence_df["submission_date"])
        antigenic_novelty_df = pd.read_csv(
            risk_out(mode)
            / "antigenic_novelty"
            / f"strain_antigenic_novelty_{subtype}.csv"
        )

        for hemi_lower, hemisphere in [("north", "North"), ("south", "South")]:
            for year in YEARS:
                print(f"[component:{mode}] {subtype} {hemi_lower} {year}", flush=True)
                date_str = f"{year}0131" if hemi_lower == "north" else f"{year - 1}0831"
                prefix = f"{subtype}_{hemisphere}_{year}"
                linear_dir = linear_out(mode) / f"{subtype}_{hemisphere}" / str(year)
                distribution_path = linear_dir / f"{prefix}_distribution.csv"
                prediction_path = linear_dir / f"{prefix}_mutations.csv"
                mutations_path = evescape_dir / f"{eve_prefix}_evescape_{date_str}.csv"
                sites_path = evescape_dir / f"{eve_prefix}_evescape_sites_{date_str}.csv"

                df = pd.read_csv(distribution_path)
                if df.empty:
                    continue
                mutations_df = pd.read_csv(mutations_path)
                sites_df = pd.read_csv(sites_path)
                prediction_df = pd.read_csv(prediction_path)
                df["mutation_count"] = df["risk_mutation_group"].apply(comp.count_mutations)
                df_filtered = df[df["mutation_count"] > 0]
                if df_filtered.empty:
                    continue
                q1 = df_filtered["mutation_count"].quantile(0.25)
                q3 = df_filtered["mutation_count"].quantile(0.75)
                iqr = q3 - q1
                non_outliers = df_filtered[
                    (df_filtered["mutation_count"] <= q3 + 3 * iqr)
                    & (df_filtered["mutation_count"] >= q1 - 3 * iqr)
                ]
                if non_outliers.empty:
                    continue
                all_mutation_groups = non_outliers["risk_mutation_group"].dropna().tolist()

                if hemi_lower == "north":
                    start_date = f"{year - 1}-09-01"
                    end_date = f"{year}-02-01"
                else:
                    start_date = f"{year - 1}-02-01"
                    end_date = f"{year - 1}-09-01"
                filtered_seq_df = sequence_df[
                    (sequence_df["collection_date"] >= start_date)
                    & (sequence_df["collection_date"] < end_date)
                    & (sequence_df["submission_date"] < end_date)
                ]

                mutations_min_escape = mutations_df["evescape"].min()
                sites_min_escape = sites_df["evescape"].min()
                mutation_escape = {
                    f"{row['i']}{row['mut']}": row["evescape"] - mutations_min_escape
                    for _, row in mutations_df.iterrows()
                }
                site_escape = {
                    str(row["i"]): row["evescape"] - sites_min_escape
                    for _, row in sites_df.iterrows()
                }
                dch_min = mutations_df["dissimilarity_charge_hydro"].min()
                mutation_dch = {
                    f"{row['i']}{row['mut']}": row["dissimilarity_charge_hydro"] - dch_min
                    for _, row in mutations_df.iterrows()
                }
                tmp_dch = mutations_df.copy()
                tmp_dch["_dch"] = tmp_dch["dissimilarity_charge_hydro"] - dch_min
                site_dch = {str(site): val for site, val in tmp_dch.groupby("i")["_dch"].mean().items()}
                awcn_min = mutations_df["accessibility_wcn"].min(skipna=True)
                mutation_awcn = {}
                for _, row in mutations_df.iterrows():
                    val = row["accessibility_wcn"]
                    mutation_awcn[f"{row['i']}{row['mut']}"] = 0.0 if pd.isna(val) else val - awcn_min
                tmp_awcn = mutations_df.copy()
                tmp_awcn["_awcn"] = tmp_awcn["accessibility_wcn"].apply(
                    lambda x: 0.0 if pd.isna(x) else x - awcn_min
                )
                site_awcn = {str(site): val for site, val in tmp_awcn.groupby("i")["_awcn"].mean().items()}
                ef_min = mutations_df["fitness_eve"].min()
                mutation_ef = {
                    f"{row['i']}{row['mut']}": row["fitness_eve"] - ef_min
                    for _, row in mutations_df.iterrows()
                }
                tmp_ef = mutations_df.copy()
                tmp_ef["_ef"] = tmp_ef["fitness_eve"] - ef_min
                site_ef = {str(site): val for site, val in tmp_ef.groupby("i")["_ef"].mean().items()}
                mutation_prevalence = dict(zip(prediction_df["risk_mutation"], prediction_df["delta"]))
                min_prev = min(mutation_prevalence.values()) if mutation_prevalence else 0
                mutation_prevalence = {k: v - min_prev for k, v in mutation_prevalence.items()}
                antigenic_tmp = antigenic_novelty_df.copy()
                antigenic_tmp["_an_norm"] = antigenic_tmp.groupby("season")[
                    "antigenic_novelty"
                ].transform(lambda x: x - x.min())
                antigenic_novelty_dict = dict(
                    zip(antigenic_tmp["accession_number"], antigenic_tmp["_an_norm"])
                )

                rows = []
                for _, row in non_outliers.iterrows():
                    mutation_group = row["risk_mutation_group"]
                    count = row["mutation_count"]
                    if pd.isna(mutation_group):
                        mutual_info = 0
                    else:
                        muts = [m.strip() for m in mutation_group.split(",")]
                        if len(muts) == 1:
                            mutual_info = comp.calculate_single_mutation_mi(
                                muts[0], filtered_seq_df, all_mutation_groups
                            )
                        else:
                            mut_matrix = comp.get_mutation_matrix_simple(filtered_seq_df, muts)
                            mutual_info = comp.calculate_group_mutual_information(mut_matrix)
                    matching_seqs = comp.get_matching_sequences(
                        mutation_group, filtered_seq_df, all_mutation_groups
                    )
                    rows.append(
                        {
                            "subtype": subtype,
                            "hemisphere": hemi_lower,
                            "year": year,
                            "risk_mutation_group": mutation_group,
                            "clade": comp.get_clade_info_from_matching(matching_seqs),
                            "mutation_count": count,
                            "mutation_group_seq_count": len(matching_seqs),
                            "total_escape": comp.calculate_total_escape_value(
                                mutation_group,
                                mutation_escape,
                                site_escape,
                                subtype,
                                sites_df,
                                mutations_df,
                            ),
                            "predicted_prevalence": comp.calculate_prevalence(
                                mutation_group, mutation_prevalence
                            ),
                            "mutual_information": mutual_info,
                            "dissimilarity_charge_hydro": comp.calculate_metric_value(
                                mutation_group, mutation_dch, site_dch, subtype, sites_df, mutations_df
                            ),
                            "accessibility_wcn": comp.calculate_metric_value(
                                mutation_group, mutation_awcn, site_awcn, subtype, sites_df, mutations_df
                            ),
                            "fitness_eve": comp.calculate_metric_value(
                                mutation_group, mutation_ef, site_ef, subtype, sites_df, mutations_df
                            ),
                            "antigenic_novelty": comp.calculate_antigenic_novelty_from_matching(
                                matching_seqs, antigenic_novelty_dict
                            ),
                        }
                    )
                results_df = comp.filter_random_single_mutations(pd.DataFrame(rows))
                frames.append(results_df)

    if not frames:
        raise RuntimeError(f"component stage produced no rows for {mode}")
    all_results = pd.concat(frames, ignore_index=True)
    columns = [
        "subtype",
        "hemisphere",
        "year",
        "risk_mutation_group",
        "clade",
        "mutation_count",
        "mutation_group_seq_count",
        "total_escape",
        "predicted_prevalence",
        "mutual_information",
        "dissimilarity_charge_hydro",
        "accessibility_wcn",
        "fitness_eve",
        "antigenic_novelty",
    ]
    out_path = risk_out(mode) / "mutation_components" / "risk_mutation_group_component.csv"
    root_copy = risk_out(mode) / "risk_mutation_group_component.csv"
    all_results[columns].to_csv(out_path, index=False)
    all_results[columns].to_csv(root_copy, index=False)
    print(f"[component:{mode}] wrote {out_path} shape={all_results.shape}", flush=True)


def season_window(year: int, hemisphere: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if hemisphere.lower() == "north":
        return pd.Timestamp(f"{year - 1}-09-01"), pd.Timestamp(f"{year}-02-01")
    return pd.Timestamp(f"{year - 1}-02-01"), pd.Timestamp(f"{year - 1}-09-01")


def target_season_window(year: int, hemisphere: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if hemisphere.lower() == "north":
        return pd.Timestamp(f"{year}-09-01"), pd.Timestamp(f"{year + 1}-02-01")
    return pd.Timestamp(f"{year}-02-01"), pd.Timestamp(f"{year}-09-01")


def valid_clade_series(df: pd.DataFrame) -> pd.Series:
    clade = df["clade"].fillna("").astype(str).str.strip()
    return clade[~clade.str.lower().isin({"", "unknown", "unassigned", "nan"})]


def counted_clade_series(df: pd.DataFrame) -> pd.Series:
    clade = df["clade"].fillna("").astype(str).str.strip()
    return clade[~clade.str.lower().isin({"", "nan"})]


def build_label_and_count_inputs(mode: str) -> None:
    count_dir = data_out(mode) / "clade_counts"
    rank_dir = data_out(mode) / "futureflu_rank"
    count_dir.mkdir(parents=True, exist_ok=True)
    rank_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for subtype in SUBTYPES:
        seq_df = pd.read_csv(sequence_csv_path(mode, subtype), usecols=BASE_COLUMNS)
        seq_df["collection_date"] = pd.to_datetime(seq_df["collection_date"])
        seq_df["submission_date"] = pd.to_datetime(seq_df["submission_date"])
        count_rows = []
        for year in range(2012, 2025):
            for hemisphere in ("south", "north"):
                start, end = target_season_window(year, hemisphere)
                collection_mask = (seq_df["collection_date"] >= start) & (seq_df["collection_date"] < end)
                submission_mask = collection_mask & (seq_df["submission_date"] < end)
                collection_counts = counted_clade_series(seq_df.loc[collection_mask]).value_counts()
                submission_counts = counted_clade_series(seq_df.loc[submission_mask]).value_counts()
                clades = sorted(set(collection_counts.index).union(set(submission_counts.index)))
                for clade in clades:
                    count_rows.append(
                        {
                            "year": year,
                            "hemisphere": hemisphere,
                            "clade": clade,
                            "submission_count": int(submission_counts.get(clade, 0)),
                            "collection_count": int(collection_counts.get(clade, 0)),
                        }
                    )
                label_counts = valid_clade_series(seq_df.loc[collection_mask]).value_counts()
                if year in YEARS and not label_counts.empty:
                    labels.append(
                        {
                            "subtype": subtype,
                            "hemisphere": hemisphere,
                            "year": year,
                            "clade": label_counts.idxmax(),
                        }
                    )
        key = subtype.lower().replace("/", "").replace(" ", "")
        pd.DataFrame(count_rows).to_csv(
            count_dir / f"submission_collection_clade_count_{key}.csv", index=False
        )
    pd.DataFrame(labels).to_csv(rank_dir / "circulating_clade.csv", index=False)


def write_local_script(src: Path, dst: Path) -> None:
    """Adapt packaged helper scripts to a mode-local experiment root."""
    text = src.read_text(encoding="utf-8")
    text = text.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
        "MODE_ROOT = Path(__file__).resolve().parents[1]\nPROJECT_ROOT = MODE_ROOT",
        1,
    )
    text = text.replace(
        "/ 'outputs'\n    / 'predictions'\n    / 'risk_components'",
        "/ 'predictions'\n    / 'risk_components'",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def run_aux(mode: str) -> None:
    ensure_mode_dirs(mode)
    build_label_and_count_inputs(mode)
    scripts_dir = mode_root(mode) / "scripts"
    acc_src = PACKAGE_ROOT / "scripts" / "calculate_clade_accuracy.py"
    combine_src = PACKAGE_ROOT / "scripts" / "combine_clade_components.py"
    acc_dst = scripts_dir / "calculate_clade_accuracy.py"
    combine_dst = scripts_dir / "combine_clade_components.py"
    write_local_script(acc_src, acc_dst)
    write_local_script(combine_src, combine_dst)

    acc_cmd = [
        sys.executable,
        str(acc_dst),
        CLADE_ACC_CSV,
        CLADE_MAX_CSV,
    ]
    combine_cmd = [
        sys.executable,
        str(combine_dst),
        "component_combinations/clade_component_combine_acc_Twindow.csv",
        "component_combinations/EGD_combine_Twindow.csv",
        "component_combinations/EGD_temperatures_Twindow.csv",
        "--elpd_output",
        "component_combinations/elpd_Twindow.csv",
        "--aic_output",
        "component_combinations/aic_Twindow.csv",
        "--pre_act_output",
        "component_combinations/clade_pre_act_Twindow.csv",
        "--divergence_inform",
        "component_combinations/divergence_Twindow.csv",
        "--escape_inform",
        "component_combinations/escape_Twindow.csv",
        "--growth_inform",
        "component_combinations/growth_Twindow.csv",
    ]
    print(f"[aux:{mode}] clade component accuracy", flush=True)
    subprocess.run(acc_cmd, check=True, cwd=str(mode_root(mode)))
    print(f"[aux:{mode}] clade component combine", flush=True)
    subprocess.run(combine_cmd, check=True, cwd=str(mode_root(mode)))


def evaluate_outputs(modes: Sequence[str]) -> None:
    results_dir = EXP_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_frames = []
    detail_frames = []

    def collect_mode(mode: str, pre_act_path: Path, label_path: Path) -> None:
        pre_act = pd.read_csv(pre_act_path)
        labels = pd.read_csv(label_path)
        for combo in COMBOS:
            for method in ("fit", "freq"):
                score_col = f"{combo}_pre_{method}"
                if score_col not in pre_act.columns:
                    continue
                tmp = pre_act.dropna(subset=[score_col]).copy()
                idx = tmp.groupby(["subtype", "year", "hemisphere"], sort=False)[score_col].idxmax()
                pred = tmp.loc[idx, ["subtype", "year", "hemisphere", "clade", score_col]].rename(
                    columns={"clade": "pred_clade", score_col: "score"}
                )
                merged = labels.merge(pred, on=["subtype", "year", "hemisphere"], how="left")
                merged = merged.rename(columns={"clade": "truth_clade"})
                merged["correct"] = merged["truth_clade"].eq(merged["pred_clade"])
                merged["mode"] = mode
                merged["combo"] = combo
                merged["method"] = method
                detail_frames.append(merged)
                for subtype, sub in list(merged.groupby("subtype")) + [("All", merged)]:
                    total = len(sub)
                    correct = int(sub["correct"].sum())
                    summary_frames.append(
                        {
                            "mode": mode,
                            "subtype": subtype,
                            "combo": combo,
                            "method": method,
                            "correct": correct,
                            "total": total,
                            "accuracy": correct / total if total else np.nan,
                            "available_predictions": int(sub["pred_clade"].notna().sum()),
                        }
                    )

    baseline_pre_act = (
        PACKAGE_ROOT
        / "outputs"
        / "predictions"
        / "risk_components"
        / "component_combinations"
        / "clade_pre_act_Twindow.csv"
    )
    baseline_labels = (
        PACKAGE_ROOT / "data" / "futureflu_rank" / "circulating_clade.csv"
    )
    collect_mode(BASELINE_MODE, baseline_pre_act, baseline_labels)

    for mode in modes:
        pre_act_path = risk_out(mode) / "component_combinations" / "clade_pre_act_Twindow.csv"
        label_path = data_out(mode) / "futureflu_rank" / "circulating_clade.csv"
        collect_mode(mode, pre_act_path, label_path)

    summary = pd.DataFrame(summary_frames)
    detail = pd.concat(detail_frames, ignore_index=True)
    summary.to_csv(results_dir / "accuracy_vs_truth_summary.csv", index=False)
    detail.to_csv(results_dir / "predictions_vs_truth_long.csv", index=False)



def selected_modes(names: Iterable[str]) -> list[str]:
    modes = []
    for name in names:
        if name not in MODES:
            raise ValueError(f"unknown mode {name!r}; choose from {', '.join(MODES)}")
        modes.append(name)
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["linear", "component", "aux", "evaluate", "all"],
    )
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--subtypes", nargs="+", default=list(SUBTYPES), choices=list(SUBTYPES))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = selected_modes(args.modes)
    start = time.time()
    for mode in modes:
        if args.stage in {"linear", "all"}:
            run_linear(mode, args.subtypes, resume=args.resume)
        if args.stage in {"component", "all"}:
            run_component(mode, args.subtypes)
        if args.stage in {"aux", "all"}:
            run_aux(mode)
    if args.stage in {"evaluate", "all"}:
        evaluate_outputs(modes)
    print(f"[done] stage={args.stage} elapsed={time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
