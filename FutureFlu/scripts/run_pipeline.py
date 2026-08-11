#!/usr/bin/env python3
"""Rebuild FutureFlu HA1 reproduction outputs in this release package.

English: Run the packaged HA1 reproduction stages and refresh release outputs.
中文：运行打包后的 HA1 复现阶段并刷新发布结果。

Raw-data stages use only explicitly configured source snapshots. Set the
FUTUREFLU_* environment variables before running those stages; the packaged
auxiliary stage uses the release-package inputs directly.
原始数据阶段只使用显式配置的源快照；运行这些阶段前请设置 FUTUREFLU_* 环境变量，
而打包后的辅助阶段直接使用发布包内输入。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, default: Path | None = None) -> Path | None:
    configured = os.environ.get(name)
    if configured:
        return Path(configured).expanduser().resolve()
    return default


def configured_workspace() -> Path:
    """Return only an explicitly configured source workspace.

    English: Never discover or use a sibling workspace implicitly.
    中文：不自动发现或隐式使用同级工作区。
    """
    return env_path("FUTUREFLU_WORKSPACE_ROOT", PACKAGE_ROOT / "external" / "workspace")


WORKSPACE = configured_workspace()
REPRO_ROOT = env_path("FUTUREFLU_PACKAGE_ROOT", PACKAGE_ROOT)
PRE_ROOT = env_path("FUTUREFLU_SOURCE_ROOT", PACKAGE_ROOT / "external" / "inputs")
REF_ROOT = env_path(
    "FUTUREFLU_REFERENCE_ROOT",
    WORKSPACE / "reference" / "predictions",
)
REF_LINEAR = REF_ROOT / "linear"
REF_RISK = REF_ROOT / "risk_components"
REF_STEP2_SCRIPT = env_path(
    "FUTUREFLU_STEP2_SCRIPT",
    PACKAGE_ROOT / "scripts" / "predict_mutations_linear.py",
)
MUTATION_COMPONENTS_SCRIPT = env_path(
    "FUTUREFLU_MUTATION_COMPONENTS_SCRIPT",
    REF_RISK / "mutation_components" / "analyze_mutation_components.py",
)

REPRO_FUTUREFLU = REPRO_ROOT / "outputs" / "predictions"
LINEAR_OUT = REPRO_FUTUREFLU / "linear" / "results"
RISK_OUT = REPRO_FUTUREFLU / "risk_components"
DATA_OUT = REPRO_ROOT / "data"

SUBTYPES = ("H1N1", "H3N2", "Victoria")
HEMISPHERES = ("North", "South")
YEARS = tuple(range(2013, 2025))
CLADE_ACC_CSV = "clade_accuracy/clade_component_acc.csv"
CLADE_MAX_CSV = "clade_accuracy/clade_component_max.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    for path in [
        DATA_OUT,
        LINEAR_OUT,
        RISK_OUT / "mutation_components",
        RISK_OUT / "clade_accuracy",
        RISK_OUT / "component_combinations",
        REPRO_ROOT / "scripts",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def require_raw_workspace() -> None:
    """Require explicit source snapshots for raw-data stages.

    English: Published summaries remain package-local; raw reruns need an
    explicitly configured source workspace.
    中文：发布汇总仅使用包内文件；原始数据重跑必须显式配置源工作区。
    """
    if "FUTUREFLU_WORKSPACE_ROOT" not in os.environ:
        raise RuntimeError(
            "This stage needs local source inputs. Place them under raw_inputs/ "
            "or point FUTUREFLU_WORKSPACE_ROOT at your data root, then retry."
        )


def read_pre_config(subtype: str) -> dict:
    cfg_name = {
        "H1N1": "h1n1_pre2024.json",
        "H3N2": "h3n2_pre2024.json",
        "Victoria": "victoria_pre2024.json",
    }[subtype]
    return json.loads((PRE_ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))


def pre_path(rel: str) -> Path:
    return (PRE_ROOT / rel).resolve()


def sequence_csv_path(subtype: str) -> Path:
    return DATA_OUT / f"{subtype}_HA_sequence_20250131.csv"


def normalize_partial_date(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    parts = text.split("-")
    if len(parts) == 1:
        return f"{text}-01-01"
    if len(parts) == 2:
        return f"{text}-01"
    return text


def load_sequence_info_lookup(info_path: Path) -> Dict[str, dict]:
    if info_path.suffix.lower() == ".csv":
        info_df = pd.read_csv(info_path)
    else:
        info_df = pd.read_excel(info_path)

    if "Isolate_Id" not in info_df.columns:
        raise KeyError(f"missing Isolate_Id in {info_path}")

    info_df = info_df.copy()
    info_df["Isolate_Id"] = info_df["Isolate_Id"].fillna("").astype(str).str.strip()
    info_df = info_df[info_df["Isolate_Id"] != ""]
    info_df = info_df.drop_duplicates(subset=["Isolate_Id"], keep="first")
    return {row["Isolate_Id"]: row for row in info_df.to_dict("records")}


def sequence_record_ids(record_id: str, subtype: str) -> Tuple[str, str]:
    parts = record_id.split("|")
    isolate_id = parts[0]
    if subtype == "Victoria":
        accession_number = parts[1] if len(parts) > 1 else isolate_id
        return accession_number, accession_number
    return isolate_id, isolate_id


def write_sequence_table(
    subtype: str,
    fasta_path: Path,
    info_path: Path,
    cutoff_date: str,
    out_path: Path,
) -> int:
    info_lookup = load_sequence_info_lookup(info_path)
    cutoff_ts = pd.Timestamp(cutoff_date)
    base_columns = [
        "accession number",
        "name",
        "clade",
        "collection_date",
        "submission_date",
        "season",
    ]

    written = 0
    seq_len = None
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            seq = str(record.seq)
            if seq.count("-") > 3:
                continue

            accession_number, lookup_key = sequence_record_ids(str(record.id), subtype)
            info = info_lookup.get(lookup_key)
            if not info:
                continue

            collection_str = normalize_partial_date(info.get("Collection_Date", ""))
            if not collection_str:
                continue
            try:
                collection_date = pd.to_datetime(collection_str)
            except (ValueError, TypeError):
                continue

            submission_str = normalize_partial_date(info.get("Submission_Date", ""))
            try:
                submission_date = pd.to_datetime(submission_str) if submission_str else None
            except (ValueError, TypeError):
                submission_date = None
            if submission_date is None:
                submission_date = collection_date

            if submission_date >= cutoff_ts or collection_date.year < 2010:
                continue

            if writer is None:
                seq_len = len(seq)
                writer = csv.writer(handle)
                writer.writerow(base_columns + [f"X{i+1}" for i in range(seq_len)])
            elif seq_len is not None and len(seq) != seq_len:
                raise ValueError(
                    f"inconsistent sequence length for {subtype}: {len(seq)} != {seq_len}"
                )

            isolate_name = info.get("Isolate_Name", "")
            clade = info.get("Clade", "")
            writer.writerow(
                [
                    accession_number,
                    "" if pd.isna(isolate_name) else str(isolate_name),
                    "" if pd.isna(clade) else str(clade),
                    collection_date.strftime("%Y-%m-%d"),
                    submission_date.strftime("%Y-%m-%d"),
                    collection_date.year - 1 if collection_date.month < 2 else collection_date.year,
                ]
                + list(seq)
            )
            written += 1

        if writer is None:
            writer = csv.writer(handle)
            writer.writerow(base_columns)

    return written


def prepare_sequences(processes: int | None) -> None:
    require_raw_workspace()
    ensure_dirs()
    for subtype in SUBTYPES:
        out_path = sequence_csv_path(subtype)
        if out_path.exists():
            print(f"[sequence] reuse {out_path}")
            continue

        cfg = read_pre_config(subtype)
        print(f"[sequence] generating {subtype} from external inputs")
        t0 = time.time()
        rows = write_sequence_table(
            subtype=subtype,
            fasta_path=pre_path(cfg["fasta_path"]),
            info_path=pre_path(cfg["info_path"]),
            cutoff_date=str(cfg.get("sequence_cutoff", "2025-02-01")),
            out_path=out_path,
        )
        print(f"[sequence] wrote {out_path} rows={rows} elapsed={time.time() - t0:.1f}s")


def run_linear() -> None:
    require_raw_workspace()
    ensure_dirs()
    step2 = load_module(
        "reference_step2",
        REF_STEP2_SCRIPT,
    )

    for subtype in SUBTYPES:
        cfg = read_pre_config(subtype)
        seq_path = sequence_csv_path(subtype)
        if not seq_path.exists():
            raise FileNotFoundError(f"missing sequence table: {seq_path}")
        print(f"[linear] loading {seq_path}")
        seq_df = pd.read_csv(seq_path)

        for hemisphere in HEMISPHERES:
            epi_path = pre_path(f"data/positivity/{subtype}_positive_rate_{hemisphere.lower()}.csv")
            epi_df = pd.read_csv(epi_path)
            for year in YEARS:
                out_dir = LINEAR_OUT / f"{subtype}_{hemisphere}" / str(year)
                out_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{subtype}_{hemisphere}_{year}"

                print(f"[linear] {subtype} {hemisphere} {year}")
                log_print = step2.setup_logging(str(out_dir))
                seq_log = f"data/{subtype}_HA_sequence_20250131.csv"
                epi_rel = f"data/positivity/{subtype}_positive_rate_{hemisphere.lower()}.csv"
                log_print(f"reproduction input seq_file: {seq_log}")
                log_print(f"reproduction input epi_file: not_in_package:{epi_rel}")

                theta_range = np.arange(0.1, 0.5 + 0.1 / 2, 0.1)
                prev_data = step2.site_prevalence(seq_df, year, hemisphere, subtype)
                prev_data.to_csv(out_dir / f"{prefix}_prevalence.csv", index=False)

                gmeasure_data = step2.gmeasure(prev_data, theta_range)
                gmeasure_data.to_csv(out_dir / f"{prefix}_gmeasure.csv", index=False)

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
                    ).to_csv(out_dir / f"{prefix}_mutations.csv", index=False)
                    pd.DataFrame(columns=["risk_mutation_group", "count", "model"]).to_csv(
                        out_dir / f"{prefix}_distribution.csv", index=False
                    )
                    continue

                mutations = step2.predict_mutations_multi_model(year, best_theta, prev_data, None)
                mutations.to_csv(out_dir / f"{prefix}_mutations.csv", index=False)

                distribution = step2.analyze_risk_mutations(seq_df, mutations, year, hemisphere)
                distribution.to_csv(out_dir / f"{prefix}_distribution.csv", index=False)


def copy_antigenic_inputs() -> None:
    require_raw_workspace()
    ensure_dirs()
    antigenic_dir = RISK_OUT / "antigenic_novelty"
    antigenic_dir.mkdir(parents=True, exist_ok=True)
    for subtype in SUBTYPES:
        src = WORKSPACE / f"strain_antigenic_novelty_{subtype}.csv"
        if not src.exists():
            src = REF_RISK / "antigenic_novelty" / f"strain_antigenic_novelty_{subtype}.csv"
        dst = antigenic_dir / f"strain_antigenic_novelty_{subtype}.csv"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        print(f"[antigenic] {dst}")


def run_component() -> None:
    require_raw_workspace()
    ensure_dirs()
    copy_antigenic_inputs()
    if not MUTATION_COMPONENTS_SCRIPT.exists():
        raise FileNotFoundError(
            "mutation-components script not found: "
            f"{MUTATION_COMPONENTS_SCRIPT}. Set FUTUREFLU_MUTATION_COMPONENTS_SCRIPT "
            "or place analyze_mutation_components.py under "
            "reference/predictions/risk_components/mutation_components/."
        )
    comp = load_module(
        "reference_component",
        MUTATION_COMPONENTS_SCRIPT,
    )

    frames: List[pd.DataFrame] = []
    for subtype in SUBTYPES:
        cfg = read_pre_config(subtype)
        eve_prefix = cfg["evescape_prefix"]
        evescape_dir = pre_path(cfg["evescape_dir"])
        sequence_df = pd.read_csv(sequence_csv_path(subtype)).rename(
            columns={"accession number": "accession_number"}
        )
        sequence_df["collection_date"] = pd.to_datetime(sequence_df["collection_date"])
        sequence_df["submission_date"] = pd.to_datetime(sequence_df["submission_date"])
        antigenic_novelty_df = pd.read_csv(
            RISK_OUT / "antigenic_novelty" / f"strain_antigenic_novelty_{subtype}.csv"
        )

        for hemi_lower, hemisphere in [("north", "North"), ("south", "South")]:
            for year in YEARS:
                print(f"[component] {subtype} {hemi_lower} {year}")
                date_str = f"{year}0131" if hemi_lower == "north" else f"{year - 1}0831"
                prefix = f"{subtype}_{hemisphere}_{year}"
                linear_dir = LINEAR_OUT / f"{subtype}_{hemisphere}" / str(year)

                risk_mutations_path = linear_dir / f"{prefix}_distribution.csv"
                prediction_path = linear_dir / f"{prefix}_mutations.csv"
                mutations_path = evescape_dir / f"{eve_prefix}_evescape_{date_str}.csv"
                sites_path = evescape_dir / f"{eve_prefix}_evescape_sites_{date_str}.csv"

                df = pd.read_csv(risk_mutations_path)
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
                upper_bound = q3 + 3 * iqr
                lower_bound = q1 - 3 * iqr
                non_outliers = df_filtered[
                    (df_filtered["mutation_count"] <= upper_bound)
                    & (df_filtered["mutation_count"] >= lower_bound)
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
                site_awcn = {
                    str(site): val for site, val in tmp_awcn.groupby("i")["_awcn"].mean().items()
                }

                ef_min = mutations_df["fitness_eve"].min()
                mutation_ef = {
                    f"{row['i']}{row['mut']}": row["fitness_eve"] - ef_min
                    for _, row in mutations_df.iterrows()
                }
                tmp_ef = mutations_df.copy()
                tmp_ef["_ef"] = tmp_ef["fitness_eve"] - ef_min
                site_ef = {str(site): val for site, val in tmp_ef.groupby("i")["_ef"].mean().items()}

                mutation_prevalence = dict(
                    zip(prediction_df["risk_mutation"], prediction_df["delta"])
                )
                min_prev = min(mutation_prevalence.values()) if mutation_prevalence else 0
                mutation_prevalence = {k: v - min_prev for k, v in mutation_prevalence.items()}

                antigenic_tmp = antigenic_novelty_df.copy()
                antigenic_tmp["_an_norm"] = antigenic_tmp.groupby("season")[
                    "antigenic_novelty"
                ].transform(lambda x: x - x.min())
                antigenic_novelty_dict = dict(
                    zip(antigenic_tmp["accession_number"], antigenic_tmp["_an_norm"])
                )

                results = []
                for _, row in non_outliers.iterrows():
                    mutation_group = row["risk_mutation_group"]
                    count = row["mutation_count"]

                    total_escape = comp.calculate_total_escape_value(
                        mutation_group,
                        mutation_escape,
                        site_escape,
                        subtype,
                        sites_df,
                        mutations_df,
                    )
                    predicted_prevalence = comp.calculate_prevalence(
                        mutation_group, mutation_prevalence
                    )

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
                    seq_count = len(matching_seqs)
                    clade = comp.get_clade_info_from_matching(matching_seqs)
                    antigenic_value = comp.calculate_antigenic_novelty_from_matching(
                        matching_seqs, antigenic_novelty_dict
                    )
                    dch_value = comp.calculate_metric_value(
                        mutation_group, mutation_dch, site_dch, subtype, sites_df, mutations_df
                    )
                    awcn_value = comp.calculate_metric_value(
                        mutation_group, mutation_awcn, site_awcn, subtype, sites_df, mutations_df
                    )
                    ef_value = comp.calculate_metric_value(
                        mutation_group, mutation_ef, site_ef, subtype, sites_df, mutations_df
                    )

                    results.append(
                        {
                            "subtype": subtype,
                            "hemisphere": hemi_lower,
                            "year": year,
                            "risk_mutation_group": mutation_group,
                            "clade": clade,
                            "mutation_count": count,
                            "mutation_group_seq_count": seq_count,
                            "total_escape": total_escape,
                            "predicted_prevalence": predicted_prevalence,
                            "mutual_information": mutual_info,
                            "dissimilarity_charge_hydro": dch_value,
                            "accessibility_wcn": awcn_value,
                            "fitness_eve": ef_value,
                            "antigenic_novelty": antigenic_value,
                        }
                    )

                results_df = pd.DataFrame(results)
                results_df = comp.filter_random_single_mutations(results_df)
                frames.append(results_df)

    if not frames:
        raise RuntimeError("component stage produced no rows")

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
    out_path = RISK_OUT / "mutation_components" / "risk_mutation_group_component.csv"
    root_copy = RISK_OUT / "risk_mutation_group_component.csv"
    all_results[columns].to_csv(out_path, index=False)
    all_results[columns].to_csv(root_copy, index=False)
    print(f"[component] wrote {out_path} shape={all_results.shape}")


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


def build_label_and_count_inputs() -> None:
    count_dir = DATA_OUT / "clade_counts"
    rank_dir = DATA_OUT / "futureflu_rank"
    count_dir.mkdir(parents=True, exist_ok=True)
    rank_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for subtype in SUBTYPES:
        seq_df = pd.read_csv(sequence_csv_path(subtype))
        seq_df["collection_date"] = pd.to_datetime(seq_df["collection_date"])
        seq_df["submission_date"] = pd.to_datetime(seq_df["submission_date"])
        count_rows = []

        for year in range(2012, 2025):
            for hemisphere in ("south", "north"):
                start, end = target_season_window(year, hemisphere)
                collection_mask = (
                    (seq_df["collection_date"] >= start)
                    & (seq_df["collection_date"] < end)
                )
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

    label_path = rank_dir / "circulating_clade.csv"
    pd.DataFrame(labels).to_csv(label_path, index=False)
    print(f"[labels] wrote {label_path} rows={len(labels)}")


def write_local_script(src: Path, dst: Path, assignments: Dict[str, Path]) -> None:
    text = src.read_text(encoding="utf-8")
    text = re.sub(
        r"risk_mutation_group_component_[vV]\d+\.csv",
        "risk_mutation_group_component.csv",
        text,
    )
    if "from pathlib import Path" not in text:
        text, count = re.subn(
            r"(?m)^import os\s*$",
            "import os\nfrom pathlib import Path",
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"cannot add pathlib import in {src}")
    replacements = {
        "OUT_DIR": (
            "PROJECT_ROOT = Path(__file__).resolve().parents[1]\n"
            "OUT_DIR = str(PROJECT_ROOT / 'outputs' / 'predictions' / 'risk_components')"
        ),
        "LABEL_PATH": (
            "LABEL_PATH = str(PROJECT_ROOT / 'data' / 'futureflu_rank' / "
            "'circulating_clade.csv')"
        ),
        "COUNT_DIR": "COUNT_DIR = str(PROJECT_ROOT / 'data' / 'clade_counts')",
    }
    for name in assignments:
        replacement = replacements[name]
        text, count = re.subn(
            rf"(?m)^{re.escape(name)}\s*=\s*(?:r|u|f|fr|rf)?['\"].*?['\"]\s*$",
            replacement,
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"cannot replace {name} assignment in {src}")
    dst.write_text(text, encoding="utf-8")


def run_auxiliary_scripts() -> None:
    ensure_dirs()
    acc_dst = REPRO_ROOT / "scripts" / "calculate_clade_accuracy.py"
    combine_dst = REPRO_ROOT / "scripts" / "combine_clade_components.py"

    # English: Run packaged helpers against packaged compact inputs.
    # 中文：直接使用包内辅助脚本和精简输入，不重新生成或读取包外 reference。
    required_paths = [
        acc_dst,
        combine_dst,
        DATA_OUT / "futureflu_rank" / "circulating_clade.csv",
        DATA_OUT / "clade_counts",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "packaged auxiliary inputs are missing: " + ", ".join(missing)
        )

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
    print("[aux] running clade component accuracy")
    subprocess.run(acc_cmd, check=True, cwd=str(REPRO_ROOT))
    print("[aux] running clade component combine")
    subprocess.run(combine_cmd, check=True, cwd=str(REPRO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce FutureFlu R1 reference outputs.")
    parser.add_argument(
        "stage",
        choices=["sequences", "linear", "component", "aux", "all"],
        help="stage to run",
    )
    parser.add_argument("--sequence-processes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processes = args.sequence_processes if args.sequence_processes > 0 else None
    if args.stage in {"sequences", "all"}:
        prepare_sequences(processes)
    if args.stage in {"linear", "all"}:
        run_linear()
    if args.stage in {"component", "all"}:
        run_component()
    if args.stage in {"aux", "all"}:
        run_auxiliary_scripts()


if __name__ == "__main__":
    main()
