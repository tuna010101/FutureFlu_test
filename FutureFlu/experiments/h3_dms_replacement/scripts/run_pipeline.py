#!/usr/bin/env python3
"""Run the H3 DMS replacement sensitivity analysis.

English: Recompute H3N2 escape components with alternative DMS score mappings.
中文：使用不同 DMS 分数映射重新计算 H3N2 免疫逃逸组件。
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPRO_ROOT = Path(__file__).resolve().parents[3]


def env_path(name: str, default: Path | None = None) -> Path | None:
    configured = os.environ.get(name)
    if configured:
        return Path(configured).expanduser().resolve()
    return default


def configured_workspace() -> Path:
    """Return only an explicitly configured source workspace.

    English: Prefer an explicitly configured external workspace.
    中文：仅使用显式配置的外部工作区。
    """
    return env_path("FUTUREFLU_WORKSPACE_ROOT", REPRO_ROOT / "external" / "workspace")


WORKSPACE = configured_workspace()
PRE_ROOT = env_path("FUTUREFLU_SOURCE_ROOT", REPRO_ROOT / "external" / "inputs")
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DMS_ROOT = EXPERIMENT_ROOT / "inputs" / "dms"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"

LINEAR_OUT = (
    REPRO_ROOT
    / "outputs"
    / "predictions"
    / "linear"
    / "results"
)
RISK_OUT = (
    REPRO_ROOT
    / "outputs"
    / "predictions"
    / "risk_components"
)

REF_ROOT = env_path(
    "FUTUREFLU_REFERENCE_ROOT",
    WORKSPACE / "reference" / "predictions",
)
REF_RISK = REF_ROOT / "risk_components"
MUTATION_COMPONENTS_SCRIPT = env_path(
    "FUTUREFLU_MUTATION_COMPONENTS_SCRIPT",
    REF_RISK / "mutation_components" / "analyze_mutation_components.py",
)
TRUTH_CSV = env_path(
    "FUTUREFLU_TRUTH_CSV",
    REPRO_ROOT / "data" / "futureflu_rank" / "circulating_clade.csv",
)

SUBTYPE = "H3N2"
YEARS = tuple(range(2013, 2025))
HEMISPHERES = (("north", "North"), ("south", "South"))
METRICS_ALL = [
    "total_escape",
    "predicted_prevalence",
    "mutual_information",
    "dissimilarity_charge_hydro",
    "accessibility_wcn",
    "fitness_eve",
    "antigenic_novelty",
]
METRICS_CORE = ["total_escape", "predicted_prevalence", "mutual_information"]
COMBINATIONS = [
    ("E", ["total_escape"]),
    ("G", ["predicted_prevalence"]),
    ("D", ["mutual_information"]),
    ("E+G", ["total_escape", "predicted_prevalence"]),
    ("E+D", ["total_escape", "mutual_information"]),
    ("G+D", ["predicted_prevalence", "mutual_information"]),
    ("E+G+D", ["total_escape", "predicted_prevalence", "mutual_information"]),
]
T_VALUES = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    dtype=float,
)
STRATEGIES = (
    "mutation_min_shift",
    "mutation_raw_site_mean",
    "mutation_positive_site_max",
    "site_mean",
    "site_positive_sum",
    "site_max_positive",
)


@dataclass(frozen=True)
class DmsDataset:
    name: str
    mutation_raw: Dict[str, float]
    mutation_positive: Dict[str, float]
    mutation_min_shift: Dict[str, float]
    site_mean: Dict[int, float]
    site_mean_positive: Dict[int, float]
    site_min_shift: Dict[int, float]
    site_positive_sum: Dict[int, float]
    site_max_positive: Dict[int, float]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_pre_config() -> dict:
    return json.loads((PRE_ROOT / "configs" / "h3n2_pre2024.json").read_text(encoding="utf-8"))


def pre_path(rel: str) -> Path:
    return (PRE_ROOT / rel).resolve()


def sequence_csv_path() -> Path:
    return REPRO_ROOT / "data" / "H3N2_HA_sequence_20250131.csv"


def normalize_site_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out["site_sequential"] = pd.to_numeric(out["site_sequential"], errors="coerce")
    out["site_avg_escape_mean"] = pd.to_numeric(out["site_avg_escape_mean"], errors="coerce")
    out = out.dropna(subset=["site_sequential", "site_avg_escape_mean"]).copy()
    out["site_sequential"] = out["site_sequential"].astype(int)
    return out


def normalize_mutation_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out["site_sequential"] = pd.to_numeric(out["site_sequential"], errors="coerce")
    out["avg_escape_mean"] = pd.to_numeric(out["avg_escape_mean"], errors="coerce")
    out["mutant"] = out["mutant"].astype(str).str.upper().str.strip()
    out = out.dropna(subset=["site_sequential", "avg_escape_mean"]).copy()
    out["site_sequential"] = out["site_sequential"].astype(int)
    out["mutation_key"] = out["site_sequential"].astype(str) + out["mutant"]
    return out


def load_dms_dataset(name: str) -> DmsDataset:
    mutation_df = normalize_mutation_df(pd.read_csv(DMS_ROOT / f"{name}_mutation_escape.csv"))
    site_df = normalize_site_df(pd.read_csv(DMS_ROOT / f"{name}_site_escape.csv"))

    mutation_raw = dict(zip(mutation_df["mutation_key"], mutation_df["avg_escape_mean"]))
    mutation_positive = {key: max(float(value), 0.0) for key, value in mutation_raw.items()}
    mutation_min = float(mutation_df["avg_escape_mean"].min())
    mutation_min_shift = {key: float(value) - mutation_min for key, value in mutation_raw.items()}

    site_mean = dict(zip(site_df["site_sequential"], site_df["site_avg_escape_mean"]))
    site_mean_positive = {int(site): max(float(value), 0.0) for site, value in site_mean.items()}
    site_min = float(site_df["site_avg_escape_mean"].min())
    site_min_shift = {int(site): float(value) - site_min for site, value in site_mean.items()}

    site_positive_sum = (
        mutation_df.assign(pos_value=mutation_df["avg_escape_mean"].clip(lower=0.0))
        .groupby("site_sequential", as_index=True)["pos_value"]
        .sum()
        .to_dict()
    )
    site_max_positive = (
        mutation_df.assign(pos_value=mutation_df["avg_escape_mean"].clip(lower=0.0))
        .groupby("site_sequential", as_index=True)["pos_value"]
        .max()
        .to_dict()
    )

    return DmsDataset(
        name=name,
        mutation_raw={str(k): float(v) for k, v in mutation_raw.items()},
        mutation_positive={str(k): float(v) for k, v in mutation_positive.items()},
        mutation_min_shift={str(k): float(v) for k, v in mutation_min_shift.items()},
        site_mean={int(k): float(v) for k, v in site_mean.items()},
        site_mean_positive={int(k): float(v) for k, v in site_mean_positive.items()},
        site_min_shift={int(k): float(v) for k, v in site_min_shift.items()},
        site_positive_sum={int(k): float(v) for k, v in site_positive_sum.items()},
        site_max_positive={int(k): float(v) for k, v in site_max_positive.items()},
    )


def mutation_site(mutation: str) -> int:
    return int("".join(filter(str.isdigit, str(mutation))))


def mutation_key(mutation: str) -> str:
    text = str(mutation).strip()
    return f"{mutation_site(text)}{text[-1].upper()}"


def score_group(group: str, dataset: DmsDataset, strategy: str) -> float:
    if pd.isna(group):
        return 0.0
    total = 0.0
    for mutation in map(str.strip, str(group).split(",")):
        if not mutation:
            continue
        site = mutation_site(mutation)
        key = mutation_key(mutation)
        if strategy == "mutation_min_shift":
            total += dataset.mutation_min_shift.get(key, dataset.site_min_shift.get(site, 0.0))
        elif strategy == "mutation_raw_site_mean":
            total += dataset.mutation_raw.get(key, dataset.site_mean.get(site, 0.0))
        elif strategy == "mutation_positive_site_max":
            total += dataset.mutation_positive.get(key, dataset.site_max_positive.get(site, 0.0))
        elif strategy == "site_mean":
            total += dataset.site_mean.get(site, 0.0)
        elif strategy == "site_positive_sum":
            total += dataset.site_positive_sum.get(site, 0.0)
        elif strategy == "site_max_positive":
            total += dataset.site_max_positive.get(site, 0.0)
        else:
            raise ValueError(f"unknown strategy: {strategy}")
    return total


def build_base_component_rows() -> pd.DataFrame:
    cache_path = RESULTS_ROOT / "base_h3_component_without_total_escape.csv"

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    if not MUTATION_COMPONENTS_SCRIPT.exists():
        raise FileNotFoundError(
            "mutation-components script not found: "
            f"{MUTATION_COMPONENTS_SCRIPT}. Set FUTUREFLU_MUTATION_COMPONENTS_SCRIPT."
        )
    comp = load_module(
        "reference_component",
        MUTATION_COMPONENTS_SCRIPT,
    )
    cfg = read_pre_config()
    eve_prefix = cfg["evescape_prefix"]
    evescape_dir = pre_path(cfg["evescape_dir"])
    sequence_df = pd.read_csv(sequence_csv_path()).rename(columns={"accession number": "accession_number"})
    sequence_df["collection_date"] = pd.to_datetime(sequence_df["collection_date"])
    sequence_df["submission_date"] = pd.to_datetime(sequence_df["submission_date"])
    antigenic_novelty_df = pd.read_csv(
        RISK_OUT / "antigenic_novelty" / "strain_antigenic_novelty_H3N2.csv"
    )

    frames: List[pd.DataFrame] = []
    for hemi_lower, hemisphere in HEMISPHERES:
        for year in YEARS:
            print(f"[base] H3N2 {hemi_lower} {year}", flush=True)
            date_str = f"{year}0131" if hemi_lower == "north" else f"{year - 1}0831"
            prefix = f"H3N2_{hemisphere}_{year}"
            linear_dir = LINEAR_OUT / f"H3N2_{hemisphere}" / str(year)

            df = pd.read_csv(linear_dir / f"{prefix}_distribution.csv")
            prediction_df = pd.read_csv(linear_dir / f"{prefix}_mutations.csv")
            mutations_df = pd.read_csv(evescape_dir / f"{eve_prefix}_evescape_{date_str}.csv")

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
            antigenic_novelty_dict = dict(zip(antigenic_tmp["accession_number"], antigenic_tmp["_an_norm"]))

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
                seq_count = len(matching_seqs)
                rows.append(
                    {
                        "subtype": SUBTYPE,
                        "hemisphere": hemi_lower,
                        "year": year,
                        "risk_mutation_group": mutation_group,
                        "clade": comp.get_clade_info_from_matching(matching_seqs),
                        "mutation_count": count,
                        "mutation_group_seq_count": seq_count,
                        "predicted_prevalence": comp.calculate_prevalence(
                            mutation_group, mutation_prevalence
                        ),
                        "mutual_information": mutual_info,
                        "dissimilarity_charge_hydro": comp.calculate_metric_value(
                            mutation_group, mutation_dch, site_dch, SUBTYPE, mutations_df, mutations_df
                        ),
                        "accessibility_wcn": comp.calculate_metric_value(
                            mutation_group, mutation_awcn, site_awcn, SUBTYPE, mutations_df, mutations_df
                        ),
                        "fitness_eve": comp.calculate_metric_value(
                            mutation_group, mutation_ef, site_ef, SUBTYPE, mutations_df, mutations_df
                        ),
                        "antigenic_novelty": comp.calculate_antigenic_novelty_from_matching(
                            matching_seqs, antigenic_novelty_dict
                        ),
                    }
                )

            result_df = comp.filter_random_single_mutations(pd.DataFrame(rows))
            frames.append(result_df)

    if not frames:
        raise RuntimeError("no H3N2 base component rows generated")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cache_path, index=False)
    return out


_CLADE_RE = re.compile(r"([^\(,]+?)\s*\((\d+\.?\d*)%\)")


def parse_clade_string(clade_str: object) -> list[tuple[str, float]]:
    if pd.isna(clade_str) or str(clade_str).strip().lower() in {"unknown", ""}:
        return []
    parsed = []
    for name, pct_str in _CLADE_RE.findall(str(clade_str)):
        name = name.strip()
        if name.lower() not in {"unassigned", "unknown"}:
            parsed.append((name, float(pct_str) / 100.0))
    return parsed


def compute_max_method(component_df: pd.DataFrame) -> pd.DataFrame:
    # English: Match the reference combine-stage H3N2 pre-filter.
    # 中文：与参考 combine 阶段的 H3N2 预筛选规则保持一致。
    component_df = component_df[component_df["mutation_group_seq_count"] >= 10].copy()
    rows = []
    for _, row in component_df.iterrows():
        parsed = parse_clade_string(row["clade"])
        if not parsed:
            continue
        dominant_clade = max(parsed, key=lambda item: item[1])[0]
        rec = {
            "subtype": row["subtype"],
            "hemisphere": row["hemisphere"],
            "year": row["year"],
            "clade_single": dominant_clade,
        }
        for metric in METRICS_ALL:
            rec[f"fit_{metric}"] = row.get(metric, np.nan)
        rows.append(rec)
    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return pd.DataFrame(columns=["subtype", "hemisphere", "year", "clade_single"])
    return (
        tmp.groupby(["subtype", "hemisphere", "year", "clade_single"], as_index=False)
        .agg({f"fit_{metric}": "max" for metric in METRICS_ALL})
    )


def z_score_array(vals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    mu = np.nanmean(arr)
    std = np.nanstd(arr, ddof=1)
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - mu) / std


def safe_sigmoid_array(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def load_clade_freq() -> pd.DataFrame:
    path = REPRO_ROOT / "data" / "clade_counts" / "submission_collection_clade_count_h3n2.csv"
    cnt = pd.read_csv(path)
    cnt["hemisphere"] = cnt["hemisphere"].str.lower()
    total = cnt.groupby(["year", "hemisphere"])["submission_count"].sum().rename("total_sc")
    cnt = cnt.merge(total.reset_index(), on=["year", "hemisphere"])
    cnt["xi_t"] = np.where(cnt["total_sc"] > 0, cnt["submission_count"] / cnt["total_sc"], np.nan)
    cnt["subtype"] = SUBTYPE
    return cnt[["subtype", "year", "hemisphere", "clade", "xi_t"]]


def find_best_temperatures(past_data: list[dict], combo_metrics: list[str]) -> tuple[dict, float]:
    if not past_data:
        return {metric: 1.0 for metric in combo_metrics}, np.nan
    grid = np.array(list(product(T_VALUES, repeat=len(combo_metrics))), dtype=float)
    total_loss = np.zeros(len(grid), dtype=float)
    valid_count = 0
    for data in past_data:
        valid_count += 1
        z_mat = data["z_mat"]
        xi_prev = data["xi_prev"]
        act_freq = data["act_freq"]
        log_sig = np.log(safe_sigmoid_array(z_mat[np.newaxis, :, :] / grid[:, :, np.newaxis]))
        fitness = log_sig.sum(axis=1)
        numerator = xi_prev[np.newaxis, :] * np.exp(fitness)
        z_total = numerator.sum(axis=1, keepdims=True)
        pred_freq = np.where(z_total > 0, numerator / z_total, 0.0)
        loss = np.abs(pred_freq - act_freq[np.newaxis, :]).sum(axis=1)
        loss[z_total.flatten() <= 0] = np.inf
        total_loss += loss
    mean_loss = total_loss / valid_count
    best_idx = int(np.argmin(mean_loss))
    return (
        {combo_metrics[i]: float(grid[best_idx, i]) for i in range(len(combo_metrics))},
        float(mean_loss[best_idx]),
    )


def compute_pre_act(max_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq_df = load_clade_freq()
    pre_act: Dict[tuple, dict] = {}
    temps: list[dict] = []

    for combo_name, combo_metrics in COMBINATIONS:
        sub_clade = max_df[max_df["subtype"] == SUBTYPE]
        all_seasons = set()
        for hemi in sub_clade["hemisphere"].unique():
            for year in sub_clade.loc[sub_clade["hemisphere"] == hemi, "year"].unique():
                all_seasons.add((int(year), str(hemi).lower()))
        sorted_seasons = sorted(all_seasons, key=lambda hy: (hy[0], 0 if hy[1] == "south" else 1))

        per_season = {}
        for year, hemi in sorted_seasons:
            grp = sub_clade[
                (sub_clade["hemisphere"].str.lower() == hemi) & (sub_clade["year"] == year)
            ].reset_index(drop=True)
            clades = grp["clade_single"].to_numpy(copy=True)
            z_mat = np.vstack(
                [z_score_array(grp[f"fit_{metric}"].fillna(0.0).to_numpy()) for metric in combo_metrics]
            )
            prev_freq = freq_df[(freq_df["hemisphere"] == hemi) & (freq_df["year"] == year - 1)]
            clade_to_prev = dict(zip(prev_freq["clade"], prev_freq["xi_t"]))
            xi_prev = np.array([np.nan_to_num(clade_to_prev.get(clade, 0.0), nan=0.0) for clade in clades])

            cur_freq = freq_df[(freq_df["hemisphere"] == hemi) & (freq_df["year"] == year)]
            cur_sub = cur_freq[cur_freq["clade"].isin(set(clades))]
            raw_sum = cur_sub["xi_t"].fillna(0.0).sum()
            if cur_sub.empty or raw_sum <= 0:
                act_freq = None
            else:
                clade_to_cur = dict(zip(cur_sub["clade"], cur_sub["xi_t"].fillna(0.0)))
                act_arr = np.array([clade_to_cur.get(clade, 0.0) for clade in clades])
                act_freq = act_arr / act_arr.sum()
            per_season[(year, hemi)] = {
                "clades": clades,
                "z_mat": z_mat,
                "xi_prev": xi_prev,
                "act_freq": act_freq,
            }

        for idx, (year, hemi) in enumerate(sorted_seasons):
            if idx == 0:
                temp = {metric: 1.0 for metric in combo_metrics}
                best_loss = np.nan
            else:
                history = sorted_seasons[: max(0, idx - 1)]
                past = [
                    per_season[season]
                    for season in history
                    if per_season[season]["act_freq"] is not None
                ]
                temp, best_loss = find_best_temperatures(past, combo_metrics)

            data = per_season[(year, hemi)]
            if len(data["clades"]) == 0:
                continue
            t_col = np.array([temp[metric] for metric in combo_metrics], dtype=float)[:, np.newaxis]
            fitness = np.log(safe_sigmoid_array(data["z_mat"] / t_col)).sum(axis=0)
            numerator = data["xi_prev"] * np.exp(fitness)
            if numerator.sum() > 0:
                pred_freq = numerator / numerator.sum()
            else:
                pred_freq = np.full(len(numerator), np.nan)

            temps.append(
                {
                    "subtype": SUBTYPE,
                    "hemisphere": hemi,
                    "year": year,
                    "combo": combo_name,
                    "temperatures": ";".join(f"{metric}={temp[metric]}" for metric in combo_metrics),
                    "mean_loss": best_loss,
                    "top_fit_clade": data["clades"][int(np.argmax(fitness))],
                    "top_freq_clade": data["clades"][int(np.nanargmax(pred_freq))]
                    if np.isfinite(pred_freq).any()
                    else "",
                }
            )

            for i, clade in enumerate(data["clades"]):
                key = (SUBTYPE, year, hemi, clade)
                if key not in pre_act:
                    pre_act[key] = {
                        "subtype": SUBTYPE,
                        "year": year,
                        "hemisphere": hemi,
                        "clade": clade,
                        "freq_prev": round(float(data["xi_prev"][i]), 6),
                    }
                pre_act[key][f"{combo_name}_pre_fit"] = round(float(fitness[i]), 6)
                pre_act[key][f"{combo_name}_pre_freq"] = (
                    round(float(pred_freq[i]), 6) if np.isfinite(pred_freq[i]) else np.nan
                )

    pre_act_df = pd.DataFrame(list(pre_act.values()))
    pre_act_cols = (
        ["subtype", "year", "hemisphere", "clade", "freq_prev"]
        + [col for combo, _ in COMBINATIONS for col in (f"{combo}_pre_fit", f"{combo}_pre_freq")]
    )
    for col in pre_act_cols:
        if col not in pre_act_df.columns:
            pre_act_df[col] = np.nan
    pre_act_df = pre_act_df[pre_act_cols].sort_values(["subtype", "year", "hemisphere", "clade"])
    return pre_act_df.reset_index(drop=True), pd.DataFrame(temps)


def load_truth_long() -> pd.DataFrame:
    if not TRUTH_CSV.exists():
        raise FileNotFoundError(f"missing truth table: {TRUTH_CSV}")
    truth = pd.read_csv(TRUTH_CSV)
    if {"subtype", "hemisphere", "year", "clade"}.issubset(truth.columns):
        rows = truth[truth["subtype"].astype(str) == SUBTYPE].copy()
        rows["hemisphere"] = rows["hemisphere"].astype(str).str.lower()
        rows = rows.rename(columns={"clade": "truth_clade"})
        return rows[["subtype", "year", "hemisphere", "truth_clade"]].reset_index(drop=True)

    truth = truth.rename(columns={"Unnamed: 0": "truth_subtype"})
    rows = []
    for _, row in truth.iterrows():
        subtype = "Victoria" if row["truth_subtype"] == "B" else str(row["truth_subtype"])
        if subtype != SUBTYPE:
            continue
        for col in truth.columns[1:]:
            year_text, hemi = col.split()
            rows.append(
                {
                    "subtype": subtype,
                    "year": int(year_text),
                    "hemisphere": hemi,
                    "truth_clade": row[col],
                }
            )
    return pd.DataFrame(rows)


def evaluate_pre_act(pre_act_df: pd.DataFrame, scenario: str, dataset: str, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = load_truth_long()
    detail_frames = []
    summary_rows = []
    for combo_name, _ in COMBINATIONS:
        for method in ("fit", "freq"):
            score_col = f"{combo_name}_pre_{method}"
            tmp = pre_act_df[["subtype", "year", "hemisphere", "clade", score_col]].dropna(subset=[score_col]).copy()
            if tmp.empty:
                pred = pd.DataFrame(columns=["subtype", "year", "hemisphere", "pred_clade", "score"])
            else:
                idx = tmp.groupby(["subtype", "year", "hemisphere"], sort=False)[score_col].idxmax()
                pred = tmp.loc[idx, ["subtype", "year", "hemisphere", "clade", score_col]].rename(
                    columns={"clade": "pred_clade", score_col: "score"}
                )
            merged = truth.merge(pred, on=["subtype", "year", "hemisphere"], how="left")
            merged["correct"] = merged["truth_clade"].eq(merged["pred_clade"])
            merged["scenario"] = scenario
            merged["dataset"] = dataset
            merged["strategy"] = strategy
            merged["combo"] = combo_name
            merged["method"] = method
            detail_frames.append(merged)

            total = len(merged)
            correct = int(merged["correct"].sum())
            summary_rows.append(
                {
                    "scenario": scenario,
                    "dataset": dataset,
                    "strategy": strategy,
                    "combo": combo_name,
                    "method": method,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total else np.nan,
                    "available_predictions": int(merged["pred_clade"].notna().sum()),
                }
            )

    return pd.DataFrame(summary_rows), pd.concat(detail_frames, ignore_index=True)


def baseline_pre_act() -> pd.DataFrame:
    path = RISK_OUT / "component_combinations" / "clade_pre_act_Twindow.csv"
    df = pd.read_csv(path)
    return df[df["subtype"] == SUBTYPE].copy()


def run_one(base_df: pd.DataFrame, dataset: DmsDataset, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = RESULTS_ROOT / f"{dataset.name}_{strategy}"
    run_dir.mkdir(parents=True, exist_ok=True)

    component_df = base_df.copy()
    component_df["total_escape"] = component_df["risk_mutation_group"].apply(
        lambda group: score_group(group, dataset, strategy)
    )
    component_cols = [
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
    component_df[component_cols].to_csv(run_dir / "risk_mutation_group_component_H3N2.csv", index=False)

    max_df = compute_max_method(component_df)
    max_df.to_csv(run_dir / "clade_component_max_H3N2.csv", index=False)

    pre_act_df, temps_df = compute_pre_act(max_df)
    pre_act_df.to_csv(run_dir / "clade_pre_act_Twindow_H3N2.csv", index=False)
    temps_df.to_csv(run_dir / "temperatures_Twindow_H3N2.csv", index=False)

    summary, detail = evaluate_pre_act(
        pre_act_df, "dms_total_escape_replacement", dataset.name, strategy
    )
    detail.to_csv(run_dir / "predictions_vs_truth_H3N2.csv", index=False)
    summary.to_csv(run_dir / "accuracy_vs_truth_H3N2.csv", index=False)
    return summary, detail


def coverage_summary(base_df: pd.DataFrame, datasets: dict[str, DmsDataset]) -> pd.DataFrame:
    unique_mutations = sorted(
        {
            mutation.strip()
            for group in base_df["risk_mutation_group"].dropna()
            for mutation in str(group).split(",")
            if mutation.strip()
        }
    )
    positions = sorted({mutation_site(mut) for mut in unique_mutations})
    rows = []
    for name, dataset in datasets.items():
        exact = sum(mutation_key(mut) in dataset.mutation_raw for mut in unique_mutations)
        site = sum(pos in dataset.site_mean for pos in positions)
        rows.append(
            {
                "dataset": name,
                "unique_mutations_total": len(unique_mutations),
                "unique_sites_total": len(positions),
                "unique_mutations_exact_covered": exact,
                "unique_sites_covered": site,
                "unique_mutation_coverage": exact / len(unique_mutations),
                "unique_site_coverage": site / len(positions),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # English: This batch script has no CLI options; reject accidental arguments.
    # 中文：该批处理脚本不提供命令行选项；拒绝误传参数。
    if len(sys.argv) != 1:
        raise SystemExit("run_pipeline.py does not accept command-line arguments")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    base_df = build_base_component_rows()
    datasets = {name: load_dms_dataset(name) for name in ("hk19", "ma22", "pe09")}
    coverage_summary(base_df, datasets).to_csv(RESULTS_ROOT / "dms_coverage_summary.csv", index=False)

    baseline_summary, baseline_detail = evaluate_pre_act(
        baseline_pre_act(), "baseline_current_evescape", "baseline", "current_total_escape"
    )
    all_summary = [baseline_summary]
    all_detail = [baseline_detail]

    for dataset in datasets.values():
        for strategy in STRATEGIES:
            print(f"[run] {dataset.name} {strategy}", flush=True)
            summary, detail = run_one(base_df, dataset, strategy)
            all_summary.append(summary)
            all_detail.append(detail)

    summary_df = pd.concat(all_summary, ignore_index=True)
    detail_df = pd.concat(all_detail, ignore_index=True)
    summary_df = summary_df.sort_values(
        ["combo", "method", "accuracy", "scenario", "dataset", "strategy"],
        ascending=[True, True, False, True, True, True],
    )
    summary_df.to_csv(RESULTS_ROOT / "accuracy_vs_truth_summary.csv", index=False)
    detail_df.to_csv(RESULTS_ROOT / "predictions_vs_truth_long.csv", index=False)

    best = summary_df.sort_values(["accuracy", "correct"], ascending=[False, False]).head(20)
    print("\nTop accuracy rows:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
