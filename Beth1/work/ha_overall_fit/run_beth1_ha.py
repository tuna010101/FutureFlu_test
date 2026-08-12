#!/usr/bin/env python3
"""HA-only Beth-1 overall-fit-once analysis (no climate).

English: Fit theta/history windows on HA sequences and recommend wild-type
strains for each subtype / hemisphere / season.
中文：在 HA 序列上拟合 theta 与历史窗口，并为各亚型/半球/季节推荐野毒株。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT


def _env_path(name: str, default: Path) -> Path:
    """Resolve an optional path environment variable.

    English: Keep relative paths relative to PACKAGE_ROOT; do not force
    absolute paths via Path.resolve().
    中文：相对路径相对于 PACKAGE_ROOT；不通过 Path.resolve() 强制转为绝对路径。
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return PACKAGE_ROOT / path
    return path


DATA_ROOT = _env_path("BETH1_DATA_ROOT", PACKAGE_ROOT / "data")
INPUT_ROOT = _env_path("BETH1_INPUT_ROOT", PACKAGE_ROOT)

from sequence_table import generate_sequence_table


H3_HA_AB_MATURE = [
    122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144,
    145, 146, 150, 152, 168, 128, 129, 155, 156, 157, 158, 159, 160, 163,
    164, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198,
]
H3_HA_ALL_MATURE = [
    122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144,
    145, 146, 150, 152, 168, 128, 129, 155, 156, 157, 158, 159, 160, 163,
    164, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196, 197, 198, 44,
    45, 46, 47, 48, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294,
    297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312, 96, 102, 103,
    117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177, 179, 182, 201,
    203, 205, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218, 219, 226,
    227, 228, 229, 230, 240, 242, 244, 246, 247, 248, 57, 59, 62, 63, 67,
    75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262,
    265,
]


SUBTYPE_CONFIG = {
    "H3N2": {
        "truth_name": "H3N2",
        "fasta": "data/dataset/fasta/msa-H3N2-all-20250131-submission.fasta",
        "meta": "data/dataset/metadata/H3N2-all-20250131-submission.csv",
        "ha_length": 566,
        "ha1_start": 17,
        "ha1_end": 345,
        "antigenic_mature": H3_HA_ALL_MATURE,
        "predictor_mature": H3_HA_AB_MATURE,
    },
    "H1N1": {
        "truth_name": "H1N1",
        "fasta": "data/dataset/fasta/msa-pdm09-all-20250131-submission.fasta",
        "meta": "data/dataset/metadata/pdm09-all_20250131_submission.csv",
        "ha_length": 566,
        "ha1_start": 18,
        "ha1_end": 344,
        "epitope_file": DATA_ROOT / "H1N1_epitope.txt",
    },
    "Victoria": {
        "truth_name": "B",
        "fasta": "data/dataset/fasta/msa-FLUBV-all-20250131-submission.fasta",
        "meta": "data/dataset/metadata/BV-all-20250131-submission.csv",
        "ha_length": 585,
        "ha1_start": 16,
        "ha1_end": 362,
        "epitope_file": DATA_ROOT / "Victoria-epitope.txt",
    },
}

EPIDEMIC_FILES = {
    ("H1N1", "north"): DATA_ROOT / "H1N1_north_epidemic_data_2024.csv",
    ("H1N1", "south"): DATA_ROOT / "H1N1_south_epidemic_data_2024.csv",
    ("H3N2", "north"): DATA_ROOT / "H3N2_north_epidemic_data_2024.csv",
    ("H3N2", "south"): DATA_ROOT / "H3N2_south_epidemic_data_2024.csv",
    ("Victoria", "north"): DATA_ROOT / "Victoria_epidemic_data_2024.csv",
    ("Victoria", "south"): DATA_ROOT / "Victoria_south_epidemic_data_2024.csv",
}

YEARS = list(range(2013, 2025))
HEMISPHERES = ["north", "south"]
THETA_RANGE = [round(x, 1) for x in np.arange(0.5, 1.0, 0.1)]
H_RANGE = list(range(10))
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY-")
HA_EM_THETA = 0.5
PREDICTED_EM_THETA = 0.5


def parse_args() -> argparse.Namespace:
    """Parse CLI options. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run HA-only beth-1 overall-fit-once (no climate) analysis."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "work" / "ha_overall_fit" / "results",
    )
    parser.add_argument(
        "--subtypes",
        nargs="+",
        default=list(SUBTYPE_CONFIG),
        choices=list(SUBTYPE_CONFIG),
        help="Subtypes to run. Defaults to all subtypes.",
    )
    parser.add_argument(
        "--h1n1-epitope-file",
        type=Path,
        default=None,
        help="Optional mature-site epitope definition to use for H1N1.",
    )
    parser.add_argument("--processes", type=int, default=None)
    return parser.parse_args()


def read_mature_sites(path: Path) -> List[int]:
    """Read mature epitope site indices from a text file. / 从表位位点文本读取成熟位点编号。"""
    text = path.read_text(encoding="utf-8")
    return [int(x) for x in re.findall(r"\d+", text)]


def mature_to_alignment_sites(sites: Iterable[int], ha1_start: int) -> List[int]:
    """Map mature HA1 sites to alignment column indices. / 将成熟 HA1 位点映射为比对列索引。"""
    offset = ha1_start - 1
    return sorted({int(site) + offset for site in sites})


def load_site_sets(subtype: str) -> Tuple[List[int], List[int]]:
    """Return antigenic and predictor site sets for one subtype. / 返回某亚型的抗原位点与预测位点集合。"""
    cfg = SUBTYPE_CONFIG[subtype]
    if "epitope_file" in cfg:
        # H1N1_epitope.txt / Victoria-epitope.txt are already full-length / alignment
        # coordinates (Canton-like). Do NOT add (ha1_start-1); that double-shifts.
        # H3 keeps mature lists + ha1_start offset below.
        sites = read_mature_sites(Path(cfg["epitope_file"]))
        aligned = sorted({int(site) for site in sites})
        return aligned, aligned

    antigenic = mature_to_alignment_sites(cfg["antigenic_mature"], int(cfg["ha1_start"]))
    predictor = mature_to_alignment_sites(cfg["predictor_mature"], int(cfg["ha1_start"]))
    return antigenic, predictor


def prediction_window(year: int, hemisphere: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return the prediction-season date window. / 返回预测季的起止日期窗口。"""
    if hemisphere == "north":
        return pd.Timestamp(f"{year-1}-09-01"), pd.Timestamp(f"{year}-02-01")
    return pd.Timestamp(f"{year-1}-02-01"), pd.Timestamp(f"{year-1}-09-01")


def history_window(season: int, hemisphere: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return one historical season date window. / 返回单个历史季节的日期窗口。"""
    if hemisphere == "north":
        return pd.Timestamp(f"{season}-09-01"), pd.Timestamp(f"{season+1}-02-01")
    return pd.Timestamp(f"{season}-02-01"), pd.Timestamp(f"{season}-09-01")


def load_truth_table(path: Path) -> pd.DataFrame:
    """Load circulating-clade truth labels. / 加载 circulating-clade 真值标签。"""
    raw = pd.read_csv(path, index_col=0)
    records: List[dict] = []
    for subtype_name, row in raw.iterrows():
        for col, clade in row.items():
            tokens = str(col).split()
            if len(tokens) != 2:
                continue
            records.append(
                {
                    "truth_name": subtype_name,
                    "year": int(tokens[0]),
                    "hemisphere": tokens[1].lower(),
                    "truth_clade": str(clade).strip(),
                }
            )
    return pd.DataFrame(records)


def load_epidemic_rate(path: Path) -> pd.DataFrame:
    """Load positivity-rate epidemic tables. / 加载阳性率流行表。"""
    df = pd.read_csv(path)
    rename = {}
    if "season" in df.columns and "Season" not in df.columns:
        rename["season"] = "Season"
    if "h3rate" in df.columns and "Positivity_Rate" not in df.columns:
        rename["h3rate"] = "Positivity_Rate"
    if "rate" in df.columns and "Positivity_Rate" not in df.columns:
        rename["rate"] = "Positivity_Rate"
    df = df.rename(columns=rename)
    required = {"Season", "Positivity_Rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns after normalization: {sorted(missing)}")
    return df


def site_columns(seq_df: pd.DataFrame, ha_length: int) -> List[str]:
    """List HA site columns (X1..Xha_length). / 列出 HA 位点列（X1..Xha_length）。"""
    cols = []
    for col in seq_df.columns:
        if col.startswith("X") and col[1:].isdigit() and int(col[1:]) <= ha_length:
            cols.append(col)
    return cols


def build_prevalence(
    seq_df: pd.DataFrame,
    predict_year: int,
    hemisphere: str,
    sites: List[str],
) -> pd.DataFrame:
    """Build yearly site prevalences for selected sites.
    
    English: Restrict sequences to the prediction window and compute amino-acid frequencies.
    中文：按预测窗口筛选序列并计算氨基酸频率。
    """
    _, submission_cutoff = prediction_window(predict_year, hemisphere)
    seasons = sorted(int(y) for y in seq_df["season"].unique() if 2010 <= int(y) < predict_year)
    rows: List[pd.Series] = []

    for season in seasons:
        start, end = history_window(season, hemisphere)
        season_df = seq_df[
            (seq_df["collection_date"] >= start)
            & (seq_df["collection_date"] < end)
            & (seq_df["submission_date"] < submission_cutoff)
        ]
        if season_df.empty:
            continue

        row: Dict[str, float] = {}
        for col in sites:
            values = season_df[col].fillna("X").astype(str).str.upper()
            values = values[values != "X"]
            if values.empty:
                continue
            freq = values.value_counts(normalize=True)
            for aa in AMINO_ACIDS:
                row[f"{col}{aa}"] = float(freq.get(aa, 0.0))
        rows.append(pd.Series(row, name=season))

    if not rows:
        return pd.DataFrame()
    prev = pd.concat(rows, axis=1).T.fillna(0.0).sort_index()
    prev.index = prev.index.astype(int)
    return prev


def build_prevalence_all(
    seq_df: pd.DataFrame,
    hemisphere: str,
    sites: List[str],
) -> pd.DataFrame:
    """Build prevalences over all HA site columns.
    
    English: Same as build_prevalence but for every X* column.
    中文：对全部 X* 位点列构建流行度。
    """
    seasons = sorted(int(y) for y in seq_df["season"].unique() if 2010 <= int(y) <= max(YEARS))
    rows: List[pd.Series] = []

    for season in seasons:
        start, end = history_window(season, hemisphere)
        season_df = seq_df[
            (seq_df["collection_date"] >= start)
            & (seq_df["collection_date"] < end)
        ]
        if season_df.empty:
            continue

        row: Dict[str, float] = {}
        for col in sites:
            values = season_df[col].fillna("X").astype(str).str.upper()
            values = values[values != "X"]
            if values.empty:
                continue
            freq = values.value_counts(normalize=True)
            for aa in AMINO_ACIDS:
                row[f"{col}{aa}"] = float(freq.get(aa, 0.0))
        rows.append(pd.Series(row, name=season))

    if not rows:
        return pd.DataFrame()
    prev = pd.concat(rows, axis=1).T.fillna(0.0).sort_index()
    prev.index = prev.index.astype(int)
    return prev


def get_gmeasure(prev: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute g-measure summaries across theta values.
    
    English: Aggregate prevalence dynamics used to choose theta.
    中文：汇总用于选择 theta 的流行度动态指标。
    """
    if prev.empty:
        return pd.DataFrame(), {}

    values = prev.to_numpy(dtype=float)
    years = prev.index.tolist()
    results: Dict[str, np.ndarray] = {}
    tau_by_param: Dict[str, float] = {}
    n_years, n_cols = values.shape

    for theta in THETA_RANGE:
        for h in H_RANGE:
            mut = np.zeros_like(values, dtype=np.int8)
            transition_times: List[int] = []
            for col_idx in range(n_cols):
                series = values[:, col_idx]
                start = 0
                for r in range(n_years):
                    if series[r] >= theta and np.any(series[start : r + 1] == 0):
                        zero_positions = np.where(series[: r + 1] == 0)[0]
                        if zero_positions.size == 0:
                            continue
                        a = int(zero_positions[-1])
                        if a + 1 <= r:
                            mut[a + 1 : r + 1, col_idx] = 1
                        transition_times.append(max(1, r - a))
                        start = r + 1

                        if h:
                            for j in range(1, h + 1):
                                idx = r + j
                                if idx >= n_years or series[idx] < theta:
                                    break
                                mut[idx, col_idx] = 1

            name = f"theta={theta:.1f},h={h}"
            results[name] = (values * mut).sum(axis=1)
            tau_by_param[name] = float(np.mean(transition_times)) if transition_times else 1.0

    return pd.DataFrame(results, index=years), tau_by_param


def _ols_r2_and_last_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """OLS R2 and last-coefficient p-value. / 普通最小二乘 R2 与末项系数 p 值。"""
    n_obs, n_params = x.shape
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, np.nan

    y_hat = x @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    df = n_obs - n_params
    if df <= 0:
        return r2, np.nan

    xtx_inv = np.linalg.pinv(x.T @ x)
    mse = ss_res / df
    se = float(np.sqrt(max(mse * xtx_inv[-1, -1], 0.0)))
    if se == 0.0 or not np.isfinite(se):
        return r2, np.nan
    t_stat = float(beta[-1] / se)
    pvalue = float(2 * stats.t.sf(abs(t_stat), df))
    return r2, pvalue


def fit_theta_h_tau(
    gmeasure: pd.DataFrame,
    tau_by_param: Dict[str, float],
    epi_df: pd.DataFrame,
) -> dict:
    """Fit theta, history length h, and lag tau.
    
    English: Grid-search parameters that best track historical prevalence.
    中文：网格搜索最能跟踪历史流行度的参数。
    """
    if gmeasure.empty:
        return {"theta": 0.5, "h": 0, "tau": 1, "r2": 0.0, "years_used": 0}

    epi = epi_df.copy()
    epi["Season"] = epi["Season"].astype(int)
    common = sorted(set(gmeasure.index.astype(int)).intersection(set(epi["Season"])))
    if len(common) < 3:
        return {"theta": 0.5, "h": 0, "tau": 1, "r2": 0.0, "years_used": len(common)}

    y = epi.set_index("Season").loc[common, "Positivity_Rate"].to_numpy(dtype=float)
    season_values = np.array(common, dtype=float)
    season_scaled = season_values - season_values.mean()

    best_r2 = {"param": None, "r2": -np.inf, "pvalue": np.nan}
    best_pvalue = {"param": None, "r2": -np.inf, "pvalue": np.inf}
    for param in gmeasure.columns:
        g = gmeasure.loc[common, param].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(common)), season_scaled, g])
        r2, pvalue = _ols_r2_and_last_pvalue(x, y)
        if r2 > best_r2["r2"]:
            best_r2 = {"param": param, "r2": r2, "pvalue": pvalue}
        if np.isfinite(pvalue) and pvalue < best_pvalue["pvalue"]:
            best_pvalue = {"param": param, "r2": r2, "pvalue": pvalue}

    if best_r2["param"] is None:
        return {"theta": 0.5, "h": 0, "tau": 1, "r2": 0.0, "years_used": len(common)}

    tau_param = best_pvalue["param"] or best_r2["param"]
    match = re.match(r"theta=([0-9.]+),h=(\d+)", best_r2["param"])
    theta = float(match.group(1)) if match else 0.5
    h = int(match.group(2)) if match else 0
    tau_match = re.match(r"theta=([0-9.]+),h=(\d+)", tau_param)
    tau_theta = float(tau_match.group(1)) if tau_match else theta
    tau_h = int(tau_match.group(2)) if tau_match else h
    tau = max(1, int(round(tau_by_param.get(tau_param, 1.0))))
    return {
        "theta": theta,
        "h": h,
        "tau": tau,
        "r2": float(best_r2["r2"]),
        "years_used": len(common),
        "tau_theta": tau_theta,
        "tau_h": tau_h,
        "tau_pvalue": float(best_pvalue["pvalue"]) if np.isfinite(best_pvalue["pvalue"]) else np.nan,
        "tau_param_r2": float(best_pvalue["r2"]) if best_pvalue["param"] else np.nan,
    }


def effective_mutations(prev: pd.DataFrame, theta: float) -> pd.DataFrame:
    """Mark effective mutations above a theta threshold. / 标记超过 theta 阈值的有效突变。"""
    rows: List[dict] = []
    for mutation in prev.columns:
        site = int(re.match(r"X(\d+)", mutation).group(1))
        series = prev[mutation].to_numpy(dtype=float)
        years = prev.index.tolist()
        start = 0
        for r, value in enumerate(series):
            if value >= theta and np.any(series[start : r + 1] == 0):
                zero_positions = np.where(series[: r + 1] == 0)[0]
                if zero_positions.size == 0:
                    continue
                a = int(zero_positions[-1])
                rows.append(
                    {
                        "mutation": mutation,
                        "site": site,
                        "season": int(years[r]),
                        "tau": max(1, int(r - a)),
                    }
                )
                start = r + 1
    return pd.DataFrame(rows, columns=["mutation", "site", "season", "tau"])


def predict_prevalence(prev: pd.DataFrame, tau: int, em_df: pd.DataFrame, predict_year: int) -> pd.Series:
    """Predict next-season prevalences from effective mutations. / 由有效突变预测下一季流行度。"""
    predictions: Dict[str, float] = {}
    for mutation in prev.columns:
        series = prev[mutation].to_numpy(dtype=float)
        if len(series) == 0:
            predictions[mutation] = 0.0
            continue
        if len(series) == 1:
            predictions[mutation] = float(series[-1])
            continue

        mutation_em = em_df[(em_df["mutation"] == mutation) & (em_df["season"] < predict_year)]
        transition = int(mutation_em.iloc[-1]["tau"]) if not mutation_em.empty else int(tau)
        transition = max(1, min(transition, len(series) - 1))

        zero_positions = np.where(series[:-1] == 0)[0]
        if zero_positions.size:
            tzero = int(zero_positions[-1])
            r = (len(series) - 1) - max(tzero, (len(series) - 1) - transition)
            r = max(1, min(r, len(series) - 1))
        else:
            r = transition

        delta = (series[-1] - series[-1 - r]) / r
        predictions[mutation] = float(np.clip(series[-1] + delta, 0.0, 1.0))
    return pd.Series(predictions)


def consensus_from_prediction(predicted: pd.Series, sites: List[str], prev: pd.DataFrame) -> Dict[str, str]:
    """Build a consensus genotype from predicted prevalences. / 由预测流行度构建共识基因型。"""
    consensus: Dict[str, str] = {}
    for site_col in sites:
        aa_cols = [col for col in predicted.index if col.startswith(site_col)]
        if not aa_cols:
            aa_cols = [col for col in prev.columns if col.startswith(site_col)]
            if not aa_cols:
                continue
            scores = prev[aa_cols].iloc[-1]
        else:
            scores = predicted[aa_cols]
        best_col = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
        consensus[site_col] = best_col[len(site_col) :]
    return consensus


def select_predicted_em_sites(
    prev: pd.DataFrame,
    predicted: pd.Series,
    theta: float,
    predict_year: int,
) -> pd.DataFrame:
    """Mark effective-mutation sites for the prediction year.

    A site/mutation is selected when its predicted frequency reaches ``theta``
    and the historical-or-predicted series still shows a zero-frequency state
    in the preceding window (transition into prevalence).
    """
    if prev.empty:
        return pd.DataFrame(columns=["mutation", "site", "season"])

    appended = pd.concat([prev, predicted.to_frame().T], ignore_index=True)
    appended.columns = prev.columns
    rows: List[dict] = []
    for mutation in appended.columns:
        site = int(re.match(r"X(\d+)", mutation).group(1))
        series = appended[mutation].to_numpy(dtype=float)
        start = 0
        for r, value in enumerate(series):
            if value >= theta and np.any(series[start : r + 1] == 0):
                rows.append({"mutation": mutation, "site": site, "season": predict_year})
                start = r + 1
    return pd.DataFrame(rows, columns=["mutation", "site", "season"])


def hamming_distance(df: pd.DataFrame, consensus: Dict[str, str], sites: List[int]) -> pd.Series:
    """Hamming distance to a consensus on given sites. / 在给定位点上相对共识序列的汉明距离。"""
    distance = pd.Series(0, index=df.index, dtype="int32")
    for site in sites:
        col = f"X{site}"
        target = consensus.get(col)
        if target is None or col not in df.columns:
            continue
        observed = df[col].fillna("X").astype(str).str.upper()
        distance = distance + (observed != target).astype("int32")
    return distance


def select_wildtype(
    seq_df: pd.DataFrame,
    predict_year: int,
    hemisphere: str,
    consensus: Dict[str, str],
    predicted_em_history: pd.DataFrame,
    antigenic_sites: List[int],
    predictor_sites: List[int],
    ha_length: int,
) -> dict:
    """Select recommended wild-type strains.
    
    English: Rank candidates by antigenic proximity and prevalence support.
    中文：按抗原邻近度与流行支持度排序候选株。
    """
    _, cutoff = prediction_window(predict_year, hemisphere)
    candidates = seq_df[
        (seq_df["collection_date"] < cutoff)
        & (seq_df["submission_date"] < cutoff)
    ].copy()
    if candidates.empty:
        return {
            "recommended_accession": "",
            "recommended_isolate": "",
            "recommended_clade": "",
            "stage1_distance": pd.NA,
            "stage2_distance": pd.NA,
            "stage3_distance": pd.NA,
            "candidate_count": 0,
        }

    em_sites = []
    if not predicted_em_history.empty:
        em_sites = sorted(
            set(predicted_em_history.loc[predicted_em_history["season"] <= predict_year, "site"].astype(int))
        )

    stage1_sites = sorted(set(em_sites).intersection(predictor_sites))
    stage2_sites = sorted((set(em_sites).union(antigenic_sites)) - set(stage1_sites))
    stage3_sites = sorted(set(range(1, ha_length + 1)) - set(stage1_sites) - set(stage2_sites))

    candidates["stage1_distance"] = hamming_distance(candidates, consensus, stage1_sites)
    candidates = candidates[candidates["stage1_distance"] == candidates["stage1_distance"].min()].copy()

    candidates["stage2_distance"] = hamming_distance(candidates, consensus, stage2_sites)
    candidates = candidates[candidates["stage2_distance"] == candidates["stage2_distance"].min()].copy()

    candidates["stage3_distance"] = hamming_distance(candidates, consensus, stage3_sites)
    candidates = candidates[candidates["stage3_distance"] == candidates["stage3_distance"].min()].copy()
    top = candidates.iloc[-1]
    return {
        "recommended_accession": str(top["accession number"]),
        "recommended_isolate": str(top["name"]),
        "recommended_clade": str(top["clade"]),
        "stage1_distance": int(top["stage1_distance"]),
        "stage2_distance": int(top["stage2_distance"]),
        "stage3_distance": int(top["stage3_distance"]),
        "candidate_count": int(len(seq_df[(seq_df["collection_date"] < cutoff) & (seq_df["submission_date"] < cutoff)])),
    }


def normalize_clade(value: object) -> str:
    """Normalize clade label strings. / 规范化 clade 标签字符串。"""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "unknown", "unassigned"}:
        return ""
    return text


def run(output_dir: Path, processes: int | None, subtypes: List[str] | None = None) -> None:
    """Run the full HA-only overall-fit workflow.
    
    English: Generate sequence tables, fit parameters, and write recommendation CSVs.
    中文：生成序列表、拟合参数并写出推荐结果 CSV。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = load_truth_table(DATA_ROOT / "truth.csv")

    recommendation_rows: List[dict] = []
    theta_rows: List[dict] = []
    em_rows: List[pd.DataFrame] = []

    for subtype in subtypes or list(SUBTYPE_CONFIG):
        cfg = SUBTYPE_CONFIG[subtype]
        print(f"[load] {subtype}", flush=True)
        seq_df = generate_sequence_table(
            subtype=subtype,
            fasta_path=INPUT_ROOT / cfg["fasta"],
            info_path=INPUT_ROOT / cfg["meta"],
            cutoff_date="2025-02-01",
            processes=processes,
        ).copy()
        seq_df["collection_date"] = pd.to_datetime(seq_df["collection_date"])
        seq_df["submission_date"] = pd.to_datetime(seq_df["submission_date"])

        sites = site_columns(seq_df, int(cfg["ha_length"]))
        antigenic_sites, predictor_sites = load_site_sets(subtype)

        for hemisphere in HEMISPHERES:
            epi_path = EPIDEMIC_FILES.get((subtype, hemisphere))
            if epi_path is None or not epi_path.exists():
                raise FileNotFoundError(
                    f"Missing packaged epidemic table for {subtype} {hemisphere}: {epi_path}"
                )
            epi_df = load_epidemic_rate(epi_path)
            fit_prev = build_prevalence_all(seq_df, hemisphere, sites)
            fit_gmeasure, fit_tau_by_param = get_gmeasure(fit_prev)
            fixed_fit = fit_theta_h_tau(fit_gmeasure, fit_tau_by_param, epi_df)
            fixed_history_em = effective_mutations(fit_prev, HA_EM_THETA)
            predicted_em_history = pd.DataFrame(columns=["mutation", "site", "season"])

            for year in YEARS:
                print(f"[run] {subtype} {hemisphere} {year}", flush=True)
                prev = build_prevalence(seq_df, year, hemisphere, sites)
                fit = fixed_fit
                hist_em = fixed_history_em[fixed_history_em["season"] < year].copy()
                pred_prev = predict_prevalence(prev, int(fit["tau"]), hist_em, year)
                consensus = consensus_from_prediction(pred_prev, sites, prev)
                new_pred_em = select_predicted_em_sites(
                    prev,
                    pred_prev,
                    PREDICTED_EM_THETA,
                    year,
                )
                if not new_pred_em.empty:
                    predicted_em_history = pd.concat([predicted_em_history, new_pred_em], ignore_index=True)
                    tmp_em = new_pred_em.copy()
                    tmp_em.insert(0, "hemisphere", hemisphere)
                    tmp_em.insert(0, "subtype", subtype)
                    em_rows.append(tmp_em)

                picked = select_wildtype(
                    seq_df=seq_df,
                    predict_year=year,
                    hemisphere=hemisphere,
                    consensus=consensus,
                    predicted_em_history=predicted_em_history,
                    antigenic_sites=antigenic_sites,
                    predictor_sites=predictor_sites,
                    ha_length=int(cfg["ha_length"]),
                )
                recommendation_rows.append(
                    {
                        "subtype": subtype,
                        "truth_name": cfg["truth_name"],
                        "hemisphere": hemisphere,
                        "year": year,
                        "consensus_sites_used": len(consensus),
                        "predicted_em_sites_used": int(
                            predicted_em_history[predicted_em_history["season"] <= year]["site"].nunique()
                        ),
                        **picked,
                    }
                )
                theta_rows.append(
                    {
                        "subtype": subtype,
                        "hemisphere": hemisphere,
                        "year": year,
                        "theta": fit["theta"],
                        "h": fit["h"],
                        "tau": fit["tau"],
                        "r2": fit["r2"],
                        "years_used": fit["years_used"],
                        "tau_theta": fit.get("tau_theta", pd.NA),
                        "tau_h": fit.get("tau_h", pd.NA),
                        "tau_pvalue": fit.get("tau_pvalue", pd.NA),
                        "tau_param_r2": fit.get("tau_param_r2", pd.NA),
                        "history_em_count": int(len(hist_em)),
                        "epidemic_file": str(epi_path.name),
                    }
                )

    recommendations = pd.DataFrame(recommendation_rows)
    merged = recommendations.merge(truth, on=["truth_name", "year", "hemisphere"], how="left")
    merged["predicted_clade_norm"] = merged["recommended_clade"].map(normalize_clade)
    merged["truth_clade_norm"] = merged["truth_clade"].map(normalize_clade)
    merged["clade_match"] = merged["predicted_clade_norm"] == merged["truth_clade_norm"]
    merged.loc[merged["predicted_clade_norm"] == "", "clade_match"] = False
    merged.loc[merged["truth_clade_norm"] == "", "clade_match"] = False

    summary = (
        merged.groupby("subtype", as_index=False)["clade_match"]
        .agg(total="count", correct="sum")
        .assign(accuracy=lambda x: x["correct"] / x["total"])
    )
    overall = pd.DataFrame(
        [
            {
                "subtype": "ALL",
                "total": int(merged["clade_match"].count()),
                "correct": int(merged["clade_match"].sum()),
                "accuracy": float(merged["clade_match"].mean()),
            }
        ]
    )
    summary = pd.concat([summary, overall], ignore_index=True)

    merged.to_csv(output_dir / "beth1_ha_recommendations.csv", index=False)
    pd.DataFrame(theta_rows).to_csv(output_dir / "beth1_ha_theta_tau.csv", index=False)
    summary.to_csv(output_dir / "beth1_ha_clade_accuracy.csv", index=False)
    if em_rows:
        pd.concat(em_rows, ignore_index=True).to_csv(output_dir / "beth1_ha_predicted_em.csv", index=False)

    print(f"[done] {output_dir}", flush=True)


def main() -> None:
    """CLI entry point. / 命令行入口。"""
    args = parse_args()
    if args.h1n1_epitope_file is not None:
        SUBTYPE_CONFIG["H1N1"]["epitope_file"] = args.h1n1_epitope_file.resolve()
    run(args.output_dir.resolve(), args.processes, args.subtypes)


if __name__ == "__main__":
    main()
