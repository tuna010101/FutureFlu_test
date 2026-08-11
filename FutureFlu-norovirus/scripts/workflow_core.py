#!/usr/bin/env python3
"""Core helpers for the package-local norovirus workflow.

English: This module keeps the reusable downstream logic in one place.
中文：这个模块集中放置可复用的 downstream 处理逻辑。
"""

from __future__ import annotations

import argparse
import math
import os
import re
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# English: These paths define the stable release-facing layout.
# 中文：这些路径定义了面向发布的稳定目录结构。
ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_CSV = Path(
    os.environ.get(
        "NORO_SEQUENCE_CSV",
        ROOT / "raw_inputs" / "norovirus_vp1_sequence.csv",
    )
).expanduser().resolve()
POSITIVE_RATE_CSV = (
    ROOT / "data" / "positivity" / "zhang2024_yearly_norovirus_positive_rates.csv"
)
EVESCAPE_SCORE_DIR = ROOT / "data" / "EVEscape" / "NORO_GII4_evescape"
OUTPUT_ROOT = Path(os.environ.get("NORO_OUTPUT_ROOT", ROOT / "outputs"))
DISABLE_POSITIVE_RATES = os.environ.get("NORO_DISABLE_POSITIVE_RATES", "0") == "1"

SUBTYPE = "NORO_GII4"
HEMISPHERE = "global"
MODEL = "linear"
DEFAULT_START_YEAR = 2001
DEFAULT_END_YEAR = 2015
DEFAULT_MAX_GAPS = 3
SUBMISSION_DATE_COLUMNS = (
    "Submission_Date",
    "Submission date",
    "submission_date",
    "SubmissionDate",
    "date_submitted",
    "Date_Submitted",
)
EPITOPE_START = 279
EPITOPE_END = 405
AA_LIST = [
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
METRICS = [
    "total_escape",
    "predicted_prevalence",
    "mutual_information",
    "dissimilarity_charge_hydro",
    "accessibility_wcn",
    "fitness_eve",
]
METRICS_3 = ["total_escape", "predicted_prevalence", "mutual_information"]
COMBINATIONS = [
    ("E", ["total_escape"]),
    ("G", ["predicted_prevalence"]),
    ("D", ["mutual_information"]),
    ("E+G", ["total_escape", "predicted_prevalence"]),
    ("E+D", ["total_escape", "mutual_information"]),
    ("G+D", ["predicted_prevalence", "mutual_information"]),
    ("E+G+D", ["total_escape", "predicted_prevalence", "mutual_information"]),
]
COMBO_NAMES_ORDERED = ["E", "G", "D", "E+G", "E+D", "G+D", "E+G+D"]
T_VALUES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]
FIT_COLS = [f"fit_{metric}" for metric in METRICS]
PRE_ACT_COL_ORDER = (
    ["subtype", "year", "hemisphere", "clade", "act_freq", "freq_prev"]
    + [
        col
        for combo in COMBO_NAMES_ORDERED
        for col in (f"{combo}_pre_fit", f"{combo}_pre_freq")
    ]
)
INFORM_BASE_COLS = [
    "subtype",
    "hemisphere",
    "year",
    "clade",
    "risk_mutation_group",
    "mutation_count",
    "mutation_group_seq_count",
]
DIVERGENCE_COL_ORDER = INFORM_BASE_COLS + ["mutual_information"]
ESCAPE_COL_ORDER = INFORM_BASE_COLS + ["total_escape"]
GROWTH_COL_ORDER = INFORM_BASE_COLS + ["predicted_prevalence"]
CLADE_RE = re.compile(r"([^\(,]+?)\s*\((\d+\.?\d*)%\)")
EXCLUDED_CLADE_NAMES = {"", "unknown", "unassigned", "nan", "other"}


def is_excluded_clade(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in EXCLUDED_CLADE_NAMES


def theta_label(theta: float) -> str:
    return f"theta_{str(theta).replace('.', 'p')}"


def format_theta(theta: float) -> str:
    """Stable theta text for run_config (avoid trailing zeros such as 0.10)."""
    return f"{theta:g}"


def process_work_root(run_root: Path) -> Path:
    """Scratch tree for process-only files (gitignored; not part of release tables)."""
    default_outputs = (ROOT / "outputs").resolve()
    if run_root.resolve() == default_outputs:
        path = ROOT / "_step_outputs" / "release"
    else:
        path = run_root.parent / "_step_outputs" / run_root.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def process_metadata_dir(run_root: Path) -> Path:
    path = process_work_root(run_root) / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def release_metadata_dir(run_root: Path) -> Path:
    path = run_root / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


def x_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if re.fullmatch(r"X\d+", str(col))]
    cols = sorted(cols, key=lambda col: int(str(col)[1:]))
    if not cols:
        raise ValueError("no X1..XN columns found in sequence table")
    return cols


def position_from_x_column(col: object) -> int:
    return int(str(col)[1:])


def in_epitope_region(position: object) -> bool:
    return EPITOPE_START <= int(position) <= EPITOPE_END


def epitope_x_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in x_columns(df) if in_epitope_region(position_from_x_column(col))]
    if not cols:
        raise ValueError(f"no X columns found in epitope region {EPITOPE_START}-{EPITOPE_END}")
    return cols


def filter_epitope_score_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "i" not in df.columns:
        raise KeyError("EVEscape score table is missing required column: i")
    positions = pd.to_numeric(df["i"], errors="coerce")
    return df.loc[positions.between(EPITOPE_START, EPITOPE_END)].copy()


def subtype_key(subtype: str = SUBTYPE) -> str:
    return subtype.lower().replace("/", "").replace(" ", "")


def target_season_window(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=int(year), month=1, day=1)
    end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
    return start, end


def submission_cutoff(year: int) -> pd.Timestamp:
    return target_season_window(year)[1]


def target_collection_mask(seq_df: pd.DataFrame, year: int) -> pd.Series:
    start, end = target_season_window(year)
    return (seq_df["collection_date"] >= start) & (seq_df["collection_date"] < end)


def submitted_before_cutoff_mask(seq_df: pd.DataFrame, year: int) -> pd.Series:
    return seq_df["submission_date"] < submission_cutoff(year)


def candidate_sequence_year(predict_year: int) -> int:
    return int(predict_year) - 1


def candidate_sequences_for_year(seq_df: pd.DataFrame, predict_year: int) -> pd.DataFrame:
    candidate_year = candidate_sequence_year(predict_year)
    mask = target_collection_mask(seq_df, candidate_year) & submitted_before_cutoff_mask(
        seq_df, candidate_year
    )
    return seq_df.loc[mask].copy()


def find_submission_date_column(df: pd.DataFrame) -> str | None:
    for col in SUBMISSION_DATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def data_root_for_run(run_root: Path) -> Path:
    """Use package-level data for default release runs; run-local data for experiments."""
    try:
        run_root.resolve().relative_to((ROOT / "outputs").resolve())
        return ROOT / "data"
    except ValueError:
        return run_root / "data"


def futureflu_rank_dir(run_root: Path) -> Path:
    return data_root_for_run(run_root) / "futureflu_rank"


def clade_counts_dir(run_root: Path) -> Path:
    return data_root_for_run(run_root) / "clade_counts"


def ensure_dirs(run_root: Path) -> None:
    """Create release and process output directories. / 创建发布与过程输出目录。"""
    paths = [
        process_work_root(run_root) / "sequences",
        process_metadata_dir(run_root),
        futureflu_rank_dir(run_root),
        clade_counts_dir(run_root),
        release_metadata_dir(run_root),
        linear_out(run_root) / f"{SUBTYPE}_{HEMISPHERE}",
        risk_out(run_root) / "mutation_components",
        risk_out(run_root) / "component_combinations",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def linear_out(run_root: Path) -> Path:
    return run_root / "predictions" / "linear" / "results"


def risk_out(run_root: Path) -> Path:
    return run_root / "predictions" / "risk_components"


def evescape_paths(year: int) -> tuple[Path, Path]:
    mutations_path = EVESCAPE_SCORE_DIR / f"NORO_GII4_evescape_{year}.csv"
    sites_path = EVESCAPE_SCORE_DIR / f"NORO_GII4_evescape_sites_{year}.csv"
    return mutations_path, sites_path


def validate_evescape_outputs(start_year: int, end_year: int) -> None:
    """Check packaged EVEscape yearly score files exist. / 检查打包的 EVEscape 逐年打分文件是否存在。"""
    missing: list[str] = []
    for year in range(start_year, end_year + 1):
        mutations_path, sites_path = evescape_paths(year)
        if not mutations_path.exists():
            missing.append(str(mutations_path))
        if not sites_path.exists():
            missing.append(str(sites_path))
    if missing:
        raise FileNotFoundError(
            "missing completed EVEscape output files:\n" + "\n".join(missing)
        )


def load_sequence_table(max_gaps: int) -> pd.DataFrame:
    """Load and normalize the norovirus VP1 sequence table.

    English: Read raw_inputs CSV, join X* columns into sequences, filter by gap count.
    中文：读取 raw_inputs CSV，拼接 X* 列为序列，并按 gap 数过滤。
    """
    df = pd.read_csv(SEQUENCE_CSV)
    required = {"Isolate_Name", "Collection_Date", "year", "genotype"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{SEQUENCE_CSV} is missing columns: {sorted(missing)}")

    cols = x_columns(df)
    out = df.copy()
    for col in cols:
        out[col] = out[col].fillna("-").astype(str).str.upper()
    out["sequence"] = out[cols].agg("".join, axis=1)
    out["gap_count"] = out["sequence"].str.count("-")
    out["collection_date"] = pd.to_datetime(out["Collection_Date"], errors="coerce")
    submission_col = find_submission_date_column(out)
    if submission_col is None:
        out["submission_date"] = out["collection_date"]
        submission_source = "collection_date_fallback"
    else:
        out["submission_date"] = pd.to_datetime(out[submission_col], errors="coerce")
        out["submission_date"] = out["submission_date"].fillna(out["collection_date"])
        submission_source = submission_col
    out["season"] = out["year"].astype(int)
    out["clade"] = out["genotype"].fillna("Unknown").astype(str)
    out["accession_number"] = (
        out["Isolate_Name"].fillna("unknown").astype(str) + "|idx" + out.index.astype(str)
    )
    out["name"] = out["Isolate_Name"].fillna("").astype(str)
    out = out[out["collection_date"].notna()].copy()
    out = out[out["gap_count"] <= max_gaps].copy()
    out = out.reset_index(drop=True)
    out.attrs["submission_date_source"] = submission_source
    return out


def load_positive_rates() -> pd.DataFrame:
    """Load yearly positivity rates when enabled. / 在启用时加载逐年阳性率。"""
    if DISABLE_POSITIVE_RATES:
        return pd.DataFrame(columns=["year", "norovirus_positive_rate_percent", "source"])
    if not POSITIVE_RATE_CSV.exists():
        return pd.DataFrame(columns=["year", "norovirus_positive_rate_percent", "source"])
    rates = pd.read_csv(POSITIVE_RATE_CSV)
    rates["year"] = rates["year"].astype(int)
    return rates


def positive_rate_file_label() -> str:
    if DISABLE_POSITIVE_RATES:
        return "disabled"
    return str(POSITIVE_RATE_CSV.relative_to(ROOT))


def write_preprocessed_sequence_table(seq_df: pd.DataFrame, run_root: Path) -> None:
    """Write process-only epitope sequence table under _step_outputs. / 将过程用表位序列表写入 _step_outputs。"""
    base_cols = [
        "accession_number",
        "name",
        "clade",
        "collection_date",
        "submission_date",
        "season",
        "year",
        "genotype",
        "gap_count",
    ]
    cols = epitope_x_columns(seq_df)
    output = seq_df[base_cols + cols].copy()
    output["collection_date"] = output["collection_date"].dt.strftime("%Y-%m-%d")
    output["submission_date"] = output["submission_date"].dt.strftime("%Y-%m-%d")
    output.to_csv(
        process_work_root(run_root) / "sequences" / f"{SUBTYPE}_sequence.csv",
        index=False,
    )


def get_consensus_sequence(data: pd.DataFrame, site_columns: list[str]) -> dict[str, str]:
    """Consensus amino acid per site (ignore X). / 各位点共识氨基酸（忽略 X）。"""
    consensus: dict[str, str] = {}
    for col in site_columns:
        valid_data = data[col][data[col] != "X"]
        if len(valid_data) > 0:
            consensus[col] = valid_data.mode().iloc[0]
    return consensus


def calculate_significance(
    new_aa_current: int,
    new_aa_prev: int,
    total_current: int,
    total_prev: int,
) -> float:
    """Fisher enrichment p-value for amino-acid rise. / 氨基酸上升富集的 Fisher p 值。"""
    from scipy import stats

    observed = np.array(
        [
            [new_aa_current, total_current - new_aa_current],
            [new_aa_prev, total_prev - new_aa_prev],
        ]
    )
    if np.any(observed == 0):
        return 0.0
    return float(stats.chi2_contingency(observed)[1])


def site_prevalence_all_sites(seq: pd.DataFrame, predict_year: int) -> pd.DataFrame:
    """Compute yearly amino-acid prevalences for all sites. / 计算全部位点的逐年氨基酸流行度。"""
    years = sorted(int(y) for y in seq["season"].unique() if int(y) < predict_year)
    site_columns = epitope_x_columns(seq)

    prevalence_data: list[pd.Series] = []
    annual_data: dict[int, dict[str, object]] = {}

    for year in years:
        season_data = seq[seq["season"] == year].copy()

        year_freq_data: dict[str, float] = {}
        year_counts_data: dict[str, pd.Series] = {}
        for site_col in site_columns:
            valid_data = season_data[site_col][season_data[site_col] != "X"]
            freq = valid_data.value_counts(normalize=True)
            counts = valid_data.value_counts()
            for aa in AA_LIST:
                year_freq_data[f"{site_col}{aa}"] = float(freq.get(aa, 0.0))
            year_counts_data[site_col] = counts

        prevalence_data.append(pd.Series(year_freq_data, name=year))
        annual_data[year] = {
            "counts": year_counts_data,
            "dominant_mutations": [],
        }

        prev_data = seq[seq["season"] == year - 1].copy()
        if prev_data.empty:
            continue

        current_consensus = get_consensus_sequence(season_data, site_columns)
        prev_consensus = get_consensus_sequence(prev_data, site_columns)
        prev_counts_data = {
            site_col: prev_data[site_col][prev_data[site_col] != "X"].value_counts()
            for site_col in site_columns
        }

        identified_mutations: list[str] = []
        for site_col in site_columns:
            if (
                site_col in current_consensus
                and site_col in prev_consensus
                and current_consensus[site_col] != prev_consensus[site_col]
            ):
                new_aa = current_consensus[site_col]
                current_counts = year_counts_data[site_col]
                prev_counts = prev_counts_data[site_col]
                p_value = calculate_significance(
                    int(current_counts.get(new_aa, 0)),
                    int(prev_counts.get(new_aa, 0)),
                    int(sum(current_counts)),
                    int(sum(prev_counts)),
                )
                if p_value < 0.05:
                    identified_mutations.append(f"{site_col[1:]}{new_aa}")

        annual_data[year]["dominant_mutations"] = sorted(identified_mutations)

    if not prevalence_data:
        return pd.DataFrame(columns=["season", "dominant_mutation"])

    prevalence_result = pd.concat(prevalence_data, axis=1).T
    prevalence_result["dominant_mutation"] = [
        ", ".join(annual_data[y]["dominant_mutations"]) for y in years
    ]
    mutation_columns = [
        col for col in prevalence_result.columns if col != "dominant_mutation"
    ]
    return (
        prevalence_result[mutation_columns + ["dominant_mutation"]]
        .reset_index()
        .rename(columns={"index": "season"})
    )


def gmeasure(prev_data: pd.DataFrame, theta_range: Sequence[float]) -> pd.DataFrame:
    """Compute g-measure tables across theta values. / 计算不同 theta 下的 g-measure 表。"""
    valid_years = prev_data["season"].tolist()
    year_theta_gsum = {year: {theta: 0.0 for theta in theta_range} for year in valid_years}

    for theta in theta_range:
        for col in prev_data.columns:
            if not str(col).startswith("X"):
                continue
            values = prev_data[col].fillna(0).values
            n_years = len(values)
            mut = np.zeros(n_years, dtype=int)
            start = 0

            for r in range(n_years):
                if values[r] >= theta and np.any(values[start:r] < theta):
                    low_pos = np.where(values[:r] < theta)[0]
                    if low_pos.size > 0:
                        a = low_pos[-1]
                        mut[a + 1 : r + 1] = 1
                        start = r + 1

            yearly_gsum = values * mut
            for idx, year in enumerate(valid_years):
                year_theta_gsum[year][theta] += float(yearly_gsum[idx])

    gsum_df = pd.DataFrame.from_dict(year_theta_gsum, orient="index")
    gsum_df.columns = [f"theta={theta:.2f}" for theta in theta_range]
    return gsum_df.reset_index().rename(columns={"index": "season"})


def predict_mutations_linear(
    predict_year: int,
    theta: float,
    prev_data: pd.DataFrame,
) -> pd.DataFrame:
    """Predict risk mutations with a linear prevalence model.

    English: Extrapolate site frequencies and select mutations crossing theta rules.
    中文：外推位点频率并按 theta 规则筛选风险突变。
    """
    columns = [
        "predict_season",
        "risk_mutation",
        "previous_prevalence",
        "predicted_prevalence",
        "delta",
        "model",
    ]
    if prev_data.empty:
        return pd.DataFrame(columns=columns)

    mutation_columns = [col for col in prev_data.columns if str(col).startswith("X")]
    historical_data = prev_data[prev_data.index < predict_year][mutation_columns]
    if historical_data.empty:
        return pd.DataFrame(columns=columns)

    mutation_dominant_years: dict[str, int] = {}
    if "dominant_mutation" in prev_data.columns:
        for year, row in prev_data.iterrows():
            if int(year) >= predict_year:
                continue
            dominant_muts = row["dominant_mutation"]
            if isinstance(dominant_muts, str) and dominant_muts.strip():
                for mut in dominant_muts.split(","):
                    mut = mut.strip()
                    if mut and (
                        mut not in mutation_dominant_years
                        or int(year) > mutation_dominant_years[mut]
                    ):
                        mutation_dominant_years[mut] = int(year)

    pred_prev = np.zeros(len(historical_data.columns))
    deltas = np.zeros(len(historical_data.columns))

    for col_idx, col in enumerate(historical_data.columns):
        freqs = historical_data[col].fillna(0).values
        if len(freqs) < 2:
            pred_prev[col_idx] = freqs[-1] if len(freqs) > 0 else 0
            deltas[col_idx] = 0
            continue
        last_value = freqs[-1]
        prev_value = freqs[-2]
        delta = last_value - prev_value
        predicted = last_value + delta
        deltas[col_idx] = delta
        pred_prev[col_idx] = float(np.clip(predicted, 0, 1))

    risk_muts: list[str] = []
    historical_years = prev_data[prev_data.index < predict_year].index.tolist()
    for col_idx, col in enumerate(historical_data.columns):
        pred = pred_prev[col_idx]
        formatted_mut = col[1:] if str(col).startswith("X") else str(col)

        if formatted_mut in mutation_dominant_years:
            start_year = mutation_dominant_years[formatted_mut]
            try:
                start_idx = historical_years.index(start_year)
                freqs = historical_data[col].values[start_idx:]
            except ValueError:
                freqs = historical_data[col].values
        else:
            freqs = historical_data[col].values

        condition1 = (pred >= theta) and np.any(freqs < theta)

        condition2 = False
        if theta / 10 >= 0.01:
            condition2 = (pred >= theta / 10) and np.any(freqs < theta / 10)
        elif theta * 10 < 1:
            condition2 = (pred >= theta * 10) and np.any(freqs < theta * 10)

        current_prev = historical_data[col].iloc[-1] if len(historical_data[col]) > 0 else 0
        condition3 = current_prev < 0.75

        if (condition1 or condition2) and condition3:
            risk_muts.append(col)

    records = []
    for mut in risk_muts:
        formatted_mut = mut[1:] if str(mut).startswith("X") else str(mut)
        prev_prev = historical_data[mut].iloc[-1] if len(historical_data[mut]) > 0 else 0
        mut_idx = list(historical_data.columns).index(mut)
        records.append(
            {
                "predict_season": predict_year,
                "risk_mutation": formatted_mut,
                "previous_prevalence": round(float(prev_prev), 4),
                "predicted_prevalence": round(float(pred_prev[mut_idx]), 4),
                "delta": round(float(deltas[mut_idx]), 4),
                "model": MODEL,
            }
        )

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values(["risk_mutation", "model"]).reset_index(drop=True)


def parse_mutation(mutation_str: str) -> tuple[int, str]:
    position = int("".join(filter(str.isdigit, mutation_str)))
    amino_acid = mutation_str.replace(str(position), "")
    return position, amino_acid


def analyze_risk_mutations_year(
    sequences_df: pd.DataFrame,
    mutations_df: pd.DataFrame,
    predict_year: int,
    model: str | None = None,
) -> pd.DataFrame:
    """Analyze risk mutations for one prediction year. / 分析单个预测年的风险突变。"""
    columns = ["risk_mutation_group", "count", "model"]
    if mutations_df.empty:
        return pd.DataFrame(columns=columns)

    filtered_sequences = candidate_sequences_for_year(sequences_df, predict_year)
    if filtered_sequences.empty:
        return pd.DataFrame(columns=columns)

    if model is not None and "model" in mutations_df.columns:
        mutations_df = mutations_df[mutations_df["model"] == model]
    if mutations_df.empty:
        return pd.DataFrame(columns=columns)

    result_dfs: list[pd.DataFrame] = []
    models_to_analyze = (
        mutations_df["model"].unique().tolist()
        if "model" in mutations_df.columns
        else ["undefined"]
    )

    for current_model in models_to_analyze:
        if current_model != "undefined" and "model" in mutations_df.columns:
            model_mutations = mutations_df[mutations_df["model"] == current_model]
        else:
            model_mutations = mutations_df.copy()
        risk_mutations = [
            parse_mutation(mut) for mut in model_mutations["risk_mutation"].astype(str)
        ]
        if not risk_mutations:
            continue

        model_sequences = filtered_sequences.copy()

        def find_risk_mutations(row: pd.Series) -> str | None:
            mutations_found = []
            for pos, aa in risk_mutations:
                column_name = f"X{pos}"
                if column_name not in row:
                    continue
                if aa == "-":
                    if pd.isna(row[column_name]) or row[column_name] == "-":
                        mutations_found.append(f"{pos}{aa}")
                elif row[column_name] == aa:
                    mutations_found.append(f"{pos}{aa}")
            return ",".join(sorted(mutations_found)) if mutations_found else None

        model_sequences["risk_mutation_group"] = model_sequences.apply(
            find_risk_mutations, axis=1
        )
        model_sequences = model_sequences[model_sequences["risk_mutation_group"].notna()]
        if model_sequences.empty:
            continue

        mutation_counts = (
            model_sequences["risk_mutation_group"]
            .value_counts()
            .reset_index(name="count")
            .rename(columns={"index": "risk_mutation_group"})
        )
        mutation_counts["model"] = current_model
        result_dfs.append(mutation_counts)

    if not result_dfs:
        return pd.DataFrame(columns=columns)
    return pd.concat(result_dfs, ignore_index=True)[columns]


def count_mutations(mutation_group: object) -> int:
    if pd.isna(mutation_group) or mutation_group == "":
        return 0
    return len(str(mutation_group).split(","))


def calculate_single_mutation_mi(
    mutation: str,
    seq_df: pd.DataFrame,
    all_mutation_groups: Sequence[str],
) -> float:
    site = int("".join(filter(str.isdigit, mutation)))
    col_name = f"X{site}"
    if col_name not in seq_df.columns:
        return 0.0

    has_mutation = seq_df[seq_df[col_name] == mutation[-1]]
    total_occurrences = len(has_mutation)
    if total_occurrences == 0:
        return 0.0

    co_mutations: set[str] = set()
    for group in all_mutation_groups:
        muts_in_group = [m.strip() for m in str(group).split(",")]
        if mutation in muts_in_group:
            co_mutations.update(m for m in muts_in_group if m != mutation)

    solo_seqs = has_mutation.copy()
    for co_mut in co_mutations:
        co_site = int("".join(filter(str.isdigit, co_mut)))
        co_col = f"X{co_site}"
        if co_col in seq_df.columns:
            solo_seqs = solo_seqs[solo_seqs[co_col] != co_mut[-1]]

    return float(len(solo_seqs) / total_occurrences)


def calculate_group_mutual_information(mutation_matrix: pd.DataFrame) -> float:
    if mutation_matrix.shape[1] <= 1:
        return 0.0
    n = len(mutation_matrix)
    if n == 0:
        return 0.0
    marginal_probs = mutation_matrix.mean()
    patterns = mutation_matrix.apply(lambda row: "".join(row.astype(str)), axis=1)
    joint_counts = patterns.value_counts()
    mi_value = 0.0
    for pattern, count in joint_counts.items():
        p_joint = count / n
        if p_joint <= 0:
            continue
        binary_pattern = [int(bit) for bit in pattern]
        p_indep = 1.0
        for idx, mut in enumerate(mutation_matrix.columns):
            p_i = (
                marginal_probs[mut]
                if binary_pattern[idx] == 1
                else 1 - marginal_probs[mut]
            )
            p_indep *= p_i
        if p_indep > 0:
            mi_value += p_joint * np.log2(p_joint / p_indep)
    return float(mi_value / mutation_matrix.shape[1])


def get_mutation_matrix_simple(seq_df: pd.DataFrame, mutations: list[str]) -> pd.DataFrame:
    mutation_matrix = pd.DataFrame(index=seq_df.index)
    for mut in mutations:
        site = int("".join(filter(str.isdigit, mut)))
        col_name = f"X{site}"
        if col_name in seq_df.columns:
            mutation_matrix[mut] = (seq_df[col_name] == mut[-1]).astype(int)
        else:
            mutation_matrix[mut] = 0
    return mutation_matrix


def get_matching_sequences(
    mutation_group: object,
    sequence_df: pd.DataFrame,
    all_mutation_groups: Sequence[str],
) -> pd.DataFrame:
    if pd.isna(mutation_group):
        return pd.DataFrame()
    mutations = [m.strip() for m in str(mutation_group).split(",")]
    matching = sequence_df.copy()
    for mut in mutations:
        site = int("".join(filter(str.isdigit, mut)))
        col_name = f"X{site}"
        if col_name in sequence_df.columns:
            matching = matching.loc[matching[col_name] == mut[-1]]

    for other_group in all_mutation_groups:
        other_muts = [m.strip() for m in str(other_group).split(",")]
        if set(other_muts) > set(mutations):
            for extra_mut in set(other_muts) - set(mutations):
                site = int("".join(filter(str.isdigit, extra_mut)))
                col_name = f"X{site}"
                if col_name in sequence_df.columns:
                    matching = matching.loc[matching[col_name] != extra_mut[-1]]
    return matching


def get_clade_info_from_matching(matching: pd.DataFrame) -> str:
    if len(matching) == 0:
        return "Unknown"
    clade_counts = matching["clade"].fillna("Unknown").astype(str).value_counts()
    if len(clade_counts) == 0:
        return "Unknown"
    total = len(matching)
    clade_percentages = [
        f"{clade} ({count / total * 100:.1f}%)"
        for clade, count in clade_counts.items()
        if not is_excluded_clade(clade)
    ]
    return ", ".join(clade_percentages) if clade_percentages else "unassigned (100.0%)"


def calculate_total_escape_value(
    mutation_group: object,
    mutation_escape: dict[str, float],
    site_escape: dict[str, float],
) -> float:
    if pd.isna(mutation_group):
        return 0.0
    total = 0.0
    for mut in [m.strip() for m in str(mutation_group).split(",")]:
        site = int("".join(filter(str.isdigit, mut)))
        if mut in mutation_escape:
            total += mutation_escape[mut]
        elif str(site) in site_escape:
            total += site_escape[str(site)]
    return float(total)


def calculate_metric_value(
    mutation_group: object,
    mutation_metric: dict[str, float],
    site_metric: dict[str, float],
) -> float:
    if pd.isna(mutation_group):
        return 0.0
    total = 0.0
    for mut in [m.strip() for m in str(mutation_group).split(",")]:
        site = int("".join(filter(str.isdigit, mut)))
        if mut in mutation_metric:
            total += mutation_metric[mut]
        elif str(site) in site_metric:
            total += site_metric[str(site)]
    return float(total)


def calculate_prevalence(
    mutation_group: object,
    mutation_prevalence: dict[str, float],
) -> float:
    if pd.isna(mutation_group):
        return 0.0
    mutations = [m.strip() for m in str(mutation_group).split(",")]
    total = sum(mutation_prevalence.get(m, 0.0) for m in mutations)
    return float(total / len(mutations)) if mutations else 0.0


def get_dominant_clade(clade_str: object) -> str | None:
    if is_excluded_clade(clade_str):
        return None
    first_entry = str(clade_str).split(",")[0].strip()
    clade_name = first_entry.split("(")[0].strip()
    if is_excluded_clade(clade_name):
        return None
    return clade_name if clade_name else None


def filter_random_single_mutations(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    results_df = results_df.copy()
    results_df["_dom_clade"] = results_df["clade"].apply(get_dominant_clade)

    multi_rows = results_df[results_df["mutation_count"] > 1]
    multi_mc_pairs: dict[tuple[str, str], float] = {}
    for _, row_b in multi_rows.iterrows():
        dc_b = row_b["_dom_clade"]
        seq_count_b = row_b["mutation_group_seq_count"]
        if dc_b is None:
            continue
        for mutation in [m.strip() for m in str(row_b["risk_mutation_group"]).split(",")]:
            key = (mutation, dc_b)
            if key not in multi_mc_pairs or seq_count_b > multi_mc_pairs[key]:
                multi_mc_pairs[key] = seq_count_b

    def should_exclude(row: pd.Series) -> bool:
        if row["mutation_count"] != 1:
            return False
        mutation = str(row["risk_mutation_group"]).strip()
        dominant = row["_dom_clade"]
        seq_count = row["mutation_group_seq_count"]
        if dominant is None:
            return False
        key = (mutation, dominant)
        if key not in multi_mc_pairs:
            return False
        return bool(multi_mc_pairs[key] >= seq_count * 0.1)

    mask = results_df.apply(should_exclude, axis=1)
    return results_df[~mask].drop(columns=["_dom_clade"])


def shifted_map(
    df: pd.DataFrame,
    value_col: str,
    key_col: str = "mutation",
    fill_nan_with_zero: bool = False,
) -> dict[str, float]:
    values = df[value_col]
    min_value = values.min(skipna=True)
    if pd.isna(min_value):
        min_value = 0.0
    result: dict[str, float] = {}
    for _, row in df.iterrows():
        if key_col == "mutation":
            key = f"{int(row['i'])}{row['mut']}"
        else:
            key = str(int(row["i"]))
        value = row[value_col]
        if pd.isna(value) and fill_nan_with_zero:
            result[key] = 0.0
        elif pd.isna(value):
            result[key] = np.nan
        else:
            result[key] = float(value - min_value)
    return result


def site_mean_map_from_mutations(
    df: pd.DataFrame,
    value_col: str,
    fill_nan_with_zero: bool = False,
) -> dict[str, float]:
    values = df[value_col]
    min_value = values.min(skipna=True)
    if pd.isna(min_value):
        min_value = 0.0
    tmp = df[["i", value_col]].copy()
    if fill_nan_with_zero:
        tmp["_value"] = tmp[value_col].apply(
            lambda value: 0.0 if pd.isna(value) else float(value - min_value)
        )
    else:
        tmp["_value"] = tmp[value_col] - min_value
    return {str(int(site)): float(value) for site, value in tmp.groupby("i")["_value"].mean().items()}


def build_component_table(
    seq_df: pd.DataFrame,
    run_root: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Build E/G/D risk-component tables. / 构建 E/G/D 风险组分表。"""
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    for year in range(start_year, end_year + 1):
        print(f"[component] {SUBTYPE} {HEMISPHERE} {year}", flush=True)
        prefix = f"{SUBTYPE}_{HEMISPHERE}_{year}"
        year_linear_dir = linear_out(run_root) / f"{SUBTYPE}_{HEMISPHERE}" / str(year)
        distribution_path = year_linear_dir / f"{prefix}_distribution.csv"
        prediction_path = year_linear_dir / f"{prefix}_mutations.csv"
        mutations_path, sites_path = evescape_paths(year)

        distribution_df = pd.read_csv(distribution_path)
        prediction_df = pd.read_csv(prediction_path)
        if distribution_df.empty:
            status_rows.append(
                {
                    "year": year,
                    "distribution_rows": 0,
                    "component_rows": 0,
                    "status": "no_risk_mutation_groups",
                }
            )
            continue

        mutations_df = filter_epitope_score_rows(pd.read_csv(mutations_path))
        sites_df = filter_epitope_score_rows(pd.read_csv(sites_path))
        if mutations_df.empty or sites_df.empty:
            status_rows.append(
                {
                    "year": year,
                    "distribution_rows": len(distribution_df),
                    "component_rows": 0,
                    "status": "no_epitope_evescape_scores",
                }
            )
            continue
        distribution_df["mutation_count"] = distribution_df["risk_mutation_group"].apply(
            count_mutations
        )
        df_filtered = distribution_df[distribution_df["mutation_count"] > 0]
        if df_filtered.empty:
            status_rows.append(
                {
                    "year": year,
                    "distribution_rows": len(distribution_df),
                    "component_rows": 0,
                    "status": "no_nonempty_mutation_groups",
                }
            )
            continue

        q1 = df_filtered["mutation_count"].quantile(0.25)
        q3 = df_filtered["mutation_count"].quantile(0.75)
        iqr = q3 - q1
        non_outliers = df_filtered[
            (df_filtered["mutation_count"] <= q3 + 3 * iqr)
            & (df_filtered["mutation_count"] >= q1 - 3 * iqr)
        ].copy()
        if non_outliers.empty:
            status_rows.append(
                {
                    "year": year,
                    "distribution_rows": len(distribution_df),
                    "component_rows": 0,
                    "status": "no_non_outlier_mutation_groups",
                }
            )
            continue

        all_mutation_groups = non_outliers["risk_mutation_group"].dropna().tolist()
        filtered_seq_df = candidate_sequences_for_year(seq_df, year)

        mutations_min_escape = mutations_df["evescape"].min()
        sites_min_escape = sites_df["evescape"].min()
        mutation_escape = {
            f"{int(row['i'])}{row['mut']}": float(row["evescape"] - mutations_min_escape)
            for _, row in mutations_df.iterrows()
        }
        site_escape = {
            str(int(row["i"])): float(row["evescape"] - sites_min_escape)
            for _, row in sites_df.iterrows()
        }

        mutation_dch = shifted_map(mutations_df, "dissimilarity_charge_hydro")
        site_dch = site_mean_map_from_mutations(mutations_df, "dissimilarity_charge_hydro")
        mutation_awcn = shifted_map(
            mutations_df, "accessibility_wcn", fill_nan_with_zero=True
        )
        site_awcn = site_mean_map_from_mutations(
            mutations_df, "accessibility_wcn", fill_nan_with_zero=True
        )
        mutation_ef = shifted_map(mutations_df, "fitness_eve")
        site_ef = site_mean_map_from_mutations(mutations_df, "fitness_eve")

        mutation_prevalence = dict(
            zip(prediction_df["risk_mutation"], prediction_df["delta"])
        )
        min_prev = min(mutation_prevalence.values()) if mutation_prevalence else 0.0
        mutation_prevalence = {
            key: float(value - min_prev) for key, value in mutation_prevalence.items()
        }

        rows: list[dict[str, object]] = []
        filtered_singleton_low_support = 0
        for _, row in non_outliers.iterrows():
            mutation_group = row["risk_mutation_group"]
            count = int(row["mutation_count"])
            if pd.isna(mutation_group):
                mutual_info = 0.0
                matching_seqs = pd.DataFrame()
            else:
                muts = [m.strip() for m in str(mutation_group).split(",")]
                if len(muts) == 1:
                    mutual_info = calculate_single_mutation_mi(
                        muts[0], filtered_seq_df, all_mutation_groups
                    )
                else:
                    mut_matrix = get_mutation_matrix_simple(filtered_seq_df, muts)
                    mutual_info = calculate_group_mutual_information(mut_matrix)
                matching_seqs = get_matching_sequences(
                    mutation_group, filtered_seq_df, all_mutation_groups
                )

            if count == 1 and len(matching_seqs) < 3:
                filtered_singleton_low_support += 1
                continue

            rows.append(
                {
                    "subtype": SUBTYPE,
                    "hemisphere": HEMISPHERE,
                    "year": year,
                    "risk_mutation_group": mutation_group,
                    "clade": get_clade_info_from_matching(matching_seqs),
                    "mutation_count": count,
                    "mutation_group_seq_count": len(matching_seqs),
                    "total_escape": calculate_total_escape_value(
                        mutation_group, mutation_escape, site_escape
                    ),
                    "predicted_prevalence": calculate_prevalence(
                        mutation_group, mutation_prevalence
                    ),
                    "mutual_information": mutual_info,
                    "dissimilarity_charge_hydro": calculate_metric_value(
                        mutation_group, mutation_dch, site_dch
                    ),
                    "accessibility_wcn": calculate_metric_value(
                        mutation_group, mutation_awcn, site_awcn
                    ),
                    "fitness_eve": calculate_metric_value(mutation_group, mutation_ef, site_ef),
                }
            )

        year_df = filter_random_single_mutations(pd.DataFrame(rows))
        frames.append(year_df)
        status_rows.append(
            {
                "year": year,
                "distribution_rows": len(distribution_df),
                "component_rows": len(year_df),
                "filtered_singleton_low_support": filtered_singleton_low_support,
                "status": "ok" if len(year_df) else "filtered_to_empty",
            }
        )

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
    ]
    if frames:
        component_df = pd.concat(frames, ignore_index=True)
        component_df = component_df[columns]
    else:
        component_df = pd.DataFrame(columns=columns)

    component_dir = risk_out(run_root) / "mutation_components"
    component_path = component_dir / "risk_mutation_group_component.csv"
    root_copy = risk_out(run_root) / "risk_mutation_group_component.csv"
    component_df.to_csv(component_path, index=False)
    component_df.to_csv(root_copy, index=False)
    pd.DataFrame(status_rows).to_csv(
        process_metadata_dir(run_root) / "component_status_by_year.csv", index=False
    )
    print(f"[component] wrote {component_path} shape={component_df.shape}", flush=True)
    return component_df


def valid_clade_series(df: pd.DataFrame) -> pd.Series:
    clade = df["clade"].fillna("").astype(str).str.strip()
    return clade[~clade.apply(is_excluded_clade)]


def counted_clade_series(df: pd.DataFrame) -> pd.Series:
    clade = df["clade"].fillna("").astype(str).str.strip()
    return clade[~clade.apply(is_excluded_clade)]


def build_label_and_count_inputs(
    seq_df: pd.DataFrame,
    run_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare clade labels and count inputs. / 准备 clade 标签与计数输入。"""
    count_dir = clade_counts_dir(run_root)
    rank_dir = futureflu_rank_dir(run_root)
    labels: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for year in range(start_year - 1, end_year + 1):
        collection_mask = target_collection_mask(seq_df, year)
        submission_mask = collection_mask & submitted_before_cutoff_mask(seq_df, year)
        collection_df = seq_df.loc[collection_mask].copy()
        submission_df = seq_df.loc[submission_mask].copy()
        collection_counts = counted_clade_series(collection_df).value_counts()
        submission_counts = counted_clade_series(submission_df).value_counts()
        clades = sorted(set(collection_counts.index).union(set(submission_counts.index)))
        for clade in clades:
            count_rows.append(
                {
                    "year": year,
                    "hemisphere": HEMISPHERE,
                    "clade": clade,
                    "submission_count": int(submission_counts.get(clade, 0)),
                    "collection_count": int(collection_counts.get(clade, 0)),
                }
            )

        label_counts = valid_clade_series(collection_df).value_counts()
        if start_year <= year <= end_year and not label_counts.empty:
            labels.append(
                {
                    "subtype": SUBTYPE,
                    "hemisphere": HEMISPHERE,
                    "year": year,
                    "clade": label_counts.idxmax(),
                }
            )

    count_df = pd.DataFrame(count_rows)
    label_df = pd.DataFrame(labels)
    count_df.to_csv(
        count_dir / f"submission_collection_clade_count_{subtype_key()}.csv",
        index=False,
    )
    label_df.to_csv(rank_dir / "circulating_clade.csv", index=False)
    return label_df, count_df


def parse_clade_string(clade_str: object) -> list[tuple[str, float]]:
    if pd.isna(clade_str) or str(clade_str).strip().lower() in {"unknown", ""}:
        return []
    results = []
    for name, pct_str in CLADE_RE.findall(str(clade_str)):
        name = name.strip()
        if not is_excluded_clade(name):
            results.append((name, float(pct_str) / 100.0))
    return results


def compute_max_method(component_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
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
        for metric in METRICS:
            rec[f"fit_{metric}"] = row.get(metric, np.nan)
        rows.append(rec)

    if not rows:
        return pd.DataFrame(
            columns=["subtype", "hemisphere", "year", "clade_single"] + FIT_COLS
        )
    return (
        pd.DataFrame(rows)
        .groupby(["subtype", "hemisphere", "year", "clade_single"], as_index=False)
        .agg({f"fit_{metric}": "max" for metric in METRICS})
    )


def normalize_fit_columns(clade_df: pd.DataFrame) -> pd.DataFrame:
    result = clade_df.copy()
    group_keys = ["subtype", "hemisphere", "year"]

    for _, group in result.groupby(group_keys):
        idx = group.index
        for col in FIT_COLS:
            if col not in result.columns:
                continue
            vals = result.loc[idx, col]
            v_min = vals.min()
            v_max = vals.max()
            if pd.notna(v_min) and pd.notna(v_max) and v_max > v_min:
                result.loc[idx, col] = (vals - v_min) / (v_max - v_min)
            else:
                result.loc[idx, col] = 0.0
    return result


def compute_accuracy(
    clade_df: pd.DataFrame,
    label_df: pd.DataFrame,
    method_name: str,
) -> pd.DataFrame:
    acc_rows: list[dict[str, object]] = []
    if clade_df.empty or label_df.empty:
        return pd.DataFrame(
            columns=["subtype", "metric", "accuracy", "hit_count", "total_count", "methods"]
        )

    for subtype in sorted(clade_df["subtype"].unique()):
        sub_clade = clade_df[clade_df["subtype"] == subtype]
        sub_label = label_df[label_df["subtype"] == subtype]
        if sub_label.empty:
            continue
        label_hy = sub_label[["hemisphere", "year"]].drop_duplicates().reset_index(drop=True)
        total_count = len(label_hy)

        for metric in METRICS:
            col = f"fit_{metric}"
            if col not in sub_clade.columns:
                continue
            hit_count = 0
            for _, label_row in label_hy.iterrows():
                hemi = label_row["hemisphere"]
                year = label_row["year"]
                true_rows = sub_label.loc[
                    (sub_label["hemisphere"] == hemi) & (sub_label["year"] == year),
                    "clade",
                ]
                if true_rows.empty:
                    continue
                true_clade = true_rows.iloc[0]
                group = sub_clade[
                    (sub_clade["hemisphere"] == hemi) & (sub_clade["year"] == year)
                ]
                if group.empty or group[col].isna().all():
                    continue
                pred_clade = group.loc[group[col].idxmax(), "clade_single"]
                if pred_clade == true_clade:
                    hit_count += 1

            acc_rows.append(
                {
                    "subtype": subtype,
                    "metric": metric,
                    "accuracy": round(hit_count / total_count, 4)
                    if total_count > 0
                    else np.nan,
                    "hit_count": hit_count,
                    "total_count": total_count,
                    "methods": method_name,
                }
            )

    return pd.DataFrame(acc_rows)


def compute_max_method_with_info(
    component_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_keys = ["subtype", "hemisphere", "year", "clade_single"]
    rows: list[dict[str, object]] = []
    for _, row in component_df.iterrows():
        parsed = parse_clade_string(row["clade"])
        if not parsed:
            continue
        dominant = max(parsed, key=lambda item: item[1])[0]
        rec = {
            "subtype": row["subtype"],
            "hemisphere": row["hemisphere"],
            "year": row["year"],
            "clade_single": dominant,
            "risk_mutation_group": row.get("risk_mutation_group", np.nan),
            "mutation_count": row.get("mutation_count", np.nan),
            "mutation_group_seq_count": row.get("mutation_group_seq_count", np.nan),
        }
        for metric in METRICS_3:
            rec[f"fit_{metric}"] = row.get(metric, np.nan)
        rows.append(rec)

    empty_max = pd.DataFrame(columns=group_keys + [f"fit_{metric}" for metric in METRICS_3])
    empty_div = pd.DataFrame(columns=DIVERGENCE_COL_ORDER)
    empty_esc = pd.DataFrame(columns=ESCAPE_COL_ORDER)
    empty_gro = pd.DataFrame(columns=GROWTH_COL_ORDER)
    if not rows:
        return empty_max, empty_div, empty_esc, empty_gro

    tmp = pd.DataFrame(rows)
    max_df = (
        tmp.groupby(group_keys, as_index=False)
        .agg({f"fit_{metric}": "max" for metric in METRICS_3})
    )

    inform_dfs: dict[str, pd.DataFrame] = {}
    metric_info_map = [
        ("fit_mutual_information", "mutual_information", "divergence"),
        ("fit_total_escape", "total_escape", "escape"),
        ("fit_predicted_prevalence", "predicted_prevalence", "growth"),
    ]
    for fit_col, metric_col_name, out_key in metric_info_map:
        idx_max = tmp.groupby(group_keys, sort=False)[fit_col].idxmax().dropna().astype(int)
        select_cols = group_keys + [
            "risk_mutation_group",
            "mutation_count",
            "mutation_group_seq_count",
            fit_col,
        ]
        info_df = (
            tmp.loc[idx_max.values, select_cols]
            .copy()
            .rename(columns={"clade_single": "clade", fit_col: metric_col_name})
            .reset_index(drop=True)
        )
        inform_dfs[out_key] = info_df

    return max_df, inform_dfs["divergence"], inform_dfs["escape"], inform_dfs["growth"]


def load_clade_freq(run_root: Path, collection_based: bool = False) -> pd.DataFrame:
    """Load clade frequency tables. / 加载 clade 频率表。"""
    path = (
        clade_counts_dir(run_root)
        / f"submission_collection_clade_count_{subtype_key()}.csv"
    )
    cnt = pd.read_csv(path)
    cnt["hemisphere"] = cnt["hemisphere"].str.lower()
    count_col = "collection_count" if collection_based else "submission_count"
    total = cnt.groupby(["year", "hemisphere"])[count_col].sum().rename("total_count")
    cnt = cnt.merge(total.reset_index(), on=["year", "hemisphere"])
    cnt["xi_t"] = np.where(
        cnt["total_count"] > 0, cnt[count_col] / cnt["total_count"], np.nan
    )
    cnt["subtype"] = SUBTYPE
    return cnt[["subtype", "year", "hemisphere", "clade", "xi_t"]]


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def z_score_arr(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    mu = np.nanmean(vals)
    std = np.nanstd(vals, ddof=1)
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(vals)
    return (vals - mu) / std


def find_best_temperatures(
    past_list: list[dict[str, np.ndarray]],
    combo_metrics: list[str],
) -> tuple[dict[str, float], float]:
    n_m = len(combo_metrics)
    if not past_list:
        return {metric: 1.0 for metric in combo_metrics}, np.nan

    t_grid = np.array(list(product(T_VALUES, repeat=n_m)), dtype=float)
    total_loss = np.zeros(len(t_grid))
    valid_cnt = 0

    for data in past_list:
        z_mat = data["z_matrix"]
        xi_prev = data["xi_prev"]
        act_freq = data["actual_freq"]
        valid_cnt += 1

        log_sig = np.log(safe_sigmoid(z_mat[np.newaxis, :, :] / t_grid[:, :, np.newaxis]))
        fitness = log_sig.sum(axis=1)
        numerator = xi_prev[np.newaxis, :] * np.exp(fitness)
        denominator = numerator.sum(axis=1, keepdims=True)
        valid_mask = denominator.flatten() > 0
        pred_freq = np.where(denominator > 0, numerator / denominator, 0.0)
        loss = np.abs(pred_freq - act_freq[np.newaxis, :]).sum(axis=1)
        loss[~valid_mask] = np.inf
        total_loss += loss

    if valid_cnt == 0:
        return {metric: 1.0 for metric in combo_metrics}, np.nan

    mean_loss = total_loss / valid_cnt
    best_idx = int(np.argmin(mean_loss))
    best_mean_loss = float(mean_loss[best_idx])
    if np.isinf(best_mean_loss):
        best_mean_loss = np.nan
    return (
        {combo_metrics[i]: float(t_grid[best_idx, i]) for i in range(n_m)},
        best_mean_loss,
    )


def compute_combination_accuracy(
    max_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    label_df: pd.DataFrame,
    freq_df_coll: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc_rows: list[dict[str, object]] = []
    egdfit_rows: list[dict[str, object]] = []
    egdtemp_rows: list[dict[str, object]] = []
    lpd_rows: list[dict[str, object]] = []
    pre_act_dict: dict[tuple[object, object, object, object], dict[str, object]] = {}

    if max_df.empty or label_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(columns=PRE_ACT_COL_ORDER),
        )

    for combo_name, combo_metrics in COMBINATIONS:
        for subtype in sorted(max_df["subtype"].unique()):
            sub_clade = max_df[max_df["subtype"] == subtype]
            sub_label = label_df[label_df["subtype"] == subtype]
            sub_freq = freq_df[freq_df["subtype"] == subtype]
            sub_freq_coll = freq_df_coll[freq_df_coll["subtype"] == subtype]
            if sub_label.empty:
                continue

            label_hy = sub_label[["hemisphere", "year"]].drop_duplicates().reset_index(drop=True)
            total_count = len(label_hy)

            all_seasons: set[tuple[int, str]] = set()
            for hemi in sub_clade["hemisphere"].unique():
                years = sub_clade.loc[sub_clade["hemisphere"] == hemi, "year"].unique()
                for year in years:
                    all_seasons.add((int(year), str(hemi).lower()))

            def season_sort_key(hy: tuple[int, str]) -> tuple[int, int]:
                year, hemi = hy
                hemi_order = 0 if hemi.lower() == "south" else 1
                return year, hemi_order

            sorted_seasons = sorted(all_seasons, key=season_sort_key)
            if not sorted_seasons:
                continue

            per_season: dict[tuple[int, str], dict[str, np.ndarray | None]] = {}
            for year, hemi in sorted_seasons:
                hemi_clade = sub_clade[sub_clade["hemisphere"].str.lower() == hemi]
                hemi_freq = sub_freq[sub_freq["hemisphere"].str.lower() == hemi]
                group = hemi_clade[hemi_clade["year"] == year].reset_index(drop=True)
                clades = group["clade_single"].values.copy()
                z_mat = np.vstack(
                    [
                        z_score_arr(group[f"fit_{metric}"].fillna(0.0).values)
                        for metric in combo_metrics
                    ]
                )

                freq_prev = hemi_freq[hemi_freq["year"] == year - 1]
                clade_to_prev = dict(zip(freq_prev["clade"], freq_prev["xi_t"]))
                xi_prev = np.array(
                    [np.nan_to_num(clade_to_prev.get(clade, 0.0), nan=0.0) for clade in clades]
                )

                freq_cur = hemi_freq[hemi_freq["year"] == year]
                present_set = set(clades)
                freq_cur_sub = freq_cur[freq_cur["clade"].isin(present_set)]
                raw_sum = freq_cur_sub["xi_t"].fillna(0.0).sum()
                if freq_cur_sub.empty or raw_sum <= 0:
                    act_freq = None
                else:
                    clade_to_cur = dict(zip(freq_cur_sub["clade"], freq_cur_sub["xi_t"].fillna(0.0)))
                    act_arr = np.array([clade_to_cur.get(clade, 0.0) for clade in clades])
                    act_freq = act_arr / act_arr.sum()

                per_season[(year, hemi)] = {
                    "clades": clades,
                    "z_mat": z_mat,
                    "xi_prev": xi_prev,
                    "act_freq": act_freq,
                }

            stored: dict[tuple[str, int], dict[str, object]] = {}
            for idx, (year, hemi) in enumerate(sorted_seasons):
                if idx == 0:
                    t_best = {metric: 1.0 for metric in combo_metrics}
                    best_mean_loss = np.nan
                else:
                    history_seasons = sorted_seasons[: max(0, idx - 1)]
                    past_data = [
                        {
                            "z_matrix": per_season[(py, ph)]["z_mat"],
                            "xi_prev": per_season[(py, ph)]["xi_prev"],
                            "actual_freq": per_season[(py, ph)]["act_freq"],
                        }
                        for py, ph in history_seasons
                        if per_season[(py, ph)]["act_freq"] is not None
                    ]
                    t_best, best_mean_loss = find_best_temperatures(past_data, combo_metrics)

                stored[(hemi, year)] = {
                    "clades": per_season[(year, hemi)]["clades"],
                    "z_mat": per_season[(year, hemi)]["z_mat"],
                    "xi_prev": per_season[(year, hemi)]["xi_prev"],
                    "T": t_best,
                    "best_mean_loss": best_mean_loss,
                }

            if combo_name == "E+G+D":
                for (hemi, year), data in stored.items():
                    clades = data["clades"]
                    z_mat = data["z_mat"]
                    t_best = data["T"]
                    t_col = np.array([t_best[metric] for metric in combo_metrics], dtype=float)[
                        :, np.newaxis
                    ]
                    fitness = np.log(safe_sigmoid(z_mat / t_col)).sum(axis=0)
                    for idx, clade in enumerate(clades):
                        egdfit_rows.append(
                            {
                                "subtype": subtype,
                                "hemisphere": hemi,
                                "year": year,
                                "clade": clade,
                                "fit_E+G+D": round(float(fitness[idx]), 6),
                            }
                        )

                    top1_clade = clades[np.argmax(fitness)] if len(clades) > 0 else None
                    true_rows = sub_label[
                        (sub_label["hemisphere"].str.lower() == hemi)
                        & (sub_label["year"] == year)
                    ]
                    if not true_rows.empty and top1_clade is not None:
                        true_clade = true_rows["clade"].iloc[0]
                        tf_value = "T" if top1_clade == true_clade else "F"
                    else:
                        tf_value = None

                    egdtemp_rows.append(
                        {
                            "subtype": subtype,
                            "hemisphere": hemi,
                            "year": year,
                            "tem_E": t_best["total_escape"],
                            "tem_G": t_best["predicted_prevalence"],
                            "tem_D": t_best["mutual_information"],
                            "mean_l1_loss": data["best_mean_loss"],
                            "TOP1_clade": top1_clade,
                            "T/F": tf_value,
                        }
                    )

            for (hemi, year), data in stored.items():
                clades = data["clades"]
                z_mat = data["z_mat"]
                xi_prev = data["xi_prev"]
                t_best = data["T"]
                if len(clades) == 0:
                    continue

                t_col = np.array([t_best[metric] for metric in combo_metrics], dtype=float)[
                    :, np.newaxis
                ]
                fitness = np.log(safe_sigmoid(z_mat / t_col)).sum(axis=0)
                fit_freq_arr = fitness.copy()

                numerator = xi_prev * np.exp(fitness)
                denominator = numerator.sum()
                if denominator > 0:
                    freq_freq_arr = numerator / denominator
                else:
                    freq_freq_arr = np.full(len(clades), np.nan)

                hemi_freq_coll = sub_freq_coll[sub_freq_coll["hemisphere"].str.lower() == hemi]
                fc_coll = hemi_freq_coll[hemi_freq_coll["year"] == year]
                fc_coll_sub = fc_coll[fc_coll["clade"].isin(set(clades))]
                raw_sum_coll = fc_coll_sub["xi_t"].fillna(0.0).sum()
                if fc_coll_sub.empty or raw_sum_coll <= 0:
                    act_freq_coll = None
                else:
                    clade_to_act_coll = dict(
                        zip(fc_coll_sub["clade"], fc_coll_sub["xi_t"].fillna(0.0))
                    )
                    act_arr_coll = np.array(
                        [clade_to_act_coll.get(clade, 0.0) for clade in clades]
                    )
                    act_freq_coll = act_arr_coll / act_arr_coll.sum()

                for idx, clade in enumerate(clades):
                    pk = (subtype, hemi, year, clade)
                    if pk not in pre_act_dict:
                        act_val = (
                            float(act_freq_coll[idx])
                            if act_freq_coll is not None
                            else np.nan
                        )
                        prev_val = float(xi_prev[idx])
                        pre_act_dict[pk] = {
                            "subtype": subtype,
                            "year": year,
                            "hemisphere": hemi,
                            "clade": clade,
                            "act_freq": round(act_val, 6) if np.isfinite(act_val) else np.nan,
                            "freq_prev": round(prev_val, 6)
                            if np.isfinite(prev_val)
                            else np.nan,
                        }
                    ffit = float(fit_freq_arr[idx])
                    ffreq = float(freq_freq_arr[idx])
                    pre_act_dict[pk][f"{combo_name}_pre_fit"] = (
                        round(ffit, 6) if np.isfinite(ffit) else np.nan
                    )
                    pre_act_dict[pk][f"{combo_name}_pre_freq"] = (
                        round(ffreq, 6) if np.isfinite(ffreq) else np.nan
                    )

            hit_fit = 0
            hit_freq = 0
            for _, label_row in label_hy.iterrows():
                hemi = str(label_row["hemisphere"]).lower()
                year = int(label_row["year"])
                true_rows = sub_label[
                    (sub_label["hemisphere"].str.lower() == hemi)
                    & (sub_label["year"] == year)
                ]
                if true_rows.empty:
                    continue
                true_clade = true_rows["clade"].iloc[0]

                key = (hemi, year)
                if key not in stored:
                    lpd_rows.append(
                        {
                            "subtype": subtype,
                            "hemisphere": hemi,
                            "year": year,
                            "combo": combo_name,
                            "lpd": np.nan,
                        }
                    )
                    continue

                data = stored[key]
                clades = data["clades"]
                z_mat = data["z_mat"]
                xi_prev = data["xi_prev"]
                t_best = data["T"]
                if len(clades) == 0:
                    lpd_rows.append(
                        {
                            "subtype": subtype,
                            "hemisphere": hemi,
                            "year": year,
                            "combo": combo_name,
                            "lpd": np.nan,
                        }
                    )
                    continue

                t_col = np.array([t_best[metric] for metric in combo_metrics], dtype=float)[
                    :, np.newaxis
                ]
                fitness = np.log(safe_sigmoid(z_mat / t_col)).sum(axis=0)
                pred_fit_clade = clades[np.argmax(fitness)]
                if pred_fit_clade == true_clade:
                    hit_fit += 1

                numerator = xi_prev * np.exp(fitness)
                denominator = numerator.sum()
                if denominator > 0:
                    pred_freq_arr = numerator / denominator
                    pred_freq_clade = clades[np.argmax(numerator)]
                    if pred_freq_clade == true_clade:
                        hit_freq += 1
                    true_idx_arr = np.where(clades == true_clade)[0]
                    if len(true_idx_arr) > 0:
                        pred_prob = max(float(pred_freq_arr[true_idx_arr[0]]), 1e-10)
                    else:
                        pred_prob = 1e-10
                    lpd_value = float(np.log(pred_prob))
                else:
                    lpd_value = float(np.log(1e-10))

                lpd_rows.append(
                    {
                        "subtype": subtype,
                        "hemisphere": hemi,
                        "year": year,
                        "combo": combo_name,
                        "lpd": lpd_value,
                    }
                )

            for method_name, hit in [("fit", hit_fit), ("freq", hit_freq)]:
                acc_rows.append(
                    {
                        "subtype": subtype,
                        "metric_combine": combo_name,
                        "accuracy": round(hit / total_count, 4)
                        if total_count > 0
                        else np.nan,
                        "hit_count": hit,
                        "total_count": total_count,
                        "methods": method_name,
                    }
                )

    if pre_act_dict:
        pre_act_df = pd.DataFrame(list(pre_act_dict.values()))
        for col in PRE_ACT_COL_ORDER:
            if col not in pre_act_df.columns:
                pre_act_df[col] = np.nan
        pre_act_df = pre_act_df[PRE_ACT_COL_ORDER].sort_values(
            ["subtype", "year", "hemisphere", "clade"]
        )
    else:
        pre_act_df = pd.DataFrame(columns=PRE_ACT_COL_ORDER)

    return (
        pd.DataFrame(acc_rows),
        pd.DataFrame(egdfit_rows),
        pd.DataFrame(egdtemp_rows),
        pd.DataFrame(lpd_rows),
        pre_act_df.reset_index(drop=True),
    )


def build_elpd_aic(lpd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ELPD and AIC summary tables. / 计算 ELPD 与 AIC 汇总表。"""
    if lpd_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    elpd_pivot = (
        lpd_df.pivot_table(
            index=["subtype", "hemisphere", "year"],
            columns="combo",
            values="lpd",
            aggfunc="first",
        )
        .reset_index()
    )
    elpd_pivot.columns.name = None
    for combo in COMBO_NAMES_ORDERED:
        if combo not in elpd_pivot.columns:
            elpd_pivot[combo] = np.nan
    elpd_pivot = elpd_pivot.rename(
        columns={combo: f"ELPD_{combo}" for combo in COMBO_NAMES_ORDERED}
    )
    elpd_cols = [f"ELPD_{combo}" for combo in COMBO_NAMES_ORDERED]
    elpd_pivot = elpd_pivot[["subtype", "hemisphere", "year"] + elpd_cols]

    summary_rows = []
    for subtype in sorted(elpd_pivot["subtype"].unique()):
        sub = elpd_pivot[elpd_pivot["subtype"] == subtype]
        row = {"subtype": subtype, "hemisphere": "Summary", "year": "All"}
        for col in elpd_cols:
            row[col] = round(float(np.nansum(sub[col])), 4)
        summary_rows.append(row)
    elpd_final = pd.concat([elpd_pivot, pd.DataFrame(summary_rows)], ignore_index=True)

    aic_rows = []
    n_params = {combo: len(metrics) for combo, metrics in COMBINATIONS}
    for subtype in sorted(elpd_pivot["subtype"].unique()):
        sub = elpd_pivot[elpd_pivot["subtype"] == subtype]
        row = {"subtype": subtype}
        for combo in COMBO_NAMES_ORDERED:
            col = f"ELPD_{combo}"
            elpd_sum = float(np.nansum(sub[col]))
            row[f"AIC_{combo}"] = round(2 * n_params[combo] - 2 * elpd_sum, 4)
        aic_rows.append(row)
    return elpd_final, pd.DataFrame(aic_rows)


def run_auxiliary_tables(
    component_df: pd.DataFrame,
    label_df: pd.DataFrame,
    run_root: Path,
) -> None:
    """Write E/G/D combination, temperature, ELPD/AIC, and pre_act tables.

    Single-metric clade_accuracy tables (HA1 calculate_clade_accuracy /
    analyze_components style) are intentionally omitted.
    """
    combine_dir = risk_out(run_root) / "component_combinations"
    combine_dir.mkdir(parents=True, exist_ok=True)

    combine_max_df, divergence_info, escape_info, growth_info = compute_max_method_with_info(
        component_df
    )
    freq_df = load_clade_freq(run_root, collection_based=False)
    freq_coll = load_clade_freq(run_root, collection_based=True)
    acc_df, egdfit_df, egdtemp_df, lpd_df, pre_act_df = compute_combination_accuracy(
        combine_max_df, freq_df, label_df, freq_coll
    )

    acc_df.to_csv(combine_dir / "clade_component_combine_acc_Twindow.csv", index=False)
    egdfit_df.to_csv(combine_dir / "EGD_combine_Twindow.csv", index=False)
    egdtemp_df.to_csv(combine_dir / "EGD_temperatures_Twindow.csv", index=False)
    pre_act_df.to_csv(combine_dir / "clade_pre_act_Twindow.csv", index=False)

    elpd_df, aic_df = build_elpd_aic(lpd_df)
    elpd_df.to_csv(combine_dir / "elpd_Twindow.csv", index=False)
    aic_df.to_csv(combine_dir / "aic_Twindow.csv", index=False)

    for path, df_info, columns in [
        (combine_dir / "divergence_Twindow.csv", divergence_info, DIVERGENCE_COL_ORDER),
        (combine_dir / "escape_Twindow.csv", escape_info, ESCAPE_COL_ORDER),
        (combine_dir / "growth_Twindow.csv", growth_info, GROWTH_COL_ORDER),
    ]:
        for col in columns:
            if col not in df_info.columns:
                df_info[col] = np.nan
        df_info[columns].sort_values(["subtype", "hemisphere", "year", "clade"]).to_csv(
            path, index=False
        )


def write_year_outputs(
    seq_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    run_root: Path,
    start_year: int,
    end_year: int,
    theta: float,
) -> pd.DataFrame:
    """Write yearly linear result tables. / 写出逐年线性结果表。"""
    status_rows: list[dict[str, object]] = []
    rate_lookup = rates_df.set_index("year").to_dict("index") if not rates_df.empty else {}

    for year in range(start_year, end_year + 1):
        print(f"[linear] theta={theta:.2f} {SUBTYPE} {HEMISPHERE} {year}", flush=True)
        prefix = f"{SUBTYPE}_{HEMISPHERE}_{year}"
        year_dir = linear_out(run_root) / f"{SUBTYPE}_{HEMISPHERE}" / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        prevalence = site_prevalence_all_sites(seq_df, year)
        prevalence_path = year_dir / f"{prefix}_prevalence.csv"
        prevalence.to_csv(prevalence_path, index=False)

        gmeasure_df = gmeasure(prevalence, [theta])
        gmeasure_path = year_dir / f"{prefix}_gmeasure.csv"
        gmeasure_df.to_csv(gmeasure_path, index=False)

        prev_indexed = prevalence.set_index("season") if not prevalence.empty else prevalence
        mutations = predict_mutations_linear(year, theta, prev_indexed)
        mutations_path = year_dir / f"{prefix}_mutations.csv"
        mutations.to_csv(mutations_path, index=False)

        distribution = analyze_risk_mutations_year(seq_df, mutations, year)
        distribution_path = year_dir / f"{prefix}_distribution.csv"
        distribution.to_csv(distribution_path, index=False)

        rate_info = rate_lookup.get(year, {})
        # Release layout matches FutureFlu package layout: keep yearly distribution /
        # gmeasure / mutations / prevalence only (no theta_fitting side table).

        status_rows.append(
            {
                "year": year,
                "theta": theta,
                "historical_years": int(len(prevalence)),
                "risk_mutations": int(len(mutations)),
                "risk_mutation_groups": int(len(distribution)),
                "candidate_sequence_year": candidate_sequence_year(year),
                "candidate_sequences": int(
                    len(candidate_sequences_for_year(seq_df, year))
                ),
                "target_sequences": int(target_collection_mask(seq_df, year).sum()),
                "positive_rate_available": bool(rate_info),
                "positive_rate_percent": rate_info.get(
                    "norovirus_positive_rate_percent", np.nan
                ),
            }
        )

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(process_metadata_dir(run_root) / "linear_status_by_year.csv", index=False)
    return status_df


def run_theta(
    seq_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    theta: float,
) -> None:
    """Run the FutureFlu-like workflow for one or more thetas.

    English: Orchestrate prevalence, mutation prediction, and component outputs.
    中文：编排流行度、突变预测与组分输出。
    """
    run_root = OUTPUT_ROOT
    default_outputs = (ROOT / "outputs").resolve()
    if Path(OUTPUT_ROOT).resolve() != default_outputs:
        run_root = OUTPUT_ROOT / theta_label(theta)
    ensure_dirs(run_root)
    write_preprocessed_sequence_table(seq_df, run_root)
    rates_df.to_csv(process_metadata_dir(run_root) / "positive_rates_used.csv", index=False)

    label_df, count_df = build_label_and_count_inputs(seq_df, run_root, start_year, end_year)
    count_df.to_csv(process_metadata_dir(run_root) / "genotype_counts_by_year.csv", index=False)
    label_df.to_csv(process_metadata_dir(run_root) / "dominant_genotype_labels.csv", index=False)

    linear_status = write_year_outputs(seq_df, rates_df, run_root, start_year, end_year, theta)
    component_df = build_component_table(seq_df, run_root, start_year, end_year)
    run_auxiliary_tables(component_df, label_df, run_root)

    component_status_path = process_metadata_dir(run_root) / "component_status_by_year.csv"
    component_status = (
        pd.read_csv(component_status_path)
        if component_status_path.exists()
        else pd.DataFrame(columns=["year", "component_rows"])
    )
    summary = linear_status.merge(
        component_status[["year", "component_rows", "status"]],
        on="year",
        how="left",
    )
    summary.to_csv(release_metadata_dir(run_root) / "summary_by_year.csv", index=False)
    print(f"[done] theta={format_theta(theta)} output={run_root}", flush=True)


def parse_thetas(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one theta is required")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the package-local norovirus FutureFlu-like theta=0.1 workflow."
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--thetas", default="0.1")
    parser.add_argument("--max-gaps", type=int, default=DEFAULT_MAX_GAPS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be >= --start-year")
    thetas = parse_thetas(args.thetas)

    validate_evescape_outputs(args.start_year, args.end_year)
    seq_df = load_sequence_table(args.max_gaps)
    rates_df = load_positive_rates()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "start_year": args.start_year,
                "end_year": args.end_year,
                "thetas": ",".join(format_theta(theta) for theta in thetas),
                "max_gaps": args.max_gaps,
                "sequence_rows_after_gap_filter": len(seq_df),
                "subtype": SUBTYPE,
                "hemisphere": HEMISPHERE,
                "clade_column": "genotype",
                "season_window": "calendar_year",
                "submission_cutoff": "season_end_exclusive",
                "candidate_sequence_window": "previous_calendar_year",
                "submission_date_source": seq_df.attrs.get(
                    "submission_date_source", "unknown"
                ),
                "site_region": f"{EPITOPE_START}-{EPITOPE_END}",
                "site_region_start": EPITOPE_START,
                "site_region_end": EPITOPE_END,
                "positive_rate_file": positive_rate_file_label(),
                "evescape_source": str(EVESCAPE_SCORE_DIR.relative_to(ROOT)),
            }
        ]
    ).to_csv(OUTPUT_ROOT / "run_config.csv", index=False)

    for theta in thetas:
        run_theta(seq_df, rates_df, args.start_year, args.end_year, theta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
