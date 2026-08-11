"""Build FutureFlu issue-date run inputs and timepoint summaries.

English: Runs prepare, timepoint, aggregate, and support-table steps for one configured lineage.
中文：为单条配置 lineage 运行 prepare、timepoint、aggregate 和辅助表步骤。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))

import numpy as np
import pandas as pd
from Bio import Phylo
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT
FUTUREFLU_DATA_ROOT = PROJECT_ROOT / "data" / "futureflu"
FUTUREFLU_RESULTS_ROOT = PROJECT_ROOT / "results" / "futureflu"
FUTUREFLU_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "futureflu"


def resolve_flu_forecasting_root() -> Path:
    """Locate the external blab/flu-forecasting checkout used for engine scripts.

    This package no longer vendors upstream Snakefile/rules/src/scripts.
    Set FLU_FORECASTING_ROOT, or place a checkout next to this package
    (../flu-forecasting). An optional local ./flu-forecasting is also accepted
    but should not be committed (see .gitignore).
    """
    env_value = os.environ.get("FLU_FORECASTING_ROOT", "").strip()
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT.parent / "flu-forecasting",
            PROJECT_ROOT / "flu-forecasting",
        ]
    )
    for candidate in candidates:
        if (candidate / "scripts" / "frequencies.py").exists() and (candidate / "src").exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "(none)"
    raise FileNotFoundError(
        "flu-forecasting upstream checkout not found. Clone "
        "https://github.com/blab/flu-forecasting and set FLU_FORECASTING_ROOT. "
        f"Searched: {searched}"
    )


FLU_ROOT: Path | None = None


def flu_root() -> Path:
    """Return the resolved flu-forecasting root (cached)."""
    global FLU_ROOT
    if FLU_ROOT is None:
        FLU_ROOT = resolve_flu_forecasting_root()
    return FLU_ROOT


INPUT_METADATA = PROJECT_ROOT / "data" / "sequences" / "h3n2_futureflu_metadata.tsv"
INPUT_FASTA = PROJECT_ROOT / "data" / "sequences" / "h3n2_futureflu_aa_sequences.fasta"
INPUT_SEQUENCE_TABLE = PROJECT_ROOT / "data" / "sequences" / "h3n2_sequence_table.tsv"
REPORT_LABEL = "H3N2"

RUN_DIR = FUTUREFLU_RESULTS_ROOT / "runs" / "H3N2"
INPUTS_DIR = RUN_DIR / "inputs"
TIMEPOINTS_DIR = RUN_DIR / "timepoints"
FORECASTS_DIR = RUN_DIR / "forecasts"
ARTIFACTS_DIR = FUTUREFLU_RESULTS_ROOT / "artifacts"

PREPARE_REPORT = ARTIFACTS_DIR / "H3N2_prepare.md"
TIMEPOINTS_REPORT = ARTIFACTS_DIR / "H3N2_timepoints.md"
AGGREGATE_REPORT = ARTIFACTS_DIR / "H3N2_aggregate.md"
MODELING_REPORT = ARTIFACTS_DIR / "H3N2_modeling.md"

NORMALIZED_METADATA = INPUTS_DIR / "normalized_metadata.tsv"
GLOBAL_STRAINS = INPUTS_DIR / "global_subsampled_strains.txt"
GLOBAL_METADATA = INPUTS_DIR / "global_subsampled_metadata.tsv"
GLOBAL_FASTA = INPUTS_DIR / "global_subsampled_aa_sequences.fasta"
ISSUE_MANIFEST = INPUTS_DIR / "issue_manifest.tsv"

AGGREGATED_TIP_ATTRIBUTES = RUN_DIR / "tip_attributes.tsv"
AGGREGATED_TIP_ATTRIBUTES_NAIVE = RUN_DIR / "tip_attributes_with_naive.tsv"

GLOBAL_START = pd.Timestamp("2004-02-01")
GLOBAL_END = pd.Timestamp("2015-02-01")
ISSUE_START = pd.Timestamp("2010-02-01")
ISSUE_END = pd.Timestamp("2015-02-01")
FORECAST_YEARS = [2013, 2014]
YEARS_BACK_TO_BUILD = 6
YEARS_BACK_FOR_TITER_COMPARISON = 5
DIFFUSION_YEARS_BACK = 2
PIVOT_INTERVAL_MONTHS = 6
VIRUSES_PER_MONTH = 90
RANDOM_SEED = 0
DELTA_PIVOTS = 1
MAX_SEQUENCE_GAPS = 3
ENABLE_HI_PREDICTORS = True
ALLOW_HI_FALLBACK = False
ENABLE_DMS_PREDICTORS = True
TIMEPOINT_WORKERS = 1
IQTREE_THREADS = "AUTO"

SIGPEP_SLICE = slice(0, 16)
HA1_SLICE = slice(16, 345)
HA2_SLICE = slice(345, 566)

ADDED_PREDICTOR_COLUMNS = [
    "ep",
    "ep_wolf",
    "ne",
    "rb",
    "dms",
    "ep_star",
    "ne_star",
    "dms_star",
    "dms_entropy",
    "dms_nonepitope",
    "delta_frequency",
    "distance_from_consensus",
    "ep_x",
    "ep_x_wolf",
    "ep_x_koel",
    "oracle_x",
]

ISSUE_DATE_TABLE_COLUMNS = ADDED_PREDICTOR_COLUMNS + [
    "clade_membership",
    "country_entropy",
    "country_metadata",
    "region_entropy",
    "region_metadata",
    "cTiterSub",
    "dTiterSub",
]

FORECAST_REGIONS = [
    "africa",
    "europe",
    "north_america",
    "china",
    "south_asia",
    "japan_korea",
    "oceania",
    "south_america",
    "southeast_asia",
    "west_asia",
]

ASIA_BUCKETS = {
    "china": "china",
    "hong kong": "china",
    "hong_kong": "china",
    "taiwan": "china",
    "japan": "japan_korea",
    "south korea": "japan_korea",
    "korea": "japan_korea",
    "india": "south_asia",
    "pakistan": "south_asia",
    "bangladesh": "south_asia",
    "sri lanka": "south_asia",
    "nepal": "south_asia",
    "bhutan": "south_asia",
    "maldives": "south_asia",
    "indonesia": "southeast_asia",
    "thailand": "southeast_asia",
    "vietnam": "southeast_asia",
    "singapore": "southeast_asia",
    "malaysia": "southeast_asia",
    "philippines": "southeast_asia",
    "cambodia": "southeast_asia",
    "laos": "southeast_asia",
    "lao, people's democratic republic": "southeast_asia",
    "myanmar": "southeast_asia",
    "brunei": "southeast_asia",
    "timor-leste": "southeast_asia",
    "united arab emirates": "west_asia",
    "saudi arabia": "west_asia",
    "qatar": "west_asia",
    "oman": "west_asia",
    "kuwait": "west_asia",
    "bahrain": "west_asia",
    "iraq": "west_asia",
    "iran": "west_asia",
    "jordan": "west_asia",
    "lebanon": "west_asia",
    "israel": "west_asia",
    "palestine": "west_asia",
    "yemen": "west_asia",
    "syria": "west_asia",
    "turkey": "west_asia",
    "georgia": "west_asia",
    "armenia": "west_asia",
    "azerbaijan": "west_asia",
}

def get_iqtree_args() -> list[str]:
    return [
        "iqtree",
        "-st",
        "AA",
        "-m",
        "LG",
        "-nt",
        str(IQTREE_THREADS),
        "-redo",
    ]

DISTANCE_MAP_ROOT: Path | None = None


def default_h3n2_distance_map_root() -> Path:
    """H3N2 epitope/DMS maps come from upstream flu-forecasting, not this package."""
    return flu_root() / "config" / "distance_maps" / "h3n2" / "ha"


def upstream_titer_model_root() -> Path:
    return (
        flu_root()
        / "results"
        / "builds"
        / "natural"
        / "natural_sample_20191001"
        / "timepoints"
    )

DISTANCE_MODEL_SPECS = [
    ("root", "ep", "luksza.json"),
    ("root", "ep_wolf", "wolf.json"),
    ("root", "ne", "luksza_nonepitope.json"),
    ("root", "rb", "koel.json"),
    ("root", "dms", "dms_mutation_effect.json"),
    ("ancestor", "ep_star", "luksza.json"),
    ("ancestor", "ne_star", "luksza_nonepitope.json"),
    ("ancestor", "dms_star", "dms_mutation_effect.json"),
    ("ancestor", "dms_entropy", "dms_entropy.json"),
    ("ancestor", "dms_nonepitope", "dms_nonepitope.json"),
]

MODEL_PREDICTORS = [
    "naive",
    "lbi",
    "cTiterSub_x",
    "ep_x",
    "ep_x_wolf",
    "ep_x_koel",
    "oracle_x",
    "delta_frequency",
    "distance_from_consensus",
    "ep_star",
    "ne_star",
    "dms_star",
    "dms_nonepitope",
    "dms_entropy",
    "ep_star,ne_star",
    "ep_x,ne_star",
    "ne_star,lbi",
    "cTiterSub_x,ne_star",
]


def configure_paths(run_dir: Path) -> None:
    global RUN_DIR, INPUTS_DIR, TIMEPOINTS_DIR, FORECASTS_DIR
    global PREPARE_REPORT, TIMEPOINTS_REPORT, AGGREGATE_REPORT, MODELING_REPORT
    global NORMALIZED_METADATA, GLOBAL_STRAINS, GLOBAL_METADATA, GLOBAL_FASTA, ISSUE_MANIFEST
    global AGGREGATED_TIP_ATTRIBUTES, AGGREGATED_TIP_ATTRIBUTES_NAIVE

    if not run_dir.is_absolute():
        if run_dir.parts and run_dir.parts[0] == "results":
            run_dir = PROJECT_ROOT / run_dir
        elif run_dir.parts and run_dir.parts[0] == "runs":
            run_dir = FUTUREFLU_RESULTS_ROOT / run_dir
        else:
            run_dir = FUTUREFLU_RESULTS_ROOT / "runs" / run_dir

    RUN_DIR = run_dir.resolve()
    INPUTS_DIR = RUN_DIR / "inputs"
    TIMEPOINTS_DIR = RUN_DIR / "timepoints"
    FORECASTS_DIR = RUN_DIR / "forecasts"

    report_stem = RUN_DIR.name
    PREPARE_REPORT = ARTIFACTS_DIR / f"{report_stem}_prepare.md"
    TIMEPOINTS_REPORT = ARTIFACTS_DIR / f"{report_stem}_timepoints.md"
    AGGREGATE_REPORT = ARTIFACTS_DIR / f"{report_stem}_aggregate.md"
    MODELING_REPORT = ARTIFACTS_DIR / f"{report_stem}_modeling.md"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    NORMALIZED_METADATA = INPUTS_DIR / "normalized_metadata.tsv"
    GLOBAL_STRAINS = INPUTS_DIR / "global_subsampled_strains.txt"
    GLOBAL_METADATA = INPUTS_DIR / "global_subsampled_metadata.tsv"
    GLOBAL_FASTA = INPUTS_DIR / "global_subsampled_aa_sequences.fasta"
    ISSUE_MANIFEST = INPUTS_DIR / "issue_manifest.tsv"

    AGGREGATED_TIP_ATTRIBUTES = RUN_DIR / "tip_attributes.tsv"
    AGGREGATED_TIP_ATTRIBUTES_NAIVE = RUN_DIR / "tip_attributes_with_naive.tsv"


def get_required_table_columns() -> list[str]:
    added_predictor_columns = list(ADDED_PREDICTOR_COLUMNS)
    if not ENABLE_DMS_PREDICTORS:
        added_predictor_columns = [
            column for column in added_predictor_columns if not column.startswith("dms")
        ]
    columns = [
        *added_predictor_columns,
        "clade_membership",
        "country_entropy",
        "country_metadata",
        "region_entropy",
        "region_metadata",
        "age",
        "gender",
    ]
    if ENABLE_HI_PREDICTORS:
        columns.extend(["cTiterSub", "dTiterSub"])
    return columns


def apply_runtime_configuration(args: argparse.Namespace) -> None:
    global INPUT_METADATA, INPUT_FASTA, INPUT_SEQUENCE_TABLE, DISTANCE_MAP_ROOT, REPORT_LABEL
    global GLOBAL_START, GLOBAL_END, ISSUE_START, ISSUE_END, FORECAST_YEARS
    global ENABLE_HI_PREDICTORS, ALLOW_HI_FALLBACK, ENABLE_DMS_PREDICTORS, MODEL_PREDICTORS
    global TIMEPOINT_WORKERS, IQTREE_THREADS

    if args.run_dir:
        configure_paths(Path(args.run_dir))
    if args.input_metadata:
        INPUT_METADATA = Path(args.input_metadata).resolve()
    if args.input_fasta:
        INPUT_FASTA = Path(args.input_fasta).resolve()
    if args.input_sequence_table:
        INPUT_SEQUENCE_TABLE = Path(args.input_sequence_table).resolve()
    if args.distance_map_root:
        DISTANCE_MAP_ROOT = Path(args.distance_map_root).resolve()
    elif DISTANCE_MAP_ROOT is None:
        DISTANCE_MAP_ROOT = default_h3n2_distance_map_root()
    if args.report_label:
        REPORT_LABEL = args.report_label
    if args.global_start:
        GLOBAL_START = pd.Timestamp(args.global_start)
    if args.global_end:
        GLOBAL_END = pd.Timestamp(args.global_end)
    if args.issue_start:
        ISSUE_START = pd.Timestamp(args.issue_start)
    if args.issue_end:
        ISSUE_END = pd.Timestamp(args.issue_end)
    if args.forecast_years:
        FORECAST_YEARS = args.forecast_years
    if args.disable_hi:
        ENABLE_HI_PREDICTORS = False
    if args.disable_dms:
        ENABLE_DMS_PREDICTORS = False
    if args.allow_hi_fallback:
        ALLOW_HI_FALLBACK = True
    if args.predictors:
        MODEL_PREDICTORS = args.predictors
    if args.timepoint_workers:
        TIMEPOINT_WORKERS = args.timepoint_workers
    if args.iqtree_threads:
        IQTREE_THREADS = args.iqtree_threads
    if args.disable_hi:
        MODEL_PREDICTORS = [
            predictor for predictor in MODEL_PREDICTORS if "cTiterSub_x" not in predictor.split(",")
        ]
    if args.disable_dms:
        MODEL_PREDICTORS = [
            predictor
            for predictor in MODEL_PREDICTORS
            if not any(part.startswith("dms") for part in predictor.split(","))
        ]


def get_distance_model_specs() -> list[tuple[str, str, str]]:
    if ENABLE_DMS_PREDICTORS:
        return DISTANCE_MODEL_SPECS
    return [spec for spec in DISTANCE_MODEL_SPECS if not spec[1].startswith("dms")]


def timestamp_to_float(timepoint: pd.Timestamp) -> float:
    return timepoint.year + ((timepoint.month - 1) / 12.0)


def build_issue_dates() -> list[pd.Timestamp]:
    issue_dates = []
    for year in range(ISSUE_START.year, ISSUE_END.year + 1):
        for month in [2, 9]:
            issue_date = pd.Timestamp(f"{year}-{month:02d}-01")
            if ISSUE_START <= issue_date <= ISSUE_END:
                issue_dates.append(issue_date)
    return sorted(issue_dates)


def issue_to_hemisphere(issue_date: pd.Timestamp) -> str:
    return "north" if issue_date.month == 2 else "south"


def issue_to_forecast_year(issue_date: pd.Timestamp) -> int:
    return issue_date.year if issue_date.month == 2 else issue_date.year + 1


def issue_to_titer_model_timepoint(issue_date: pd.Timestamp) -> str:
    if issue_date.month == 2:
        return f"{issue_date.year - 1}-10-01"
    return f"{issue_date.year}-04-01"


def list_available_titer_model_timepoints() -> list[pd.Timestamp]:
    timepoints = []
    if not upstream_titer_model_root().exists():
        return timepoints

    for path in upstream_titer_model_root().iterdir():
        if not path.is_dir() or not (path / "titers-sub-model.json").exists():
            continue
        try:
            timepoints.append(pd.Timestamp(path.name))
        except ValueError:
            continue

    return sorted(timepoints)


def parse_bool(value: object) -> bool:
    """Parse boolean values from cached TSV cells.

    English: Pandas may load manifest booleans as bools or strings.
    中文：Pandas 读取 manifest 时可能得到 bool 或字符串。
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def cached_issue_manifest_paths() -> list[Path]:
    """Return package-local issue manifests for HI fallback resolution.

    English: Only official run/input paths are consulted; no silent trash/ cache
    fallbacks. Missing manifests fail later in resolve_cached_titer_model.
    中文：只使用正式 run/input 路径；不静默回退到 trash/ 缓存。缺失时由
    resolve_cached_titer_model 后续失败。
    """
    return [
        ISSUE_MANIFEST,
        PROJECT_ROOT / "data" / "run_inputs" / RUN_DIR.name / "issue_manifest.tsv",
    ]


def resolve_cached_titer_model(issue_date: pd.Timestamp) -> tuple[str, str, bool] | None:
    """Resolve HI model timepoints from a cached issue manifest.

    English: This keeps prepare reproducible when HI model files are not bundled.
    中文：当发布包未包含 HI model 文件时，该缓存使 prepare 仍可复现。
    """
    issue_date_string = issue_date.strftime("%Y-%m-%d")
    for path in cached_issue_manifest_paths():
        if not path.exists():
            continue
        cached = pd.read_csv(path, sep="\t")
        row = cached[cached["issue_date"].astype(str) == issue_date_string]
        if row.empty:
            continue
        record = row.iloc[0]
        requested = str(record.get("requested_titer_model_timepoint", issue_to_titer_model_timepoint(issue_date)))
        resolved = str(record.get("original_titer_model_timepoint", "N/A"))
        if resolved and resolved != "N/A":
            return requested, resolved, parse_bool(record.get("titer_model_fallback", False))
    return None


def resolve_original_titer_model(issue_date: pd.Timestamp) -> tuple[str, str, bool]:
    requested = pd.Timestamp(issue_to_titer_model_timepoint(issue_date))
    requested_path = upstream_titer_model_root() / requested.strftime("%Y-%m-%d") / "titers-sub-model.json"
    if requested_path.exists():
        return requested.strftime("%Y-%m-%d"), requested.strftime("%Y-%m-%d"), False

    if not ALLOW_HI_FALLBACK:
        raise FileNotFoundError(
            "Missing HI substitution model "
            f"{requested_path}. Re-run with --disable-hi for non-HI predictors, "
            "or use --allow-hi-fallback to explicitly reuse the latest available stored HI model."
        )

    available = list_available_titer_model_timepoints()
    if not available:
        cached = resolve_cached_titer_model(issue_date)
        if cached is not None:
            return cached
        raise FileNotFoundError(f"No titers-sub-model.json files found under {upstream_titer_model_root()}")

    previous_or_equal = [timepoint for timepoint in available if timepoint <= requested]
    resolved = previous_or_equal[-1] if previous_or_equal else available[-1]
    return requested.strftime("%Y-%m-%d"), resolved.strftime("%Y-%m-%d"), True


def run_command(cmd: list[str], cwd: Path | None = None, log_path: Path | None = None) -> None:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(" ".join(cmd) + "\n")
            subprocess.run(cmd, cwd=cwd, check=True, stdout=log_handle, stderr=subprocess.STDOUT)
        return

    subprocess.run(cmd, cwd=cwd, check=True)


def table_has_columns(path: Path, columns: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        existing_columns = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    except pd.errors.EmptyDataError:
        return False
    return set(columns).issubset(existing_columns)


def table_matches_hi_policy(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        existing_columns = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    except pd.errors.EmptyDataError:
        return False

    hi_columns = {"cTiterSub", "dTiterSub", "cTiterSub_x"}
    if ENABLE_HI_PREDICTORS:
        return {"cTiterSub", "dTiterSub"}.issubset(existing_columns)
    return existing_columns.isdisjoint(hi_columns)


def status_matches_hi_policy(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False

    if ENABLE_HI_PREDICTORS:
        return status.get("original_titer_model_timepoint") not in {None, "N/A"}
    # Older no-HI runs may have stale HI metadata in status.json while their
    # tables correctly omit HI columns. In no-HI mode, the table policy is the
    # source of truth; any completed status can be resumed from safely.
    return status.get("status") == "completed"


def tree_has_expected_tips(tree_path: Path, expected_strains: list[str]) -> bool:
    if not tree_path.exists():
        return False
    try:
        current_recursion_limit = sys.getrecursionlimit()
        required_recursion_limit = max(current_recursion_limit, len(expected_strains) * 4, 10000)
        if required_recursion_limit > current_recursion_limit:
            sys.setrecursionlimit(required_recursion_limit)
        tree = Phylo.read(str(tree_path), "newick")
    except Exception:
        return False
    tips = {terminal.name for terminal in tree.get_terminals()}
    return set(expected_strains).issubset(tips)


def remove_refine_and_downstream_outputs(timepoint_dir: Path) -> None:
    for filename in [
        "tree.nwk",
        "branch_lengths.json",
        "frequencies.json",
        "frequencies.log",
        "frequencies.tsv",
        "diffusion_frequencies.json",
        "diffusion_frequencies.log",
        "diffusion_frequencies.tsv",
        "lbi.json",
        "lbi.log",
        "aa-seq_SigPep_with_internal.fasta",
        "aa-seq_HA1_with_internal.fasta",
        "aa-seq_HA2_with_internal.fasta",
        "aa-seq_SigPep_ancestral.json",
        "aa-seq_HA1_ancestral.json",
        "aa-seq_HA2_ancestral.json",
        "aa_ancestral_SigPep.log",
        "aa_ancestral_HA1.log",
        "aa_ancestral_HA2.log",
        "distances.json",
        "distances.log",
        "delta_frequency.json",
        "delta_frequency.log",
        "traits.json",
        "traits.log",
        "clades.json",
        "tip_clades.tsv",
        "clades.log",
        "aa_seq.json",
        "distance_from_consensus.json",
        "pairwise_antigenic_distances.json",
        "pairwise_antigenic_distances.log",
        "antigenic_cross_immunity.json",
        "antigenic_cross_immunity.log",
        "titer_substitution_distance_map.json",
        "titer_substitution_distances.json",
        "titer_substitution_distances.log",
        "pairwise_titer_distances.json",
        "pairwise_titer_distances.log",
        "cross_immunity_frequencies.json",
        "titer_substitution_cross_immunity.json",
        "titer_cross_immunity.log",
        "node_data.tsv",
        "tip_attributes.tsv",
        "tip_attributes_with_naive.tsv",
        "status.json",
        "report.md",
        "refine.log",
    ]:
        path = timepoint_dir / filename
        if path.exists():
            path.unlink()


def normalize_region(row: pd.Series) -> str:
    region = str(row.get("region", "") or "")
    if region in FORECAST_REGIONS:
        return region

    country = str(row.get("country", "") or "")
    division = str(row.get("division", "") or "")
    location = str(row.get("location", "") or "")
    strain = str(row.get("strain", "") or "")

    combined = " ".join([country, division, location, strain]).lower().replace("_", " ")

    if "spain" in combined or region == "spain":
        return "europe"
    if "laos" in combined or "lao, people" in combined or region == "laos":
        return "southeast_asia"
    if "taiwan" in combined or region == "taiwan":
        return "china"
    if "south sudan" in combined or "southsudan" in combined or region == "sudan_south":
        return "africa"

    country_key = country.lower()
    if country_key in ASIA_BUCKETS:
        return ASIA_BUCKETS[country_key]

    return region


def build_sequence_lookup(fasta_path: Path) -> dict[str, str]:
    return {record.id: str(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")}


def write_fasta(sequences_by_strain: dict[str, str], strains: list[str], output_path: Path) -> None:
    records = []
    for strain in strains:
        sequence = sequences_by_strain[strain]
        record = SeqRecord(Seq(sequence), id=strain, name=strain, description=strain)
        records.append(record)
    SeqIO.write(records, str(output_path), "fasta")


def get_metadata_for_selection() -> pd.DataFrame:
    metadata = pd.read_csv(
        INPUT_METADATA,
        sep="\t",
        parse_dates=["date", "collection_date", "submission_date"],
    )

    sequence_table = pd.read_csv(INPUT_SEQUENCE_TABLE, sep="\t", usecols=["accession_number"])
    valid_accessions = set(sequence_table["accession_number"].astype(str))

    metadata = metadata[metadata["source_isolate_id"].astype(str).isin(valid_accessions)].copy()
    metadata["region"] = metadata.apply(normalize_region, axis=1)
    metadata = metadata[metadata["region"].isin(FORECAST_REGIONS)].copy()
    metadata = metadata[
        (metadata["collection_date"] >= GLOBAL_START) & (metadata["collection_date"] <= GLOBAL_END)
    ].copy()
    for column in ["age", "gender"]:
        if column not in metadata.columns:
            metadata[column] = "?"
        metadata[column] = metadata[column].fillna("?").replace("", "?")
    metadata["year"] = metadata["collection_date"].dt.year
    metadata["month"] = metadata["collection_date"].dt.month
    metadata = metadata.sort_values(["collection_date", "strain"]).reset_index(drop=True)

    return metadata


def populate_categories(metadata: dict[str, dict]) -> tuple[defaultdict, defaultdict]:
    super_category = lambda x: (x["year"], x["month"])
    category = lambda x: (x["region"], x["year"], x["month"])

    virus_by_category = defaultdict(list)
    virus_by_super_category = defaultdict(list)
    for strain in metadata:
        virus_by_category[category(metadata[strain])].append(strain)
        virus_by_super_category[super_category(metadata[strain])].append(strain)

    return virus_by_super_category, virus_by_category


def flu_subsampling(metadata: dict[str, dict], viruses_per_month: int) -> list[str]:
    np.random.seed(RANDOM_SEED)

    def priority(_strain: str) -> float:
        return float(np.random.random())

    subcat_threshold = int(np.ceil(1.0 * viruses_per_month / len(FORECAST_REGIONS)))
    virus_by_super_category, virus_by_category = populate_categories(metadata)

    def threshold_fn(category_tuple: tuple[str, int, int]) -> int:
        if len(virus_by_super_category[category_tuple[1:]]) < viruses_per_month:
            return viruses_per_month

        sub_counts = sorted(
            [
                (
                    region,
                    virus_by_category.get((region, category_tuple[1], category_tuple[2]), []),
                )
                for region in FORECAST_REGIONS
            ],
            key=lambda item: len(item[1]),
        )

        if len(sub_counts[0][1]) > subcat_threshold:
            return subcat_threshold

        strains_selected = 0
        for region_index, (region, strains) in enumerate(sub_counts):
            current_threshold = int(
                np.ceil(1.0 * (viruses_per_month - strains_selected) / (len(FORECAST_REGIONS) - region_index))
            )
            if region == category_tuple[0]:
                return current_threshold
            strains_selected += min(len(strains), current_threshold)

        return subcat_threshold

    selected_strains: list[str] = []
    for category_tuple, strains in virus_by_category.items():
        sorted_strains = sorted(strains, key=priority, reverse=True)
        selected_strains.extend(sorted_strains[: threshold_fn(category_tuple)])

    return sorted(set(selected_strains))


def prepare_inputs() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TIMEPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = get_metadata_for_selection()
    sequence_lookup = build_sequence_lookup(INPUT_FASTA)
    rows_before_sequence_filter = len(metadata)
    metadata = metadata[metadata["strain"].isin(sequence_lookup)].copy()
    missing_sequence_rows = rows_before_sequence_filter - len(metadata)
    gap_counts = metadata["strain"].map(lambda strain: sequence_lookup[strain].count("-"))
    too_many_gap_rows = int((gap_counts > MAX_SEQUENCE_GAPS).sum())
    metadata = metadata.loc[gap_counts <= MAX_SEQUENCE_GAPS].copy()
    metadata_to_write = metadata.copy()
    for column in ["date", "collection_date", "submission_date"]:
        metadata_to_write[column] = metadata_to_write[column].dt.strftime("%Y-%m-%d")
    metadata_to_write.to_csv(NORMALIZED_METADATA, sep="\t", index=False)

    metadata_by_strain = {
        row.strain: {"region": row.region, "year": int(row.year), "month": int(row.month)}
        for row in metadata.loc[:, ["strain", "region", "year", "month"]].itertuples(index=False)
    }

    selected_strains = flu_subsampling(metadata_by_strain, VIRUSES_PER_MONTH)
    with GLOBAL_STRAINS.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(selected_strains) + "\n")

    subset_metadata = metadata[metadata["strain"].isin(selected_strains)].copy()
    subset_to_write = subset_metadata.copy()
    for column in ["date", "collection_date", "submission_date"]:
        subset_to_write[column] = subset_to_write[column].dt.strftime("%Y-%m-%d")
    subset_to_write.to_csv(GLOBAL_METADATA, sep="\t", index=False)

    global_sequences = [strain for strain in selected_strains if strain in sequence_lookup]
    write_fasta(sequence_lookup, global_sequences, GLOBAL_FASTA)

    issue_records = []
    for issue_date in build_issue_dates():
        requested_titer_timepoint = issue_to_titer_model_timepoint(issue_date)
        if ENABLE_HI_PREDICTORS:
            requested_titer_timepoint, resolved_titer_timepoint, titer_model_fallback = resolve_original_titer_model(issue_date)
        else:
            resolved_titer_timepoint = "N/A"
            titer_model_fallback = False
        issue_records.append(
            {
                "issue_date": issue_date.strftime("%Y-%m-%d"),
                "hemisphere": issue_to_hemisphere(issue_date),
                "forecast_year": issue_to_forecast_year(issue_date),
                "requested_titer_model_timepoint": requested_titer_timepoint,
                "original_titer_model_timepoint": resolved_titer_timepoint,
                "titer_model_fallback": titer_model_fallback,
            }
        )
    issue_df = pd.DataFrame(issue_records)
    issue_df.to_csv(ISSUE_MANIFEST, sep="\t", index=False)

    region_counts = subset_metadata["region"].value_counts().sort_index()
    PREPARE_REPORT.write_text(
        "\n".join(
            [
                f"# {REPORT_LABEL} {ISSUE_START.year}-{ISSUE_END.year} Issue-Date Prepare",
                "",
                "## Inputs",
                "",
                f"- normalized metadata: `{NORMALIZED_METADATA.relative_to(WORK_ROOT)}`",
                f"- global selected strains: `{GLOBAL_STRAINS.relative_to(WORK_ROOT)}`",
                f"- global selected metadata: `{GLOBAL_METADATA.relative_to(WORK_ROOT)}`",
                f"- global selected amino-acid sequences: `{GLOBAL_FASTA.relative_to(WORK_ROOT)}`",
                f"- issue manifest: `{ISSUE_MANIFEST.relative_to(WORK_ROOT)}`",
                f"- source metadata: `{INPUT_METADATA}`",
                f"- source amino-acid FASTA: `{INPUT_FASTA}`",
                f"- source sequence table: `{INPUT_SEQUENCE_TABLE}`",
                f"- distance map root: `{DISTANCE_MAP_ROOT}`",
                "",
                "## Notes",
                "",
                f"- metadata pool inherits the existing `sequence.py`-style prefilter through `{INPUT_SEQUENCE_TABLE.name}`",
                f"- final amino-acid FASTA strains with >{MAX_SEQUENCE_GAPS} gaps are excluded before subsampling",
                "- global subsampling balances strain counts by region, year, and month with `viruses_per_month = 90`",
                "- no titer-count priority is available here, so random priorities are used with a fixed seed for reproducibility",
                f"- HI substitution predictors enabled: `{ENABLE_HI_PREDICTORS}`",
                f"- DMS predictors enabled: `{ENABLE_DMS_PREDICTORS}`",
                f"- HI model fallback allowed: `{ALLOW_HI_FALLBACK}`",
                f"- intended global collection interval: `{GLOBAL_START.strftime('%Y-%m-%d')}` to `{GLOBAL_END.strftime('%Y-%m-%d')}`",
                f"- intended issue-date interval: `{ISSUE_START.strftime('%Y-%m-%d')}` to `{ISSUE_END.strftime('%Y-%m-%d')}`",
                f"- forecast years: `{','.join(str(year) for year in FORECAST_YEARS)}`",
                "",
                "## Counts",
                "",
                f"- normalized rows: {len(metadata)}",
                f"- metadata rows missing final FASTA sequence: {missing_sequence_rows}",
                f"- metadata rows with >{MAX_SEQUENCE_GAPS} gaps in final FASTA sequence: {too_many_gap_rows}",
                f"- globally selected strains: {len(global_sequences)}",
                f"- issue dates listed: {len(issue_df)}",
                "",
                "## Selected Region Counts",
                "",
                *[f"- {region}: {count}" for region, count in region_counts.items()],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_segment_fastas(timepoint_dir: Path, sequences_by_strain: dict[str, str], strains: list[str]) -> dict[str, Path]:
    segment_paths = {
        "SigPep": timepoint_dir / "aa-seq_SigPep.fasta",
        "HA1": timepoint_dir / "aa-seq_HA1.fasta",
        "HA2": timepoint_dir / "aa-seq_HA2.fasta",
    }

    segment_records: dict[str, list] = {"SigPep": [], "HA1": [], "HA2": []}
    for strain in strains:
        sequence = sequences_by_strain[strain]
        segment_sequences = {
            "SigPep": sequence[SIGPEP_SLICE],
            "HA1": sequence[HA1_SLICE],
            "HA2": sequence[HA2_SLICE],
        }
        for gene, gene_sequence in segment_sequences.items():
            record = SeqRecord(Seq(gene_sequence), id=strain, name=strain, description=strain)
            segment_records[gene].append(record)

    for gene, records in segment_records.items():
        SeqIO.write(records, str(segment_paths[gene]), "fasta")

    return segment_paths


def prepare_cross_immunity_frequencies(source_frequencies: Path, output_frequencies: Path) -> None:
    with source_frequencies.open("r", encoding="utf-8") as handle:
        frequencies = json.load(handle)

    flattened = {
        "pivots": frequencies["data"]["pivots"],
        "generated_by": "run_issue_date_pipeline.py",
    }
    for node_name, node_frequencies in frequencies["data"]["frequencies"].items():
        flattened[node_name] = {"frequencies": node_frequencies}

    output_frequencies.write_text(json.dumps(flattened, indent=1, sort_keys=True), encoding="utf-8")


def write_tip_trait_annotations(metadata_path: Path, output_path: Path) -> None:
    """Write tip-level trait node data for table-format compatibility.

    English: The output provides terminal-node region and country annotations.
    中文：输出提供终端节点的 region 和 country 注释。
    """
    metadata = pd.read_csv(metadata_path, sep="\t")
    nodes = {}
    for row in metadata.loc[:, ["strain", "region", "country"]].itertuples(index=False):
        nodes[row.strain] = {
            "region": row.region,
            "country": row.country,
            "region_entropy": 0.0,
            "country_entropy": 0.0,
        }

    output_path.write_text(
        json.dumps(
            {
                "generated_by": "run_issue_date_pipeline.py tip trait fallback",
                "nodes": nodes,
            },
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def validate_issue_date(issue_date_string: str) -> tuple[str, pd.Timestamp]:
    """Validate one issue date before using it as a directory name.

    English: Only canonical YYYY-MM-DD dates may select a timepoint directory.
    中文：只有规范的 YYYY-MM-DD 日期才能用于选择 timepoint 目录。
    """
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", issue_date_string) is None:
        raise ValueError(f"invalid issue date {issue_date_string!r}; expected YYYY-MM-DD")
    try:
        issue_date = pd.Timestamp(issue_date_string)
    except ValueError as error:
        raise ValueError(f"invalid issue date {issue_date_string!r}") from error
    if issue_date.strftime("%Y-%m-%d") != issue_date_string:
        raise ValueError(f"invalid issue date {issue_date_string!r}; expected canonical YYYY-MM-DD")
    return issue_date_string, issue_date


def run_timepoint(issue_date_string: str) -> dict:
    issue_date_string, issue_date = validate_issue_date(issue_date_string)
    timepoint_dir = TIMEPOINTS_DIR / issue_date_string
    timepoint_dir.mkdir(parents=True, exist_ok=True)

    tip_attributes_with_naive = timepoint_dir / "tip_attributes_with_naive.tsv"
    status_json = timepoint_dir / "status.json"
    report_path = timepoint_dir / "report.md"

    if (
        tip_attributes_with_naive.exists()
        and status_json.exists()
        and table_has_columns(tip_attributes_with_naive, get_required_table_columns())
        and table_matches_hi_policy(tip_attributes_with_naive)
        and status_matches_hi_policy(status_json)
    ):
        with status_json.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    global_metadata = pd.read_csv(
        GLOBAL_METADATA,
        sep="\t",
        parse_dates=["date", "collection_date", "submission_date"],
    )
    for column in ["age", "gender"]:
        if column not in global_metadata.columns:
            global_metadata[column] = "?"
        global_metadata[column] = global_metadata[column].fillna("?").replace("", "?")
    sequence_lookup = build_sequence_lookup(GLOBAL_FASTA)

    timepoint_metadata = global_metadata[
        (global_metadata["collection_date"] <= issue_date)
        & (global_metadata["collection_date"] >= issue_date - pd.DateOffset(years=YEARS_BACK_TO_BUILD))
        & (global_metadata["submission_date"] < issue_date)
    ].copy()

    timepoint_metadata["timepoint"] = issue_date_string
    timepoint_metadata["hemisphere"] = issue_to_hemisphere(issue_date)
    timepoint_metadata["forecast_year"] = issue_to_forecast_year(issue_date)
    requested_titer_timepoint = issue_to_titer_model_timepoint(issue_date)
    if ENABLE_HI_PREDICTORS:
        requested_titer_timepoint, mapped_titer_timepoint, titer_model_fallback = resolve_original_titer_model(issue_date)
    else:
        mapped_titer_timepoint = "N/A"
        titer_model_fallback = False
    timepoint_metadata["requested_titer_model_timepoint"] = requested_titer_timepoint
    timepoint_metadata["original_titer_model_timepoint"] = mapped_titer_timepoint
    timepoint_metadata["titer_model_fallback"] = titer_model_fallback
    timepoint_metadata["aa_sequence"] = timepoint_metadata["strain"].map(sequence_lookup)
    timepoint_metadata = timepoint_metadata.dropna(subset=["aa_sequence"]).copy()
    timepoint_metadata = timepoint_metadata.sort_values(["collection_date", "strain"]).reset_index(drop=True)

    strains = timepoint_metadata["strain"].tolist()
    strains_path = timepoint_dir / "strains.txt"
    with strains_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(strains) + ("\n" if strains else ""))

    metadata_path = timepoint_dir / "metadata.tsv"
    timepoint_metadata_to_write = timepoint_metadata.copy()
    for column in ["date", "collection_date", "submission_date"]:
        timepoint_metadata_to_write[column] = timepoint_metadata_to_write[column].dt.strftime("%Y-%m-%d")
    timepoint_metadata_to_write.to_csv(metadata_path, sep="\t", index=False)

    if len(strains) < 4:
        status = {
            "issue_date": issue_date_string,
            "hemisphere": issue_to_hemisphere(issue_date),
            "forecast_year": issue_to_forecast_year(issue_date),
            "n_strains": len(strains),
            "status": "skipped_not_enough_strains",
        }
        status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")
        report_path.write_text(
            "\n".join(
                [
                    f"# Issue Date {issue_date_string}",
                    "",
                    f"- status: `{status['status']}`",
                    f"- strains: {len(strains)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return status

    fasta_path = timepoint_dir / "aligned_aa_sequences.fasta"
    write_fasta(sequence_lookup, strains, fasta_path)
    segment_paths = write_segment_fastas(timepoint_dir, sequence_lookup, strains)

    tree_raw_path = timepoint_dir / "tree_raw.nwk"
    if tree_raw_path.exists() and not tree_has_expected_tips(tree_raw_path, strains):
        tree_raw_path.unlink()
        remove_refine_and_downstream_outputs(timepoint_dir)

    if not tree_raw_path.exists():
        iqtree_treefile = timepoint_dir / "iqtree.treefile"
        if iqtree_treefile.exists() and tree_has_expected_tips(iqtree_treefile, strains):
            iqtree_treefile.replace(tree_raw_path)
        else:
            run_command(
                get_iqtree_args() + ["-s", str(fasta_path), "-pre", str(timepoint_dir / "iqtree")],
                log_path=timepoint_dir / "iqtree.log",
            )
            if not tree_has_expected_tips(iqtree_treefile, strains):
                raise RuntimeError(f"IQ-TREE output is missing expected tips for {issue_date_string}")
            (timepoint_dir / "iqtree.treefile").replace(tree_raw_path)

    tree_path = timepoint_dir / "tree.nwk"
    branch_lengths_path = timepoint_dir / "branch_lengths.json"
    if tree_path.exists() and not tree_has_expected_tips(tree_path, strains):
        remove_refine_and_downstream_outputs(timepoint_dir)

    if not tree_path.exists() or not branch_lengths_path.exists():
        run_command(
            [
                sys.executable,
                str(FUTUREFLU_SCRIPT_ROOT / "refine_aa_with_treetime.py"),
                "--tree",
                str(tree_raw_path),
                "--alignment",
                str(fasta_path),
                "--metadata",
                str(metadata_path),
                "--output-tree",
                str(tree_path),
                "--output-node-data",
                str(branch_lengths_path),
                "--coalescent",
                "const",
                "--date-inference",
                "marginal",
                "--date-confidence",
                "--no-covariance",
            ],
            log_path=timepoint_dir / "refine.log",
        )

    timepoint_start = max(GLOBAL_START, issue_date - pd.DateOffset(years=YEARS_BACK_TO_BUILD))
    frequencies_path = timepoint_dir / "frequencies.json"
    if not frequencies_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "frequencies.py"),
                str(tree_path),
                str(metadata_path),
                str(frequencies_path),
                "--narrow-bandwidth",
                "0.1667",
                "--wide-bandwidth",
                "0.25",
                "--proportion-wide",
                "0.0",
                "--pivot-frequency",
                str(PIVOT_INTERVAL_MONTHS),
                "--start-date",
                timepoint_start.strftime("%Y-%m-%d"),
                "--end-date",
                issue_date_string,
                "--include-internal-nodes",
            ],
            log_path=timepoint_dir / "frequencies.log",
        )

    diffusion_frequencies_path = timepoint_dir / "diffusion_frequencies.json"
    if not diffusion_frequencies_path.exists():
        min_diffusion_date = max(timepoint_start, issue_date - pd.DateOffset(years=DIFFUSION_YEARS_BACK))
        run_command(
            [
                "augur",
                "frequencies",
                "--method",
                "diffusion",
                "--tree",
                str(tree_path),
                "--metadata",
                str(metadata_path),
                "--output",
                str(diffusion_frequencies_path),
                "--include-internal-nodes",
                "--stiffness",
                "20",
                "--inertia",
                "0.2",
                "--pivot-interval",
                str(PIVOT_INTERVAL_MONTHS),
                "--min-date",
                str(timestamp_to_float(min_diffusion_date)),
                "--max-date",
                str(timestamp_to_float(issue_date)),
            ],
            log_path=timepoint_dir / "diffusion_frequencies.log",
        )

    frequencies_table = timepoint_dir / "frequencies.tsv"
    if not frequencies_table.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "frequencies_to_table.py"),
                "--tree",
                str(tree_path),
                "--frequencies",
                str(frequencies_path),
                "--method",
                "kde",
                "--annotations",
                f"timepoint={issue_date_string}",
                "--output",
                str(frequencies_table),
            ]
        )

    diffusion_frequencies_table = timepoint_dir / "diffusion_frequencies.tsv"
    if not diffusion_frequencies_table.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "frequencies_to_table.py"),
                "--tree",
                str(tree_path),
                "--frequencies",
                str(diffusion_frequencies_path),
                "--method",
                "diffusion",
                "--annotations",
                f"timepoint={issue_date_string}",
                "--output",
                str(diffusion_frequencies_table),
            ]
        )

    lbi_path = timepoint_dir / "lbi.json"
    if not lbi_path.exists():
        run_command(
            [
                "augur",
                "lbi",
                "--tree",
                str(tree_path),
                "--branch-lengths",
                str(branch_lengths_path),
                "--output",
                str(lbi_path),
                "--attribute-names",
                "lbi",
                "--tau",
                "0.3",
                "--window",
                "0.5",
            ],
            log_path=timepoint_dir / "lbi.log",
        )

    traits_path = timepoint_dir / "traits.json"
    if not traits_path.exists():
        write_tip_trait_annotations(metadata_path, traits_path)
        (timepoint_dir / "traits.log").write_text(
            "\n".join(
                [
                    "Generated tip-level region/country trait annotations directly.",
                    "Reason: augur 5.4.1 cannot encode the current number of country states as ASCII pseudo sequences.",
                    "This pipeline exports terminal rows only, so tip-level annotations are sufficient for table compatibility.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    ancestral_segment_paths = {
        gene: timepoint_dir / f"aa-seq_{gene}_with_internal.fasta"
        for gene in ["SigPep", "HA1", "HA2"]
    }
    for gene, output_path in ancestral_segment_paths.items():
        if not output_path.exists():
            run_command(
                [
                    sys.executable,
                    str(FUTUREFLU_SCRIPT_ROOT / "infer_aa_ancestral_sequences.py"),
                    "--tree",
                    str(tree_path),
                    "--alignment",
                    str(segment_paths[gene]),
                    "--output-sequences",
                    str(output_path),
                    "--output-node-data",
                    str(timepoint_dir / f"aa-seq_{gene}_ancestral.json"),
                    "--gtr-model",
                    "JTT92",
                ],
                log_path=timepoint_dir / f"aa_ancestral_{gene}.log",
            )

    clades_path = timepoint_dir / "clades.json"
    tip_clades_path = timepoint_dir / "tip_clades.tsv"
    if not clades_path.exists() or not tip_clades_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "nonoverlapping_clades.py"),
                "--tree",
                str(tree_path),
                "--translations",
                str(ancestral_segment_paths["SigPep"]),
                str(ancestral_segment_paths["HA1"]),
                str(ancestral_segment_paths["HA2"]),
                "--gene-names",
                "SigPep",
                "HA1",
                "HA2",
                "--annotations",
                f"timepoint={issue_date_string}",
                "--output",
                str(clades_path),
                "--output-tip-clade-table",
                str(tip_clades_path),
            ],
            log_path=timepoint_dir / "clades.log",
        )

    latest_distance_date = issue_date - pd.DateOffset(months=12)
    earliest_distance_date = latest_distance_date - pd.DateOffset(years=YEARS_BACK_FOR_TITER_COMPARISON)

    distances_path = timepoint_dir / "distances.json"
    if not distances_path.exists():
        run_command(
            [
                "augur",
                "distance",
                "--tree",
                str(tree_path),
                "--alignment",
                str(ancestral_segment_paths["SigPep"]),
                str(ancestral_segment_paths["HA1"]),
                str(ancestral_segment_paths["HA2"]),
                "--gene-names",
                "SigPep",
                "HA1",
                "HA2",
                "--compare-to",
                *[compare_to for compare_to, _, _ in get_distance_model_specs()],
                "--attribute-name",
                *[attribute for _, attribute, _ in get_distance_model_specs()],
                "--map",
                *[str(DISTANCE_MAP_ROOT / map_name) for _, _, map_name in get_distance_model_specs()],
                "--date-annotations",
                str(branch_lengths_path),
                "--earliest-date",
                earliest_distance_date.strftime("%Y-%m-%d"),
                "--latest-date",
                latest_distance_date.strftime("%Y-%m-%d"),
                "--output",
                str(distances_path),
            ],
            log_path=timepoint_dir / "distances.log",
        )

    delta_frequency_path = timepoint_dir / "delta_frequency.json"
    if not delta_frequency_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "calculate_delta_frequency.py"),
                "--tree",
                str(tree_path),
                "--frequencies",
                str(diffusion_frequencies_path),
                "--frequency-method",
                "diffusion",
                "--delta-pivots",
                str(DELTA_PIVOTS),
                "--output",
                str(delta_frequency_path),
            ],
            log_path=timepoint_dir / "delta_frequency.log",
        )

    aa_sequence_json_path = timepoint_dir / "aa_seq.json"
    if not aa_sequence_json_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "convert_translations_to_json.py"),
                "--tree",
                str(tree_path),
                "--alignment",
                str(segment_paths["SigPep"]),
                str(segment_paths["HA1"]),
                str(segment_paths["HA2"]),
                "--gene-names",
                "SigPep",
                "HA1",
                "HA2",
                "--output",
                str(aa_sequence_json_path),
            ]
        )

    distance_from_consensus_path = timepoint_dir / "distance_from_consensus.json"
    if not distance_from_consensus_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "distance_from_consensus.py"),
                "--sequences",
                str(aa_sequence_json_path),
                "--frequencies",
                str(frequencies_table),
                "--output",
                str(distance_from_consensus_path),
            ]
        )

    antigenic_pairwise_distances_path = timepoint_dir / "pairwise_antigenic_distances.json"
    if not antigenic_pairwise_distances_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "pairwise_distances.py"),
                "--tree",
                str(tree_path),
                "--frequencies",
                str(frequencies_path),
                "--alignment",
                str(segment_paths["SigPep"]),
                str(segment_paths["HA1"]),
                str(segment_paths["HA2"]),
                "--gene-names",
                "SigPep",
                "HA1",
                "HA2",
                "--attribute-name",
                "ep_pairwise",
                "ep_pairwise_wolf",
                "ep_pairwise_koel",
                "oracle_pairwise",
                "--map",
                str(DISTANCE_MAP_ROOT / "luksza.json"),
                str(DISTANCE_MAP_ROOT / "wolf.json"),
                str(DISTANCE_MAP_ROOT / "koel.json"),
                str(DISTANCE_MAP_ROOT / "oracle.json"),
                "--date-annotations",
                str(branch_lengths_path),
                "--years-back-to-compare",
                str(YEARS_BACK_FOR_TITER_COMPARISON),
                "--output",
                str(antigenic_pairwise_distances_path),
            ],
            log_path=timepoint_dir / "pairwise_antigenic_distances.log",
        )

    antigenic_cross_immunity_path = timepoint_dir / "antigenic_cross_immunity.json"
    if not antigenic_cross_immunity_path.exists():
        cross_immunity_frequencies_path = timepoint_dir / "cross_immunity_frequencies.json"
        if not cross_immunity_frequencies_path.exists():
            prepare_cross_immunity_frequencies(frequencies_path, cross_immunity_frequencies_path)
        run_command(
            [
                sys.executable,
                str(flu_root() / "src" / "cross_immunity.py"),
                "--frequencies",
                str(cross_immunity_frequencies_path),
                "--distances",
                str(antigenic_pairwise_distances_path),
                "--date-annotations",
                str(branch_lengths_path),
                "--distance-attributes",
                "ep_pairwise",
                "ep_pairwise_wolf",
                "ep_pairwise_koel",
                "oracle_pairwise",
                "--immunity-attributes",
                "ep_x",
                "ep_x_wolf",
                "ep_x_koel",
                "oracle_x",
                "--decay-factors",
                "14.0",
                "14.0",
                "14.0",
                "14.0",
                "--output",
                str(antigenic_cross_immunity_path),
            ],
            log_path=timepoint_dir / "antigenic_cross_immunity.log",
        )

    original_titer_model = upstream_titer_model_root() / mapped_titer_timepoint / "titers-sub-model.json"
    titer_distance_map_path = timepoint_dir / "titer_substitution_distance_map.json"
    titer_substitution_distances_path = timepoint_dir / "titer_substitution_distances.json"
    pairwise_titer_distances_path = timepoint_dir / "pairwise_titer_distances.json"
    titer_cross_immunity_path = timepoint_dir / "titer_substitution_cross_immunity.json"

    if ENABLE_HI_PREDICTORS and not titer_distance_map_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "titer_model_to_distance_map.py"),
                "--model",
                str(original_titer_model),
                "--output",
                str(titer_distance_map_path),
            ]
        )

    if ENABLE_HI_PREDICTORS and not titer_substitution_distances_path.exists():
        run_command(
            [
                "augur",
                "distance",
                "--tree",
                str(tree_path),
                "--alignment",
                str(ancestral_segment_paths["SigPep"]),
                str(ancestral_segment_paths["HA1"]),
                str(ancestral_segment_paths["HA2"]),
                "--gene-names",
                "SigPep",
                "HA1",
                "HA2",
                "--compare-to",
                "root",
                "ancestor",
                "--attribute-name",
                "cTiterSub",
                "dTiterSub",
                "--map",
                str(titer_distance_map_path),
                str(titer_distance_map_path),
                "--date-annotations",
                str(branch_lengths_path),
                "--earliest-date",
                earliest_distance_date.strftime("%Y-%m-%d"),
                "--latest-date",
                latest_distance_date.strftime("%Y-%m-%d"),
                "--output",
                str(titer_substitution_distances_path),
            ],
            log_path=timepoint_dir / "titer_substitution_distances.log",
        )

    if ENABLE_HI_PREDICTORS and not pairwise_titer_distances_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "pairwise_distances.py"),
                "--tree",
                str(tree_path),
                "--frequencies",
                str(frequencies_path),
                "--alignment",
                str(segment_paths["SigPep"]),
                str(segment_paths["HA1"]),
                str(segment_paths["HA2"]),
                "--gene-names",
                "SigPep",
                "HA1",
                "HA2",
                "--attribute-name",
                "cTiterSub_pairwise",
                "--map",
                str(titer_distance_map_path),
                "--date-annotations",
                str(branch_lengths_path),
                "--years-back-to-compare",
                str(YEARS_BACK_FOR_TITER_COMPARISON),
                "--output",
                str(pairwise_titer_distances_path),
            ],
            log_path=timepoint_dir / "pairwise_titer_distances.log",
        )

    if ENABLE_HI_PREDICTORS and not titer_cross_immunity_path.exists():
        cross_immunity_frequencies_path = timepoint_dir / "cross_immunity_frequencies.json"
        if not cross_immunity_frequencies_path.exists():
            prepare_cross_immunity_frequencies(frequencies_path, cross_immunity_frequencies_path)
        run_command(
            [
                sys.executable,
                str(flu_root() / "src" / "cross_immunity.py"),
                "--frequencies",
                str(cross_immunity_frequencies_path),
                "--distances",
                str(pairwise_titer_distances_path),
                "--date-annotations",
                str(branch_lengths_path),
                "--distance-attributes",
                "cTiterSub_pairwise",
                "--immunity-attributes",
                "cTiterSub_x",
                "--decay-factors",
                "14.0",
                "--years-to-wane",
                str(YEARS_BACK_FOR_TITER_COMPARISON),
                "--output",
                str(titer_cross_immunity_path),
            ],
            log_path=timepoint_dir / "titer_cross_immunity.log",
        )

    node_data_table = timepoint_dir / "node_data.tsv"
    if node_data_table.exists() and not table_has_columns(node_data_table, get_required_table_columns()):
        node_data_table.unlink()
    if not node_data_table.exists():
        node_data_jsons = [
            str(branch_lengths_path),
            str(lbi_path),
            str(traits_path),
            str(clades_path),
            str(distances_path),
            str(delta_frequency_path),
            str(distance_from_consensus_path),
            str(antigenic_cross_immunity_path),
        ]
        if ENABLE_HI_PREDICTORS:
            node_data_jsons.extend(
                [
                    str(titer_substitution_distances_path),
                    str(titer_cross_immunity_path),
                ]
            )

        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "node_data_to_table.py"),
                "--tree",
                str(tree_path),
                "--metadata",
                str(metadata_path),
                "--jsons",
                *node_data_jsons,
                "--excluded-fields",
                "_none",
                "--annotations",
                f"timepoint={issue_date_string}",
                "--output",
                str(node_data_table),
            ]
        )

    tip_attributes_path = timepoint_dir / "tip_attributes.tsv"
    if tip_attributes_path.exists() and not table_has_columns(tip_attributes_path, get_required_table_columns()):
        tip_attributes_path.unlink()
    if not tip_attributes_path.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "merge_node_data_and_frequencies.py"),
                "--node-data",
                str(node_data_table),
                "--kde-frequencies",
                str(frequencies_table),
                "--diffusion-frequencies",
                str(diffusion_frequencies_table),
                "--preferred-frequency-method",
                "kde",
                "--output",
                str(tip_attributes_path),
            ]
        )

    if tip_attributes_with_naive.exists() and not table_has_columns(tip_attributes_with_naive, get_required_table_columns()):
        tip_attributes_with_naive.unlink()
    if not tip_attributes_with_naive.exists():
        run_command(
            [
                sys.executable,
                str(flu_root() / "scripts" / "annotate_naive_tip_attribute.py"),
                "--tip-attributes",
                str(tip_attributes_path),
                "--output",
                str(tip_attributes_with_naive),
            ]
        )

    status = {
        "issue_date": issue_date_string,
        "hemisphere": issue_to_hemisphere(issue_date),
        "forecast_year": issue_to_forecast_year(issue_date),
        "n_strains": len(strains),
        "status": "completed",
        "requested_titer_model_timepoint": requested_titer_timepoint,
        "original_titer_model_timepoint": mapped_titer_timepoint,
        "titer_model_fallback": titer_model_fallback,
    }
    status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")
    output_lines = [
        f"- `{strains_path.relative_to(WORK_ROOT)}`",
        f"- `{metadata_path.relative_to(WORK_ROOT)}`",
        f"- `{fasta_path.relative_to(WORK_ROOT)}`",
        f"- `{tree_path.relative_to(WORK_ROOT)}`",
        f"- `{branch_lengths_path.relative_to(WORK_ROOT)}`",
        f"- `{frequencies_path.relative_to(WORK_ROOT)}`",
        f"- `{diffusion_frequencies_path.relative_to(WORK_ROOT)}`",
        f"- `{lbi_path.relative_to(WORK_ROOT)}`",
        f"- `{traits_path.relative_to(WORK_ROOT)}`",
        f"- `{clades_path.relative_to(WORK_ROOT)}`",
        f"- `{tip_clades_path.relative_to(WORK_ROOT)}`",
        f"- `{distances_path.relative_to(WORK_ROOT)}`",
        f"- `{delta_frequency_path.relative_to(WORK_ROOT)}`",
        f"- `{aa_sequence_json_path.relative_to(WORK_ROOT)}`",
        f"- `{distance_from_consensus_path.relative_to(WORK_ROOT)}`",
        f"- `{antigenic_pairwise_distances_path.relative_to(WORK_ROOT)}`",
        f"- `{antigenic_cross_immunity_path.relative_to(WORK_ROOT)}`",
    ]
    if ENABLE_HI_PREDICTORS:
        output_lines.extend(
            [
                f"- `{titer_distance_map_path.relative_to(WORK_ROOT)}`",
                f"- `{titer_substitution_distances_path.relative_to(WORK_ROOT)}`",
                f"- `{pairwise_titer_distances_path.relative_to(WORK_ROOT)}`",
                f"- `{titer_cross_immunity_path.relative_to(WORK_ROOT)}`",
            ]
        )
    output_lines.extend(
        [
            f"- `{tip_attributes_path.relative_to(WORK_ROOT)}`",
            f"- `{tip_attributes_with_naive.relative_to(WORK_ROOT)}`",
        ]
    )
    report_path.write_text(
        "\n".join(
            [
                f"# Issue Date {issue_date_string}",
                "",
                f"- status: `{status['status']}`",
                f"- hemisphere: `{status['hemisphere']}`",
                f"- forecast year: {status['forecast_year']}",
                f"- strains: {status['n_strains']}",
                f"- requested HI substitution model: `{requested_titer_timepoint}`",
                f"- resolved HI substitution model: `{mapped_titer_timepoint}`",
                f"- HI model fallback used: `{titer_model_fallback}`",
                "",
                "## Outputs",
                "",
                *output_lines,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return status


def run_all_timepoints() -> list[dict]:
    issue_date_strings = [issue_date.strftime("%Y-%m-%d") for issue_date in build_issue_dates()]
    if TIMEPOINT_WORKERS == 1:
        statuses = [run_timepoint(issue_date_string) for issue_date_string in issue_date_strings]
    else:
        statuses_by_issue_date = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=TIMEPOINT_WORKERS) as executor:
            future_to_issue_date = {
                executor.submit(run_timepoint, issue_date_string): issue_date_string
                for issue_date_string in issue_date_strings
            }
            for future in concurrent.futures.as_completed(future_to_issue_date):
                issue_date_string = future_to_issue_date[future]
                statuses_by_issue_date[issue_date_string] = future.result()
                print(
                    f"[timepoints] completed {issue_date_string}: "
                    f"{statuses_by_issue_date[issue_date_string].get('status')}",
                    flush=True,
                )
        statuses = [statuses_by_issue_date[issue_date_string] for issue_date_string in issue_date_strings]

    TIMEPOINTS_REPORT.write_text(
        "\n".join(
            [
                f"# {REPORT_LABEL} {ISSUE_START.year}-{ISSUE_END.year} Issue-Date Timepoints",
                "",
                "## Status",
                "",
                *[
                    f"- {record['issue_date']}: {record['status']} ({record['n_strains']} strains, {record['hemisphere']})"
                    for record in statuses
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return statuses


def aggregate_tip_attributes() -> None:
    """Aggregate tip-attribute tables from the official run timepoints directory.

    English: Fail fast if the active run tree has no completed timepoint tables.
    Do not silently fall back to trash/ or other machine-local caches.
    中文：若正式 run 树下没有已完成的 timepoint 表则立即失败；不静默回退到
    trash/ 或其他本机缓存。
    """
    completed_timepoints = []
    timepoints_dir = TIMEPOINTS_DIR
    for issue_date in build_issue_dates():
        issue_date_string = issue_date.strftime("%Y-%m-%d")
        tip_path = timepoints_dir / issue_date_string / "tip_attributes.tsv"
        naive_path = timepoints_dir / issue_date_string / "tip_attributes_with_naive.tsv"
        if tip_path.exists() and naive_path.exists():
            completed_timepoints.append((issue_date_string, tip_path, naive_path))

    if not completed_timepoints:
        raise FileNotFoundError(
            "aggregate requires completed timepoint tables under "
            f"{timepoints_dir} (expected */tip_attributes.tsv and "
            "*/tip_attributes_with_naive.tsv). Run the timepoints step first; "
            "silent fallback to trash/intermediate_runs is not supported."
        )

    tip_frames = [pd.read_csv(path, sep="\t", parse_dates=["timepoint"]) for _, path, _ in completed_timepoints]
    naive_frames = [pd.read_csv(path, sep="\t", parse_dates=["timepoint"]) for _, _, path in completed_timepoints]

    if tip_frames:
        aggregated = pd.concat(tip_frames, ignore_index=True)
        aggregated["timepoint"] = aggregated["timepoint"].dt.strftime("%Y-%m-%d")
        aggregated.to_csv(AGGREGATED_TIP_ATTRIBUTES, sep="\t", index=False)
    if naive_frames:
        aggregated_naive = pd.concat(naive_frames, ignore_index=True)
        aggregated_naive["timepoint"] = aggregated_naive["timepoint"].dt.strftime("%Y-%m-%d")
        aggregated_naive.to_csv(
            AGGREGATED_TIP_ATTRIBUTES_NAIVE,
            sep="\t",
            index=False,
        )

    AGGREGATE_REPORT.write_text(
        "\n".join(
            [
                f"# {REPORT_LABEL} {ISSUE_START.year}-{ISSUE_END.year} Issue-Date Aggregate",
                "",
                "## Inputs",
                "",
                *[
                    f"- `{(timepoints_dir / issue_date / 'tip_attributes_with_naive.tsv').relative_to(WORK_ROOT)}`"
                    for issue_date, _, _ in completed_timepoints
                ],
                "",
                "## Outputs",
                "",
                f"- `{AGGREGATED_TIP_ATTRIBUTES.relative_to(WORK_ROOT)}`",
                f"- `{AGGREGATED_TIP_ATTRIBUTES_NAIVE.relative_to(WORK_ROOT)}`",
                "",
                "## Counts",
                "",
                f"- completed timepoints aggregated: {len(completed_timepoints)}",
                f"- aggregated rows with naive predictor: {sum(len(frame) for frame in naive_frames)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_modeling() -> None:
    predictor_args = []
    for predictor in MODEL_PREDICTORS:
        predictor_args.extend(["--predictor", predictor])

    run_command(
        [
            sys.executable,
            str(FUTUREFLU_SCRIPT_ROOT / "run_issue_date_pair_modeling.py"),
            "--run-dir",
            str(RUN_DIR),
            *predictor_args,
            "--forecast-years",
            *[str(year) for year in FORECAST_YEARS],
        ]
    )

    forecast_files = sorted(path.relative_to(WORK_ROOT) for path in FORECASTS_DIR.glob("*.tsv"))
    MODELING_REPORT.write_text(
        "\n".join(
            [
                f"# {REPORT_LABEL} {ISSUE_START.year}-{ISSUE_END.year} Issue-Date Modeling",
                "",
                "## Inputs",
                "",
                f"- `{AGGREGATED_TIP_ATTRIBUTES_NAIVE.relative_to(WORK_ROOT)}`",
                f"- forecast years: `{','.join(str(year) for year in FORECAST_YEARS)}`",
                "",
                "## Forecast Outputs",
                "",
                *[f"- `{path}`" for path in forecast_files],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FutureFlu issue-date pipeline with resumable steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "step",
        choices=["prepare", "timepoints", "aggregate", "model", "all", "timepoint"],
        help="which step to run",
    )
    parser.add_argument("--issue-date", help="single issue date for the timepoint step")
    parser.add_argument(
        "--reset-from-refine",
        action="store_true",
        help="for a single timepoint, delete refine-and-downstream outputs but keep IQ-TREE outputs",
    )
    parser.add_argument("--run-dir", help="override output run directory")
    parser.add_argument("--input-metadata", help="override normalized FutureFlu metadata TSV")
    parser.add_argument("--input-fasta", help="override normalized FutureFlu amino-acid FASTA")
    parser.add_argument("--input-sequence-table", help="override FutureFlu sequence table TSV")
    parser.add_argument("--distance-map-root", help="override HA distance map directory")
    parser.add_argument("--report-label", help="label used in generated markdown reports")
    parser.add_argument("--global-start", help="collection-date start for global subsampling")
    parser.add_argument("--global-end", help="collection-date end for global subsampling")
    parser.add_argument("--issue-start", help="first FutureFlu issue date to build")
    parser.add_argument("--issue-end", help="last FutureFlu issue date to build")
    parser.add_argument("--forecast-years", nargs="+", type=int, help="forecast years to emit in the model step")
    parser.add_argument(
        "--disable-hi",
        action="store_true",
        help="skip HI substitution predictors and remove HI predictor sets from modeling",
    )
    parser.add_argument(
        "--disable-dms",
        action="store_true",
        help="skip DMS predictors and remove DMS predictor sets from modeling",
    )
    parser.add_argument(
        "--allow-hi-fallback",
        action="store_true",
        help="if a requested stored HI substitution model is unavailable, reuse the latest available model explicitly",
    )
    parser.add_argument(
        "--predictor",
        dest="predictors",
        action="append",
        help="override predictor set for modeling; repeat for multiple sets",
    )
    parser.add_argument(
        "--timepoint-workers",
        type=int,
        default=1,
        help="number of issue dates to build in parallel during the timepoints step",
    )
    parser.add_argument(
        "--iqtree-threads",
        default="AUTO",
        help="thread count passed to IQ-TREE -nt for each timepoint worker",
    )
    args = parser.parse_args()
    apply_runtime_configuration(args)

    if args.step in {"prepare", "all"}:
        prepare_inputs()

    if args.step == "timepoint":
        if not args.issue_date:
            raise SystemExit("--issue-date is required for the 'timepoint' step")
        issue_date_string, _ = validate_issue_date(args.issue_date)
        if args.reset_from_refine:
            remove_refine_and_downstream_outputs(TIMEPOINTS_DIR / issue_date_string)
        run_timepoint(issue_date_string)
        return

    if args.step in {"timepoints", "all"}:
        run_all_timepoints()

    if args.step in {"aggregate", "all"}:
        aggregate_tip_attributes()

    if args.step in {"model", "all"}:
        run_modeling()


if __name__ == "__main__":
    main()
