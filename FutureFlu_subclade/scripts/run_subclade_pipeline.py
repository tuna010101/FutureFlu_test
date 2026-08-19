#!/usr/bin/env python3
"""Run the 2025/2026 subclade prediction pipeline stages.

English: Prepare definitions, annotations, component tables, and published
subclade prediction outputs using package-local paths only.
中文：仅使用包内路径，生成亚分支定义、注释、组件表和发布预测结果。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from Bio import SeqIO


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# English: Package-local paths only. Raw sequence inputs are omitted from final.
# 中文：仅使用包内路径；原始序列输入不进入 final 包。
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
LINEAR_MODULE = SCRIPTS_DIR / "predict_mutations_linear.py"
COMPONENT_MODULE = SCRIPTS_DIR / "component_metrics.py"

DATA_OUT = PACKAGE_ROOT / "data"
CONFIG_DIR = DATA_OUT / "configs"
POSITIVITY_DIR = DATA_OUT / "positivity"
EVESCAPE_DIR = DATA_OUT / "EVEscape"
RAW_INPUTS = PACKAGE_ROOT / "raw_inputs"
PROVENANCE_OUT = RAW_INPUTS / "run_logs"
LINEAR_OUT = PACKAGE_ROOT / "outputs" / "predictions" / "linear" / "results"
RISK_OUT = PACKAGE_ROOT / "outputs" / "predictions" / "risk_components"
ACC_OUT = RISK_OUT / "subclade_accuracy"
COMBINE_OUT = RISK_OUT / "component_combinations"
MUTATION_COMPONENTS_OUT = RISK_OUT / "mutation_components"
ANTIGENIC_OUT = RISK_OUT / "antigenic_novelty"

SUBTYPES = ("H1N1", "H3N2", "Victoria")
SUBTYPE_REPOS = {
    "H1N1": "seasonal_A-H1N1pdm_HA",
    "H3N2": "seasonal_A-H3N2_HA",
    "Victoria": "seasonal_B-Vic_HA",
}
RELEASE_INPUTS = {
    "H1N1": {
        "meta": "H1N1_pdm09_from20240201_to20260131_meta.csv",
        "nuc": "H1N1_pdm09_from20240201_to20260131_nuc.fasta",
        "pro": "H1N1_pdm09_from20240201_to20260131_pro.fasta",
        "reference": "pdm09-reference.fasta",
        "nextclade_dataset": "nextstrain/flu/h1n1pdm/ha/MW626062",
    },
    "H3N2": {
        "meta": "H3N2_human_from20240201_to20260130_meta.csv",
        "nuc": "H3N2_human_from20240201_to20260130_nuc.fasta",
        "pro": "H3N2_human_from20240201_to20260130_pro.fasta",
        "reference": "H3N2-reference.fasta",
        "nextclade_dataset": "nextstrain/flu/h3n2/ha/EPI1857216",
    },
    "Victoria": {
        "meta": "Victoria_from20240201_to20260130_meta.csv",
        "nuc": "Victoria_from20240201_to20260130_nuc.fasta",
        "pro": "Victoria_from20240201_to20260130_pro.fasta",
        "reference": "Victoria-reference.fasta",
        "nextclade_dataset": "nextstrain/flu/vic/ha/KX058884",
    },
}
REPO_OWNER = "influenza-clade-nomenclature"
TARGETS = (
    ("North", 2025),
    ("South", 2025),
    ("South", 2026),
)
HA1_RANGES = {
    "H3N2": (17, 345),
    "H1N1": (18, 344),
    "Victoria": (16, 362),
}
METRICS = [
    "total_escape",
    "predicted_prevalence",
    "mutual_information",
    "dissimilarity_charge_hydro",
    "accessibility_wcn",
    "fitness_eve",
    "antigenic_novelty",
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
COMBO_NAMES_ORDERED = [name for name, _ in COMBINATIONS]
N_PARAMS = {name: len(metrics) for name, metrics in COMBINATIONS}
T_VALUES = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    dtype=float,
)
PRE_ACT_COL_ORDER = (
    ["subtype", "year", "hemisphere", "subclade", "act_freq", "freq_prev"]
    + [
        col
        for combo_name in COMBO_NAMES_ORDERED
        for col in (f"{combo_name}_pre_fit", f"{combo_name}_pre_freq")
    ]
)


@dataclass(frozen=True)
class Target:
    subtype: str
    hemisphere: str
    year: int

    @property
    def hemi_lower(self) -> str:
        return self.hemisphere.lower()

    @property
    def label(self) -> str:
        return f"{self.subtype}_{self.hemisphere}_{self.year}"

    @property
    def cutoff(self) -> pd.Timestamp:
        if self.hemisphere == "North":
            return pd.Timestamp(f"{self.year}-02-01", tz="UTC")
        return pd.Timestamp(f"{self.year - 1}-09-01", tz="UTC")

    @property
    def evescape_date(self) -> str:
        if self.hemisphere == "North":
            return f"{self.year}0131"
        return f"{self.year - 1}0831"


def all_targets() -> list[Target]:
    return [Target(subtype, hemi, year) for subtype in SUBTYPES for hemi, year in TARGETS]


def ensure_dirs() -> None:
    for path in [
        DATA_OUT,
        DATA_OUT / "subclade_definitions",
        DATA_OUT / "subclade_counts",
        DATA_OUT / "futureflu_rank",
        RAW_INPUTS,
        SCRIPTS_DIR,
        LINEAR_OUT,
        MUTATION_COMPONENTS_OUT,
        ACC_OUT,
        COMBINE_OUT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_provenance_csv(name: str, frame: pd.DataFrame) -> None:
    PROVENANCE_OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROVENANCE_OUT / name, index=False)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def url_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as handle:
        return json.loads(handle.read().decode("utf-8"))


def url_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as handle:
        return handle.read().decode("utf-8")


def run_json_command(cmd: list[str]) -> object:
    result = None
    for attempt in range(1, 4):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            break
        if attempt < 3:
            print(f"[nextclade] retry command attempt={attempt + 1}: {' '.join(cmd)}")
            time.sleep(5)
    if result is None or result.returncode != 0:
        stderr = "" if result is None else result.stderr
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{stderr}")
    return json.loads(result.stdout)


def nextclade_dataset_name(subtype: str) -> str:
    return str(RELEASE_INPUTS[subtype]["nextclade_dataset"])


def nextclade_dataset_slug(name: str) -> str:
    return name.replace("/", "__")


def nextclade_result_path(target: Target) -> Path:
    return (
        RAW_INPUTS
        / "nextclade"
        / "results"
        / f"{target.subtype}_{target.hemi_lower}_{target.year}_nextclade.tsv"
    )


def nextclade_dataset_dir(subtype: str, tag: str) -> Path:
    return RAW_INPUTS / "nextclade" / "datasets" / f"{subtype}_{tag}"


def nextclade_dataset_zip(subtype: str, tag: str) -> Path:
    return RAW_INPUTS / "nextclade" / "datasets" / f"{subtype}_{tag}.zip"


def list_nextclade_dataset_versions(subtype: str) -> list[dict]:
    name = nextclade_dataset_name(subtype)
    data = run_json_command(["nextclade", "dataset", "list", "--name", name, "--json"])
    if not data:
        raise RuntimeError(f"no Nextclade dataset found for {subtype}: {name}")
    versions = data[0].get("versions", [])
    if not versions:
        raise RuntimeError(f"no Nextclade dataset versions for {subtype}: {name}")
    return versions


def selected_nextclade_tag(subtype: str, cutoff: pd.Timestamp) -> tuple[str, str]:
    versions = list_nextclade_dataset_versions(subtype)
    rows = []
    for item in versions:
        updated = pd.to_datetime(item["updatedAt"], utc=True)
        if updated < cutoff:
            rows.append((updated, item["tag"]))
    if not rows:
        raise RuntimeError(f"no Nextclade dataset for {subtype} before {cutoff.isoformat()}")
    updated, tag = sorted(rows)[-1]
    return tag, updated.isoformat()


def ensure_nextclade_dataset(subtype: str, tag: str) -> Path:
    out_dir = nextclade_dataset_dir(subtype, tag)
    out_zip = nextclade_dataset_zip(subtype, tag)
    if (out_dir / "pathogen.json").exists():
        return out_dir
    if out_zip.exists() and out_zip.stat().st_size > 0:
        return out_zip
    print(f"[nextclade] downloading {subtype} dataset tag={tag}")
    result = None
    for attempt in range(1, 4):
        result = subprocess.run(
            [
                "nextclade",
                "dataset",
                "get",
                "--name",
                nextclade_dataset_name(subtype),
                "--tag",
                tag,
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            break
        if attempt < 3:
            print(f"[nextclade] retry dataset download {subtype} tag={tag} attempt={attempt + 1}")
            time.sleep(5)
    if result is None or result.returncode != 0:
        print(f"[nextclade] directory download failed; trying zip for {subtype} tag={tag}")
        zip_result = subprocess.run(
            [
                "nextclade",
                "dataset",
                "get",
                "--name",
                nextclade_dataset_name(subtype),
                "--tag",
                tag,
                "--output-zip",
                str(out_zip),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if zip_result.returncode == 0:
            return out_zip
        stderr = "" if result is None else result.stderr
        raise RuntimeError(
            f"nextclade dataset get failed for {subtype} {tag}:\n"
            f"directory stderr:\n{stderr}\nzip stderr:\n{zip_result.stderr}"
        )
    return out_dir


def run_nextclade() -> None:
    ensure_dirs()
    rows = []
    for target in all_targets():
        tag, updated_at = selected_nextclade_tag(target.subtype, target.cutoff)
        dataset_dir = ensure_nextclade_dataset(target.subtype, tag)
        out_path = nextclade_result_path(target)
        nuc_path = release_input_path(target.subtype, "nuc")
        print(f"[nextclade] {target.label} tag={tag}")
        out_current = (
            out_path.exists()
            and out_path.stat().st_size > 0
            and out_path.stat().st_mtime >= nuc_path.stat().st_mtime
        )
        if out_current:
            print(f"[nextclade] reuse {out_path}")
        else:
            result = subprocess.run(
                [
                    "nextclade",
                    "run",
                    "--input-dataset",
                    str(dataset_dir),
                    "--replace-unknown",
                    "true",
                    "--output-tsv",
                    str(out_path),
                    str(nuc_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"nextclade run failed for {target.label}:\n{result.stderr}")
        rows.append(
            {
                "subtype": target.subtype,
                "hemisphere": target.hemi_lower,
                "year": target.year,
                "prediction_cutoff": target.cutoff.isoformat(),
                "dataset_name": nextclade_dataset_name(target.subtype),
                "dataset_tag": tag,
                "dataset_updated_at": updated_at,
                "dataset_path": str(dataset_dir.relative_to(PACKAGE_ROOT)),
                "nextclade_tsv": str(out_path.relative_to(PACKAGE_ROOT)),
            }
        )
    pd.DataFrame(rows).to_csv(RAW_INPUTS / "nextclade" / "selected_datasets.csv", index=False)


def fetch_commits(repo: str) -> list[dict]:
    commits: list[dict] = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{REPO_OWNER}/{repo}/commits?"
            f"per_page=100&page={page}"
        )
        batch = url_json(url)
        if not batch:
            break
        commits.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return commits


def fetch_subclades_tsv(repo: str, sha: str) -> str | None:
    url = (
        f"https://raw.githubusercontent.com/{REPO_OWNER}/{repo}/"
        f"{sha}/.auto-generated/subclades.tsv"
    )
    try:
        return url_text(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def run_definitions() -> None:
    ensure_dirs()
    bundled_targets = [target for target in all_targets() if definition_path(target).exists()]
    if len(bundled_targets) == len(all_targets()):
        selected_rows = []
        for target in all_targets():
            selected_rows.append(
                {
                    "subtype": target.subtype,
                    "hemisphere": target.hemi_lower,
                    "year": target.year,
                    "prediction_cutoff": target.cutoff.isoformat(),
                    "repo": "packaged release input",
                    "sha": "",
                    "commit_date": "",
                    "definition_path": str(definition_path(target).relative_to(PACKAGE_ROOT)),
                    "status": "bundled",
                }
            )
        pd.DataFrame(selected_rows).to_csv(
            DATA_OUT / "subclade_definitions" / "selected_definitions.csv",
            index=False,
        )
        print("[definitions] using package-bundled subclade definitions")
        return

    manifest_rows = []
    selected_rows = []
    for subtype, repo in SUBTYPE_REPOS.items():
        print(f"[definitions] fetching commits for {subtype} {repo}")
        commits = fetch_commits(repo)
        for item in commits:
            commit = item["commit"]
            manifest_rows.append(
                {
                    "subtype": subtype,
                    "repo": repo,
                    "sha": item["sha"],
                    "commit_date": commit["committer"]["date"],
                    "message": commit["message"].splitlines()[0],
                }
            )

        commit_df = pd.DataFrame(manifest_rows)
        subtype_commits = commit_df[commit_df["subtype"] == subtype].copy()
        subtype_commits["commit_ts"] = pd.to_datetime(subtype_commits["commit_date"], utc=True)

        for target in [t for t in all_targets() if t.subtype == subtype]:
            eligible = subtype_commits[subtype_commits["commit_ts"] < target.cutoff]
            if eligible.empty:
                selected_rows.append(
                    {
                        "subtype": subtype,
                        "hemisphere": target.hemi_lower,
                        "year": target.year,
                        "prediction_cutoff": target.cutoff.isoformat(),
                        "repo": repo,
                        "sha": "",
                        "commit_date": "",
                        "definition_path": "",
                        "status": "no_commit_before_cutoff",
                    }
                )
                continue
            chosen = eligible.sort_values("commit_ts").iloc[-1]
            text = fetch_subclades_tsv(repo, chosen["sha"])
            status = "ok" if text else "missing_subclades_tsv"
            out_name = f"{subtype}_{target.hemi_lower}_{target.year}_subclades.tsv"
            out_path = DATA_OUT / "subclade_definitions" / out_name
            if text:
                out_path.write_text(text, encoding="utf-8")
            selected_rows.append(
                {
                    "subtype": subtype,
                    "hemisphere": target.hemi_lower,
                    "year": target.year,
                    "prediction_cutoff": target.cutoff.isoformat(),
                    "repo": repo,
                    "sha": chosen["sha"],
                    "commit_date": chosen["commit_date"],
                    "definition_path": str(out_path.relative_to(PACKAGE_ROOT)) if text else "",
                    "status": status,
                }
            )
            print(f"[definitions] {target.label} {chosen['sha'][:12]} {status}")

    pd.DataFrame(manifest_rows).to_csv(
        DATA_OUT / "subclade_definitions" / "commit_manifest.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        DATA_OUT / "subclade_definitions" / "selected_definitions.csv", index=False
    )


def sequence_csv_path(subtype: str) -> Path:
    return RAW_INPUTS / f"{subtype}_HA_sequence_20260131.csv"


def read_pre_config(subtype: str) -> dict:
    cfg_name = {
        "H1N1": "h1n1_pre2024.json",
        "H3N2": "h3n2_pre2024.json",
        "Victoria": "victoria_pre2024.json",
    }[subtype]
    return json.loads((CONFIG_DIR / cfg_name).read_text(encoding="utf-8"))


def old_sequence_csv_path(subtype: str) -> Path:
    candidates = [
        RAW_INPUTS / f"{subtype}_HA_sequence_20250131.csv",
        RAW_INPUTS / "historical" / f"{subtype}_HA_sequence_20250131.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"missing historical sequence table for {subtype} under raw_inputs/: "
        + ", ".join(str(path) for path in candidates)
    )


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


def release_input_path(subtype: str, key: str) -> Path:
    base = RAW_INPUTS / RELEASE_INPUTS[subtype][key]
    if not base.exists():
        raise FileNotFoundError(
            f"missing external input for {subtype}: {base}. "
            "Place separately obtained raw inputs under raw_inputs/."
        )
    return base


def reference_fasta_path(subtype: str) -> Path:
    return release_input_path(subtype, "reference")


def aligned_fasta_path(subtype: str) -> Path:
    return RAW_INPUTS / "aligned_pro" / f"{subtype}_aligned_to_reference.fasta"


def run_mafft_alignment(subtype: str, force: bool = False) -> Path:
    out_path = aligned_fasta_path(subtype)
    reference_path = reference_fasta_path(subtype)
    fasta_path = release_input_path(subtype, "pro")
    if not reference_path.exists():
        raise FileNotFoundError(f"missing reference FASTA for {subtype}: {reference_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"missing release protein FASTA for {subtype}: {fasta_path}")
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        input_mtime = max(reference_path.stat().st_mtime, fasta_path.stat().st_mtime)
        if out_path.stat().st_mtime >= input_mtime:
            return out_path

    print(f"[sequences] mafft aligning {subtype}")
    with out_path.open("w", encoding="utf-8") as out_handle:
        result = subprocess.run(
            [
                "mafft",
                "--anysymbol",
                "--thread",
                "-1",
                "--retree",
                "1",
                "--maxiterate",
                "0",
                "--keeplength",
                "--addfragments",
                str(fasta_path),
                str(reference_path),
            ],
            stdout=out_handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"mafft failed for {subtype}: {result.stderr}")
    return out_path


def reference_id(subtype: str) -> str:
    records = list(SeqIO.parse(str(reference_fasta_path(subtype)), "fasta"))
    if len(records) != 1:
        raise ValueError(f"expected one reference record for {subtype}")
    return str(records[0].id)


def reference_aligned_positions(records: dict[str, str], subtype: str) -> list[int]:
    ref_id = reference_id(subtype)
    ref_seq = records.get(ref_id)
    if ref_seq is None:
        raise KeyError(f"reference {ref_id} not found in aligned FASTA for {subtype}")
    positions = [idx for idx, aa in enumerate(ref_seq) if aa != "-"]
    _, xcols = old_sequence_columns(subtype)
    expected_len = len(xcols)
    if len(positions) != expected_len:
        raise ValueError(
            f"reference ungapped length mismatch for {subtype}: {len(positions)} != {expected_len}"
        )
    return positions


def extract_reference_coordinate_sequence(aligned_seq: str, ref_positions: list[int]) -> list[str]:
    values = []
    for idx in ref_positions:
        aa = aligned_seq[idx].upper() if idx < len(aligned_seq) else "X"
        values.append(aa)
    return values


def old_sequence_columns(subtype: str) -> tuple[list[str], list[str]]:
    path = old_sequence_csv_path(subtype)
    header = list(pd.read_csv(path, nrows=0).columns)
    xcols = [col for col in header if re.fullmatch(r"X\d+", col)]
    return header, xcols


def load_release_info_lookup(meta_path: Path) -> dict[str, dict]:
    info_df = pd.read_csv(meta_path, dtype=str, encoding="utf-8-sig").fillna("")
    info_df.columns = [str(col).lstrip("\ufeff") for col in info_df.columns]
    if "Isolate_Id" not in info_df.columns:
        raise KeyError(f"missing Isolate_Id in {meta_path}")
    if "HA Segment_Id" not in info_df.columns:
        raise KeyError(f"missing HA Segment_Id in {meta_path}")

    lookup: dict[str, dict] = {}
    for row in info_df.to_dict("records"):
        isolate_id = str(row.get("Isolate_Id", "")).strip()
        if isolate_id:
            lookup.setdefault(isolate_id, row)
        for token in str(row.get("HA Segment_Id", "")).split("|"):
            key = token.strip()
            if key:
                lookup.setdefault(key, row)
    return lookup


def release_record_keys(record_id: str) -> tuple[str, list[str]]:
    parts = [part.strip() for part in str(record_id).split("|") if part.strip()]
    protein_accession = parts[0] if parts else str(record_id).strip()
    isolate_id = next((part for part in parts if part.startswith("EPI_ISL_")), "")
    keys = [key for key in (isolate_id, protein_accession) if key]
    accession_number = isolate_id or protein_accession
    return accession_number, keys


def parsed_date_or_none(value: object) -> pd.Timestamp | None:
    text = normalize_partial_date(value)
    if not text:
        return None
    try:
        return pd.to_datetime(text)
    except (TypeError, ValueError):
        return None


def nextclade_record_keys(seq_name: object) -> list[str]:
    parts = [part.strip() for part in str(seq_name).split("|") if part.strip()]
    protein_or_nuc_accession = parts[0] if parts else str(seq_name).strip()
    isolate_id = next((part for part in parts if part.startswith("EPI_ISL_")), "")
    return [key for key in (isolate_id, protein_or_nuc_accession, str(seq_name).strip()) if key]


def clean_nextclade_label(value: object) -> str:
    label = str(value).strip()
    if label.lower() in {"", "nan", "unknown", "unassigned", "n/a", "na", "none"}:
        return ""
    return label


def nextclade_assignment_lookup(target: Target) -> tuple[dict[str, dict[str, str]], dict[str, int | str]]:
    path = nextclade_result_path(target)
    if not path.exists():
        return {}, {
            "nextclade_status": "missing",
            "nextclade_rows": 0,
            "nextclade_clade_rows": 0,
            "nextclade_subclade_rows": 0,
            "nextclade_subclade_columns": "",
        }
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    clade_col = None
    for candidate in ["clade", "Nextclade_pango", "legacy-clade"]:
        if candidate in df.columns:
            clade_col = candidate
            break
    subclade_cols = [candidate for candidate in ["subclade", "proposedSubclade"] if candidate in df.columns]
    if clade_col is None and not subclade_cols:
        return {}, {
            "nextclade_status": "missing_clade_and_subclade_columns",
            "nextclade_rows": len(df),
            "nextclade_clade_rows": 0,
            "nextclade_subclade_rows": 0,
            "nextclade_subclade_columns": "",
        }
    if "seqName" not in df.columns:
        return {}, {
            "nextclade_status": "missing_seqName_column",
            "nextclade_rows": len(df),
            "nextclade_clade_rows": 0,
            "nextclade_subclade_rows": 0,
            "nextclade_subclade_columns": ",".join(subclade_cols),
        }
    lookup: dict[str, dict[str, str]] = {}
    clade_rows = 0
    subclade_rows = 0
    for _, row in df.iterrows():
        clade = clean_nextclade_label(row.get(clade_col, "")) if clade_col else ""
        subclade = ""
        for col in subclade_cols:
            subclade = clean_nextclade_label(row.get(col, ""))
            if subclade:
                break
        if not clade and not subclade:
            continue
        if clade:
            clade_rows += 1
        if subclade:
            subclade_rows += 1
        for key in nextclade_record_keys(row["seqName"]):
            assignment = lookup.setdefault(key, {})
            if clade and "clade" not in assignment:
                assignment["clade"] = clade
            if subclade and "subclade" not in assignment:
                assignment["subclade"] = subclade
    return lookup, {
        "nextclade_status": "ok",
        "nextclade_rows": len(df),
        "nextclade_clade_rows": clade_rows,
        "nextclade_subclade_rows": subclade_rows,
        "nextclade_subclade_columns": ",".join(subclade_cols),
    }


def write_combined_sequence_table(subtype: str) -> dict:
    old_path = old_sequence_csv_path(subtype)
    out_path = sequence_csv_path(subtype)
    meta_path = release_input_path(subtype, "meta")
    fasta_path = release_input_path(subtype, "pro")
    aligned_path = run_mafft_alignment(subtype, force=False)
    header, xcols = old_sequence_columns(subtype)
    expected_len = len(xcols)
    info_lookup = load_release_info_lookup(meta_path)
    aligned_records = {str(record.id): str(record.seq) for record in SeqIO.parse(str(aligned_path), "fasta")}
    ref_positions = reference_aligned_positions(aligned_records, subtype)

    stats = {
        "subtype": subtype,
        "output_path": str(out_path.relative_to(PACKAGE_ROOT)),
        "old_sequence_path": str(old_path.relative_to(PACKAGE_ROOT)),
        "release_meta_path": str(meta_path.relative_to(PACKAGE_ROOT)),
        "release_pro_path": str(fasta_path.relative_to(PACKAGE_ROOT)),
        "reference_path": str(reference_fasta_path(subtype).relative_to(PACKAGE_ROOT)),
        "aligned_pro_path": str(aligned_path.relative_to(PACKAGE_ROOT)),
        "expected_aa_length": expected_len,
        "old_rows": 0,
        "release_rows_written": 0,
        "skipped_duplicate": 0,
        "skipped_missing_meta": 0,
        "skipped_bad_collection_date": 0,
        "skipped_gap_count": 0,
        "skipped_alignment_failed": 0,
    }

    seen_accessions: set[str] = set()
    with out_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(header)

        with old_path.open("r", encoding="utf-8", newline="") as old_handle:
            reader = csv.reader(old_handle)
            old_header = next(reader)
            accession_idx = old_header.index("accession number")
            for row in reader:
                accession = row[accession_idx].strip()
                if accession in seen_accessions:
                    stats["skipped_duplicate"] += 1
                    continue
                seen_accessions.add(accession)
                writer.writerow(row)
                stats["old_rows"] += 1

        for record in SeqIO.parse(str(fasta_path), "fasta"):
            aligned_seq = aligned_records.get(str(record.id))
            if aligned_seq is None:
                stats["skipped_alignment_failed"] += 1
                continue
            ref_seq = extract_reference_coordinate_sequence(aligned_seq, ref_positions)
            if not any(aa != "X" for aa in ref_seq):
                stats["skipped_alignment_failed"] += 1
                continue
            if ref_seq.count("-") > 3:
                stats["skipped_gap_count"] += 1
                continue

            accession_number, lookup_keys = release_record_keys(str(record.id))
            info = None
            for key in lookup_keys:
                info = info_lookup.get(key)
                if info is not None:
                    break
            if info is None:
                stats["skipped_missing_meta"] += 1
                continue

            isolate_id = str(info.get("Isolate_Id", "")).strip()
            if isolate_id:
                accession_number = isolate_id
            if accession_number in seen_accessions:
                stats["skipped_duplicate"] += 1
                continue

            collection_date = parsed_date_or_none(info.get("Collection_Date", ""))
            if collection_date is None:
                stats["skipped_bad_collection_date"] += 1
                continue
            submission_date = parsed_date_or_none(info.get("Submission_Date", ""))
            if submission_date is None:
                submission_date = collection_date

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
                + ref_seq
            )
            seen_accessions.add(accession_number)
            stats["release_rows_written"] += 1

    return stats


def run_sequences() -> None:
    ensure_dirs()
    rows = []
    for subtype in SUBTYPES:
        out_path = sequence_csv_path(subtype)
        print(f"[sequences] building {out_path}")
        t0 = time.time()
        stats = write_combined_sequence_table(subtype)
        stats["status"] = "ok"
        stats["elapsed_seconds"] = round(time.time() - t0, 3)
        rows.append(stats)
        print(
            "[sequences] "
            f"{subtype} old={stats['old_rows']} "
            f"release={stats['release_rows_written']} "
            f"elapsed={stats['elapsed_seconds']}s"
        )
    write_provenance_csv("sequence_generation.csv", pd.DataFrame(rows))


def definition_path(target: Target) -> Path:
    return (
        DATA_OUT
        / "subclade_definitions"
        / f"{target.subtype}_{target.hemi_lower}_{target.year}_subclades.tsv"
    )


def target_season_window(year: int, hemisphere: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if hemisphere.lower() == "north":
        return pd.Timestamp(f"{year}-09-01"), pd.Timestamp(f"{year + 1}-02-01")
    return pd.Timestamp(f"{year}-02-01"), pd.Timestamp(f"{year}-09-01")


def candidate_window(year: int, hemisphere: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if hemisphere.lower() == "north":
        return pd.Timestamp(f"{year - 1}-09-01"), pd.Timestamp(f"{year}-02-01")
    return pd.Timestamp(f"{year - 1}-02-01"), pd.Timestamp(f"{year - 1}-09-01")


def parse_definition_rules(path: Path, subtype: str) -> tuple[dict[str, list[tuple[str, str]]], dict]:
    raw = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    parent: dict[str, str] = {}
    direct: dict[str, list[tuple[str, str]]] = {}
    ha1_start, ha1_end = HA1_RANGES[subtype]

    for _, row in raw.iterrows():
        name = row["clade"].strip()
        gene = row["gene"].strip()
        site = row["site"].strip()
        alt = row["alt"].strip()
        if not name:
            continue
        if gene == "clade":
            if site:
                parent[name] = site
            continue
        if gene not in {"HA1", "HA2"}:
            continue
        try:
            pos = int(float(site))
        except ValueError:
            continue
        if gene == "HA1":
            x_pos = ha1_start + pos - 1
        else:
            x_pos = ha1_end + pos
        direct.setdefault(name, []).append((f"X{x_pos}", alt.upper()))

    inherited: dict[str, list[tuple[str, str]]] = {}

    def collect(name: str, stack: tuple[str, ...] = ()) -> list[tuple[str, str]]:
        if name in inherited:
            return inherited[name]
        if name in stack:
            inherited[name] = direct.get(name, [])
            return inherited[name]
        rules: list[tuple[str, str]] = []
        if parent.get(name):
            rules.extend(collect(parent[name], stack + (name,)))
        rules.extend(direct.get(name, []))
        by_column: dict[str, str] = {}
        column_order: list[str] = []
        for col, state in rules:
            if col not in by_column:
                column_order.append(col)
            by_column[col] = state
        deduped = [(col, by_column[col]) for col in column_order]
        inherited[name] = deduped
        return deduped

    for name in sorted(set(parent) | set(direct)):
        collect(name)

    inherited = {k: v for k, v in inherited.items() if v}
    return inherited, {}


def annotate_one(target: Target) -> dict:
    def_path = definition_path(target)
    if not def_path.exists():
        return {
            "subtype": target.subtype,
            "hemisphere": target.hemi_lower,
            "year": target.year,
            "status": "missing_definition",
        }
    seq_path = sequence_csv_path(target.subtype)
    rules, meta = parse_definition_rules(def_path, target.subtype)
    usecols = ["accession number", "name", "clade", "collection_date", "submission_date", "season"]
    xcols = sorted(
        {x for rule_list in rules.values() for x, _ in rule_list},
        key=lambda value: int(value[1:]),
    )
    header = pd.read_csv(seq_path, nrows=0).columns
    xcols = [col for col in xcols if col in header]
    df = pd.read_csv(seq_path, usecols=usecols + xcols)
    nextclade_lookup, nextclade_meta = nextclade_assignment_lookup(target)
    nextclade_clade_hits = 0
    nextclade_subclade_hits = 0
    nextclade_subclades = pd.Series("", index=df.index, dtype=object)
    if nextclade_lookup:
        mapped_clades = []
        for _, row in df.iterrows():
            accession = str(row["accession number"]).strip()
            assignment = nextclade_lookup.get(accession, {})
            clade = assignment.get("clade", "")
            if clade:
                nextclade_clade_hits += 1
                mapped_clades.append(clade)
            else:
                mapped_clades.append(row["clade"])
            subclade = assignment.get("subclade", "")
            if subclade:
                nextclade_subclade_hits += 1
                nextclade_subclades.loc[row.name] = subclade
        df["clade"] = mapped_clades
    for col in xcols:
        df[col] = df[col].fillna("").astype(str).str.upper()

    labels = pd.Series("unknown", index=df.index, dtype=object)
    matched_rule_counts = pd.Series(0, index=df.index, dtype=np.int16)
    matched_depth = pd.Series(-1, index=df.index, dtype=np.int16)
    ordered = sorted(rules.items(), key=lambda item: (len(item[1]), len(item[0].split(".")), item[0]))
    for name, rule_list in ordered:
        valid_rules = [(col, state) for col, state in rule_list if col in df.columns]
        if not valid_rules:
            continue
        mask = pd.Series(True, index=df.index)
        for col, state in valid_rules:
            mask &= df[col].eq(state)
        if mask.any():
            depth = len(name.split("."))
            better = mask & (
                (len(valid_rules) > matched_rule_counts)
                | ((len(valid_rules) == matched_rule_counts) & (depth >= matched_depth))
            )
            labels.loc[better] = name
            matched_rule_counts.loc[better] = len(valid_rules)
            matched_depth.loc[better] = depth

    final_labels = labels.copy()
    nextclade_mask = nextclade_subclades.ne("")
    final_labels.loc[nextclade_mask] = nextclade_subclades.loc[nextclade_mask]
    source = pd.Series("unknown", index=df.index, dtype=object)
    source.loc[labels.ne("unknown")] = "ha_rule"
    source.loc[nextclade_mask] = "nextclade"

    out = df[usecols].copy()
    out["subtype"] = target.subtype
    out["hemisphere"] = target.hemi_lower
    out["target_year"] = target.year
    out["subclade"] = final_labels
    out["ha_rule_subclade"] = labels
    out["nextclade_subclade"] = nextclade_subclades
    out["subclade_source"] = source
    out["matched_ha_rule_count"] = matched_rule_counts
    out_path = annotation_path(target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    assigned = int(out["subclade"].ne("unknown").sum())
    return {
        "subtype": target.subtype,
        "hemisphere": target.hemi_lower,
        "year": target.year,
        "status": "ok",
        "rows": len(out),
        "assigned_rows": assigned,
        "assigned_fraction": assigned / len(out) if len(out) else math.nan,
        "annotation_path": str(out_path.relative_to(PACKAGE_ROOT)),
        "nextclade_clade_hits": nextclade_clade_hits,
        "nextclade_clade_hit_fraction": nextclade_clade_hits / len(out) if len(out) else math.nan,
        "nextclade_subclade_hits": nextclade_subclade_hits,
        "nextclade_subclade_hit_fraction": nextclade_subclade_hits / len(out) if len(out) else math.nan,
        **nextclade_meta,
        **meta,
    }


def run_annotate() -> None:
    ensure_dirs()
    rows = []
    for target in all_targets():
        print(f"[annotate] {target.label}")
        rows.append(annotate_one(target))
    write_provenance_csv("subclade_definition_coverage.csv", pd.DataFrame(rows))


def run_linear() -> None:
    ensure_dirs()
    if not LINEAR_MODULE.exists():
        raise FileNotFoundError(f"missing package-local linear module: {LINEAR_MODULE}")
    step2 = load_module("predict_mutations_linear", LINEAR_MODULE)
    for subtype in SUBTYPES:
        seq_df = pd.read_csv(sequence_csv_path(subtype))
        for hemisphere, year in TARGETS:
            target = Target(subtype, hemisphere, year)
            epi_path = POSITIVITY_DIR / f"{subtype}_positive_rate_{target.hemi_lower}.csv"
            epi_df = pd.read_csv(epi_path)
            out_dir = LINEAR_OUT / f"{subtype}_{hemisphere}" / str(year)
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = target.label
            print(f"[linear] {prefix}")
            log_print = step2.setup_logging(str(out_dir))
            log_print(f"subclade pipeline input seq_file: {sequence_csv_path(subtype)}")
            log_print(f"subclade pipeline input epi_file: {epi_path}")
            theta_range = np.arange(0.1, 0.5 + 0.1 / 2, 0.1)
            prev_data = step2.site_prevalence(seq_df, year, hemisphere, subtype)
            prev_data.to_csv(out_dir / f"{prefix}_prevalence.csv", index=False)
            gmeasure_data = step2.gmeasure(prev_data, theta_range)
            gmeasure_data.to_csv(out_dir / f"{prefix}_gmeasure.csv", index=False)
            best_theta, best_r2, years_used = step2.fit_regression(gmeasure_data, epi_df, log_print)
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


def valid_label_series(df: pd.DataFrame) -> pd.Series:
    vals = df["subclade"].fillna("").astype(str).str.strip()
    return vals[~vals.str.lower().isin({"", "unknown", "nan", "unassigned"})]


def counted_label_series(df: pd.DataFrame) -> pd.Series:
    vals = df["subclade"].fillna("").astype(str).str.strip()
    return vals[~vals.str.lower().isin({"", "nan"})]


def annotation_path(target: Target) -> Path:
    return (
        RAW_INPUTS
        / "subclade_annotations"
        / f"{target.subtype}_{target.hemi_lower}_{target.year}_subclade_annotations.csv"
    )


def build_count_and_truth_tables() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    labels = []
    count_tables: dict[str, pd.DataFrame] = {}
    for subtype in SUBTYPES:
        count_rows = []
        for hemisphere, year in TARGETS:
            target = Target(subtype, hemisphere, year)
            ann = pd.read_csv(annotation_path(target), low_memory=False)
            ann["collection_date"] = pd.to_datetime(ann["collection_date"])
            ann["submission_date"] = pd.to_datetime(ann["submission_date"])
            for count_year in sorted({year - 1, year}):
                start, end = target_season_window(count_year, target.hemi_lower)
                collection_mask = (ann["collection_date"] >= start) & (ann["collection_date"] < end)
                submission_mask = collection_mask & (ann["submission_date"] < end)
                collection_counts = counted_label_series(ann.loc[collection_mask]).value_counts()
                submission_counts = counted_label_series(ann.loc[submission_mask]).value_counts()
                subclades = sorted(set(collection_counts.index).union(set(submission_counts.index)))
                for subclade in subclades:
                    count_rows.append(
                        {
                            "target_year": year,
                            "target_hemisphere": target.hemi_lower,
                            "year": count_year,
                            "hemisphere": target.hemi_lower,
                            "subclade": subclade,
                            "submission_count": int(submission_counts.get(subclade, 0)),
                            "collection_count": int(collection_counts.get(subclade, 0)),
                        }
                    )
            start, end = target_season_window(year, target.hemi_lower)
            label_counts = valid_label_series(
                ann.loc[(ann["collection_date"] >= start) & (ann["collection_date"] < end)]
            ).value_counts()
            if not label_counts.empty:
                labels.append(
                    {
                        "subtype": subtype,
                        "hemisphere": target.hemi_lower,
                        "year": year,
                        "subclade": label_counts.idxmax(),
                    }
                )
        key = subtype.lower()
        count_tables[key] = pd.DataFrame(count_rows).drop_duplicates()
    truth = pd.DataFrame(labels)
    return count_tables, truth


def write_prediction_truth_definition_alignment_report() -> None:
    definitions_path = DATA_OUT / "subclade_definitions" / "selected_definitions.csv"
    nextclade_path = RAW_INPUTS / "nextclade" / "selected_datasets.csv"
    if not definitions_path.exists() or not nextclade_path.exists():
        return

    definitions = pd.read_csv(definitions_path)
    nextclade = pd.read_csv(nextclade_path)
    merged = definitions.merge(
        nextclade,
        on=["subtype", "hemisphere", "year", "prediction_cutoff"],
        how="left",
        suffixes=("_definition", "_nextclade"),
    )
    rows = []
    for _, row in merged.iterrows():
        target = Target(row["subtype"], str(row["hemisphere"]).capitalize(), int(row["year"]))
        rows.append(
            {
                "subtype": target.subtype,
                "hemisphere": target.hemi_lower,
                "year": target.year,
                "prediction_cutoff": row["prediction_cutoff"],
                "annotation_used_by_prediction_and_truth": str(
                    annotation_path(target).relative_to(PACKAGE_ROOT)
                ),
                "definition_sha": row.get("sha", ""),
                "definition_commit_date": row.get("commit_date", ""),
                "nextclade_dataset_tag": row.get("dataset_tag", ""),
                "nextclade_dataset_updated_at": row.get("dataset_updated_at", ""),
            }
        )
    write_provenance_csv("prediction_truth_subclade_definition_alignment.csv", pd.DataFrame(rows))


def write_count_and_truth_tables() -> pd.DataFrame:
    count_tables, truth = build_count_and_truth_tables()
    for key, count_df in count_tables.items():
        count_df.to_csv(
            DATA_OUT / "subclade_counts" / f"submission_collection_subclade_count_{key}.csv",
            index=False,
        )
    truth.to_csv(DATA_OUT / "futureflu_rank" / "circulating_subclade.csv", index=False)
    write_prediction_truth_definition_alignment_report()
    return truth


def run_counts() -> None:
    ensure_dirs()
    write_count_and_truth_tables()


def parse_mutation_group(value: object) -> list[tuple[str, str]]:
    if pd.isna(value):
        return []
    rules = []
    for token in str(value).split(","):
        token = token.strip()
        match = re.fullmatch(r"(\d+)([A-Za-z-])", token)
        if match:
            rules.append((f"X{int(match.group(1))}", match.group(2).upper()))
    return rules


def subclade_info(values: pd.Series) -> str:
    clean = values.fillna("").astype(str).str.strip()
    clean = clean[~clean.str.lower().isin({"", "unknown", "nan"})]
    if clean.empty:
        return "unknown"
    counts = clean.value_counts()
    total = counts.sum()
    return ", ".join(f"{name} ({count / total * 100:.1f}%)" for name, count in counts.items())


def calculate_prevalence(group: object, mutation_prevalence: dict[str, float]) -> float:
    total = 0.0
    for _, state in parse_mutation_group(group):
        # English: This fallback is unused because mutation keys include site and state.
        # 中文：mutation key 已包含位点和状态，因此该回退分支不会被使用。
        _ = state
    if pd.isna(group):
        return 0.0
    for token in str(group).split(","):
        mut = token.strip()
        if mut:
            total += float(mutation_prevalence.get(mut, 0.0))
    return total


def antigenic_source_path(subtype: str) -> Path:
    candidates = [
        ANTIGENIC_OUT / f"strain_antigenic_novelty_{subtype}.csv",
        RAW_INPUTS / f"strain_antigenic_novelty_{subtype}.csv",
        RAW_INPUTS / "antigenic_novelty" / f"strain_antigenic_novelty_{subtype}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing antigenic novelty input for {subtype}")


def prepare_antigenic_inputs() -> dict[str, Path]:
    ensure_dirs()
    out = {}
    for subtype in SUBTYPES:
        src = antigenic_source_path(subtype)
        dst = ANTIGENIC_OUT / f"strain_antigenic_novelty_{subtype}.csv"
        ANTIGENIC_OUT.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        out[subtype] = dst
    return out


def evescape_pair_for_target(subtype: str, required_date: str) -> tuple[Path, Path, str, bool]:
    cfg = read_pre_config(subtype)
    prefix = cfg["evescape_prefix"]
    evescape_dir = DATA_OUT / cfg["evescape_dir"]
    available = []
    for mutation_path in evescape_dir.glob(f"{prefix}_evescape_*.csv"):
        match = re.fullmatch(rf"{re.escape(prefix)}_evescape_(\d{{8}})\.csv", mutation_path.name)
        if not match:
            continue
        date_str = match.group(1)
        sites_path = evescape_dir / f"{prefix}_evescape_sites_{date_str}.csv"
        if sites_path.exists() and date_str <= required_date:
            available.append((date_str, mutation_path, sites_path))
    if not available:
        raise FileNotFoundError(
            f"missing EVEscape files for {subtype} with date <= {required_date} in {evescape_dir}"
        )
    used_date, mutation_path, sites_path = max(available, key=lambda item: item[0])
    return mutation_path, sites_path, used_date, used_date == required_date


def subclade_info_from_matching(matching: pd.DataFrame) -> str:
    if matching.empty or "subclade" not in matching.columns:
        return "unknown"
    return subclade_info(matching["subclade"])


def run_component() -> None:
    ensure_dirs()
    comp = load_module("component_metrics", COMPONENT_MODULE)
    antigenic_paths = prepare_antigenic_inputs()
    frames = []
    availability = []
    for target in all_targets():
        print(f"[component] {target.label}")
        ann = pd.read_csv(annotation_path(target), low_memory=False)
        seq = pd.read_csv(sequence_csv_path(target.subtype))
        ann_small = ann[["accession number", "subclade"]].rename(
            columns={"accession number": "accession_number"}
        )
        seq = seq.rename(columns={"accession number": "accession_number"})
        seq = seq.merge(ann_small, on="accession_number", how="left")
        seq["collection_date"] = pd.to_datetime(seq["collection_date"])
        seq["submission_date"] = pd.to_datetime(seq["submission_date"])

        linear_dir = LINEAR_OUT / f"{target.subtype}_{target.hemisphere}" / str(target.year)
        prefix = target.label
        dist_path = linear_dir / f"{prefix}_distribution.csv"
        mut_path = linear_dir / f"{prefix}_mutations.csv"
        if not dist_path.exists() or not mut_path.exists():
            availability.append({"target": target.label, "status": "missing_linear"})
            continue
        dist = pd.read_csv(dist_path)
        prediction_df = pd.read_csv(mut_path)
        mutations_path, sites_path, used_date, exact_match = evescape_pair_for_target(
            target.subtype, target.evescape_date
        )
        mutations_df = pd.read_csv(mutations_path)
        sites_df = pd.read_csv(sites_path)
        antigenic_novelty_df = pd.read_csv(antigenic_paths[target.subtype])

        dist["mutation_count"] = dist["risk_mutation_group"].apply(comp.count_mutations)
        df_filtered = dist[dist["mutation_count"] > 0]
        if df_filtered.empty:
            availability.append(
                {
                    "target": target.label,
                    "status": "empty_distribution",
                    "required_evescape_date": target.evescape_date,
                    "used_evescape_date": used_date,
                    "exact_match": exact_match,
                    "mutations_path": str(mutations_path.relative_to(PACKAGE_ROOT)),
                    "sites_path": str(sites_path.relative_to(PACKAGE_ROOT)),
                }
            )
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
            availability.append(
                {
                    "target": target.label,
                    "status": "empty_after_outlier_filter",
                    "required_evescape_date": target.evescape_date,
                    "used_evescape_date": used_date,
                    "exact_match": exact_match,
                    "mutations_path": str(mutations_path.relative_to(PACKAGE_ROOT)),
                    "sites_path": str(sites_path.relative_to(PACKAGE_ROOT)),
                }
            )
            continue
        all_mutation_groups = non_outliers["risk_mutation_group"].dropna().tolist()

        start, end = candidate_window(target.year, target.hemisphere)
        filtered_seq_df = seq[
            (seq["collection_date"] >= start)
            & (seq["collection_date"] < end)
            & (seq["submission_date"] < end)
        ].copy()

        mutations_min_escape = mutations_df["evescape"].min()
        sites_min_escape = sites_df["evescape"].min()
        mutation_escape = {
            f"{row['i']}{row['mut']}": row["evescape"] - mutations_min_escape
            for _, row in mutations_df.iterrows()
        }
        site_escape = {
            str(row["i"]): row["evescape"] - sites_min_escape for _, row in sites_df.iterrows()
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
            lambda value: 0.0 if pd.isna(value) else value - awcn_min
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
        min_prev = min(mutation_prevalence.values()) if mutation_prevalence else 0.0
        mutation_prevalence = {k: v - min_prev for k, v in mutation_prevalence.items()}

        antigenic_tmp = antigenic_novelty_df.copy()
        antigenic_tmp["_an_norm"] = antigenic_tmp.groupby("season")[
            "antigenic_novelty"
        ].transform(lambda values: values - values.min())
        antigenic_novelty_dict = dict(
            zip(antigenic_tmp["accession_number"], antigenic_tmp["_an_norm"])
        )

        result_rows = []
        for _, row in non_outliers.iterrows():
            mutation_group = row["risk_mutation_group"]
            count = row["mutation_count"]
            total_escape = comp.calculate_total_escape_value(
                mutation_group,
                mutation_escape,
                site_escape,
                target.subtype,
                sites_df,
                mutations_df,
            )
            predicted_prevalence = comp.calculate_prevalence(
                mutation_group, mutation_prevalence
            )
            if pd.isna(mutation_group):
                mutual_info = 0
            else:
                muts = [mut.strip() for mut in mutation_group.split(",")]
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
            subclade = subclade_info_from_matching(matching_seqs)
            antigenic_value = comp.calculate_antigenic_novelty_from_matching(
                matching_seqs, antigenic_novelty_dict
            )
            dch_value = comp.calculate_metric_value(
                mutation_group, mutation_dch, site_dch, target.subtype, sites_df, mutations_df
            )
            awcn_value = comp.calculate_metric_value(
                mutation_group, mutation_awcn, site_awcn, target.subtype, sites_df, mutations_df
            )
            ef_value = comp.calculate_metric_value(
                mutation_group, mutation_ef, site_ef, target.subtype, sites_df, mutations_df
            )
            result_rows.append(
                {
                    "subtype": target.subtype,
                    "hemisphere": target.hemi_lower,
                    "year": target.year,
                    "risk_mutation_group": mutation_group,
                    "subclade": subclade,
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
        results_df = pd.DataFrame(result_rows)
        if not results_df.empty:
            filter_df = results_df.rename(columns={"subclade": "clade"})
            filter_df = comp.filter_random_single_mutations(filter_df)
            results_df = filter_df.rename(columns={"clade": "subclade"})
            frames.append(results_df)
        availability.append(
            {
                "target": target.label,
                "status": "ok",
                "required_evescape_date": target.evescape_date,
                "used_evescape_date": used_date,
                "exact_match": exact_match,
                "mutations_path": str(mutations_path.relative_to(PACKAGE_ROOT)),
                "sites_path": str(sites_path.relative_to(PACKAGE_ROOT)),
                "antigenic_path": str(antigenic_paths[target.subtype].relative_to(PACKAGE_ROOT)),
            }
        )

    if not frames:
        raise RuntimeError("component stage produced no rows")
    out = pd.concat(frames, ignore_index=True)
    columns = [
        "subtype",
        "hemisphere",
        "year",
        "risk_mutation_group",
        "subclade",
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
    out_path = MUTATION_COMPONENTS_OUT / "risk_mutation_group_component.csv"
    out[columns].to_csv(out_path, index=False)
    out[columns].to_csv(RISK_OUT / "risk_mutation_group_component.csv", index=False)
    write_provenance_csv("component_input_availability.csv", pd.DataFrame(availability))


_LABEL_RE = re.compile(r"([^,()]+)\s*\(([-+0-9.]+)%\)")


def parse_subclade_string(value: object) -> list[tuple[str, float]]:
    if pd.isna(value):
        return []
    parsed = []
    for name, pct in _LABEL_RE.findall(str(value)):
        parsed.append((name.strip(), float(pct)))
    return parsed


def dominant_subclade(value: object) -> str | None:
    parsed = parse_subclade_string(value)
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[1])[0]


def zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    if len(values) == 1:
        return np.zeros_like(values, dtype=float)
    std = np.nanstd(values, ddof=1)
    if std == 0 or np.isnan(std):
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / std


def safe_sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))


def min_seq_count(subtype: str) -> int:
    return 10 if "H3N2" in str(subtype).upper() else 0


def normalize_fit_columns(df: pd.DataFrame, fit_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for _, grp in out.groupby(["subtype", "hemisphere", "year"]):
        idx = grp.index
        for col in fit_cols:
            vals = out.loc[idx, col]
            v_min = vals.min()
            v_max = vals.max()
            if pd.notna(v_min) and pd.notna(v_max) and v_max > v_min:
                out.loc[idx, col] = (vals - v_min) / (v_max - v_min)
            else:
                out.loc[idx, col] = 0.0
    return out


def filtered_component_df(comp: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for subtype, sub_df in comp.groupby("subtype"):
        frames.append(sub_df[sub_df["mutation_group_seq_count"] >= min_seq_count(subtype)])
    if not frames:
        return comp.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def compute_max_tables(comp: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    for _, row in comp.iterrows():
        dom = dominant_subclade(row.get("subclade"))
        if dom is None or str(dom).lower() == "unknown":
            continue
        rec = {
            "subtype": row["subtype"],
            "hemisphere": row["hemisphere"],
            "year": row["year"],
            "subclade_single": dom,
            "risk_mutation_group": row.get("risk_mutation_group", np.nan),
            "mutation_count": row.get("mutation_count", np.nan),
            "mutation_group_seq_count": row.get("mutation_group_seq_count", np.nan),
        }
        for metric in METRICS:
            rec[f"fit_{metric}"] = row.get(metric, np.nan)
        rows.append(rec)

    fit_cols = [f"fit_{metric}" for metric in METRICS]
    if not rows:
        empty = pd.DataFrame(
            columns=["subtype", "hemisphere", "year", "subclade_single"] + fit_cols
        )
        return empty, {metric: pd.DataFrame() for metric in METRICS_3}

    tmp = pd.DataFrame(rows)
    max_df = (
        tmp.groupby(["subtype", "hemisphere", "year", "subclade_single"], as_index=False)
        .agg({col: "max" for col in fit_cols})
    )

    inform = {}
    for metric in METRICS_3:
        col = f"fit_{metric}"
        source_rows = []
        for keys, grp in tmp.groupby(["subtype", "hemisphere", "year", "subclade_single"]):
            if grp[col].isna().all():
                continue
            row = grp.loc[grp[col].idxmax()]
            source_rows.append(
                {
                    "subtype": keys[0],
                    "hemisphere": keys[1],
                    "year": keys[2],
                    "subclade": keys[3],
                    "risk_mutation_group": row.get("risk_mutation_group", np.nan),
                    "mutation_count": row.get("mutation_count", np.nan),
                    "mutation_group_seq_count": row.get("mutation_group_seq_count", np.nan),
                    metric: row.get(col, np.nan),
                }
            )
        inform[metric] = pd.DataFrame(source_rows)
    return max_df, inform


def compute_metric_accuracy(max_df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subtype in sorted(max_df["subtype"].unique()):
        sub_max = max_df[max_df["subtype"] == subtype]
        sub_truth = truth[truth["subtype"] == subtype]
        label_hy = sub_truth[["hemisphere", "year"]].drop_duplicates()
        total_count = len(label_hy)
        for metric in METRICS:
            col = f"fit_{metric}"
            hit_count = 0
            for _, label in label_hy.iterrows():
                hemi = label["hemisphere"]
                year = label["year"]
                truth_rows = sub_truth[
                    (sub_truth["hemisphere"] == hemi) & (sub_truth["year"] == year)
                ]
                grp = sub_max[(sub_max["hemisphere"] == hemi) & (sub_max["year"] == year)]
                if truth_rows.empty or grp.empty or grp[col].isna().all():
                    continue
                pred = grp.loc[grp[col].idxmax(), "subclade_single"]
                if pred == truth_rows["subclade"].iloc[0]:
                    hit_count += 1
            rows.append(
                {
                    "subtype": subtype,
                    "metric": metric,
                    "hit_count": hit_count,
                    "total_count": total_count,
                    "accuracy": round(hit_count / total_count, 4) if total_count else np.nan,
                    "methods": "max",
                }
            )
    return pd.DataFrame(rows)


def subclade_frequency_dict(
    subtype: str,
    target_year: int,
    hemisphere: str,
    count_year: int,
    count_col: str,
) -> dict[str, float]:
    path = DATA_OUT / "subclade_counts" / f"submission_collection_subclade_count_{subtype.lower()}.csv"
    counts = pd.read_csv(path)
    hemi = hemisphere.lower()
    sub = counts[
        (counts["target_year"] == target_year)
        & (counts["target_hemisphere"] == hemi)
        & (counts["year"] == count_year)
        & (counts["hemisphere"] == hemi)
    ]
    total = sub[count_col].sum()
    if total <= 0:
        return {}
    return {row["subclade"]: row[count_col] / total for _, row in sub.iterrows()}


def ha1_clade_component_max_path() -> Path:
    return (
        DATA_OUT
        / "ha1_clade_priors"
        / "clade_component_max.csv"
    )


def ha1_clade_count_path(subtype: str) -> Path:
    return (
        DATA_OUT
        / "ha1_clade_priors"
        / f"submission_collection_clade_count_{subtype.lower()}.csv"
    )


def ha1_clade_frequency_dict(
    subtype: str, year: int, hemisphere: str, count_col: str = "submission_count"
) -> dict[str, float]:
    path = ha1_clade_count_path(subtype)
    if not path.exists():
        return {}
    counts = pd.read_csv(path)
    hemi = hemisphere.lower()
    sub = counts[(counts["year"] == year) & (counts["hemisphere"] == hemi)]
    total = sub[count_col].sum()
    if total <= 0:
        return {}
    return {row["clade"]: row[count_col] / total for _, row in sub.iterrows()}


def ha1_clade_temperature_training_data(subtype: str, combo_metrics: list[str]) -> list[dict]:
    max_path = ha1_clade_component_max_path()
    if not max_path.exists():
        return []
    old_max = pd.read_csv(max_path)
    required_cols = {"subtype", "hemisphere", "year", "clade_single"} | {
        f"fit_{metric}" for metric in combo_metrics
    }
    if not required_cols.issubset(old_max.columns):
        return []
    sub_max = old_max[old_max["subtype"] == subtype].copy()
    if sub_max.empty:
        return []

    seasons = sorted(
        {
            (int(row["year"]), str(row["hemisphere"]))
            for _, row in sub_max[["year", "hemisphere"]].drop_duplicates().iterrows()
        },
        key=season_sort_key,
    )
    past = []
    for year, hemi in seasons:
        grp = sub_max[(sub_max["year"] == year) & (sub_max["hemisphere"] == hemi)].reset_index(
            drop=True
        )
        clades = grp["clade_single"].to_numpy(copy=True)
        if len(clades) == 0:
            continue
        z_mat = np.vstack(
            [
                zscore(grp[f"fit_{metric}"].fillna(0.0).to_numpy(dtype=float))
                for metric in combo_metrics
            ]
        )
        prev_freq = ha1_clade_frequency_dict(subtype, year - 1, hemi, "submission_count")
        cur_freq = ha1_clade_frequency_dict(subtype, year, hemi, "submission_count")
        xi_prev = np.array([np.nan_to_num(prev_freq.get(c, 0.0), nan=0.0) for c in clades])
        cur_arr = np.array([np.nan_to_num(cur_freq.get(c, 0.0), nan=0.0) for c in clades])
        actual_freq = cur_arr / cur_arr.sum() if cur_arr.sum() > 0 else None
        if actual_freq is None:
            continue
        past.append(
            {
                "z_mat": z_mat,
                "xi_prev": xi_prev,
                "actual_freq": actual_freq,
                "source": "ha1_clade_priors",
                "year": year,
                "hemisphere": hemi,
            }
        )
    return past


def build_elpd_aic(lpd_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lpd_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    elpd = lpd_df.pivot_table(
        index=["subtype", "hemisphere", "year"], columns="combo", values="lpd", aggfunc="first"
    ).reset_index()
    elpd.columns.name = None
    for combo_name in COMBO_NAMES_ORDERED:
        if combo_name in elpd.columns:
            elpd = elpd.rename(columns={combo_name: f"ELPD_{combo_name}"})
        elif f"ELPD_{combo_name}" not in elpd.columns:
            elpd[f"ELPD_{combo_name}"] = np.nan
    elpd_cols = [f"ELPD_{combo_name}" for combo_name in COMBO_NAMES_ORDERED]
    elpd = elpd[["subtype", "hemisphere", "year"] + elpd_cols]
    summary_rows = []
    for subtype in sorted(elpd["subtype"].unique()):
        sub = elpd[elpd["subtype"] == subtype]
        row = {"subtype": subtype, "hemisphere": "Summary", "year": "All"}
        for col in elpd_cols:
            row[col] = round(float(np.nansum(sub[col])), 4)
        summary_rows.append(row)
    elpd_final = pd.concat([elpd, pd.DataFrame(summary_rows)], ignore_index=True)

    aic_rows = []
    for subtype in sorted(elpd["subtype"].unique()):
        sub = elpd[elpd["subtype"] == subtype]
        row = {"subtype": subtype}
        for combo_name in COMBO_NAMES_ORDERED:
            elpd_sum = float(np.nansum(sub[f"ELPD_{combo_name}"]))
            row[f"AIC_{combo_name}"] = round(2 * N_PARAMS[combo_name] - 2 * elpd_sum, 4)
        aic_rows.append(row)
    return elpd_final, pd.DataFrame(aic_rows)


def find_best_temperatures(past_list: list[dict], combo_metrics: list[str]) -> tuple[dict, float]:
    if not past_list:
        return {metric: 1.0 for metric in combo_metrics}, np.nan
    grid = np.array(list(product(T_VALUES, repeat=len(combo_metrics))), dtype=float)
    total_loss = np.zeros(len(grid), dtype=float)
    valid_cnt = 0
    for data in past_list:
        z_mat = data["z_mat"]
        xi_prev = data["xi_prev"]
        actual_freq = data["actual_freq"]
        if actual_freq is None:
            continue
        valid_cnt += 1
        log_sig = np.log(safe_sigmoid(z_mat[np.newaxis, :, :] / grid[:, :, np.newaxis]))
        fitness = log_sig.sum(axis=1)
        numerator = xi_prev[np.newaxis, :] * np.exp(fitness)
        denom = numerator.sum(axis=1, keepdims=True)
        valid = denom.flatten() > 0
        pred_freq = np.where(denom > 0, numerator / denom, 0.0)
        loss = np.abs(pred_freq - actual_freq[np.newaxis, :]).sum(axis=1)
        loss[~valid] = np.inf
        total_loss += loss
    if valid_cnt == 0:
        return {metric: 1.0 for metric in combo_metrics}, np.nan
    mean_loss = total_loss / valid_cnt
    best_idx = int(np.argmin(mean_loss))
    best_loss = float(mean_loss[best_idx])
    return (
        {combo_metrics[i]: float(grid[best_idx, i]) for i in range(len(combo_metrics))},
        np.nan if np.isinf(best_loss) else best_loss,
    )


def season_sort_key(item: tuple[int, str]) -> tuple[int, int]:
    year, hemi = item
    return int(year), 0 if str(hemi).lower() == "south" else 1


def compute_combination_outputs(
    max_df: pd.DataFrame, truth: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc_rows = []
    egdfit_rows = []
    egdtemp_rows = []
    lpd_rows = []
    pre_act = {}

    for combo_name, combo_metrics in COMBINATIONS:
        for subtype in sorted(max_df["subtype"].unique()):
            sub_max = max_df[max_df["subtype"] == subtype]
            sub_truth = truth[truth["subtype"] == subtype]
            label_hy = sub_truth[["hemisphere", "year"]].drop_duplicates()
            total_count = len(label_hy)
            ha1_past = ha1_clade_temperature_training_data(subtype, combo_metrics)
            seasons = sorted(
                {
                    (int(row["year"]), str(row["hemisphere"]))
                    for _, row in sub_max[["year", "hemisphere"]].drop_duplicates().iterrows()
                },
                key=season_sort_key,
            )
            per_season = {}
            for year, hemi in seasons:
                grp = sub_max[(sub_max["year"] == year) & (sub_max["hemisphere"] == hemi)].reset_index(
                    drop=True
                )
                subclades = grp["subclade_single"].to_numpy(copy=True)
                z_mat = np.vstack(
                    [
                        zscore(grp[f"fit_{metric}"].fillna(0.0).to_numpy(dtype=float))
                        for metric in combo_metrics
                    ]
                )
                prev_freq = subclade_frequency_dict(
                    subtype, year, hemi, year - 1, "submission_count"
                )
                cur_freq = subclade_frequency_dict(subtype, year, hemi, year, "submission_count")
                xi_prev = np.array([np.nan_to_num(prev_freq.get(c, 0.0), nan=0.0) for c in subclades])
                cur_arr = np.array([np.nan_to_num(cur_freq.get(c, 0.0), nan=0.0) for c in subclades])
                actual_freq = cur_arr / cur_arr.sum() if cur_arr.sum() > 0 else None
                per_season[(year, hemi)] = {
                    "subclades": subclades,
                    "z_mat": z_mat,
                    "xi_prev": xi_prev,
                    "actual_freq": actual_freq,
                }

            stored = {}
            for idx, season in enumerate(seasons):
                past = ha1_past + [
                    per_season[s] for s in seasons[:idx] if per_season[s]["actual_freq"] is not None
                ]
                temps, best_loss = find_best_temperatures(past, combo_metrics)
                stored[season] = {"temps": temps, "best_loss": best_loss, **per_season[season]}

            hit_fit = 0
            hit_freq = 0
            for year, hemi in seasons:
                season = (year, hemi)
                truth_rows = sub_truth[
                    (sub_truth["hemisphere"] == hemi) & (sub_truth["year"] == year)
                ]
                if season not in stored:
                    continue
                true_subclade = truth_rows["subclade"].iloc[0] if not truth_rows.empty else None
                data = stored[season]
                subclades = data["subclades"]
                if len(subclades) == 0:
                    continue
                t_col = np.array([data["temps"][metric] for metric in combo_metrics])[:, np.newaxis]
                fitness = np.log(safe_sigmoid(data["z_mat"] / t_col)).sum(axis=0)
                pred_fit = subclades[int(np.argmax(fitness))]
                if true_subclade is not None and pred_fit == true_subclade:
                    hit_fit += 1
                numerator = data["xi_prev"] * np.exp(fitness)
                denom = numerator.sum()
                if denom > 0:
                    freq_arr = numerator / denom
                    pred_freq = subclades[int(np.argmax(numerator))]
                    if true_subclade is not None and pred_freq == true_subclade:
                        hit_freq += 1
                else:
                    freq_arr = np.full(len(subclades), np.nan)
                if true_subclade is not None:
                    if denom > 0:
                        true_idx = np.where(subclades == true_subclade)[0]
                        prob = max(float(freq_arr[true_idx[0]]), 1e-10) if len(true_idx) else 1e-10
                    else:
                        prob = 1e-10
                    lpd_rows.append(
                        {
                            "subtype": subtype,
                            "hemisphere": hemi,
                            "year": year,
                            "combo": combo_name,
                            "lpd": float(np.log(prob)),
                        }
                    )

                if combo_name == "E+G+D":
                    for i, subclade in enumerate(subclades):
                        egdfit_rows.append(
                            {
                                "subtype": subtype,
                                "hemisphere": hemi,
                                "year": year,
                                "subclade": subclade,
                                "fit_E+G+D": round(float(fitness[i]), 6),
                            }
                        )
                    egdtemp_rows.append(
                        {
                            "subtype": subtype,
                            "hemisphere": hemi,
                            "year": year,
                            "tem_E": data["temps"]["total_escape"],
                            "tem_G": data["temps"]["predicted_prevalence"],
                            "tem_D": data["temps"]["mutual_information"],
                            "平均L1损失": data["best_loss"],
                            "TOP1_subclade": pred_fit,
                            "T/F": (
                                "T"
                                if true_subclade is not None and pred_fit == true_subclade
                                else ("F" if true_subclade is not None else np.nan)
                            ),
                        }
                    )

                coll_freq = subclade_frequency_dict(subtype, year, hemi, year, "collection_count")
                coll_arr = np.array([np.nan_to_num(coll_freq.get(c, 0.0), nan=0.0) for c in subclades])
                act_arr = coll_arr / coll_arr.sum() if coll_arr.sum() > 0 else np.full(len(subclades), np.nan)
                for i, subclade in enumerate(subclades):
                    key = (subtype, hemi, year, subclade)
                    if key not in pre_act:
                        pre_act[key] = {
                            "subtype": subtype,
                            "year": year,
                            "hemisphere": hemi,
                            "subclade": subclade,
                            "act_freq": round(float(act_arr[i]), 6)
                            if np.isfinite(act_arr[i])
                            else np.nan,
                            "freq_prev": round(float(data["xi_prev"][i]), 6)
                            if np.isfinite(data["xi_prev"][i])
                            else np.nan,
                        }
                    pre_act[key][f"{combo_name}_pre_fit"] = round(float(fitness[i]), 6)
                    pre_act[key][f"{combo_name}_pre_freq"] = (
                        round(float(freq_arr[i]), 6) if np.isfinite(freq_arr[i]) else np.nan
                    )

            for method, hit in [("fit", hit_fit), ("freq", hit_freq)]:
                acc_rows.append(
                    {
                        "subtype": subtype,
                        "metric_combine": combo_name,
                        "accuracy": round(hit / total_count, 4) if total_count else np.nan,
                        "hit_count": hit,
                        "total_count": total_count,
                        "methods": method,
                    }
                )

    pre_act_df = pd.DataFrame(list(pre_act.values()))
    if pre_act_df.empty:
        pre_act_df = pd.DataFrame(columns=PRE_ACT_COL_ORDER)
    else:
        for col in PRE_ACT_COL_ORDER:
            if col not in pre_act_df.columns:
                pre_act_df[col] = np.nan
        pre_act_df = pre_act_df[PRE_ACT_COL_ORDER].sort_values(
            ["subtype", "year", "hemisphere", "subclade"]
        )
    return (
        pd.DataFrame(acc_rows),
        pd.DataFrame(egdfit_rows),
        pd.DataFrame(egdtemp_rows),
        pd.DataFrame(lpd_rows),
        pre_act_df,
    )


def run_aux() -> None:
    ensure_dirs()
    annotation_paths = [annotation_path(target) for target in all_targets()]
    if all(path.is_file() for path in annotation_paths):
        truth = write_count_and_truth_tables()
    else:
        truth_path = DATA_OUT / "futureflu_rank" / "circulating_subclade.csv"
        if not truth_path.is_file():
            raise FileNotFoundError(
                "strain-level annotations are omitted and the packaged truth "
                f"table is missing: {truth_path}"
            )
        truth = pd.read_csv(truth_path)
        required_truth_columns = {"subtype", "hemisphere", "year", "subclade"}
        missing_columns = required_truth_columns - set(truth.columns)
        if missing_columns:
            raise KeyError(
                f"{truth_path} is missing columns: {sorted(missing_columns)}"
            )
        print(f"[aux] using packaged truth table: {truth_path.relative_to(PACKAGE_ROOT)}")
    comp_path = RISK_OUT / "risk_mutation_group_component.csv"
    comp = filtered_component_df(pd.read_csv(comp_path))
    raw_max_df, inform = compute_max_tables(comp)
    if raw_max_df.empty:
        raise RuntimeError("no subclade component rows available")
    fit_cols = [f"fit_{metric}" for metric in METRICS]
    norm_max_df = normalize_fit_columns(raw_max_df, fit_cols)
    norm_max_df.to_csv(
        ACC_OUT / "subclade_component_max.csv",
        index=False,
    )
    compute_metric_accuracy(raw_max_df, truth).to_csv(
        ACC_OUT / "subclade_component_acc.csv",
        index=False,
    )

    acc_df, egdfit_df, egdtemp_df, lpd_df, pre_act_df = compute_combination_outputs(
        raw_max_df, truth
    )
    combine_dir = COMBINE_OUT
    acc_df.to_csv(combine_dir / "subclade_component_combine_acc_Twindow.csv", index=False)
    egdfit_df.to_csv(combine_dir / "EGD_combine_Twindow.csv", index=False)
    egdtemp_df.to_csv(combine_dir / "EGD_temperatures_Twindow.csv", index=False)
    pre_act_df.to_csv(combine_dir / "subclade_pre_act_Twindow.csv", index=False)
    elpd_df, aic_df = build_elpd_aic(lpd_df)
    elpd_df.to_csv(combine_dir / "elpd_Twindow.csv", index=False)
    aic_df.to_csv(combine_dir / "aic_Twindow.csv", index=False)
    inform["mutual_information"].to_csv(combine_dir / "divergence_Twindow.csv", index=False)
    inform["total_escape"].to_csv(combine_dir / "escape_Twindow.csv", index=False)
    inform["predicted_prevalence"].to_csv(combine_dir / "growth_Twindow.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2025/2026 subclade prediction pipeline.")
    parser.add_argument(
        "stage",
        choices=[
            "sequences",
            "definitions",
            "nextclade",
            "annotate",
            "linear",
            "counts",
            "component",
            "aux",
            "all",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"sequences", "all"}:
        run_sequences()
    if args.stage in {"definitions", "all"}:
        run_definitions()
    if args.stage in {"nextclade", "all"}:
        run_nextclade()
    if args.stage in {"annotate", "all"}:
        run_annotate()
    if args.stage in {"linear", "all"}:
        run_linear()
    if args.stage in {"counts", "all"}:
        run_counts()
    if args.stage in {"component", "all"}:
        run_component()
    if args.stage in {"aux", "all"}:
        run_aux()


if __name__ == "__main__":
    main()
