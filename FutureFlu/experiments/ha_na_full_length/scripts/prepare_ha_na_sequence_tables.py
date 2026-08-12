#!/usr/bin/env python3
"""Build paired HA/NA metadata and sequence tables for three experiment modes.

English: Match HA and NA records and generate the packaged control inputs.
中文：匹配 HA 与 NA 记录并生成打包后的对照实验输入。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from Bio import SeqIO


EXP_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = EXP_ROOT.parents[1]


def configured_workspace() -> Path:
    """Return only an explicitly configured source workspace.

    English: Do not discover a sibling workspace implicitly.
    中文：不自动发现同级工作区。
    """
    configured = os.environ.get("FUTUREFLU_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return PACKAGE_ROOT / "external" / "workspace"


WORKSPACE = configured_workspace()
PRE_ROOT = Path(
    os.environ.get(
        "FUTUREFLU_SOURCE_ROOT",
        PACKAGE_ROOT / "external" / "inputs",
    )
).expanduser().resolve()
NA_ALIGNED_DIR = EXP_ROOT / "data" / "na_aligned"
DATA_DIR = EXP_ROOT / "data"

BASE_COLUMNS = [
    "accession number",
    "name",
    "clade",
    "collection_date",
    "submission_date",
    "season",
]


@dataclass(frozen=True)
class SubtypeSpec:
    subtype: str
    stem: str
    metadata: Path
    ha_fasta: Path
    na_aligned_fasta: Path
    ha_length: int
    na_length: int
    ha1_start: int
    ha1_end: int


SUBTYPES = {
    "H1N1": SubtypeSpec(
        subtype="H1N1",
        stem="h1n1",
        metadata=PRE_ROOT / "data/dataset/metadata/pdm09-all_20250131_submission.csv",
        ha_fasta=PRE_ROOT / "data/dataset/fasta/msa-pdm09-all-20250131-submission.fasta",
        na_aligned_fasta=NA_ALIGNED_DIR / "h1n1_na_aligned_to_NC_026434.1.fasta",
        ha_length=566,
        na_length=469,
        ha1_start=18,
        ha1_end=344,
    ),
    "H3N2": SubtypeSpec(
        subtype="H3N2",
        stem="h3n2",
        metadata=PRE_ROOT / "data/dataset/metadata/H3N2-all-20250131-submission.csv",
        ha_fasta=PRE_ROOT / "data/dataset/fasta/msa-H3N2-all-20250131-submission.fasta",
        na_aligned_fasta=NA_ALIGNED_DIR / "h3n2_na_aligned_to_NC_007368.1.fasta",
        ha_length=566,
        na_length=469,
        ha1_start=17,
        ha1_end=345,
    ),
    "Victoria": SubtypeSpec(
        subtype="Victoria",
        stem="victoria",
        metadata=PRE_ROOT / "data/dataset/metadata/BV-all-20250131-submission.csv",
        ha_fasta=PRE_ROOT / "data/dataset/fasta/msa-FLUBV-all-20250131-submission.fasta",
        na_aligned_fasta=NA_ALIGNED_DIR / "victoria_na_aligned_to_FJ766839.1.fasta",
        ha_length=585,
        na_length=466,
        ha1_start=16,
        ha1_end=362,
    ),
}

MODES = ("ha_full", "ha_full_na_full", "ha1_na_full")


def first_segment_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    first_entry = re.split(r"\s*,\s*", text, maxsplit=1)[0]
    return first_entry.split("|", 1)[0].strip()


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


def ha_lookup_key(record_id: str, subtype: str) -> str:
    parts = record_id.split("|")
    if subtype == "Victoria" and len(parts) > 1:
        return parts[1].strip()
    return parts[0].strip()


def load_sequence_dict(path: Path, key_fn, expected_length: int | None = None) -> tuple[dict[str, str], dict[str, int]]:
    seqs: dict[str, str] = {}
    stats = Counter()
    for record in SeqIO.parse(str(path), "fasta"):
        stats["records"] += 1
        key = key_fn(str(record.id))
        seq = str(record.seq).upper()
        stats[f"len_{len(seq)}"] += 1
        if expected_length is not None and len(seq) != expected_length:
            stats["unexpected_length"] += 1
            continue
        if key in seqs:
            stats["duplicate_keys"] += 1
            continue
        seqs[key] = seq
    stats["unique_keys"] = len(seqs)
    return seqs, dict(stats)


def load_ha_sequences(spec: SubtypeSpec) -> tuple[dict[str, str], dict[str, int]]:
    return load_sequence_dict(
        spec.ha_fasta,
        key_fn=lambda record_id: ha_lookup_key(record_id, spec.subtype),
        expected_length=spec.ha_length,
    )


def load_na_sequences(spec: SubtypeSpec) -> tuple[dict[str, str], dict[str, int]]:
    def key_fn(record_id: str) -> str:
        if "_NA_REF|" in record_id:
            return ""
        return record_id.split("|", 1)[0].strip()

    seqs: dict[str, str] = {}
    stats = Counter()
    for record in SeqIO.parse(str(spec.na_aligned_fasta), "fasta"):
        stats["records"] += 1
        record_id = str(record.id)
        if "_NA_REF|" in record_id:
            stats["reference_records"] += 1
            continue
        key = key_fn(record_id)
        seq = str(record.seq).upper()
        stats[f"len_{len(seq)}"] += 1
        if len(seq) != spec.na_length:
            stats["unexpected_length"] += 1
            continue
        if key in seqs:
            stats["duplicate_keys"] += 1
            continue
        seqs[key] = seq
    stats["unique_keys"] = len(seqs)
    return seqs, dict(stats)


def paired_metadata(spec: SubtypeSpec) -> tuple[pd.DataFrame, int]:
    metadata = pd.read_csv(spec.metadata, low_memory=False)
    for column in ("Isolate_Id", "HA Segment_Id", "NA Segment_Id"):
        if column not in metadata.columns:
            raise KeyError(f"missing {column!r} in {spec.metadata}")

    prepared = metadata.copy()
    prepared.insert(1, "Original_Isolate_Id", prepared["Isolate_Id"])
    prepared.insert(2, "HA_Matched_Segment_Id", prepared["HA Segment_Id"].map(first_segment_token))
    prepared.insert(3, "NA_Matched_Segment_Id", prepared["NA Segment_Id"].map(first_segment_token))
    has_ha = prepared["HA_Matched_Segment_Id"].astype(str).str.len() > 0
    has_na = prepared["NA_Matched_Segment_Id"].astype(str).str.len() > 0
    return prepared[has_ha & has_na].copy(), len(metadata)


def mode_site_positions(spec: SubtypeSpec, mode: str) -> list[tuple[str, int, int]]:
    positions: list[tuple[str, int, int]] = []
    if mode == "ha_full":
        positions.extend(("HA", pos, pos) for pos in range(1, spec.ha_length + 1))
        return positions

    if mode == "ha_full_na_full":
        positions.extend(("HA", pos, pos) for pos in range(1, spec.ha_length + 1))
    elif mode == "ha1_na_full":
        positions.extend(("HA", pos, pos) for pos in range(spec.ha1_start, spec.ha1_end + 1))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    na_offset = spec.ha_length
    positions.extend(("NA", pos, na_offset + pos) for pos in range(1, spec.na_length + 1))
    return positions


def write_site_map(spec: SubtypeSpec, mode: str) -> None:
    rows = []
    for segment, original_pos, feature_pos in mode_site_positions(spec, mode):
        rows.append(
            {
                "subtype": spec.subtype,
                "mode": mode,
                "feature_column": f"X{feature_pos}",
                "segment": segment,
                "segment_position": original_pos,
                "feature_position": feature_pos,
                "ha_length": spec.ha_length,
                "na_length": spec.na_length,
            }
        )
    path = DATA_DIR / "site_maps" / f"{spec.stem}_{mode}_site_map.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def row_sequence_values(ha_seq: str, na_seq: str, positions: list[tuple[str, int, int]]) -> list[str]:
    values = []
    for segment, segment_pos, _feature_pos in positions:
        if segment == "HA":
            values.append(ha_seq[segment_pos - 1])
        else:
            values.append(na_seq[segment_pos - 1])
    return values


def build_tables_for_subtype(spec: SubtypeSpec, cutoff_date: str, max_gaps: int) -> list[dict[str, object]]:
    start = time.time()
    paired, metadata_rows = paired_metadata(spec)
    paired_path = DATA_DIR / "paired_metadata" / f"{spec.stem}_paired_ha_na_metadata.csv"
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(paired_path, index=False)

    ha_seqs, ha_stats = load_ha_sequences(spec)
    na_seqs, na_stats = load_na_sequences(spec)

    mode_paths = {
        mode: DATA_DIR / "sequences" / mode / f"{spec.subtype}_sequence_20250131.csv"
        for mode in MODES
    }
    for path in mode_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        write_site_map(spec, mode)

    mode_positions = {mode: mode_site_positions(spec, mode) for mode in MODES}
    writers: dict[str, csv.writer] = {}
    handles = {}
    written = Counter()
    filter_counts = Counter()
    cutoff_ts = pd.Timestamp(cutoff_date)

    try:
        for mode, path in mode_paths.items():
            handle = path.open("w", encoding="utf-8", newline="")
            handles[mode] = handle
            writer = csv.writer(handle)
            feature_cols = [f"X{feature_pos}" for _seg, _seg_pos, feature_pos in mode_positions[mode]]
            writer.writerow(BASE_COLUMNS + feature_cols)
            writers[mode] = writer

        for row in paired.to_dict("records"):
            isolate_id = str(row.get("Original_Isolate_Id", row.get("Isolate_Id", ""))).strip()
            ha_id = str(row.get("HA_Matched_Segment_Id", "")).strip()
            na_id = str(row.get("NA_Matched_Segment_Id", "")).strip()
            ha_seq = ha_seqs.get(isolate_id)
            na_seq = na_seqs.get(na_id)
            if not ha_seq:
                filter_counts["missing_ha_fasta"] += 1
                continue
            if not na_seq:
                filter_counts["missing_na_fasta"] += 1
                continue
            if ha_seq.count("-") > max_gaps:
                filter_counts["ha_gap_filtered"] += 1
                continue
            if na_seq.count("-") > max_gaps:
                filter_counts["na_gap_filtered"] += 1
                continue

            collection_str = normalize_partial_date(row.get("Collection_Date", ""))
            if not collection_str:
                filter_counts["missing_collection_date"] += 1
                continue
            try:
                collection_date = pd.to_datetime(collection_str)
            except (ValueError, TypeError):
                filter_counts["invalid_collection_date"] += 1
                continue

            submission_str = normalize_partial_date(row.get("Submission_Date", ""))
            try:
                submission_date = pd.to_datetime(submission_str) if submission_str else None
            except (ValueError, TypeError):
                submission_date = None
            if submission_date is None:
                submission_date = collection_date

            if submission_date >= cutoff_ts:
                filter_counts["submission_cutoff_filtered"] += 1
                continue
            if collection_date.year < 2010:
                filter_counts["pre_2010_filtered"] += 1
                continue

            isolate_name = row.get("Isolate_Name", "")
            clade = row.get("Clade", "")
            base = [
                isolate_id,
                "" if pd.isna(isolate_name) else str(isolate_name),
                "" if pd.isna(clade) else str(clade),
                collection_date.strftime("%Y-%m-%d"),
                submission_date.strftime("%Y-%m-%d"),
                collection_date.year - 1 if collection_date.month < 2 else collection_date.year,
            ]
            for mode in MODES:
                writers[mode].writerow(base + row_sequence_values(ha_seq, na_seq, mode_positions[mode]))
                written[mode] += 1
    finally:
        for handle in handles.values():
            handle.close()

    rows: list[dict[str, object]] = []
    for mode in MODES:
        summary = {
            "subtype": spec.subtype,
            "mode": mode,
            "metadata_rows": metadata_rows,
            "paired_metadata_rows": len(paired),
            "sequence_rows": written[mode],
            "paired_metadata_path": str(paired_path),
            "sequence_path": str(mode_paths[mode]),
            "elapsed_seconds_for_subtype": round(time.time() - start, 3),
            "ha_fasta_unique_keys": ha_stats.get("unique_keys", 0),
            "na_aligned_unique_keys": na_stats.get("unique_keys", 0),
        }
        for key, value in sorted(filter_counts.items()):
            summary[key] = value
        rows.append(summary)
    print(
        f"[prepare] {spec.subtype} paired={len(paired)} "
        f"rows={{mode: written[mode] for mode in MODES}} "
        f"filters={dict(filter_counts)} elapsed={time.time() - start:.1f}s"
    )
    return rows


def selected_specs(names: Iterable[str]) -> list[SubtypeSpec]:
    specs = []
    for name in names:
        if name not in SUBTYPES:
            raise ValueError(f"unknown subtype {name!r}; choose from {', '.join(SUBTYPES)}")
        specs.append(SUBTYPES[name])
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtypes", nargs="+", default=list(SUBTYPES), choices=list(SUBTYPES))
    parser.add_argument("--cutoff-date", default="2025-02-01")
    parser.add_argument("--max-gaps-per-segment", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for spec in selected_specs(args.subtypes):
        rows.extend(
            build_tables_for_subtype(
                spec,
                cutoff_date=args.cutoff_date,
                max_gaps=args.max_gaps_per_segment,
            )
        )
    summary_path = DATA_DIR / "prepare_sequence_summary.csv"
    columns = sorted({key for row in rows for key in row})
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[prepare] wrote {summary_path}")


if __name__ == "__main__":
    main()
