#!/usr/bin/env python3
"""Convert external HA sequence tables into FutureFlu sequence inputs.

English: The release package keeps an empty package-local source-table directory
for path stability. Provide a directory containing the omitted CSV tables with
--sources-root when rebuilding the sequence-input layer.
中文：发布包为路径稳定性保留空的包内源表目录。若需重建序列输入层，
请通过 --sources-root 指向另行提供的 HA sequence table CSV 目录。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES_ROOT = PROJECT_ROOT / "data" / "sources"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sequences"
DEFAULT_SCAFFOLD_DIR = PROJECT_ROOT / "data" / "sequences"
ARTIFACT_DIR = PROJECT_ROOT / "results" / "futureflu" / "artifacts"

BASE_SEQUENCE_COLUMNS = [
    "accession number",
    "name",
    "clade",
    "collection_date",
    "submission_date",
    "season",
]

METADATA_COLUMNS = [
    "strain",
    "accession",
    "age",
    "collection_date",
    "date",
    "submission_date",
    "gender",
    "region",
    "country",
    "division",
    "location",
    "passage",
    "submitting_lab",
    "clade",
    "virus",
    "lineage",
    "segment",
    "source_isolate_id",
    "source_ha_segment_id",
    "source_fasta_id",
    "sequence_type",
]

LINEAGE_SPECS = {
    "h1n1pdm": {
        "label": "A/H1N1pdm09",
        "lineage": "h1n1pdm",
        "source_table": "H1N1_HA_sequence_20250131.csv",
    },
    "h3n2": {
        "label": "A/H3N2",
        "lineage": "h3n2",
        "source_table": "H3N2_HA_sequence_20250131.csv",
    },
    "victoria": {
        "label": "B/Victoria",
        "lineage": "victoria",
        "source_table": "Victoria_HA_sequence_20250131.csv",
    },
}


def public_path(path: Path | None) -> str | None:
    """Return a repository-relative path when possible.

    English: Conversion summaries should not expose machine-specific roots.
    中文：转换摘要不暴露本机绝对路径。
    """
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(resolved.relative_to(PROJECT_ROOT.parent))
        except ValueError:
            return str(path)


def sanitize_token(value: object) -> str:
    """Return a stable strain-safe token.

    English: Keep slash-separated influenza strain names readable while removing
    characters that can break downstream table joins.
    中文：保留流感 strain 名称中的斜杠结构，同时去除会影响下游表连接的字符。
    """
    text = re.sub(r"[^A-Za-z0-9._/-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_date(value: object) -> str:
    """Normalize partial or full dates to YYYY-MM-DD.

    English: Empty or invalid values are returned as empty strings so they can
    be counted and reported instead of silently passing into the workflow.
    中文：空值或无效日期返回空字符串，便于统计报告，而不是静默进入 workflow。
    """
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    parts = text.split("-")
    if len(parts) == 1 and parts[0].isdigit():
        text = f"{parts[0]}-01-01"
    elif len(parts) == 2 and all(part.isdigit() for part in parts):
        text = f"{parts[0]}-{parts[1]}-01"

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed) or parsed.year < 1990 or parsed.year > 2100:
        return ""
    return parsed.strftime("%Y-%m-%d")


def sequence_column_sort_key(column: str) -> int:
    match = re.fullmatch(r"X(\d+)", column)
    if match is None:
        raise ValueError(f"Unexpected sequence column name: {column}")
    return int(match.group(1))


def find_sequence_columns(columns: Iterable[str]) -> list[str]:
    """Find and sort X-position amino-acid columns.

    English: The source sequence tables encode aligned HA amino-acid positions as
    X1, X2, ... columns; preserving numeric order is required for FASTA output.
    中文：源序列表用 X1、X2 等列表示对齐后的 HA 氨基酸位点；
    生成 FASTA 时必须按数字顺序拼接。
    """
    sequence_columns = [column for column in columns if re.fullmatch(r"X\d+", column)]
    if not sequence_columns:
        raise ValueError("No X-position amino-acid columns found")
    return sorted(sequence_columns, key=sequence_column_sort_key)


def check_required_columns(table: pd.DataFrame, path: Path) -> None:
    missing = [column for column in BASE_SEQUENCE_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def clean_residue(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "-"
    return text[0].upper()


def normalize_sequence_table(table: pd.DataFrame, sequence_columns: list[str]) -> pd.DataFrame:
    """Build the workflow sequence table.

    English: This keeps the source-table row order and only normalizes the
    accession column name expected by the issue-date workflow.
    中文：该步骤保留源序列表的行顺序，只把 accession 列名规范为
    issue-date workflow 所需格式。
    """
    output = table.loc[:, [*BASE_SEQUENCE_COLUMNS, *sequence_columns]].copy()
    output = output.rename(columns={"accession number": "accession_number"})
    for column in ["collection_date", "submission_date"]:
        output[column] = output[column].map(normalize_date)
    output["season"] = output["season"].fillna("").astype(str)
    for column in sequence_columns:
        output[column] = output[column].map(clean_residue)
    return output


def build_sequence_lookup(sequence_table: pd.DataFrame, sequence_columns: list[str]) -> tuple[dict[str, str], int]:
    """Build one accession-to-sequence map.

    English: Duplicate accessions are allowed in the source table. The first
    observed sequence is used; conflicting duplicates are reported in the summary.
    中文：源表允许重复 accession。脚本使用第一次出现的序列，并在 summary 中记录
    不一致重复项数量。
    """
    sequences: dict[str, str] = {}
    conflicting_duplicates = 0
    values = sequence_table.loc[:, sequence_columns].to_numpy(dtype=str)
    for accession, residues in zip(sequence_table["accession_number"].astype(str), values):
        sequence = "".join(residues)
        if accession in sequences and sequences[accession] != sequence:
            conflicting_duplicates += 1
            continue
        sequences.setdefault(accession, sequence)
    return sequences, conflicting_duplicates


def make_minimal_metadata(sequence_table: pd.DataFrame, lineage: str) -> pd.DataFrame:
    """Create fallback metadata when no metadata source is available.

    English: The fallback is structurally valid but lacks geographic metadata
    that is not present in the source sequence table.
    中文：fallback metadata 在结构上有效，但缺少源 sequence table 中没有的
    地理信息。
    """
    rows = []
    seen: dict[str, int] = {}
    for record in sequence_table.itertuples(index=False):
        accession = str(getattr(record, "accession_number"))
        strain_name = sanitize_token(f"{getattr(record, 'name')}_{accession}")
        seen[strain_name] = seen.get(strain_name, 0) + 1
        strain = strain_name if seen[strain_name] == 1 else f"{strain_name}__dup{seen[strain_name]}"
        collection_date = normalize_date(getattr(record, "collection_date"))
        submission_date = normalize_date(getattr(record, "submission_date")) or collection_date
        rows.append(
            {
                "strain": strain,
                "accession": accession,
                "age": "?",
                "collection_date": collection_date,
                "date": collection_date,
                "submission_date": submission_date,
                "gender": "?",
                "region": "unknown",
                "country": "unknown",
                "division": "unknown",
                "location": "unknown",
                "passage": "undetermined",
                "submitting_lab": "",
                "clade": str(getattr(record, "clade")),
                "virus": "flu",
                "lineage": lineage,
                "segment": "ha",
                "source_isolate_id": accession,
                "source_ha_segment_id": "",
                "source_fasta_id": f"{accession}|{getattr(record, 'name')}",
                "sequence_type": "amino_acid",
            }
        )
    return pd.DataFrame(rows, columns=METADATA_COLUMNS)


def load_scaffold_metadata(prefix: str, scaffold_dir: Path) -> pd.DataFrame | None:
    path = scaffold_dir / f"{prefix}_futureflu_metadata.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def prepare_metadata(
    prefix: str,
    lineage: str,
    sequence_table: pd.DataFrame,
    scaffold_dir: Path | None,
) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    """Prepare derived metadata.

    English: Optional external metadata supplies region, country, passage, and
    source identifiers that are not present in the source CSVs.
    中文：可选的外部 metadata 补充源 CSV 中没有的 region、country、
    passage 和 source identifier 等字段。
    """
    sequence_accessions = set(sequence_table["accession_number"].astype(str))
    scaffold = load_scaffold_metadata(prefix, scaffold_dir) if scaffold_dir else None
    used_scaffold = scaffold is not None and not scaffold.empty

    if used_scaffold:
        accession_column = "accession" if "accession" in scaffold.columns else "source_isolate_id"
        metadata = scaffold[scaffold[accession_column].astype(str).isin(sequence_accessions)].copy()
        for column in METADATA_COLUMNS:
            if column not in metadata.columns:
                metadata[column] = ""
        metadata = metadata.loc[:, METADATA_COLUMNS]
    else:
        metadata = make_minimal_metadata(sequence_table, lineage)

    represented_accessions = set(metadata["accession"].astype(str))
    missing_accessions = sequence_accessions - represented_accessions
    if missing_accessions:
        fallback_rows = make_minimal_metadata(
            sequence_table[sequence_table["accession_number"].astype(str).isin(missing_accessions)].copy(),
            lineage,
        )
        metadata = pd.concat([metadata, fallback_rows], ignore_index=True)

    for column in ["collection_date", "date", "submission_date"]:
        metadata[column] = metadata[column].map(normalize_date)
    metadata["lineage"] = lineage
    metadata["virus"] = "flu"
    metadata["segment"] = "ha"
    metadata["sequence_type"] = "amino_acid"
    metadata = metadata[metadata["collection_date"] != ""].copy()
    metadata = metadata.sort_values(["collection_date", "strain"]).reset_index(drop=True)

    stats = {
        "used_metadata_scaffold": used_scaffold,
        "metadata_rows": int(len(metadata)),
        "sequence_accessions": int(len(sequence_accessions)),
        "missing_scaffold_accessions": int(len(missing_accessions)),
    }
    return metadata, stats


def fasta_records_from_metadata(metadata: pd.DataFrame, sequence_lookup: dict[str, str]) -> tuple[list[tuple[str, str]], int]:
    """Create FASTA records keyed by metadata strain.

    English: `prepare` later joins metadata to FASTA by the `strain` field, so
    FASTA IDs must use metadata strain names instead of accession numbers.
    中文：后续 `prepare` 按 `strain` 字段连接 metadata 和 FASTA，因此 FASTA ID
    必须使用 metadata 中的 strain 名称，而不是 accession。
    """
    records: list[tuple[str, str]] = []
    missing_sequences = 0
    for row in metadata.itertuples(index=False):
        accession = str(getattr(row, "accession"))
        sequence = sequence_lookup.get(accession)
        if sequence is None:
            missing_sequences += 1
            continue
        records.append((str(getattr(row, "strain")), sequence))
    return records, missing_sequences


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for strain, sequence in records:
            handle.write(f">{strain}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def planned_outputs(prefix: str, output_dir: Path) -> list[Path]:
    return [
        output_dir / f"{prefix}_futureflu_aa_sequences.fasta",
        output_dir / f"{prefix}_futureflu_metadata.tsv",
        output_dir / f"{prefix}_futureflu_summary.json",
        output_dir / f"{prefix}_sequence_table.tsv",
    ]


def ensure_outputs_are_writable(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing derived outputs without --allow-overwrite:\n"
            f"{formatted}"
        )


def source_table_path(sources_root: Path, source_name: str) -> Path:
    """Resolve an external HA sequence table.

    English: The final package keeps the source-table directory but omits the
    source tables themselves.
    中文：final 包保留源表目录，但省略实际源 sequence table。
    """
    path = sources_root / source_name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing HA sequence table: {path}"
        )
    return path


def convert_lineage(
    prefix: str,
    sources_root: Path,
    output_dir: Path,
    scaffold_dir: Path | None,
    allow_overwrite: bool,
    dry_run: bool,
) -> dict:
    spec = LINEAGE_SPECS[prefix]
    source_table = source_table_path(sources_root, spec["source_table"])
    outputs = planned_outputs(prefix, output_dir)

    if dry_run:
        return {
            "lineage": spec["label"],
            "prefix": prefix,
            "source_table": public_path(source_table),
            "metadata_scaffold_dir": public_path(scaffold_dir),
            "output_files": [public_path(path) for path in outputs],
            "dry_run": True,
        }

    if not source_table.exists():
        raise FileNotFoundError(source_table)
    ensure_outputs_are_writable(outputs, allow_overwrite)

    table = pd.read_csv(source_table, dtype=str, low_memory=False)
    check_required_columns(table, source_table)
    sequence_columns = find_sequence_columns(table.columns)
    sequence_table = normalize_sequence_table(table, sequence_columns)
    sequence_lookup, conflicting_duplicates = build_sequence_lookup(sequence_table, sequence_columns)
    metadata, metadata_stats = prepare_metadata(prefix, spec["lineage"], sequence_table, scaffold_dir)
    fasta_records, missing_fasta_sequences = fasta_records_from_metadata(metadata, sequence_lookup)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_sequence_table = output_dir / f"{prefix}_sequence_table.tsv"
    output_metadata = output_dir / f"{prefix}_futureflu_metadata.tsv"
    output_fasta = output_dir / f"{prefix}_futureflu_aa_sequences.fasta"
    output_summary = output_dir / f"{prefix}_futureflu_summary.json"

    sequence_table.to_csv(output_sequence_table, sep="\t", index=False)
    metadata.to_csv(output_metadata, sep="\t", index=False)
    write_fasta(fasta_records, output_fasta)

    summary = {
        "lineage": spec["label"],
        "prefix": prefix,
        "source_table": public_path(source_table),
        "metadata_scaffold_dir": public_path(scaffold_dir),
        "output_dir": public_path(output_dir),
        "sequence_table_rows": int(len(sequence_table)),
        "sequence_columns": int(len(sequence_columns)),
        "unique_accessions": int(sequence_table["accession_number"].nunique()),
        "metadata_rows": metadata_stats["metadata_rows"],
        "fasta_records": int(len(fasta_records)),
        "used_metadata_scaffold": metadata_stats["used_metadata_scaffold"],
        "missing_scaffold_accessions": metadata_stats["missing_scaffold_accessions"],
        "missing_fasta_sequences": int(missing_fasta_sequences),
        "conflicting_duplicate_accessions": int(conflicting_duplicates),
        "collection_date_min": metadata["collection_date"].min() if len(metadata) else None,
        "collection_date_max": metadata["collection_date"].max() if len(metadata) else None,
        "outputs": {
            "metadata": public_path(output_metadata),
            "fasta": public_path(output_fasta),
            "sequence_table_tsv": public_path(output_sequence_table),
            "summary": public_path(output_summary),
        },
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_reports(summaries: list[dict], output_dir: Path) -> None:
    """Write bilingual conversion reports.

    English: Reports document the raw-to-derived bridge without changing the
    primary workflow outputs.
    中文：报告记录 raw-to-derived 的衔接方式，不改变主 workflow 输出。
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        "| Lineage | Source rows | Metadata rows | FASTA records | Scaffold | Output directory |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    zh_rows = [
        "| Lineage | 源表行数 | Metadata 行数 | FASTA 记录数 | Scaffold | 输出目录 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for summary in summaries:
        rows.append(
            "| {lineage} | {sequence_table_rows} | {metadata_rows} | {fasta_records} | {used_metadata_scaffold} | `{output_dir}` |".format(
                **summary
            )
        )
        zh_rows.append(
            "| {lineage} | {sequence_table_rows} | {metadata_rows} | {fasta_records} | {used_metadata_scaffold} | `{output_dir}` |".format(
                **summary
            )
        )

    report = "\n".join(
        [
            "# Source-Table Sequence Inputs",
            "",
            "[简体中文](source_table_derived_inputs.zh.md)",
            "",
            "This report documents conversion from externally supplied HA sequence tables to the FutureFlu sequence-input layer.",
            "",
            "Place primary HA sequence tables under `data/sources/`, or pass "
            "`--sources-root` to another directory.",
            "",
            "External metadata can be used as a scaffold when available because the source CSVs do not contain region, country, passage, host, or source HA segment fields required by the issue-date workflow.",
            "",
            *rows,
            "",
            "To replace canonical sequence inputs in a clean copy, pass `--output-dir data/sequences`. Existing files are never overwritten unless `--allow-overwrite` is supplied.",
            "",
        ]
    )
    zh_report = "\n".join(
        [
            "# 源表序列输入",
            "",
            "[English](source_table_derived_inputs.md)",
            "",
            "本报告说明如何从另行提供的 HA sequence table 转换到 FutureFlu 序列输入层。",
            "",
            "请把 HA sequence table 放入 `data/sources/`，或用 `--sources-root` 指向其他目录。",
            "",
            "如有外部 metadata，可作为 scaffold 使用，因为源 CSV 不包含 issue-date workflow 需要的 region、country、passage、host 和 source HA segment 等字段。",
            "",
            *zh_rows,
            "",
            "如需在干净副本中替换正式序列输入，可传入 `--output-dir data/sequences`。除非显式使用 `--allow-overwrite`，脚本不会覆盖已有文件。",
            "",
        ]
    )
    (ARTIFACT_DIR / "source_table_derived_inputs.md").write_text(report, encoding="utf-8")
    (ARTIFACT_DIR / "source_table_derived_inputs.zh.md").write_text(zh_report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert external HA sequence tables into FutureFlu sequence inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lineage", choices=["all", *LINEAGE_SPECS.keys()], default="all")
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=DEFAULT_SOURCES_ROOT,
        help="directory containing primary HA sequence-table CSVs (default: data/sources)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-scaffold-dir", type=Path, default=DEFAULT_SCAFFOLD_DIR)
    parser.add_argument(
        "--without-metadata-scaffold",
        action="store_true",
        help="write structurally valid minimal metadata instead of using bundled metadata as scaffold",
    )
    parser.add_argument("--allow-overwrite", action="store_true", help="allow existing output files to be overwritten")
    parser.add_argument("--dry-run", action="store_true", help="print planned inputs and outputs without writing files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected = list(LINEAGE_SPECS) if args.lineage == "all" else [args.lineage]
    scaffold_dir = None if args.without_metadata_scaffold else args.metadata_scaffold_dir
    try:
        summaries = [
            convert_lineage(
                prefix=prefix,
                sources_root=args.sources_root.resolve(),
                output_dir=args.output_dir.resolve(),
                scaffold_dir=scaffold_dir.resolve() if scaffold_dir else None,
                allow_overwrite=args.allow_overwrite,
                dry_run=args.dry_run,
            )
            for prefix in selected
        ]
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summaries, indent=2))
    if not args.dry_run:
        write_reports(summaries, args.output_dir.resolve())


if __name__ == "__main__":
    main()
