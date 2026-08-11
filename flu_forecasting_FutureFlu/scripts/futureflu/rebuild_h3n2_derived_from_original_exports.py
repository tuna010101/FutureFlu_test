#!/usr/bin/env python3
"""Build H3N2 derived inputs from FutureFlu FASTA and metadata exports.

English: This script runs the H3N2 FASTA, metadata, and sequence-table
conversion steps and writes the derived input set used by the workflow.
中文：该脚本运行 H3N2 FASTA、metadata 和 sequence-table 转换步骤，
并写出 workflow 使用的 derived 输入集。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from Bio import SeqIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "sources"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sequences"
STAGING_OUTPUT_DIR = PROJECT_ROOT / "data" / "sequences" / ".staging"
ARTIFACT_ROOT = PROJECT_ROOT / "results" / "futureflu" / "artifacts"

H3N2_CONVERTER = PROJECT_ROOT / "scripts" / "futureflu" / "convert_futureflu_h3n2_to_flu_forecasting.py"
H3N2_SEQUENCE_TABLE = PROJECT_ROOT / "scripts" / "futureflu" / "prepare_h3n2_sequence_table.py"
H3N2_INPUT_FASTA = DATASET_ROOT / "msa-H3N2-all-20250131-submission.fasta"
H3N2_INPUT_METADATA = DATASET_ROOT / "H3N2-all-20250131-submission.csv"
H3N2_SEQUENCE_TABLE_TSV = STAGING_OUTPUT_DIR / "h3n2_sequence_table.tsv"
H3N2_SEQUENCE_TABLE_PKL = STAGING_OUTPUT_DIR / "h3n2_sequence_table.pkl"
H3N2_SEQUENCE_TABLE_REPORT = ARTIFACT_ROOT / "h3n2_sequence_table_artifacts.md"

REQUIRED_DATASET_FILES = [
    H3N2_INPUT_FASTA,
    H3N2_INPUT_METADATA,
]

DERIVED_PRODUCTS = {
    "h3n2_futureflu_aa_sequences.fasta": STAGING_OUTPUT_DIR / "h3n2_futureflu_aa_sequences.fasta",
    "h3n2_futureflu_metadata.tsv": STAGING_OUTPUT_DIR / "h3n2_futureflu_metadata.tsv",
    "h3n2_futureflu_summary.json": STAGING_OUTPUT_DIR / "h3n2_futureflu_summary.json",
    "h3n2_sequence_table.tsv": STAGING_OUTPUT_DIR / "h3n2_sequence_table.tsv",
}


@contextmanager
def fasta_input(path: Path):
    """Open FASTA input. / 打开 FASTA 输入。"""
    with path.open("r", encoding="utf-8") as handle:
        yield handle


def public_path(path: Path) -> str:
    """Return a repository-relative path when possible.

    English: Derived-input reports should be portable across machines.
    中文：derived 输入报告使用可迁移的相对路径。
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(resolved.relative_to(PROJECT_ROOT.parent))
        except ValueError:
            return str(path)


def quote_command(command: list[str]) -> str:
    """Render a copyable command line.

    English: Reports show the exact converter commands.
    中文：报告展示实际转换命令。
    """
    return " ".join(str(part) for part in command)


def require_inputs() -> None:
    missing = [
        path
        for path in [*REQUIRED_DATASET_FILES, H3N2_CONVERTER, H3N2_SEQUENCE_TABLE]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(str(path) for path in missing))


def ensure_outputs_are_writable(output_dir: Path, allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [output_dir / name for name in DERIVED_PRODUCTS if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing H3N2 derived outputs without --allow-overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )


def run_converter_script(command: list[str], dry_run: bool) -> None:
    print(f"[cwd] {PROJECT_ROOT}")
    print(f"[cmd] {quote_command(command)}")
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def normalize_partial_date(value: object) -> pd.Timestamp | None:
    """Parse partial collection dates.

    English: Year and year-month values are expanded for date filtering.
    中文：对 year 和 year-month 形式进行补全以支持日期过滤。
    """
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    parts = text.split("-")
    if len(parts) == 1:
        text = f"{text}-01-01"
    elif len(parts) == 2:
        text = f"{text}-01"
    try:
        return pd.to_datetime(text)
    except (ValueError, TypeError):
        return None


def build_sequence_table_fast(cutoff_date: str = "2025-02-01") -> int:
    """Build the H3N2 sequence table with indexed metadata lookup.

    English: Indexed metadata lookup keeps sequence-table construction practical.
    中文：使用 metadata 索引以提升 sequence table 构建效率。
    """
    print("[fast-sequence-table] loading metadata", flush=True)
    metadata = pd.read_csv(H3N2_INPUT_METADATA, dtype=str, low_memory=False)
    metadata = metadata.drop_duplicates("Isolate_Id", keep="first").set_index("Isolate_Id", drop=False)
    cutoff = pd.to_datetime(cutoff_date)

    rows = []
    with fasta_input(H3N2_INPUT_FASTA) as fasta_handle:
        records = SeqIO.parse(fasta_handle, "fasta")
        for index, record in enumerate(records, start=1):
            if record.id == "H3N2_reference":
                continue
            sequence = str(record.seq)
            if sequence.count("-") > 3:
                continue

            isolate_id = record.id.split("|")[0]
            if isolate_id not in metadata.index:
                continue
            row = metadata.loc[isolate_id]

            collection_date = normalize_partial_date(row.get("Collection_Date", ""))
            if collection_date is None:
                continue

            submission_date = normalize_partial_date(row.get("Submission_Date", ""))
            if submission_date is None:
                submission_date = collection_date
            if submission_date >= cutoff:
                continue
            if collection_date.year < 2010:
                continue

            row_data = {
                "accession_number": isolate_id,
                "name": str(row.get("Isolate_Name", "")) if pd.notna(row.get("Isolate_Name", "")) else "",
                "clade": row.get("Clade", "") if pd.notna(row.get("Clade", "")) else "",
                "collection_date": collection_date.strftime("%Y-%m-%d"),
                "submission_date": submission_date.strftime("%Y-%m-%d"),
                "season": collection_date.year - 1 if collection_date.month < 2 else collection_date.year,
            }
            for position, aa in enumerate(sequence, start=1):
                row_data[f"X{position}"] = aa
            rows.append(row_data)

            if index % 25000 == 0:
                print(f"[fast-sequence-table] scanned {index} FASTA records; retained {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    H3N2_SEQUENCE_TABLE_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(H3N2_SEQUENCE_TABLE_TSV, sep="\t", index=False)
    df.to_pickle(H3N2_SEQUENCE_TABLE_PKL)
    H3N2_SEQUENCE_TABLE_REPORT.write_text(
        "\n".join(
            [
                "# H3N2 Sequence Table Artifacts",
                "",
                "## Command",
                "",
                "```bash",
                "python scripts/futureflu/rebuild_h3n2_derived_from_original_exports.py",
                "```",
                "",
                "## Logic source",
                "",
                "This step mirrors the filtering logic in `scripts/futureflu/prepare_h3n2_sequence_table.py`, but uses indexed metadata lookup for practical runtime.",
                "",
                "## Outputs",
                "",
                "- `data/sequences/.staging/h3n2_sequence_table.tsv`",
                "- `data/sequences/.staging/h3n2_sequence_table.pkl`",
                "",
                "## Summary",
                "",
                f"- rows written: {len(df)}",
                f"- seasons covered: {df['season'].min()} to {df['season'].max()}",
                f"- collection date min: {df['collection_date'].min()}",
                f"- collection date max: {df['collection_date'].max()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {H3N2_SEQUENCE_TABLE_TSV}", flush=True)
    print(f"Wrote {H3N2_SEQUENCE_TABLE_PKL}", flush=True)
    print(f"Wrote {H3N2_SEQUENCE_TABLE_REPORT}", flush=True)
    return len(df)


def copy_outputs(output_dir: Path, dry_run: bool) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name, source in DERIVED_PRODUCTS.items():
        destination = output_dir / name
        print(f"[copy] {source} -> {destination}")
        if not dry_run:
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination)
        output_paths[name] = public_path(destination)
    return output_paths


def write_report(output_dir: Path, output_paths: dict[str, str], dry_run: bool) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_root": public_path(DATASET_ROOT),
        "staging_output_dir": public_path(STAGING_OUTPUT_DIR),
        "output_dir": public_path(output_dir),
        "converter_script": public_path(H3N2_CONVERTER),
        "sequence_table_script": public_path(H3N2_SEQUENCE_TABLE),
        "output_paths": output_paths,
        "dry_run": dry_run,
    }
    report_path = ARTIFACT_ROOT / "h3n2_original_export_derived_inputs.json"
    if not dry_run:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = ARTIFACT_ROOT / "h3n2_original_export_derived_inputs.md"
    zh_path = ARTIFACT_ROOT / "h3n2_original_export_derived_inputs.zh.md"
    if not dry_run:
        md_path.write_text(
            "\n".join(
                [
                    "# H3N2 Export Derived Inputs",
                    "",
                    "[简体中文](h3n2_original_export_derived_inputs.zh.md)",
                    "",
                    "This report summarizes the H3N2 FASTA and metadata export conversion.",
                    "",
                    "## Command",
                    "",
                    "```bash",
                    "python scripts/run_futureflu_workflow.py derive --derive-source h3n2-exports --lineage h3n2",
                    "```",
                    "",
                    "## Inputs",
                    "",
                    f"- `{H3N2_INPUT_FASTA.relative_to(PROJECT_ROOT)}`",
                    f"- `{H3N2_INPUT_METADATA.relative_to(PROJECT_ROOT)}`",
                    "",
                    "## Outputs",
                    "",
                    f"- `{(output_dir / 'h3n2_futureflu_aa_sequences.fasta').relative_to(PROJECT_ROOT)}`",
                    f"- `{(output_dir / 'h3n2_futureflu_metadata.tsv').relative_to(PROJECT_ROOT)}`",
                    f"- `{(output_dir / 'h3n2_futureflu_summary.json').relative_to(PROJECT_ROOT)}`",
                    f"- `{(output_dir / 'h3n2_sequence_table.tsv').relative_to(PROJECT_ROOT)}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        zh_path.write_text(
            "\n".join(
                [
                    "# H3N2 导出 Derived 输入",
                    "",
                    "[English](h3n2_original_export_derived_inputs.md)",
                    "",
                    "本报告汇总 H3N2 FASTA 和 metadata 导出的转换流程。",
                    "",
                    "## 命令",
                    "",
                    "```bash",
                    "python scripts/run_futureflu_workflow.py derive --derive-source h3n2-exports --lineage h3n2",
                    "```",
                    "",
                    "## 输入",
                    "",
                    f"- `{H3N2_INPUT_FASTA.relative_to(PROJECT_ROOT)}`",
                    f"- `{H3N2_INPUT_METADATA.relative_to(PROJECT_ROOT)}`",
                    "",
                    "## 输出",
                    "",
                    f"- `{(output_dir / 'h3n2_futureflu_aa_sequences.fasta').relative_to(PROJECT_ROOT)}`",
                    f"- `{(output_dir / 'h3n2_futureflu_metadata.tsv').relative_to(PROJECT_ROOT)}`",
                    f"- `{(output_dir / 'h3n2_futureflu_summary.json').relative_to(PROJECT_ROOT)}`",
                    f"- `{(output_dir / 'h3n2_sequence_table.tsv').relative_to(PROJECT_ROOT)}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build H3N2 derived inputs from FutureFlu FASTA and metadata exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--use-sequence-table-script",
        action="store_true",
        help="call the sequence-table script directly instead of the indexed builder",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    try:
        require_inputs()
        ensure_outputs_are_writable(output_dir, args.allow_overwrite)
    except (FileExistsError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    run_converter_script([args.python_bin, str(H3N2_CONVERTER)], dry_run=args.dry_run)
    if args.use_sequence_table_script:
        run_converter_script([args.python_bin, str(H3N2_SEQUENCE_TABLE)], dry_run=args.dry_run)
    elif args.dry_run:
        print("[cmd] build indexed-equivalent H3N2 sequence table")
    else:
        build_sequence_table_fast()
    output_paths = copy_outputs(output_dir, dry_run=args.dry_run)
    write_report(output_dir, output_paths, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
