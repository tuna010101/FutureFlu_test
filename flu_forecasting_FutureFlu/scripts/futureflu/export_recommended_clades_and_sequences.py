#!/usr/bin/env python3
"""Export recommended clade tables from completed FutureFlu runs.

English: Read top1 comparison outputs and write the recommended-clade summary CSV.
中文：读取 top1 对比结果并写出推荐 clade 汇总 CSV。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUTUREFLU_RESULTS_ROOT = PROJECT_ROOT / "results" / "futureflu"
DEFAULT_RUNS_ROOT = FUTUREFLU_RESULTS_ROOT / "runs"
DEFAULT_OUT_DIR = (
    FUTUREFLU_RESULTS_ROOT
    / "recommended_clades"
    / "primary"
)

RUNS = {
    "Victoria": "Victoria",
    "H3N2": "H3N2",
    "H1N1": "H1N1",
}

METHODS = ["ep_x-ne_star", "cTiterSub_x-ne_star"]
REGIONS = ["north", "south"]
YEARS = list(range(2013, 2025))


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface.

    English: Defaults match the repository layout used by the workflow runner.
    中文：默认路径与 workflow runner 使用的仓库结构一致。
    """
    parser = argparse.ArgumentParser(
        description="Export recommended clade tables from top1 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def load_method_rows(subtype: str, run_name: str, runs_root: Path) -> pd.DataFrame:
    run_dir = runs_root / run_name
    top1_path = run_dir / "top1_strain_clade_comparison" / "top1_strain_clades.tsv"
    if not top1_path.exists():
        raise FileNotFoundError(
            f"Missing {top1_path}. The `export` step rebuilds recommended clade "
            "tables from top1 outputs. Run the `top1` step first, or use the "
            "published CSV under results/futureflu/recommended_clades/primary/."
        )

    top1 = pd.read_csv(
        top1_path,
        sep="\t",
        usecols=[
            "hemisphere",
            "forecast_year",
            "model",
            "predicted_clade",
        ],
    )
    top1 = top1[top1["model"].isin(METHODS)].copy()
    if subtype != "H3N2":
        top1 = top1[top1["model"] != "cTiterSub_x-ne_star"].copy()

    top1.insert(0, "subtype", subtype)
    return top1


def write_summary(rows: pd.DataFrame, output_dir: Path) -> Path:
    records = []
    for subtype in RUNS:
        for region in REGIONS:
            for year in YEARS:
                record = {
                    "subtype": subtype,
                    "region": region,
                    "year": year,
                    "ep_x-ne_star": "",
                    "cTiterSub_x-ne_star": "",
                }
                subset = rows[
                    (rows["subtype"] == subtype)
                    & (rows["hemisphere"] == region)
                    & (rows["forecast_year"] == year)
                ]
                for method in METHODS:
                    method_rows = subset[subset["model"] == method]
                    if not method_rows.empty:
                        record[method] = str(method_rows.iloc[0]["predicted_clade"])
                records.append(record)

    out_path = output_dir / "recommended_clades.csv"
    pd.DataFrame.from_records(records).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = build_parser().parse_args()
    runs_root = args.runs_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.concat(
        [load_method_rows(subtype, run_name, runs_root) for subtype, run_name in RUNS.items()],
        ignore_index=True,
    )
    summary_path = write_summary(rows, output_dir)

    manifest = output_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"summary_csv\t{summary_path.name}",
                "detail_tsv\tnot included",
                "fasta_count\t0",
                "fasta_note\tnot included",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
