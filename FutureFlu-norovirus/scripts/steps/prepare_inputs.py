#!/usr/bin/env python3
"""Prepare release inputs for one workflow run.

English: This step shapes the release-facing inputs and metadata for a theta run.
中文：这个步骤用于整理单个 theta 运行所需的发布输入和元数据。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_helpers import load_project_lib, run_root_for_theta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare noro theta-run inputs.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--max-gaps", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lib = load_project_lib(args.project_root.resolve())

    if args.end_year < args.start_year:
        raise ValueError("--end-year must be >= --start-year")

    lib.validate_evescape_outputs(args.start_year, args.end_year)
    seq_df = lib.load_sequence_table(args.max_gaps)
    rates_df = lib.load_positive_rates()

    # English: A short run card is written first so the output bundle stays readable.
    # 中文：先写出简短运行卡片，便于直接理解输出包内容。
    lib.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "start_year": args.start_year,
                "end_year": args.end_year,
                "thetas": lib.format_theta(args.theta),
                "max_gaps": args.max_gaps,
                "sequence_rows_after_gap_filter": len(seq_df),
                "subtype": lib.SUBTYPE,
                "hemisphere": lib.HEMISPHERE,
                "clade_column": "genotype",
                "season_window": "calendar_year",
                "submission_cutoff": "season_end_exclusive",
                "candidate_sequence_window": "previous_calendar_year",
                "submission_date_source": seq_df.attrs.get(
                    "submission_date_source", "unknown"
                ),
                "site_region": f"{lib.EPITOPE_START}-{lib.EPITOPE_END}",
                "site_region_start": lib.EPITOPE_START,
                "site_region_end": lib.EPITOPE_END,
                "positive_rate_file": lib.positive_rate_file_label(),
                "evescape_source": str(lib.EVESCAPE_SCORE_DIR.relative_to(lib.ROOT)),
            }
        ]
    ).to_csv(lib.OUTPUT_ROOT / "run_config.csv", index=False)

    run_root = run_root_for_theta(lib, args.theta)
    lib.ensure_dirs(run_root)
    lib.write_preprocessed_sequence_table(seq_df, run_root)
    process_meta = lib.process_metadata_dir(run_root)
    rates_df.to_csv(process_meta / "positive_rates_used.csv", index=False)

    label_df, count_df = lib.build_label_and_count_inputs(
        seq_df, run_root, args.start_year, args.end_year
    )
    count_df.to_csv(process_meta / "genotype_counts_by_year.csv", index=False)
    label_df.to_csv(process_meta / "dominant_genotype_labels.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
