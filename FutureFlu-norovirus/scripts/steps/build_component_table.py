#!/usr/bin/env python3
"""Build the release component table.

English: This step gathers the component-ready table and preserves nearby extra metrics.
中文：这个步骤汇总组件主表，并顺带保留相关扩展指标。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow_helpers import component_part_dir, load_project_lib, run_root_for_theta


BASE_COLUMNS = [
    "subtype",
    "hemisphere",
    "year",
    "risk_mutation_group",
    "clade",
    "mutation_count",
    "mutation_group_seq_count",
    "dissimilarity_charge_hydro",
    "accessibility_wcn",
    "fitness_eve",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the base noro component table.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--max-gaps", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lib = load_project_lib(args.project_root.resolve())

    seq_df = lib.load_sequence_table(args.max_gaps)
    run_root = run_root_for_theta(lib, args.theta)
    lib.ensure_dirs(run_root)
    # English: The component table acts as the compact handoff into later summaries.
    # 中文：组件主表是通向后续汇总结果的紧凑交接层。
    component_df = lib.build_component_table(seq_df, run_root, args.start_year, args.end_year)

    part_dir = component_part_dir(lib, run_root)
    component_df.to_pickle(part_dir / "component_full.pkl")
    component_df[BASE_COLUMNS].to_csv(part_dir / "component_base.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
