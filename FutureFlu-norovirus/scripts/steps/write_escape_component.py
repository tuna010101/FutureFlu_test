#!/usr/bin/env python3
"""Write the release escape component slice.

English: This step keeps the escape signal in a small standalone table.
中文：这个步骤把 escape 信号整理成独立的小表。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_helpers import component_part_dir, load_project_lib, run_root_for_theta


KEY_COLUMNS = [
    "subtype",
    "hemisphere",
    "year",
    "risk_mutation_group",
    "clade",
    "mutation_count",
    "mutation_group_seq_count",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split the escape component table.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--theta", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lib = load_project_lib(args.project_root.resolve())
    run_root = run_root_for_theta(lib, args.theta)
    component_path = lib.risk_out(run_root) / "risk_mutation_group_component.csv"
    component_df = pd.read_csv(component_path)
    part_dir = component_part_dir(lib, run_root)
    # English: Only the narrow slice needed by the later consistency check is kept here.
    # 中文：这里只保留后续一致性校验所需的窄表切片。
    component_df[KEY_COLUMNS + ["total_escape"]].to_csv(
        part_dir / "component_escape.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
