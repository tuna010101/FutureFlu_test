#!/usr/bin/env python3
"""Write the release run summary.

English: This step leaves a small yearly summary beside the main release outputs.
中文：这个步骤在主结果旁边补上一份简短的逐年汇总表。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from workflow_helpers import load_project_lib, run_root_for_theta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the noro per-year summary table.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--theta", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lib = load_project_lib(args.project_root.resolve())
    run_root = run_root_for_theta(lib, args.theta)
    process_meta = lib.process_metadata_dir(run_root)

    linear_status_path = process_meta / "linear_status_by_year.csv"
    component_status_path = process_meta / "component_status_by_year.csv"
    summary_path = lib.release_metadata_dir(run_root) / "summary_by_year.csv"

    linear_status = pd.read_csv(linear_status_path)
    if component_status_path.exists():
        component_status = pd.read_csv(component_status_path)
    else:
        component_status = pd.DataFrame(columns=["year", "component_rows", "status"])

    # English: The summary keeps only a light join of the main status views.
    # 中文：这个汇总只做轻量级状态拼接，不引入新的计算口径。
    summary = linear_status.merge(
        component_status[["year", "component_rows", "status"]],
        on="year",
        how="left",
    )
    summary.to_csv(summary_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
