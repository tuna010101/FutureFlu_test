#!/usr/bin/env python3
"""Validate the release component table.

English: This step checks whether the split component slices still match the final table.
中文：这个步骤用于检查拆分后的组件切片是否仍与最终主表一致。
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
FINAL_COLUMNS = [
    "subtype",
    "hemisphere",
    "year",
    "risk_mutation_group",
    "clade",
    "mutation_count",
    "mutation_group_seq_count",
    "total_escape",
    "predicted_prevalence",
    "mutual_information",
    "dissimilarity_charge_hydro",
    "accessibility_wcn",
    "fitness_eve",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge the split noro component tables.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--theta", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lib = load_project_lib(args.project_root.resolve())
    run_root = run_root_for_theta(lib, args.theta)
    part_dir = component_part_dir(lib, run_root)

    # English: The merge result is used as a consistency check, not a second source of truth.
    # 中文：这里的合并结果主要用于一致性校验，而不是第二套正式结果来源。
    base_df = pd.read_csv(part_dir / "component_base.csv")
    escape_df = pd.read_csv(part_dir / "component_escape.csv")
    growth_df = pd.read_csv(part_dir / "component_growth.csv")
    divergence_df = pd.read_csv(part_dir / "component_divergence.csv")

    merged = base_df.merge(escape_df, on=KEY_COLUMNS, how="left")
    merged = merged.merge(growth_df, on=KEY_COLUMNS, how="left")
    merged = merged.merge(divergence_df, on=KEY_COLUMNS, how="left")
    merged = merged[FINAL_COLUMNS]

    component_dir = lib.risk_out(run_root) / "mutation_components"
    component_path = component_dir / "risk_mutation_group_component.csv"
    root_copy = lib.risk_out(run_root) / "risk_mutation_group_component.csv"

    if component_path.exists():
        current = pd.read_csv(component_path)
        sort_cols = KEY_COLUMNS
        current_sorted = current[FINAL_COLUMNS].sort_values(sort_cols).reset_index(drop=True)
        merged_sorted = merged.sort_values(sort_cols).reset_index(drop=True)
        if current_sorted.equals(merged_sorted):
            print(f"[merge] verified existing component table: {component_path}")
            if root_copy.exists():
                root_df = pd.read_csv(root_copy)[FINAL_COLUMNS].sort_values(sort_cols).reset_index(drop=True)
                if not root_df.equals(current_sorted):
                    raise RuntimeError(
                        "component table copies disagree: "
                        f"{component_path} vs {root_copy}"
                    )
        else:
            raise RuntimeError(
                "assembled component parts disagree with the published component table: "
                f"{component_path}"
            )
        return 0

    merged.to_csv(component_path, index=False)
    merged.to_csv(root_copy, index=False)
    print(f"[merge] wrote {component_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
