#!/usr/bin/env python3
"""Build release component combination tables.

English: This step derives the compact E/G/D combination views from the final component table.
中文：这个步骤从最终组件主表中整理出 E/G/D 组合视图。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from workflow_helpers import component_part_dir, load_project_lib, run_root_for_theta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build noro clade component combination tables.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--theta", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lib = load_project_lib(args.project_root.resolve())
    run_root = run_root_for_theta(lib, args.theta)

    # English: Combination views are kept separate so the release bundle stays easy to scan.
    # 中文：组合视图单独输出，便于直接浏览发布包结果。
    combine_dir = lib.risk_out(run_root) / "component_combinations"
    combine_dir.mkdir(parents=True, exist_ok=True)
    component_path = lib.risk_out(run_root) / "risk_mutation_group_component.csv"
    label_path = lib.futureflu_rank_dir(run_root) / "circulating_clade.csv"
    part_dir = component_part_dir(lib, run_root)

    component_pickle = part_dir / "component_full.pkl"
    if component_pickle.exists():
        component_df = pd.read_pickle(component_pickle)
    else:
        component_df = pd.read_csv(component_path)
    label_df = pd.read_csv(label_path)

    combine_max_df, divergence_info, escape_info, growth_info = lib.compute_max_method_with_info(
        component_df
    )
    freq_df = lib.load_clade_freq(run_root, collection_based=False)
    freq_coll = lib.load_clade_freq(run_root, collection_based=True)
    acc_df, egdfit_df, egdtemp_df, lpd_df, pre_act_df = lib.compute_combination_accuracy(
        combine_max_df,
        freq_df,
        label_df,
        freq_coll,
    )

    acc_df.to_csv(combine_dir / "clade_component_combine_acc_Twindow.csv", index=False)
    egdfit_df.to_csv(combine_dir / "EGD_combine_Twindow.csv", index=False)
    egdtemp_df.to_csv(combine_dir / "EGD_temperatures_Twindow.csv", index=False)
    pre_act_df.to_csv(combine_dir / "clade_pre_act_Twindow.csv", index=False)

    elpd_df, aic_df = lib.build_elpd_aic(lpd_df)
    elpd_df.to_csv(combine_dir / "elpd_Twindow.csv", index=False)
    aic_df.to_csv(combine_dir / "aic_Twindow.csv", index=False)

    for path, df_info, columns in [
        (combine_dir / "divergence_Twindow.csv", divergence_info, lib.DIVERGENCE_COL_ORDER),
        (combine_dir / "escape_Twindow.csv", escape_info, lib.ESCAPE_COL_ORDER),
        (combine_dir / "growth_Twindow.csv", growth_info, lib.GROWTH_COL_ORDER),
    ]:
        for col in columns:
            if col not in df_info.columns:
                df_info[col] = np.nan
        df_info[columns].sort_values(["subtype", "hemisphere", "year", "clade"]).to_csv(
            path, index=False
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
