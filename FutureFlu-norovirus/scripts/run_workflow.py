#!/usr/bin/env python3
"""Run the package-local norovirus workflow.

English: This entry script keeps the release workflow running in a fixed order.
中文：这个入口脚本按固定顺序运行发布流程。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP_SCRIPTS = PROJECT_ROOT / "scripts" / "steps"

DEFAULT_START_YEAR = 2001
DEFAULT_END_YEAR = 2015
DEFAULT_MAX_GAPS = 3
LINEAR_SUFFIXES = (
    "distribution",
    "gmeasure",
    "mutations",
    "prevalence",
)
COMBINATION_FILES = (
    "clade_component_combine_acc_Twindow.csv",
    "EGD_combine_Twindow.csv",
    "EGD_temperatures_Twindow.csv",
    "aic_Twindow.csv",
    "divergence_Twindow.csv",
    "elpd_Twindow.csv",
    "escape_Twindow.csv",
    "growth_Twindow.csv",
    "clade_pre_act_Twindow.csv",
)


def verify_release() -> None:
    """Verify the published result bundle without rebuilding raw inputs."""
    required = [
        PROJECT_ROOT / "outputs" / "run_config.csv",
        PROJECT_ROOT / "outputs" / "metadata" / "summary_by_year.csv",
        PROJECT_ROOT
        / "data"
        / "positivity"
        / "zhang2024_yearly_norovirus_positive_rates.csv",
        PROJECT_ROOT / "data" / "futureflu_rank" / "circulating_clade.csv",
        PROJECT_ROOT
        / "data"
        / "clade_counts"
        / "submission_collection_clade_count_noro_gii4.csv",
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "risk_components"
        / "risk_mutation_group_component.csv",
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "risk_components"
        / "mutation_components"
        / "risk_mutation_group_component.csv",
    ]
    combine_dir = (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "risk_components"
        / "component_combinations"
    )
    for name in COMBINATION_FILES:
        required.append(combine_dir / name)

    evescape_dir = PROJECT_ROOT / "data" / "EVEscape" / "NORO_GII4_evescape"
    for year in range(DEFAULT_START_YEAR, DEFAULT_END_YEAR + 1):
        required.append(evescape_dir / f"NORO_GII4_evescape_{year}.csv")
        required.append(evescape_dir / f"NORO_GII4_evescape_sites_{year}.csv")

    linear_root = (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "linear"
        / "results"
        / "NORO_GII4_global"
    )
    for year in range(DEFAULT_START_YEAR, DEFAULT_END_YEAR + 1):
        year_dir = linear_root / str(year)
        for suffix in LINEAR_SUFFIXES:
            required.append(year_dir / f"NORO_GII4_global_{year}_{suffix}.csv")

    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing published norovirus outputs:\n"
            + "\n".join(str(path) for path in missing)
        )

    # Reject process trees and removed single-metric accuracy / experiment trees.
    forbidden_dirs = [
        PROJECT_ROOT / "data" / "EVEscape" / "EVE",
        PROJECT_ROOT / "data" / "EVEscape" / "rolling_evescape",
        PROJECT_ROOT / "_step_outputs",
        PROJECT_ROOT / "outputs" / "theta_0p1",
        PROJECT_ROOT / "outputs" / "predictions" / "risk_components" / "clade_accuracy",
        PROJECT_ROOT / "experiments",
    ]
    present_forbidden = [path for path in forbidden_dirs if path.exists()]
    if present_forbidden:
        raise RuntimeError(
            "process directories must not be present in the release package:\n"
            + "\n".join(str(path) for path in present_forbidden)
        )

    print(f"[verified] {len(required)} published files")
    for path in required[:10]:
        print(f"[verified] {path.relative_to(PROJECT_ROOT)}")
    print(
        "[verified] "
        f"outputs/predictions/linear/results/NORO_GII4_global/"
        f"{{{DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}}}/"
        f"NORO_GII4_global_<year>_{{{'|'.join(LINEAR_SUFFIXES)}}}.csv"
    )
    print(
        "[verified] "
        f"data/EVEscape/NORO_GII4_evescape/"
        f"NORO_GII4_evescape{{,_sites}}_{{{DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}}}.csv"
    )
    print("[verified] outputs/predictions/risk_components/component_combinations/")


def parse_thetas(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one theta is required")
    return values


def run_step(
    script_name: str,
    args: Sequence[str],
    output_root: str | None = None,
    disable_positive_rates: bool = False,
) -> None:
    # English: Environment switches stay close to the launch site for easier release reuse.
    # 中文：环境开关直接放在调用入口附近，便于发布包复用。
    script_path = STEP_SCRIPTS / script_name
    cmd = [sys.executable, str(script_path), "--project-root", str(PROJECT_ROOT), *args]
    print("[step] " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    if output_root:
        env["NORO_OUTPUT_ROOT"] = output_root
    if disable_positive_rates:
        env["NORO_DISABLE_POSITIVE_RATES"] = "1"
    subprocess.run(cmd, check=True, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the package-local norovirus FutureFlu-like theta=0.1 workflow."
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--thetas", default="0.1")
    parser.add_argument("--max-gaps", type=int, default=DEFAULT_MAX_GAPS)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--disable-positive-rates", action="store_true")
    parser.add_argument("--verify-release", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_release:
        verify_release()
        return 0
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be >= --start-year")

    thetas = parse_thetas(args.thetas)
    default_outputs = str((PROJECT_ROOT / "outputs").resolve())
    effective_root = (
        str(Path(args.output_root).expanduser().resolve())
        if args.output_root
        else default_outputs
    )
    if effective_root == default_outputs and len(thetas) > 1:
        raise ValueError(
            "default outputs/ layout publishes a single theta=0.1 tree; "
            "for multiple thetas pass distinct --output-root directories "
            "(or one custom root that will nest theta_* subdirectories)"
        )

    for theta in thetas:
        common_args = [
            "--start-year",
            str(args.start_year),
            "--end-year",
            str(args.end_year),
            "--theta",
            str(theta),
            "--max-gaps",
            str(args.max_gaps),
        ]
        step_kwargs = {
            "output_root": args.output_root,
            "disable_positive_rates": args.disable_positive_rates,
        }
        # English: The active release path stays linear and predictable.
        # 中文：正式发布路径保持线性顺序，便于直接复现。
        run_step("prepare_inputs.py", common_args, **step_kwargs)
        run_step("build_linear_outputs.py", common_args, **step_kwargs)
        run_step("build_component_table.py", common_args, **step_kwargs)
        run_step("write_escape_component.py", ["--theta", str(theta)], **step_kwargs)
        run_step("write_growth_component.py", ["--theta", str(theta)], **step_kwargs)
        run_step("write_divergence_component.py", ["--theta", str(theta)], **step_kwargs)
        run_step("validate_component_table.py", ["--theta", str(theta)], **step_kwargs)
        run_step(
            "build_component_combination_tables.py",
            ["--theta", str(theta)],
            **step_kwargs,
        )
        run_step("write_run_summary.py", ["--theta", str(theta)], **step_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
