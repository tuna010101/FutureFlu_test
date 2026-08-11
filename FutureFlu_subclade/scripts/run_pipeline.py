#!/usr/bin/env python3
"""Run the 2025/2026 subclade pipeline against package-local outputs.

English: Run a selected subclade stage. Published outputs live under outputs/.
中文：运行指定亚分支阶段；发布结果位于 outputs/。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = PACKAGE_ROOT / "scripts" / "run_subclade_pipeline.py"
RELEASE_FUTUREFLU = PACKAGE_ROOT / "outputs" / "predictions"
INPUT_OMITTED_STAGES = {"sequences", "nextclade", "annotate", "linear", "component", "all"}


def parse_args() -> argparse.Namespace:
    """Parse the selected workflow stage. / 解析所选 workflow 阶段。"""
    parser = argparse.ArgumentParser(
        description="Run 2025/2026 subclade pipeline stages."
    )
    parser.add_argument(
        "stage",
        choices=[
            "sequences",
            "definitions",
            "nextclade",
            "annotate",
            "linear",
            "counts",
            "component",
            "aux",
            "all",
            "validate",
        ],
    )
    return parser.parse_args()


def validate_packaged_counts() -> None:
    """Validate retained count and truth tables without strain annotations."""
    required = [
        PACKAGE_ROOT / "data" / "futureflu_rank" / "circulating_subclade.csv",
        PACKAGE_ROOT
        / "data"
        / "subclade_counts"
        / "submission_collection_subclade_count_h1n1.csv",
        PACKAGE_ROOT
        / "data"
        / "subclade_counts"
        / "submission_collection_subclade_count_h3n2.csv",
        PACKAGE_ROOT
        / "data"
        / "subclade_counts"
        / "submission_collection_subclade_count_victoria.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "packaged count/truth tables are missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    for path in required:
        if path.stat().st_size == 0:
            raise RuntimeError(f"packaged count/truth table is empty: {path}")
        print(f"[counts] verified {path.relative_to(PACKAGE_ROOT)}")


def validate_release_layout() -> None:
    """Check that published outputs exist under outputs/predictions."""
    required_dirs = [
        RELEASE_FUTUREFLU / "linear" / "results",
        RELEASE_FUTUREFLU / "risk_components" / "subclade_accuracy",
        RELEASE_FUTUREFLU / "risk_components" / "component_combinations",
        RELEASE_FUTUREFLU / "risk_components" / "mutation_components",
    ]
    required_files = [
        RELEASE_FUTUREFLU / "risk_components" / "risk_mutation_group_component.csv",
        RELEASE_FUTUREFLU
        / "risk_components"
        / "mutation_components"
        / "risk_mutation_group_component.csv",
        RELEASE_FUTUREFLU
        / "risk_components"
        / "subclade_accuracy"
        / "subclade_component_acc.csv",
        RELEASE_FUTUREFLU
        / "risk_components"
        / "component_combinations"
        / "EGD_combine_Twindow.csv",
    ]
    missing = [path for path in required_dirs + required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "required published outputs are missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    exp_root = PACKAGE_ROOT / "experiments" / "truth_season_end_subclade"
    published_experiment_dirs = [
        exp_root / "outputs" / "predictions" / "risk_components" / "subclade_accuracy",
        exp_root / "outputs" / "predictions" / "risk_components" / "component_combinations",
    ]
    missing_exp = [path for path in published_experiment_dirs if not path.exists()]
    if missing_exp:
        raise FileNotFoundError(
            "required published experiment directories are missing:\n"
            + "\n".join(str(path) for path in missing_exp)
        )
    print("[validate] published outputs and experiment results are present")


def main() -> None:
    """Run one stage. / 运行一个阶段。"""
    args = parse_args()
    if args.stage in INPUT_OMITTED_STAGES:
        raise SystemExit(
            "This stage needs local sequence inputs under raw_inputs/. "
            "Published stages that do not need those inputs: "
            "definitions, counts, aux."
        )
    if args.stage == "counts":
        validate_packaged_counts()
    elif args.stage == "validate":
        validate_release_layout()
    else:
        subprocess.run([sys.executable, str(PIPELINE_SCRIPT), args.stage], check=True)
    if args.stage in {"definitions", "counts", "aux"}:
        validate_release_layout()


if __name__ == "__main__":
    main()
