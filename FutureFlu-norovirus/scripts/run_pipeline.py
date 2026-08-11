#!/usr/bin/env python3
"""FutureFlu-like release entry for the norovirus package.

English: Provide the same top-level stage names used by the FutureFlu / HA1 packages.
中文：提供与 FutureFlu / HA1 包一致的顶层 stage 名称。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from run_workflow import main as run_workflow_main


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the package-local norovirus FutureFlu-like workflow."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("all", "validate", "verify"),
        help="workflow stage (default: all)",
    )
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--thetas", default=None)
    parser.add_argument("--max-gaps", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--disable-positive-rates", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage in {"validate", "verify"}:
        extras = []
        if args.start_year is not None:
            extras.append("--start-year")
        if args.end_year is not None:
            extras.append("--end-year")
        if args.thetas is not None:
            extras.append("--thetas")
        if args.max_gaps is not None:
            extras.append("--max-gaps")
        if args.output_root is not None:
            extras.append("--output-root")
        if args.disable_positive_rates:
            extras.append("--disable-positive-rates")
        if extras:
            raise SystemExit(
                "validate/verify ignores run options; "
                f"remove unused arguments: {', '.join(extras)}"
            )
        return run_workflow_main(["--verify-release"])

    forwarded: list[str] = []
    if args.start_year is not None:
        forwarded.extend(["--start-year", str(args.start_year)])
    if args.end_year is not None:
        forwarded.extend(["--end-year", str(args.end_year)])
    if args.thetas is not None:
        forwarded.extend(["--thetas", args.thetas])
    if args.max_gaps is not None:
        forwarded.extend(["--max-gaps", str(args.max_gaps)])
    if args.output_root is not None:
        forwarded.extend(["--output-root", args.output_root])
    if args.disable_positive_rates:
        forwarded.append("--disable-positive-rates")
    return run_workflow_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
