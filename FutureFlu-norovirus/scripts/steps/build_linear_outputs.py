#!/usr/bin/env python3
"""Build release yearly linear outputs.

English: This step writes the direct per-year tables used by later release steps.
中文：这个步骤生成后续发布步骤所依赖的逐年线性结果表。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from workflow_helpers import load_project_lib, run_root_for_theta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build noro linear outputs.")
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
    rates_df = lib.load_positive_rates()
    run_root = run_root_for_theta(lib, args.theta)
    lib.ensure_dirs(run_root)
    # English: The yearly layer stays close to the raw inputs before component assembly.
    # 中文：逐年结果层先贴近原始输入生成，再进入组件整理阶段。
    lib.write_year_outputs(seq_df, rates_df, run_root, args.start_year, args.end_year, args.theta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
