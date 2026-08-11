#!/usr/bin/env python3
"""Small release runner for the packaged beth-1 reproduction workflow.

English: Run the primary overall-fit-once HA-only no-climate analysis.
中文：运行主方法 overall-fit-once HA-only 无气候分析。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PACKAGE_ROOT / "work" / "ha_overall_fit" / "run_beth1_ha.py"


def run_command(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=PACKAGE_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [sys.executable, str(RUNNER)]
    if args.processes:
        cmd += ["--processes", str(args.processes)]
    run_command(cmd)


if __name__ == "__main__":
    main()
