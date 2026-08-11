#!/usr/bin/env python3
"""Public entry point for FutureFlu issue-date workflows.

English: Dispatch derive/prepare/timepoints/aggregate/model/top1/export steps.
中文：调度 derive/prepare/timepoints/aggregate/model/top1/export 各步骤。
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUTUREFLU_DATA_ROOT = PROJECT_ROOT / "data" / "futureflu"
FUTUREFLU_RESULTS_ROOT = PROJECT_ROOT / "results" / "futureflu"
FUTUREFLU_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "futureflu"


def _load_lineage_specs_module():
    """Dynamically import lineage_specs.py. / 动态导入 lineage_specs.py。"""
    module_path = FUTUREFLU_SCRIPT_ROOT / "lineage_specs.py"
    spec = importlib.util.spec_from_file_location("futureflu_lineage_specs", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load lineage specs from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lineage_specs = _load_lineage_specs_module()
load_lineage_config = _lineage_specs.load_lineage_config
resolved_lineages = _lineage_specs.resolved_lineages
workflow_defaults = _lineage_specs.workflow_defaults

PIPELINE_SCRIPT = FUTUREFLU_SCRIPT_ROOT / "run_issue_date_pipeline.py"
SOURCE_TABLE_CONVERTER = FUTUREFLU_SCRIPT_ROOT / "convert_source_tables_to_derived.py"
H3N2_EXPORT_CONVERTER = FUTUREFLU_SCRIPT_ROOT / "rebuild_h3n2_derived_from_original_exports.py"
TOP1_SCRIPT = FUTUREFLU_SCRIPT_ROOT / "compare_top1_strain_clades_to_truth.py"
TOP1_TRUTH = FUTUREFLU_DATA_ROOT / "truth" / "top1_truth.csv"
EXPORT_SCRIPT = FUTUREFLU_SCRIPT_ROOT / "export_recommended_clades_and_sequences.py"
EXPORT_OUTPUT_ROOT = FUTUREFLU_RESULTS_ROOT / "recommended_clades"

_CONFIG = load_lineage_config()
_DEFAULTS = workflow_defaults(_CONFIG)
RUNS = resolved_lineages(_CONFIG)
EXPORT_RUN = str(_DEFAULTS.get("export_run", "primary"))


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser. / 构建公开命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="Run FutureFlu issue-date steps (single public entry point).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "step",
        choices=["derive", "prepare", "timepoints", "aggregate", "model", "top1", "export", "all"],
        help="workflow step to run",
    )
    parser.add_argument(
        "--lineage",
        choices=["all", *RUNS.keys()],
        default="all",
        help="lineage subset for prepare/timepoints/aggregate/model/top1",
    )
    parser.add_argument("--python-bin", default="python", help="Python executable for subprocesses")
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "sources",
        help="directory of primary HA sequence tables / submission files for derive (default: data/sources)",
    )
    parser.add_argument(
        "--derive-source",
        choices=["source-tables", "h3n2-exports"],
        default="source-tables",
        help="derive input mode: CSV sequence tables, or H3N2 FASTA+metadata exports",
    )
    parser.add_argument(
        "--sequence-output-dir",
        type=Path,
        default=None,
        help="where derive writes sequence inputs (default: data/sequences)",
    )
    parser.add_argument(
        "--sequence-input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sequences",
        help="sequence inputs for prepare/timepoints/aggregate/model (default: data/sequences)",
    )
    parser.add_argument(
        "--overwrite-sequence-inputs",
        action="store_true",
        help="allow derive to overwrite existing files in --sequence-output-dir",
    )
    parser.add_argument(
        "--without-metadata-scaffold",
        action="store_true",
        help="derive metadata without using bundled metadata as a scaffold",
    )
    parser.add_argument("--timepoint-workers", type=int, default=1, help="parallel issue-date workers")
    parser.add_argument("--iqtree-threads", default="AUTO", help="thread argument passed to IQ-TREE")
    parser.add_argument(
        "--resume-model-existing",
        action="store_true",
        help=(
            "for model steps only, skip complete forecast/error/coefficient triples already on disk; "
            "do not use for strict reproducibility after interruption"
        ),
    )
    # Developer-only; kept out of public -h text.
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    return parser


def selected_run_keys(lineage: str) -> list[str]:
    """Map --lineage to run keys. / 将 --lineage 映射为内部 run 键。"""
    if lineage == "all":
        return ["h1n1pdm", "h3n2", "victoria"]
    return [lineage]


def display_argument(value: object) -> str:
    """Pretty-print path args relative to PROJECT_ROOT. / 将路径参数相对 PROJECT_ROOT 显示。"""
    text = str(value)
    path = Path(text)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return text
    return text


def quote_command(command: list[str]) -> str:
    """Shell-quote a command list for logging. / 对命令列表做 shell 引号以便日志打印。"""
    return " ".join(shlex.quote(display_argument(part)) for part in command)


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    """Run or dry-run one subprocess. / 执行或 dry-run 一个子进程。"""
    print(f"[cwd] {display_argument(cwd)}")
    print(f"[cmd] {quote_command(command)}")
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def mode_args(spec: dict) -> list[str]:
    """Extra CLI flags from a lineage mode spec. / 由谱系 mode 配置生成额外 CLI 参数。"""
    args: list[str] = []
    if spec.get("disable_hi"):
        args.append("--disable-hi")
    if spec.get("disable_dms"):
        args.append("--disable-dms")
    if spec.get("allow_hi_fallback"):
        args.append("--allow-hi-fallback")
    return args


def common_issue_args(spec: dict) -> list[str]:
    """Shared issue-date arguments for a lineage. / 某谱系共享的 issue-date 参数。"""
    forecast_years = [str(year) for year in spec.get("forecast_years", _DEFAULTS.get("forecast_years", []))]
    return [
        "--global-start",
        str(spec.get("global_start", _DEFAULTS["global_start"])),
        "--issue-start",
        str(spec.get("issue_start", _DEFAULTS["issue_start"])),
        "--issue-end",
        str(spec.get("issue_end", _DEFAULTS["issue_end"])),
        "--forecast-years",
        *forecast_years,
    ]


def pipeline_command(step: str, run_key: str, args: argparse.Namespace) -> list[str]:
    """Build the issue-date pipeline subprocess command. / 构建 issue-date pipeline 子进程命令。"""
    spec = RUNS[run_key]
    prefix = spec["prefix"]
    predictor_args: list[str] = []
    for predictor in spec["predictors"]:
        predictor_args.extend(["--predictor", predictor])
    resume_args = ["--resume-existing"] if step == "model" and args.resume_model_existing else []

    return [
        args.python_bin,
        str(PIPELINE_SCRIPT),
        step,
        "--run-dir",
        f"results/futureflu/runs/{spec['run_name']}",
        "--input-metadata",
        str(args.sequence_input_dir / f"{prefix}_futureflu_metadata.tsv"),
        "--input-fasta",
        str(args.sequence_input_dir / f"{prefix}_futureflu_aa_sequences.fasta"),
        "--input-sequence-table",
        str(args.sequence_input_dir / f"{prefix}_sequence_table.tsv"),
        "--distance-map-root",
        str(spec["distance_map_root"]),
        "--report-label",
        spec["label"],
        "--global-end",
        str(spec["global_end"]),
        *common_issue_args(spec),
        "--timepoint-workers",
        str(args.timepoint_workers),
        "--iqtree-threads",
        str(args.iqtree_threads),
        *mode_args(spec),
        *predictor_args,
        *resume_args,
    ]


def run_lineage_step(step: str, args: argparse.Namespace) -> None:
    """Run one pipeline step for selected lineages. / 对所选谱系运行单个 pipeline 步骤。"""
    for run_key in selected_run_keys(args.lineage):
        run_command(pipeline_command(step, run_key, args), cwd=PROJECT_ROOT, dry_run=args.dry_run)


def run_derive(args: argparse.Namespace) -> None:
    """Derive sequence inputs from sources or H3N2 exports. / 从源表或 H3N2 导出物生成序列输入。"""
    if args.derive_source == "h3n2-exports":
        if args.lineage != "h3n2":
            raise SystemExit("--derive-source h3n2-exports currently supports only --lineage h3n2")
        output_dir = args.sequence_output_dir or (PROJECT_ROOT / "data" / "sequences")
        command = [
            args.python_bin,
            str(H3N2_EXPORT_CONVERTER),
            "--output-dir",
            str(output_dir),
            "--python-bin",
            args.python_bin,
        ]
        if args.overwrite_sequence_inputs:
            command.append("--allow-overwrite")
        run_command(command, cwd=PROJECT_ROOT, dry_run=args.dry_run)
        return

    output_dir = args.sequence_output_dir or (PROJECT_ROOT / "data" / "sequences")
    command = [
        args.python_bin,
        str(SOURCE_TABLE_CONVERTER),
        "--lineage",
        args.lineage,
        "--sources-root",
        str(args.sources_root),
        "--output-dir",
        str(output_dir),
    ]
    if args.overwrite_sequence_inputs:
        command.append("--allow-overwrite")
    if args.without_metadata_scaffold:
        command.append("--without-metadata-scaffold")
    run_command(command, cwd=PROJECT_ROOT, dry_run=args.dry_run)


def run_export(args: argparse.Namespace) -> None:
    """Export recommended clade tables. / 导出推荐 clade 表。"""
    run_command(
        [
            args.python_bin,
            str(EXPORT_SCRIPT),
            "--runs-root",
            str(FUTUREFLU_RESULTS_ROOT / "runs"),
            "--output-dir",
            str(EXPORT_OUTPUT_ROOT / EXPORT_RUN),
        ],
        cwd=PROJECT_ROOT,
        dry_run=args.dry_run,
    )


def run_top1(args: argparse.Namespace) -> None:
    """Compare top-1 strains/clades to truth. / 将 top-1 毒株/clade 与真值比较。"""
    for run_key in selected_run_keys(args.lineage):
        spec = RUNS[run_key]
        run_command(
            [
                args.python_bin,
                str(TOP1_SCRIPT),
                "--run-dir",
                str(FUTUREFLU_RESULTS_ROOT / "runs" / spec["run_name"]),
                "--truth",
                str(TOP1_TRUTH),
                "--virus",
                spec["virus"],
                "--score-column",
                "y",
                "--score-direction",
                "min",
            ],
            cwd=PROJECT_ROOT,
            dry_run=args.dry_run,
        )


def main() -> None:
    """CLI entry point. / 命令行入口。"""
    args = build_parser().parse_args()

    if args.step == "derive":
        run_derive(args)
        return

    if args.step in {"prepare", "timepoints", "aggregate", "model"}:
        run_lineage_step(args.step, args)
        return

    if args.step == "top1":
        run_top1(args)
        return

    if args.step == "export":
        run_export(args)
        return

    for step in ["prepare", "timepoints", "aggregate", "model"]:
        run_lineage_step(step, args)
    run_top1(args)
    run_export(args)


if __name__ == "__main__":
    main()
