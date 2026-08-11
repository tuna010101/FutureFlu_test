#!/usr/bin/env python3
"""Run the truth season-end subclade sensitivity experiment.

English: Rebuild the season-end truth definitions and comparison reports.
中文：重建季末真值定义并生成敏感性比较报告。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = EXP_ROOT.parents[1]
MAIN_SCRIPT = PACKAGE_ROOT / "scripts" / "run_subclade_pipeline.py"

EXP_DATA = EXP_ROOT / "data"
EXP_REPORTS = EXP_ROOT / "reports"
EXP_OUT = EXP_ROOT / "outputs" / "predictions" / "risk_components"


def load_main_module():
    spec = importlib.util.spec_from_file_location("run_subclade_pipeline_main", MAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repro = load_main_module()


@dataclass(frozen=True)
class TruthTarget:
    subtype: str
    hemisphere: str
    year: int

    @property
    def hemi_lower(self) -> str:
        return self.hemisphere.lower()

    @property
    def label(self) -> str:
        return f"{self.subtype}_{self.hemisphere}_{self.year}"

    @property
    def season_start(self) -> pd.Timestamp:
        start, _ = repro.target_season_window(self.year, self.hemi_lower)
        return start

    @property
    def season_end(self) -> pd.Timestamp:
        _, end = repro.target_season_window(self.year, self.hemi_lower)
        return end

    @property
    def season_end_cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.season_end, tz="UTC")


def ensure_dirs() -> None:
    for path in [
        EXP_DATA / "subclade_definitions",
        EXP_DATA / "subclade_annotations",
        EXP_DATA / "nextclade" / "datasets",
        EXP_DATA / "nextclade" / "results",
        EXP_DATA / "futureflu_rank",
        EXP_DATA / "subclade_counts",
        EXP_REPORTS,
        EXP_OUT / "subclade_accuracy",
        EXP_OUT / "component_combinations",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def all_truth_targets() -> list[TruthTarget]:
    return [
        TruthTarget(subtype, hemisphere, year)
        for subtype in repro.SUBTYPES
        for hemisphere, year in repro.TARGETS
    ]


def definition_path(target: TruthTarget) -> Path:
    return (
        EXP_DATA
        / "subclade_definitions"
        / f"{target.subtype}_{target.hemi_lower}_{target.year}_season_end_subclades.tsv"
    )


def nextclade_result_path(target: TruthTarget) -> Path:
    return (
        EXP_DATA
        / "nextclade"
        / "results"
        / f"{target.subtype}_{target.hemi_lower}_{target.year}_season_end_nextclade.tsv"
    )


def nextclade_dataset_dir(subtype: str, tag: str) -> Path:
    return EXP_DATA / "nextclade" / "datasets" / f"{subtype}_{tag}"


def nextclade_dataset_zip(subtype: str, tag: str) -> Path:
    return EXP_DATA / "nextclade" / "datasets" / f"{subtype}_{tag}.zip"


def display_path(path: Path) -> str:
    for base in (EXP_ROOT, PACKAGE_ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            pass
    return str(path)


def existing_nextclade_dataset(subtype: str, tag: str) -> Path | None:
    for base in [
        PACKAGE_ROOT / "raw_inputs" / "nextclade" / "datasets",
        EXP_DATA / "nextclade" / "datasets",
    ]:
        dataset_dir = base / f"{subtype}_{tag}"
        dataset_zip = base / f"{subtype}_{tag}.zip"
        if (dataset_dir / "pathogen.json").exists():
            return dataset_dir
        if dataset_zip.exists() and dataset_zip.stat().st_size > 0:
            return dataset_zip
    return None


def ensure_season_end_nextclade_dataset(subtype: str, tag: str) -> Path:
    existing = existing_nextclade_dataset(subtype, tag)
    if existing is not None:
        return existing
    return repro.ensure_nextclade_dataset(subtype, tag)


def annotation_path(target: TruthTarget) -> Path:
    return (
        EXP_DATA
        / "subclade_annotations"
        / f"{target.subtype}_{target.hemi_lower}_{target.year}_season_end_subclade_annotations.csv"
    )


def max_collection_date_for_subtype(subtype: str) -> pd.Timestamp:
    seq = pd.read_csv(repro.sequence_csv_path(subtype), usecols=["collection_date"])
    return pd.to_datetime(seq["collection_date"], errors="coerce").max()


def targets_with_truth_data() -> list[TruthTarget]:
    available = {}
    for subtype in repro.SUBTYPES:
        available[subtype] = max_collection_date_for_subtype(subtype)
    rows = []
    for target in all_truth_targets():
        if pd.isna(available[target.subtype]) or available[target.subtype] < target.season_start:
            continue
        rows.append(target)
    return rows


def select_definition_commit(commit_df: pd.DataFrame, target: TruthTarget) -> pd.Series | None:
    sub = commit_df[commit_df["subtype"] == target.subtype].copy()
    sub["commit_ts"] = pd.to_datetime(sub["commit_date"], utc=True)
    eligible = sub[sub["commit_ts"] < target.season_end_cutoff]
    if eligible.empty:
        return None
    return eligible.sort_values("commit_ts").iloc[-1]


def write_season_end_definitions(targets: list[TruthTarget]) -> None:
    manifest_path = PACKAGE_ROOT / "data" / "subclade_definitions" / "commit_manifest.csv"
    if manifest_path.exists():
        commit_df = pd.read_csv(manifest_path)
    else:
        rows = []
        for subtype, repo in repro.SUBTYPE_REPOS.items():
            for item in repro.fetch_commits(repo):
                commit = item["commit"]
                rows.append(
                    {
                        "subtype": subtype,
                        "repo": repo,
                        "sha": item["sha"],
                        "commit_date": commit["committer"]["date"],
                        "message": commit["message"].splitlines()[0],
                    }
                )
        commit_df = pd.DataFrame(rows)
    commit_df.to_csv(EXP_DATA / "subclade_definitions" / "commit_manifest.csv", index=False)

    selected_rows = []
    for target in targets:
        chosen = select_definition_commit(commit_df, target)
        repo = repro.SUBTYPE_REPOS[target.subtype]
        if chosen is None:
            selected_rows.append(
                {
                    "subtype": target.subtype,
                    "hemisphere": target.hemi_lower,
                    "year": target.year,
                    "season_start": target.season_start.date().isoformat(),
                    "season_end_cutoff": target.season_end_cutoff.isoformat(),
                    "repo": repo,
                    "sha": "",
                    "commit_date": "",
                    "definition_path": "",
                    "status": "no_commit_before_season_end",
                }
            )
            continue

        out_path = definition_path(target)
        text = repro.fetch_subclades_tsv(repo, chosen["sha"])
        status = "ok" if text else "missing_subclades_tsv"
        if text:
            out_path.write_text(text, encoding="utf-8")
        selected_rows.append(
            {
                "subtype": target.subtype,
                "hemisphere": target.hemi_lower,
                "year": target.year,
                "season_start": target.season_start.date().isoformat(),
                "season_end_cutoff": target.season_end_cutoff.isoformat(),
                "repo": repo,
                "sha": chosen["sha"],
                "commit_date": chosen["commit_date"],
                "definition_path": display_path(out_path) if text else "",
                "status": status,
            }
        )
        print(f"[definitions] {target.label} {chosen['sha'][:12]} {status}")

    pd.DataFrame(selected_rows).to_csv(
        EXP_DATA / "subclade_definitions" / "selected_definitions_season_end.csv",
        index=False,
    )


def patch_nextclade_paths():
    repro.DATA_OUT = EXP_DATA
    repro.nextclade_result_path = nextclade_result_path
    repro.nextclade_dataset_dir = nextclade_dataset_dir
    repro.nextclade_dataset_zip = nextclade_dataset_zip


def run_season_end_nextclade(targets: list[TruthTarget]) -> None:
    patch_nextclade_paths()
    rows = []
    for target in targets:
        tag, updated_at = repro.selected_nextclade_tag(target.subtype, target.season_end_cutoff)
        dataset = ensure_season_end_nextclade_dataset(target.subtype, tag)
        out_path = nextclade_result_path(target)
        nuc_path = repro.release_input_path(target.subtype, "nuc")
        print(f"[nextclade] {target.label} season_end tag={tag}")
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[nextclade] reuse {out_path}")
        else:
            result = repro.subprocess.run(
                [
                    "nextclade",
                    "run",
                    "--input-dataset",
                    str(dataset),
                    "--replace-unknown",
                    "true",
                    "--output-tsv",
                    str(out_path),
                    str(nuc_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"nextclade run failed for {target.label}: {result.stderr}")
        rows.append(
            {
                "subtype": target.subtype,
                "hemisphere": target.hemi_lower,
                "year": target.year,
                "season_start": target.season_start.date().isoformat(),
                "season_end_cutoff": target.season_end_cutoff.isoformat(),
                "dataset_name": repro.nextclade_dataset_name(target.subtype),
                "dataset_tag": tag,
                "dataset_updated_at": updated_at,
                "dataset_path": display_path(dataset),
                "nextclade_tsv": display_path(out_path),
            }
        )
    pd.DataFrame(rows).to_csv(EXP_DATA / "nextclade" / "selected_datasets_season_end.csv", index=False)


def nextclade_assignment_lookup(target: TruthTarget) -> tuple[dict[str, dict[str, str]], dict[str, int | str]]:
    old_func = repro.nextclade_result_path
    try:
        repro.nextclade_result_path = nextclade_result_path
        return repro.nextclade_assignment_lookup(target)
    finally:
        repro.nextclade_result_path = old_func


def annotate_one(target: TruthTarget) -> dict:
    def_path = definition_path(target)
    if not def_path.exists():
        return {
            "subtype": target.subtype,
            "hemisphere": target.hemi_lower,
            "year": target.year,
            "status": "missing_definition",
        }

    seq_path = repro.sequence_csv_path(target.subtype)
    rules, meta = repro.parse_definition_rules(def_path, target.subtype)
    usecols = ["accession number", "name", "clade", "collection_date", "submission_date", "season"]
    xcols = sorted(
        {x for rule_list in rules.values() for x, _ in rule_list},
        key=lambda value: int(value[1:]),
    )
    header = pd.read_csv(seq_path, nrows=0).columns
    xcols = [col for col in xcols if col in header]
    df = pd.read_csv(seq_path, usecols=usecols + xcols, low_memory=False)

    nextclade_lookup, nextclade_meta = nextclade_assignment_lookup(target)
    nextclade_clade_hits = 0
    nextclade_subclade_hits = 0
    nextclade_subclades = pd.Series("", index=df.index, dtype=object)
    if nextclade_lookup:
        mapped_clades = []
        for _, row in df.iterrows():
            accession = str(row["accession number"]).strip()
            assignment = nextclade_lookup.get(accession, {})
            clade = assignment.get("clade", "")
            if clade:
                nextclade_clade_hits += 1
                mapped_clades.append(clade)
            else:
                mapped_clades.append(row["clade"])
            subclade = assignment.get("subclade", "")
            if subclade:
                nextclade_subclade_hits += 1
                nextclade_subclades.loc[row.name] = subclade
        df["clade"] = mapped_clades

    for col in xcols:
        df[col] = df[col].fillna("").astype(str).str.upper()

    labels = pd.Series("unknown", index=df.index, dtype=object)
    matched_rule_counts = pd.Series(0, index=df.index, dtype=np.int16)
    matched_depth = pd.Series(-1, index=df.index, dtype=np.int16)
    ordered = sorted(rules.items(), key=lambda item: (len(item[1]), len(item[0].split(".")), item[0]))
    for name, rule_list in ordered:
        valid_rules = [(col, state) for col, state in rule_list if col in df.columns]
        if not valid_rules:
            continue
        mask = pd.Series(True, index=df.index)
        for col, state in valid_rules:
            mask &= df[col].eq(state)
        if not mask.any():
            continue
        depth = len(name.split("."))
        better = mask & (
            (len(valid_rules) > matched_rule_counts)
            | ((len(valid_rules) == matched_rule_counts) & (depth >= matched_depth))
        )
        labels.loc[better] = name
        matched_rule_counts.loc[better] = len(valid_rules)
        matched_depth.loc[better] = depth

    final_labels = labels.copy()
    nextclade_mask = nextclade_subclades.ne("")
    final_labels.loc[nextclade_mask] = nextclade_subclades.loc[nextclade_mask]
    source = pd.Series("unknown", index=df.index, dtype=object)
    source.loc[labels.ne("unknown")] = "ha_rule"
    source.loc[nextclade_mask] = "nextclade"

    out = df[usecols].copy()
    out["subtype"] = target.subtype
    out["hemisphere"] = target.hemi_lower
    out["target_year"] = target.year
    out["truth_definition_cutoff"] = target.season_end_cutoff.isoformat()
    out["subclade"] = final_labels
    out["ha_rule_subclade"] = labels
    out["nextclade_subclade"] = nextclade_subclades
    out["subclade_source"] = source
    out["matched_ha_rule_count"] = matched_rule_counts
    out_path = annotation_path(target)
    out.to_csv(out_path, index=False)

    assigned = int(out["subclade"].ne("unknown").sum())
    return {
        "subtype": target.subtype,
        "hemisphere": target.hemi_lower,
        "year": target.year,
        "status": "ok",
        "rows": len(out),
        "assigned_rows": assigned,
        "assigned_fraction": assigned / len(out) if len(out) else np.nan,
        "annotation_path": display_path(out_path),
        "nextclade_clade_hits": nextclade_clade_hits,
        "nextclade_clade_hit_fraction": nextclade_clade_hits / len(out) if len(out) else np.nan,
        "nextclade_subclade_hits": nextclade_subclade_hits,
        "nextclade_subclade_hit_fraction": nextclade_subclade_hits / len(out) if len(out) else np.nan,
        **nextclade_meta,
        **meta,
    }


def run_annotations(targets: list[TruthTarget]) -> None:
    rows = []
    for target in targets:
        print(f"[annotate] {target.label} season_end")
        rows.append(annotate_one(target))
    pd.DataFrame(rows).to_csv(EXP_REPORTS / "subclade_definition_coverage_season_end.csv", index=False)


def valid_label_series(df: pd.DataFrame) -> pd.Series:
    vals = df["subclade"].fillna("").astype(str).str.strip()
    return vals[~vals.str.lower().isin({"", "unknown", "nan", "unassigned"})]


def counted_label_series(df: pd.DataFrame) -> pd.Series:
    vals = df["subclade"].fillna("").astype(str).str.strip()
    return vals[~vals.str.lower().isin({"", "nan"})]


def run_counts_and_truth(targets: list[TruthTarget]) -> None:
    labels = []
    truth_status = []
    for subtype in repro.SUBTYPES:
        count_rows = []
        for target in [target for target in targets if target.subtype == subtype]:
            ann = pd.read_csv(annotation_path(target), low_memory=False)
            ann["collection_date"] = pd.to_datetime(ann["collection_date"])
            ann["submission_date"] = pd.to_datetime(ann["submission_date"])
            for count_year in sorted({target.year - 1, target.year}):
                start, end = repro.target_season_window(count_year, target.hemi_lower)
                collection_mask = (ann["collection_date"] >= start) & (ann["collection_date"] < end)
                submission_mask = collection_mask & (ann["submission_date"] < end)
                collection_counts = counted_label_series(ann.loc[collection_mask]).value_counts()
                submission_counts = counted_label_series(ann.loc[submission_mask]).value_counts()
                subclades = sorted(set(collection_counts.index).union(set(submission_counts.index)))
                for subclade in subclades:
                    count_rows.append(
                        {
                            "target_year": target.year,
                            "target_hemisphere": target.hemi_lower,
                            "year": count_year,
                            "hemisphere": target.hemi_lower,
                            "subclade": subclade,
                            "submission_count": int(submission_counts.get(subclade, 0)),
                            "collection_count": int(collection_counts.get(subclade, 0)),
                        }
                    )

            start, end = repro.target_season_window(target.year, target.hemi_lower)
            season_rows = ann.loc[(ann["collection_date"] >= start) & (ann["collection_date"] < end)]
            label_counts = valid_label_series(season_rows).value_counts()
            if label_counts.empty:
                truth_status.append(
                    {
                        "subtype": subtype,
                        "hemisphere": target.hemi_lower,
                        "year": target.year,
                        "season_start": start.date().isoformat(),
                        "season_end": end.date().isoformat(),
                        "collection_rows": len(season_rows),
                        "status": "no_valid_truth_label",
                    }
                )
                continue
            labels.append(
                {
                    "subtype": subtype,
                    "hemisphere": target.hemi_lower,
                    "year": target.year,
                    "subclade": label_counts.idxmax(),
                }
            )
            truth_status.append(
                {
                    "subtype": subtype,
                    "hemisphere": target.hemi_lower,
                    "year": target.year,
                    "season_start": start.date().isoformat(),
                    "season_end": end.date().isoformat(),
                    "collection_rows": len(season_rows),
                    "valid_truth_rows": int(label_counts.sum()),
                    "truth_subclade": label_counts.idxmax(),
                    "truth_count": int(label_counts.max()),
                    "status": "ok",
                }
            )

        key = subtype.lower()
        pd.DataFrame(count_rows).drop_duplicates().to_csv(
            EXP_DATA / "subclade_counts" / f"submission_collection_subclade_count_{key}.csv",
            index=False,
        )

    pd.DataFrame(labels).to_csv(EXP_DATA / "futureflu_rank" / "circulating_subclade.csv", index=False)
    pd.DataFrame(truth_status).to_csv(EXP_REPORTS / "truth_status_season_end.csv", index=False)


def run_aux_with_experiment_truth() -> None:
    old_data_out = repro.DATA_OUT
    old_risk_out = repro.RISK_OUT
    try:
        repro.DATA_OUT = EXP_DATA
        repro.RISK_OUT = EXP_OUT
        comp_path = (
            PACKAGE_ROOT
            / "outputs"
            / "predictions"
            / "risk_components"
            / "risk_mutation_group_component.csv"
        )
        comp = repro.filtered_component_df(pd.read_csv(comp_path))
        raw_max_df, inform = repro.compute_max_tables(comp)
        truth = pd.read_csv(EXP_DATA / "futureflu_rank" / "circulating_subclade.csv")
        fit_cols = [f"fit_{metric}" for metric in repro.METRICS]
        norm_max_df = repro.normalize_fit_columns(raw_max_df, fit_cols)
        norm_max_df.to_csv(EXP_OUT / "subclade_accuracy" / "subclade_component_max.csv", index=False)
        repro.compute_metric_accuracy(raw_max_df, truth).to_csv(
            EXP_OUT / "subclade_accuracy" / "subclade_component_acc.csv",
            index=False,
        )
        acc_df, egdfit_df, egdtemp_df, lpd_df, pre_act_df = repro.compute_combination_outputs(
            raw_max_df, truth
        )
        combine_dir = EXP_OUT / "component_combinations"
        acc_df.to_csv(combine_dir / "subclade_component_combine_acc_Twindow.csv", index=False)
        egdfit_df.to_csv(combine_dir / "EGD_combine_Twindow.csv", index=False)
        egdtemp_df.to_csv(combine_dir / "EGD_temperatures_Twindow.csv", index=False)
        pre_act_df.to_csv(combine_dir / "subclade_pre_act_Twindow.csv", index=False)
        elpd_df, aic_df = repro.build_elpd_aic(lpd_df)
        elpd_df.to_csv(combine_dir / "elpd_Twindow.csv", index=False)
        aic_df.to_csv(combine_dir / "aic_Twindow.csv", index=False)
        inform["mutual_information"].to_csv(combine_dir / "divergence_Twindow.csv", index=False)
        inform["total_escape"].to_csv(combine_dir / "escape_Twindow.csv", index=False)
        inform["predicted_prevalence"].to_csv(combine_dir / "growth_Twindow.csv", index=False)
    finally:
        repro.DATA_OUT = old_data_out
        repro.RISK_OUT = old_risk_out


def write_comparison_report() -> None:
    original_truth = pd.read_csv(PACKAGE_ROOT / "data" / "futureflu_rank" / "circulating_subclade.csv")
    experiment_truth = pd.read_csv(EXP_DATA / "futureflu_rank" / "circulating_subclade.csv")
    truth_cmp = original_truth.merge(
        experiment_truth,
        on=["subtype", "hemisphere", "year"],
        how="outer",
        suffixes=("_original", "_season_end"),
    )
    truth_cmp["changed"] = truth_cmp["subclade_original"].fillna("") != truth_cmp["subclade_season_end"].fillna("")
    truth_cmp.to_csv(EXP_REPORTS / "truth_comparison.csv", index=False)

    original_egd = pd.read_csv(
        PACKAGE_ROOT
        / "outputs"
        / "predictions"
        / "risk_components"
        / "component_combinations"
        / "EGD_temperatures_Twindow.csv"
    )
    experiment_egd = pd.read_csv(
        EXP_OUT
        / "component_combinations"
        / "EGD_temperatures_Twindow.csv"
    )
    original_egd = original_egd.rename(
        columns={"TOP1_subclade": "TOP1_subclade_original", "T/F": "TF_original"}
    )
    experiment_egd = experiment_egd.rename(
        columns={"TOP1_subclade": "TOP1_subclade_season_end", "T/F": "TF_season_end"}
    )
    cols = ["subtype", "hemisphere", "year"]
    merged = original_egd[cols + ["TOP1_subclade_original", "TF_original"]].merge(
        experiment_egd[cols + ["TOP1_subclade_season_end", "TF_season_end"]],
        on=cols,
        how="outer",
    )
    merged = merged.merge(experiment_truth.rename(columns={"subclade": "truth_season_end"}), on=cols, how="left")
    merged.to_csv(EXP_REPORTS / "EGD_fit_top1_comparison.csv", index=False)

    original_acc = pd.read_csv(
        PACKAGE_ROOT
        / "outputs"
        / "predictions"
        / "risk_components"
        / "component_combinations"
        / "subclade_component_combine_acc_Twindow.csv"
    )
    experiment_acc = pd.read_csv(
        EXP_OUT
        / "component_combinations"
        / "subclade_component_combine_acc_Twindow.csv"
    )
    original_edg_fit = original_acc[
        (original_acc["metric_combine"] == "E+G+D") & (original_acc["methods"] == "fit")
    ].copy()
    experiment_edg_fit = experiment_acc[
        (experiment_acc["metric_combine"] == "E+G+D") & (experiment_acc["methods"] == "fit")
    ].copy()
    acc_cmp = original_edg_fit.merge(
        experiment_edg_fit,
        on=["subtype", "metric_combine", "methods"],
        how="outer",
        suffixes=("_original", "_season_end"),
    )
    acc_cmp.to_csv(EXP_REPORTS / "EDG_fit_accuracy_comparison.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the standard command-line parser. / 构建标准命令行解析器。"""
    return argparse.ArgumentParser(
        description="Run the truth season-end subclade sensitivity experiment."
    )


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    ensure_dirs()
    targets = targets_with_truth_data()
    pd.DataFrame(
        [
            {
                "subtype": target.subtype,
                "hemisphere": target.hemi_lower,
                "year": target.year,
                "season_start": target.season_start.date().isoformat(),
                "season_end_cutoff": target.season_end_cutoff.isoformat(),
            }
            for target in targets
        ]
    ).to_csv(EXP_REPORTS / "truth_targets_with_available_data.csv", index=False)

    write_season_end_definitions(targets)
    run_season_end_nextclade(targets)
    run_annotations(targets)
    run_counts_and_truth(targets)
    run_aux_with_experiment_truth()
    write_comparison_report()
    print(f"[done] wrote experiment outputs to {EXP_ROOT}")


if __name__ == "__main__":
    main()
