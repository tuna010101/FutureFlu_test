#!/usr/bin/env python3
"""Create per-season HA1 FutureFlu component correctness tables.

English: Summarize component predictions by subtype, hemisphere, and season.
中文：按亚型、半球和季节汇总各组件预测结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SUBTYPE_ORDER = ["H1N1", "H3N2", "Victoria"]
HEM_ORDER = ["south", "north"]
YEARS = list(range(2013, 2025))
SEASON_ORDER = [(year, hemi) for year in YEARS for hemi in HEM_ORDER]
SEASON_RANK = {(year, hemi): i for i, (year, hemi) in enumerate(SEASON_ORDER)}

COMBO_ORDER = ["E", "G", "D", "E+G", "E+D", "G+D", "E+G+D"]
METHOD_ORDER = ["fit", "freq"]
COMBO_TYPE = {
    "E": "single",
    "G": "single",
    "D": "single",
    "E+G": "two",
    "E+D": "two",
    "G+D": "two",
    "E+G+D": "three",
}
COMPONENT_NAME = {
    "E": "total_escape",
    "G": "predicted_prevalence",
    "D": "mutual_information",
    "E+G": "total_escape + predicted_prevalence",
    "E+D": "total_escape + mutual_information",
    "G+D": "predicted_prevalence + mutual_information",
    "E+G+D": "total_escape + predicted_prevalence + mutual_information",
}
EVESCAPE_COMPONENTS = [
    ("total_escape", "fit_total_escape"),
    ("fitness_eve", "fit_fitness_eve"),
    ("dissimilarity_charge_hydro", "fit_dissimilarity_charge_hydro"),
    ("accessibility_wcn", "fit_accessibility_wcn"),
    ("antigenic_novelty", "fit_antigenic_novelty"),
]


def season_label(year: int, hemisphere: str) -> str:
    return f"{year}-S" if hemisphere == "south" else f"{year}-N"


def ordered(df: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    result = df.copy()
    subtype_rank = {subtype: i for i, subtype in enumerate(SUBTYPE_ORDER)}
    result["_subtype_rank"] = result["subtype"].map(subtype_rank)
    result["_season_rank"] = [
        SEASON_RANK.get((int(year), hemi), 999)
        for year, hemi in zip(result["year"], result["hemisphere"])
    ]
    sort_cols = ["_subtype_rank", "_season_rank"]
    if extra_cols:
        sort_cols.extend(extra_cols)
    return result.sort_values(sort_cols).drop(columns=["_subtype_rank", "_season_rank"])


def load_futureflu_predictions(prediction_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(prediction_path)
    ff = pred[(pred["mode"] == "baseline_ha1") & pred["combo"].isin(COMBO_ORDER)].copy()
    ff["combo_type"] = ff["combo"].map(COMBO_TYPE)
    ff["component_definition"] = ff["combo"].map(COMPONENT_NAME)
    ff["season_label"] = [season_label(int(year), hemi) for year, hemi in zip(ff["year"], ff["hemisphere"])]
    ff["_combo_rank"] = ff["combo"].map({combo: i for i, combo in enumerate(COMBO_ORDER)})
    ff["_method_rank"] = ff["method"].map({method: i for i, method in enumerate(METHOD_ORDER)})
    ff = ordered(ff, ["_combo_rank", "_method_rank"]).drop(columns=["_combo_rank", "_method_rank"])
    cols = [
        "subtype",
        "hemisphere",
        "year",
        "season_label",
        "combo_type",
        "combo",
        "component_definition",
        "method",
        "truth_clade",
        "pred_clade",
        "score",
        "correct",
    ]
    return ff[cols]


def build_truth_table(ff: pd.DataFrame) -> pd.DataFrame:
    truth = ff[["subtype", "hemisphere", "year", "truth_clade"]].drop_duplicates()
    return ordered(truth)


def _pred_missing(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "<na>"}


def fill_missing_from_previous_year(df: pd.DataFrame, series_cols: list[str]) -> pd.DataFrame:
    """Fill empty pred_clade from the same series one year earlier (same hemisphere).

    English: When a season has no prediction, reuse the previous year's value for the
    same subtype, hemisphere, and prediction series.
    中文：某季无预测时，用同亚型、同半球、同一预测系列的上一季结果补全。

    series_cols identifies one prediction series besides subtype/hemisphere/year,
    e.g. ["combo", "method"] or ["component"].
    """
    required = {"subtype", "hemisphere", "year", "pred_clade", *series_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"fill_missing_from_previous_year missing columns: {missing}")

    result = df.copy()
    result["filled_from_previous_year"] = False
    result["prediction_fill_note"] = ""

    group_cols = ["subtype", "hemisphere", *series_cols]
    for _, group in result.groupby(group_cols, sort=False):
        # Use original (pre-fill) predictions so a single pass does not chain fills.
        year_to_pred: dict[int, object] = {}
        for idx in group.index:
            year = int(result.at[idx, "year"])
            pred = result.at[idx, "pred_clade"]
            if not _pred_missing(pred):
                year_to_pred[year] = pred

        for idx in group.index:
            if not _pred_missing(result.at[idx, "pred_clade"]):
                continue
            year = int(result.at[idx, "year"])
            prev = year_to_pred.get(year - 1)
            if prev is None or _pred_missing(prev):
                continue
            result.at[idx, "pred_clade"] = prev
            result.at[idx, "filled_from_previous_year"] = True
            result.at[idx, "prediction_fill_note"] = (
                f"filled from same-hemisphere previous-year prediction ({year - 1})"
            )

    if "truth_clade" in result.columns:
        has_pred = ~result["pred_clade"].map(_pred_missing)
        result.loc[has_pred, "correct"] = (
            result.loc[has_pred, "pred_clade"] == result.loc[has_pred, "truth_clade"]
        )
        result.loc[~has_pred, "correct"] = False

    if "available_prediction" in result.columns:
        result["available_prediction"] = ~result["pred_clade"].map(_pred_missing)

    return result


def predict_evescape_components(max_path: Path, truth: pd.DataFrame) -> pd.DataFrame:
    maxdf = pd.read_csv(max_path)
    rows = []
    for (subtype, hemi, year), label in truth.groupby(["subtype", "hemisphere", "year"], sort=False):
        truth_clade = label["truth_clade"].iloc[0]
        group = maxdf[
            (maxdf["subtype"] == subtype)
            & (maxdf["hemisphere"] == hemi)
            & (maxdf["year"] == year)
        ]
        for component, col in EVESCAPE_COMPONENTS:
            n_candidate = int(group["clade_single"].nunique()) if not group.empty else 0
            if group.empty or col not in group.columns or group[col].isna().all():
                rows.append(
                    {
                        "subtype": subtype,
                        "hemisphere": hemi,
                        "year": year,
                        "season_label": season_label(int(year), hemi),
                        "source": "EVEscape",
                        "component": component,
                        "score_column": col,
                        "truth_clade": truth_clade,
                        "pred_clade": np.nan,
                        "score": np.nan,
                        "correct": False,
                        "available_prediction": False,
                        "n_candidate_clades": n_candidate,
                        "n_tied_max": 0,
                    }
                )
                continue
            idx = group[col].idxmax()
            max_score = group.loc[idx, col]
            pred_clade = group.loc[idx, "clade_single"]
            rows.append(
                {
                    "subtype": subtype,
                    "hemisphere": hemi,
                    "year": year,
                    "season_label": season_label(int(year), hemi),
                    "source": "EVEscape",
                    "component": component,
                    "score_column": col,
                    "truth_clade": truth_clade,
                    "pred_clade": pred_clade,
                    "score": max_score,
                    "correct": bool(pred_clade == truth_clade),
                    "available_prediction": True,
                    "n_candidate_clades": n_candidate,
                    "n_tied_max": int((group[col] == max_score).sum()),
                }
            )
    eve = pd.DataFrame(rows)
    eve["_component_rank"] = eve["component"].map(
        {component: i for i, (component, _) in enumerate(EVESCAPE_COMPONENTS)}
    )
    return ordered(eve, ["_component_rank"]).drop(columns=["_component_rank"])


def combine_tables(ff: pd.DataFrame, eve: pd.DataFrame) -> pd.DataFrame:
    plot_ff = ff.copy()
    plot_ff["source"] = "FutureFlu_HA1"
    plot_ff["display_method"] = "FutureFlu " + plot_ff["combo"] + " " + plot_ff["method"]
    plot_ff["available_prediction"] = ~plot_ff["pred_clade"].map(_pred_missing)
    plot_ff["component"] = plot_ff["combo"]
    plot_ff["score_column"] = plot_ff["combo"] + "_pre_" + plot_ff["method"]
    plot_ff["n_candidate_clades"] = np.nan
    plot_ff["n_tied_max"] = np.nan
    if "filled_from_previous_year" not in plot_ff.columns:
        plot_ff["filled_from_previous_year"] = False
    if "prediction_fill_note" not in plot_ff.columns:
        plot_ff["prediction_fill_note"] = ""
    cols = [
        "source",
        "subtype",
        "hemisphere",
        "year",
        "season_label",
        "display_method",
        "combo_type",
        "combo",
        "component",
        "component_definition",
        "method",
        "score_column",
        "truth_clade",
        "pred_clade",
        "score",
        "correct",
        "available_prediction",
        "filled_from_previous_year",
        "prediction_fill_note",
        "n_candidate_clades",
        "n_tied_max",
    ]
    plot_ff = plot_ff[cols]

    plot_eve = eve.copy()
    plot_eve["display_method"] = "EVEscape " + plot_eve["component"]
    plot_eve["combo_type"] = "evescape_single_component"
    plot_eve["combo"] = ""
    plot_eve["component_definition"] = plot_eve["component"]
    plot_eve["method"] = "fit_max"
    if "filled_from_previous_year" not in plot_eve.columns:
        plot_eve["filled_from_previous_year"] = False
    if "prediction_fill_note" not in plot_eve.columns:
        plot_eve["prediction_fill_note"] = ""
    plot_eve = plot_eve[cols]
    return ordered(pd.concat([plot_ff, plot_eve], ignore_index=True), ["source", "display_method"])


def write_wide_tables(out_dir: Path, ff: pd.DataFrame, eve: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    wide = truth.copy()
    wide["season_label"] = [season_label(int(year), hemi) for year, hemi in zip(wide["year"], wide["hemisphere"])]
    for method in METHOD_ORDER:
        for combo in COMBO_ORDER:
            sub = ff[(ff["method"] == method) & (ff["combo"] == combo)][
                ["subtype", "hemisphere", "year", "pred_clade", "correct"]
            ].copy()
            sub = sub.rename(
                columns={
                    "pred_clade": f"FutureFlu_{combo}_{method}_pred",
                    "correct": f"FutureFlu_{combo}_{method}_correct",
                }
            )
            wide = wide.merge(sub, on=["subtype", "hemisphere", "year"], how="left")
    for component, _ in EVESCAPE_COMPONENTS:
        sub = eve[eve["component"] == component][
            ["subtype", "hemisphere", "year", "pred_clade", "correct", "available_prediction", "n_tied_max"]
        ].copy()
        sub = sub.rename(
            columns={
                "pred_clade": f"EVEscape_{component}_pred",
                "correct": f"EVEscape_{component}_correct",
                "available_prediction": f"EVEscape_{component}_available",
                "n_tied_max": f"EVEscape_{component}_n_tied_max",
            }
        )
        wide = wide.merge(sub, on=["subtype", "hemisphere", "year"], how="left")
    wide = ordered(wide)
    wide.to_csv(out_dir / "ha1_component_predictions_wide_by_season.csv", index=False)
    for subtype in SUBTYPE_ORDER:
        wide[wide["subtype"] == subtype].to_csv(
            out_dir / f"ha1_component_predictions_wide_by_season_{subtype}.csv", index=False
        )
    return wide


def write_accuracy_summary(out_dir: Path, combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in ["EVEscape", "FutureFlu_HA1"]:
        source_df = combined[combined["source"] == source]
        for (subtype, display), group in source_df.groupby(["subtype", "display_method"], sort=False):
            total = len(group)
            correct = int(group["correct"].fillna(False).sum())
            rows.append(
                {
                    "source": source,
                    "subtype": subtype,
                    "display_method": display,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total else np.nan,
                    "available_predictions": int(group["available_prediction"].fillna(False).sum()),
                }
            )
        for display, group in source_df.groupby("display_method", sort=False):
            total = len(group)
            correct = int(group["correct"].fillna(False).sum())
            rows.append(
                {
                    "source": source,
                    "subtype": "All",
                    "display_method": display,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total else np.nan,
                    "available_predictions": int(group["available_prediction"].fillna(False).sum()),
                }
            )
    summary = pd.DataFrame(rows)
    rank = {subtype: i for i, subtype in enumerate(SUBTYPE_ORDER)}
    rank["All"] = 99
    summary["_subtype_rank"] = summary["subtype"].map(rank)
    summary = summary.sort_values(["source", "_subtype_rank", "display_method"]).drop(columns=["_subtype_rank"])
    summary.to_csv(out_dir / "ha1_component_accuracy_summary.csv", index=False)
    return summary


def status_cell(group: pd.DataFrame) -> str:
    if group.empty:
        return "NA"
    row = group.iloc[0]
    if not bool(row["available_prediction"]):
        return "NA"
    return "OK" if bool(row["correct"]) else "X"


def detail_cell(group: pd.DataFrame) -> str:
    if group.empty:
        return "NA"
    row = group.iloc[0]
    if not bool(row["available_prediction"]):
        return f"NA: pred=; truth={row['truth_clade']}"
    status = "OK" if bool(row["correct"]) else "X"
    return f"{status}: pred={row['pred_clade']}; truth={row['truth_clade']}"


def write_matrix_tables(out_dir: Path, combined: pd.DataFrame) -> None:
    all_status = []
    all_detail = []
    for subtype in SUBTYPE_ORDER:
        for method in METHOD_ORDER:
            row_defs = [
                (f"FutureFlu {combo} {method}", "FutureFlu_HA1", f"FutureFlu {combo} {method}", COMBO_TYPE[combo])
                for combo in COMBO_ORDER
            ]
            row_defs.extend(
                [
                    (
                        f"EVEscape {component}",
                        "EVEscape",
                        f"EVEscape {component}",
                        "evescape_single_component",
                    )
                    for component, _ in EVESCAPE_COMPONENTS
                ]
            )
            status_rows = []
            detail_rows = []
            for row_label, source, display, component_type in row_defs:
                status = {
                    "subtype": subtype,
                    "method_set": method,
                    "row": row_label,
                    "component_type": component_type,
                }
                detail = dict(status)
                for year, hemi in SEASON_ORDER:
                    group = combined[
                        (combined["subtype"] == subtype)
                        & (combined["source"] == source)
                        & (combined["display_method"] == display)
                        & (combined["year"] == year)
                        & (combined["hemisphere"] == hemi)
                    ]
                    label = season_label(year, hemi)
                    status[label] = status_cell(group)
                    detail[label] = detail_cell(group)
                status_rows.append(status)
                detail_rows.append(detail)
            status_df = pd.DataFrame(status_rows)
            detail_df = pd.DataFrame(detail_rows)
            status_df.to_csv(out_dir / f"correctness_matrix_{subtype}_{method}.csv", index=False)
            detail_df.to_csv(out_dir / f"prediction_detail_matrix_{subtype}_{method}.csv", index=False)
            all_status.append(status_df)
            all_detail.append(detail_df)
    pd.concat(all_status, ignore_index=True).to_csv(out_dir / "correctness_matrix_all_subtypes.csv", index=False)
    pd.concat(all_detail, ignore_index=True).to_csv(out_dir / "prediction_detail_matrix_all_subtypes.csv", index=False)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_root = repo_root / "experiments" / "ha_na_full_length"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=experiment_root,
        help="HA/NA full-length experiment directory containing results/ and runs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "ha1_components",
        help="Directory for generated tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    experiment_root = Path(args.experiment_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = experiment_root / "results" / "predictions_vs_truth_long.csv"
    max_path = (
        repo_root
        / "outputs"
        / "predictions"
        / "risk_components"
        / "clade_accuracy"
        / "clade_component_max.csv"
    )

    ff = load_futureflu_predictions(prediction_path)
    ff = fill_missing_from_previous_year(ff, ["combo", "method"])
    truth = build_truth_table(ff)
    eve = predict_evescape_components(max_path, truth)
    eve = fill_missing_from_previous_year(eve, ["component"])
    combined = combine_tables(ff, eve)

    n_ff_filled = int(ff["filled_from_previous_year"].sum())
    n_eve_filled = int(eve["filled_from_previous_year"].sum())

    ff.to_csv(out_dir / "futureflu_ha1_season_predictions_long.csv", index=False)
    eve.to_csv(out_dir / "evescape_ha1_component_predictions_long.csv", index=False)
    combined.to_csv(out_dir / "all_ha1_component_season_correctness_long.csv", index=False)
    write_wide_tables(out_dir, ff, eve, truth)
    write_accuracy_summary(out_dir, combined)
    write_matrix_tables(out_dir, combined)

    print(f"wrote {out_dir}")
    print(f"FutureFlu rows: {len(ff)} (filled from previous year: {n_ff_filled})")
    print(f"EVEscape rows: {len(eve)} (filled from previous year: {n_eve_filled})")
    print(f"Combined rows: {len(combined)}")


if __name__ == "__main__":
    main()
