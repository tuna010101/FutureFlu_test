"""Compare Top1 forecast strains with clade truth tables.

English: Selects one forecast strain per method and summarizes clade agreement.
中文：为每个方法选择一个预测 strain，并汇总 clade 匹配情况。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUTUREFLU_RESULTS_ROOT = PROJECT_ROOT / "results" / "futureflu"
FUTUREFLU_DATA_ROOT = PROJECT_ROOT / "data" / "futureflu"

DEFAULT_RUN_DIR = FUTUREFLU_RESULTS_ROOT / "runs" / "H3N2"
DEFAULT_TRUTH = FUTUREFLU_DATA_ROOT / "truth" / "top1_truth.csv"


def public_path(path: Path) -> str:
    """Return a project-relative path when possible.

    English: Relative manifest paths make the output portable.
    中文：manifest 使用相对路径，便于输出目录迁移。
    """
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the eLife-style Top1 strain per FutureFlu forecast, map it "
            "to its clade, and compare predicted clades to truth.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--virus", default="H3N2")
    parser.add_argument("--score-column", default="y", help="column used to rank Top1 strains")
    parser.add_argument(
        "--score-direction",
        choices=["min", "max"],
        default="min",
        help="use min for distance-like scores and max for reward-like scores",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def parse_forecast_name(path: Path) -> tuple[str, str]:
    match = re.match(r"forecasts_(north|south)_(.+)\.tsv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse forecast file name: {path.name}")
    return match.group(1), match.group(2)


def load_truth(truth_path: Path, virus: str) -> pd.DataFrame:
    truth = pd.read_csv(truth_path)
    virus_column = truth.columns[0]
    row = truth[truth[virus_column].astype(str).str.upper() == virus.upper()]
    if row.empty:
        raise ValueError(f"Virus {virus!r} not found in {truth_path}")

    records = []
    for column, value in row.iloc[0].items():
        match = re.match(r"^(\d{4})\s+(north|south)$", str(column))
        if not match:
            continue
        records.append(
            {
                "virus": virus,
                "forecast_year": int(match.group(1)),
                "hemisphere": match.group(2),
                "truth_clade": value,
            }
        )

    if not records:
        raise ValueError(f"No '<year> <hemisphere>' truth columns found in {truth_path}")
    return pd.DataFrame(records)


def load_tip_annotations(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "tip_attributes_with_weighted_distances.tsv"
    if not path.exists():
        path = run_dir / "tip_attributes_with_naive.tsv"
    if not path.exists():
        raise FileNotFoundError(f"No tip attributes file found under {run_dir}")

    candidate_columns = [
        "timepoint",
        "strain",
        "clade",
        "clade_membership",
        "frequency",
        "submission_date",
        "collection_date",
        "region",
        "country",
    ]
    header = pd.read_csv(path, sep="\t", nrows=0)
    columns = [column for column in candidate_columns if column in header.columns]
    tips = pd.read_csv(path, sep="\t", usecols=columns, parse_dates=["timepoint"])
    tips["timepoint"] = tips["timepoint"].dt.strftime("%Y-%m-%d")
    return tips.drop_duplicates(subset=["timepoint", "strain"])


def load_nested_hash_clades(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "tip_clades_all_timepoints.tsv"
    if not path.exists():
        # Explicit dtypes: pandas 0.x/1.x can raise on DataFrame(columns=...) alone.
        return pd.DataFrame(
            {
                "timepoint": pd.Series(dtype="object"),
                "strain": pd.Series(dtype="object"),
                "nested_hash_clades": pd.Series(dtype="object"),
            }
        )

    clades = pd.read_csv(path, sep="\t", parse_dates=["timepoint"])
    clades["timepoint"] = clades["timepoint"].dt.strftime("%Y-%m-%d")
    clades = clades.sort_values(["timepoint", "strain", "depth", "clade_membership"])
    clades["depth_hash"] = clades["depth"].astype(int).astype(str) + ":" + clades["clade_membership"].astype(str)
    nested = (
        clades.groupby(["timepoint", "strain"], as_index=False)
        .agg(nested_hash_clades=("depth_hash", ";".join))
    )
    return nested


def select_top1_from_forecast(
    forecast_path: Path,
    score_column: str,
    score_direction: str,
) -> pd.DataFrame:
    hemisphere_from_name, model_from_name = parse_forecast_name(forecast_path)
    forecasts = pd.read_csv(forecast_path, sep="\t", parse_dates=["timepoint", "future_timepoint"])
    if score_column not in forecasts.columns:
        raise ValueError(f"{forecast_path} does not contain score column {score_column!r}")

    forecasts["timepoint"] = forecasts["timepoint"].dt.strftime("%Y-%m-%d")
    forecasts["future_timepoint"] = forecasts["future_timepoint"].dt.strftime("%Y-%m-%d")
    if "hemisphere" not in forecasts.columns:
        forecasts["hemisphere"] = hemisphere_from_name
    if "model" not in forecasts.columns:
        forecasts["model"] = model_from_name
    if "forecast_year" not in forecasts.columns:
        forecasts["forecast_year"] = pd.to_datetime(forecasts["future_timepoint"]).dt.year

    group_columns = ["hemisphere", "forecast_year", "timepoint", "future_timepoint", "model"]
    ascending = score_direction == "min"
    sort_columns = group_columns + [score_column, "projected_frequency", "strain"]
    sort_ascending = [True] * len(group_columns) + [ascending, False, True]
    sorted_forecasts = forecasts.sort_values(sort_columns, ascending=sort_ascending, na_position="last")
    top1 = sorted_forecasts.groupby(group_columns, as_index=False).first()

    output_columns = [
        "hemisphere",
        "forecast_year",
        "timepoint",
        "future_timepoint",
        "model",
        "strain",
        "frequency",
        "projected_frequency",
        score_column,
    ]
    optional_columns = ["fitness", "weighted_distance_to_future", "weighted_distance_to_present"]
    output_columns.extend([column for column in optional_columns if column in top1.columns])
    top1 = top1.loc[:, output_columns].copy()
    top1 = top1.rename(
        columns={
            "timepoint": "issue_timepoint",
            "strain": "top1_strain",
            "frequency": "top1_initial_frequency",
            "projected_frequency": "top1_projected_frequency",
            score_column: "top1_score",
        }
    )
    top1["source_forecast_file"] = forecast_path.name
    top1["rank_metric"] = score_column
    top1["rank_direction"] = score_direction
    return top1


def collect_top1_predictions(
    run_dir: Path,
    truth: pd.DataFrame,
    score_column: str,
    score_direction: str,
) -> pd.DataFrame:
    tips = load_tip_annotations(run_dir)
    nested_hash_clades = load_nested_hash_clades(run_dir)

    frames = []
    for forecast_path in sorted((run_dir / "forecasts").glob("forecasts_*.tsv")):
        frames.append(select_top1_from_forecast(forecast_path, score_column, score_direction))
    if not frames:
        raise FileNotFoundError(f"No forecasts_*.tsv files found under {run_dir / 'forecasts'}")

    top1 = pd.concat(frames, ignore_index=True)
    top1 = top1.merge(
        tips,
        how="left",
        left_on=["issue_timepoint", "top1_strain"],
        right_on=["timepoint", "strain"],
    )
    top1 = top1.drop(columns=[column for column in ["timepoint", "strain"] if column in top1.columns])
    top1 = top1.rename(
        columns={
            "clade": "predicted_clade",
            "clade_membership": "predicted_hash_clade_membership",
        }
    )
    top1 = top1.merge(
        nested_hash_clades,
        how="left",
        left_on=["issue_timepoint", "top1_strain"],
        right_on=["timepoint", "strain"],
    )
    top1 = top1.drop(columns=[column for column in ["timepoint", "strain"] if column in top1.columns])
    top1 = top1.merge(truth, how="left", on=["forecast_year", "hemisphere"])
    top1["exact_clade_match"] = top1["predicted_clade"].astype(str) == top1["truth_clade"].astype(str)
    return top1.sort_values(["hemisphere", "model", "forecast_year"]).reset_index(drop=True)


def make_color_map(values: list[str]) -> dict[str, str]:
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    color_map = {}
    for index, value in enumerate(values):
        color_map[value] = mcolors.to_hex(palette[index % len(palette)])
    color_map["N/A"] = "#dddddd"
    return color_map


def text_color_for_background(color: str) -> str:
    red, green, blue = mcolors.to_rgb(color)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance > 0.58 else "white"


def plot_truth_prediction_tiles(
    rows: pd.DataFrame,
    output_path: Path,
    color_map: dict[str, str],
    title: str,
    dpi: int,
) -> None:
    years = sorted(rows["forecast_year"].unique())
    truth_values = []
    predicted_values = []
    for year in years:
        year_rows = rows[rows["forecast_year"] == year]
        truth_values.append(str(year_rows["truth_clade"].iloc[0]) if len(year_rows) else "N/A")
        predicted = year_rows["predicted_clade"].iloc[0] if len(year_rows) else "N/A"
        predicted_values.append("N/A" if pd.isna(predicted) else str(predicted))

    fig_width = max(10, len(years) * 1.05)
    fig, ax = plt.subplots(figsize=(fig_width, 2.8))
    for row_index, values in enumerate([truth_values, predicted_values]):
        for col_index, value in enumerate(values):
            color = color_map.get(value, "#dddddd")
            rect = plt.Rectangle((col_index, row_index), 1, 1, facecolor=color, edgecolor="white")
            ax.add_patch(rect)
            ax.text(
                col_index + 0.5,
                row_index + 0.5,
                value,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color_for_background(color),
                wrap=True,
            )

    ax.set_xlim(0, len(years))
    ax.set_ylim(0, 2)
    ax.set_xticks([index + 0.5 for index in range(len(years))])
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["truth", "predicted"])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_all(top1: pd.DataFrame, output_dir: Path, dpi: int) -> dict[str, str]:
    figures_dir = output_dir / "figures"
    values = sorted(
        {
            str(value)
            for value in pd.concat([top1["truth_clade"], top1["predicted_clade"]]).dropna().unique()
        }
    )
    color_map = make_color_map(values)

    figure_paths = []
    for (hemisphere, model), rows in top1.groupby(["hemisphere", "model"]):
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model))
        output_path = figures_dir / f"top1_clade_{hemisphere}_{safe_model}.png"
        title = f"{hemisphere} | {model} | Top1 by {rows['rank_metric'].iloc[0]} ({rows['rank_direction'].iloc[0]})"
        plot_truth_prediction_tiles(rows, output_path, color_map, title, dpi)
        figure_paths.append(public_path(output_path))

    pdf_path = output_dir / "top1_clade_comparison_all_methods.pdf"
    with PdfPages(pdf_path) as pdf:
        for (hemisphere, model), rows in top1.groupby(["hemisphere", "model"]):
            years = sorted(rows["forecast_year"].unique())
            truth_values = []
            predicted_values = []
            for year in years:
                year_rows = rows[rows["forecast_year"] == year]
                truth_values.append(str(year_rows["truth_clade"].iloc[0]) if len(year_rows) else "N/A")
                predicted = year_rows["predicted_clade"].iloc[0] if len(year_rows) else "N/A"
                predicted_values.append("N/A" if pd.isna(predicted) else str(predicted))

            fig_width = max(10, len(years) * 1.05)
            fig, ax = plt.subplots(figsize=(fig_width, 2.8))
            for row_index, row_values in enumerate([truth_values, predicted_values]):
                for col_index, value in enumerate(row_values):
                    color = color_map.get(value, "#dddddd")
                    ax.add_patch(
                        plt.Rectangle((col_index, row_index), 1, 1, facecolor=color, edgecolor="white")
                    )
                    ax.text(
                        col_index + 0.5,
                        row_index + 0.5,
                        value,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color_for_background(color),
                        wrap=True,
                    )
            ax.set_xlim(0, len(years))
            ax.set_ylim(0, 2)
            ax.set_xticks([index + 0.5 for index in range(len(years))])
            ax.set_xticklabels(years, rotation=45, ha="right")
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(["truth", "predicted"])
            ax.invert_yaxis()
            ax.set_title(f"{hemisphere} | {model}")
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return {
        "individual_pngs": figure_paths,
        "all_methods_pdf": public_path(pdf_path),
        "color_map": color_map,
    }


def summarize_by_model_overall(top1: pd.DataFrame) -> pd.DataFrame:
    # Use groupby().agg(...).reset_index(): pandas 0.25 drops keys with as_index=False + named agg.
    return (
        top1.groupby(["model"])
        .agg(
            n_forecasts=("forecast_year", "count"),
            exact_clade_matches=("exact_clade_match", "sum"),
            mean_top1_projected_frequency=("top1_projected_frequency", "mean"),
            mean_top1_score=("top1_score", "mean"),
        )
        .reset_index()
        .assign(
            exact_clade_accuracy=lambda df: df["exact_clade_matches"] / df["n_forecasts"],
        )
        .sort_values(["exact_clade_accuracy", "model"], ascending=[False, True])
        .reset_index(drop=True)
    )


def get_time_ordered_events(top1: pd.DataFrame) -> pd.DataFrame:
    events = top1.loc[
        :,
        ["forecast_year", "hemisphere", "issue_timepoint", "future_timepoint", "truth_clade"],
    ].drop_duplicates()
    events["issue_timepoint_dt"] = pd.to_datetime(events["issue_timepoint"])
    events = events.sort_values(["issue_timepoint_dt", "hemisphere"]).reset_index(drop=True)
    events["event_id"] = events.apply(
        lambda row: f"{int(row.forecast_year)}_{row.hemisphere}",
        axis=1,
    )
    events["event_label"] = events.apply(
        lambda row: f"{int(row.forecast_year)} {row.hemisphere}\nissue {row.issue_timepoint}",
        axis=1,
    )
    return events


def build_all_methods_matrix(top1: pd.DataFrame, overall_summary: pd.DataFrame) -> pd.DataFrame:
    events = get_time_ordered_events(top1)
    event_ids = events["event_id"].tolist()
    event_labels = events["event_label"].tolist()
    label_by_event_id = dict(zip(event_ids, event_labels))

    truth_row = {
        "row_type": "truth",
        "model": "truth.csv",
        "exact_accuracy": "",
        "exact_matches": "",
        "n_forecasts": "",
    }
    for event_id, truth_clade in zip(events["event_id"], events["truth_clade"]):
        truth_row[event_id] = truth_clade

    rows = [truth_row]
    ordered_models = overall_summary["model"].tolist()
    for model in ordered_models:
        model_rows = top1[top1["model"] == model]
        summary_row = overall_summary[overall_summary["model"] == model].iloc[0]
        row = {
            "row_type": "prediction",
            "model": model,
            "exact_accuracy": f"{summary_row.exact_clade_accuracy:.1%}",
            "exact_matches": f"{int(summary_row.exact_clade_matches)}/{int(summary_row.n_forecasts)}",
            "n_forecasts": int(summary_row.n_forecasts),
        }
        for event in events.itertuples(index=False):
            match = model_rows[
                (model_rows["forecast_year"] == event.forecast_year)
                & (model_rows["hemisphere"] == event.hemisphere)
            ]
            if len(match):
                value = match["predicted_clade"].iloc[0]
                row[event.event_id] = "N/A" if pd.isna(value) else value
            else:
                row[event.event_id] = "N/A"
        rows.append(row)

    matrix = pd.DataFrame(rows)
    matrix = matrix.loc[
        :,
        [
            "row_type",
            "model",
            *event_ids,
            "exact_accuracy",
            "exact_matches",
            "n_forecasts",
        ],
    ]
    # Return labels separately: DataFrame.attrs needs pandas>=1.0.
    return matrix, label_by_event_id


def plot_all_methods_time_ordered_matrix(
    matrix: pd.DataFrame,
    output_path: Path,
    color_map: dict[str, str],
    title: str,
    dpi: int,
    event_label_by_id: dict[str, str] | None = None,
) -> None:
    event_columns = [
        column
        for column in matrix.columns
        if re.match(r"^\d{4}_(north|south)$", str(column))
    ]
    event_label_by_id = event_label_by_id or {}
    event_labels = [event_label_by_id.get(column, column.replace("_", " ")) for column in event_columns]
    right_columns = ["exact_accuracy"]
    data_columns = event_columns + right_columns

    n_rows = len(matrix)
    n_event_columns = len(event_columns)
    n_columns = len(data_columns)
    fig_width = max(18, n_columns * 0.78 + 3.5)
    fig_height = max(8, n_rows * 0.43 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for row_index, (_, row) in enumerate(matrix.iterrows()):
        for col_index, column in enumerate(data_columns):
            value = row[column]
            if column in event_columns:
                text_value = "N/A" if pd.isna(value) else str(value)
                facecolor = color_map.get(text_value, "#dddddd")
                text_color = text_color_for_background(facecolor)
            else:
                text_value = "" if pd.isna(value) else str(value)
                facecolor = "#f7f7f7" if row["row_type"] == "prediction" else "#ffffff"
                text_color = "black"

            ax.add_patch(
                plt.Rectangle(
                    (col_index, row_index),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            ax.text(
                col_index + 0.5,
                row_index + 0.5,
                text_value,
                ha="center",
                va="center",
                fontsize=6.2 if column in event_columns else 7.2,
                color=text_color,
                wrap=True,
            )

    ax.axvline(n_event_columns, color="#222222", linewidth=1.1)
    ax.set_xlim(0, n_columns)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([index + 0.5 for index in range(n_columns)])
    ax.set_xticklabels(event_labels + ["exact\naccuracy"], rotation=60, ha="right")
    ax.set_yticks([index + 0.5 for index in range(n_rows)])
    ax.set_yticklabels(matrix["model"].tolist())
    ax.invert_yaxis()
    ax.set_title(title, pad=16)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_all_methods_comparison(
    top1: pd.DataFrame,
    output_dir: Path,
    color_map: dict[str, str],
    dpi: int,
) -> dict[str, str]:
    overall_summary = summarize_by_model_overall(top1)
    matrix, event_label_by_id = build_all_methods_matrix(top1, overall_summary)

    matrix_path = output_dir / "all_methods_time_ordered_matrix.tsv"
    summary_path = output_dir / "top1_method_summary_overall.tsv"
    png_path = output_dir / "all_methods_time_ordered_comparison.png"
    pdf_path = output_dir / "all_methods_time_ordered_comparison.pdf"

    matrix.to_csv(matrix_path, sep="\t", index=False)
    overall_summary.to_csv(summary_path, sep="\t", index=False)
    title = "All methods Top1 strain clades vs truth, north/south sorted by issue date"
    plot_all_methods_time_ordered_matrix(
        matrix, png_path, color_map, title, dpi, event_label_by_id=event_label_by_id
    )
    plot_all_methods_time_ordered_matrix(
        matrix, pdf_path, color_map, title, dpi, event_label_by_id=event_label_by_id
    )

    return {
        "all_methods_time_ordered_matrix": public_path(matrix_path),
        "overall_method_summary": public_path(summary_path),
        "all_methods_time_ordered_png": public_path(png_path),
        "all_methods_time_ordered_pdf": public_path(pdf_path),
    }


def summarize_by_method(top1: pd.DataFrame) -> pd.DataFrame:
    # Use groupby().agg(...).reset_index(): pandas 0.25 drops keys with as_index=False + named agg.
    return (
        top1.groupby(["hemisphere", "model", "source_forecast_file"])
        .agg(
            n_forecasts=("forecast_year", "count"),
            exact_clade_matches=("exact_clade_match", "sum"),
            mean_top1_projected_frequency=("top1_projected_frequency", "mean"),
            mean_top1_score=("top1_score", "mean"),
        )
        .reset_index()
        .assign(
            exact_clade_accuracy=lambda df: df["exact_clade_matches"] / df["n_forecasts"],
        )
        .sort_values(["hemisphere", "exact_clade_accuracy", "model"], ascending=[True, False, True])
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    truth_path = args.truth.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "top1_strain_clade_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = load_truth(truth_path, args.virus)
    top1 = collect_top1_predictions(run_dir, truth, args.score_column, args.score_direction)
    summary = summarize_by_method(top1)
    figures = plot_all(top1, output_dir, args.dpi)
    all_methods_outputs = write_all_methods_comparison(
        top1,
        output_dir,
        figures["color_map"],
        args.dpi,
    )

    top1_path = output_dir / "top1_strain_clades.tsv"
    summary_path = output_dir / "top1_method_summary.tsv"
    truth_long_path = output_dir / "truth_long.tsv"
    manifest_path = output_dir / "manifest.json"

    top1.to_csv(top1_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    truth.to_csv(truth_long_path, sep="\t", index=False)

    manifest = {
        "run_dir": public_path(run_dir),
        "truth": public_path(truth_path),
        "virus": args.virus,
        "score_column": args.score_column,
        "score_direction": args.score_direction,
        "interpretation": (
            "Top1 follows the eLife-style closest-to-predicted-future logic: "
            "the default ranks strains by the smallest y, not by largest projected_frequency."
        ),
        "outputs": {
            "top1_strain_clades": public_path(top1_path),
            "method_summary": public_path(summary_path),
            "truth_long": public_path(truth_long_path),
            "figures": figures,
            "all_methods_comparison": all_methods_outputs,
        },
        "n_forecast_files": int(top1["source_forecast_file"].nunique()),
        "n_rows": int(len(top1)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {top1_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {truth_long_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {figures['all_methods_pdf']}")
    print(f"Wrote {all_methods_outputs['all_methods_time_ordered_png']}")
    print(f"Wrote {all_methods_outputs['all_methods_time_ordered_pdf']}")


if __name__ == "__main__":
    main()
