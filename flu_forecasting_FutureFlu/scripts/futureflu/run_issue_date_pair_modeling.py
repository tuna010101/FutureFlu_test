"""Fit issue-date FutureFlu forecast models.

English: Builds north/south issue-date pairs and exports forecast, error, and coefficient tables.
中文：构建南北半球 issue-date 配对，并导出 forecast、error 和 coefficient 表。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
import os
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUTUREFLU_RESULTS_ROOT = PROJECT_ROOT / "results" / "futureflu"

# English: Import shared modeling code from the external flu-forecasting checkout.
# 中文：从外部 flu-forecasting checkout 导入共享建模代码。
def _resolve_flu_root_for_modeling() -> Path:
    env_value = os.environ.get("FLU_FORECASTING_ROOT", "").strip()
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT.parent / "flu-forecasting",
            PROJECT_ROOT / "flu-forecasting",
        ]
    )
    for candidate in candidates:
        if (candidate / "src" / "fit_model.py").exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "(none)"
    raise SystemExit(
        "Cannot find flu-forecasting src/fit_model.py. "
        "Clone https://github.com/blab/flu-forecasting and set FLU_FORECASTING_ROOT. "
        f"Searched: {searched}"
    )


_flu_root = _resolve_flu_root_for_modeling()
sys.path.insert(0, str(_flu_root / "src"))
from fit_model import DistanceExponentialGrowthModel
from weighted_distances import get_distances_by_sample_names


DEFAULT_RUN_DIR = FUTUREFLU_RESULTS_ROOT / "runs" / "H3N2"
MODEL_RANDOM_SEED = 314159


def public_path(path: Path) -> str:
    """Return a repository-relative path when possible.

    English: Published manifests should be portable across machines.
    中文：发布清单使用可迁移的仓库相对路径。
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def seq_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def strain_distance(
    strain_a: str,
    strain_b: str,
    distances: dict[str, dict[str, int]] | None = None,
    sequences: dict[str, str] | None = None,
) -> int:
    if distances is not None:
        try:
            return distances[strain_a][strain_b]
        except KeyError:
            return distances[strain_b][strain_a]
    if sequences is None:
        raise ValueError("sequences are required when distances are not provided")
    return seq_distance(sequences[strain_a], sequences[strain_b])


def annotate_weighted_distances(
    df: pd.DataFrame,
    pair_map: dict[str, str],
    distances: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    grouped = {pd.to_datetime(t).strftime("%Y-%m-%d"): g.copy() for t, g in df.groupby("timepoint")}
    sequences = None if distances is not None else df.drop_duplicates("strain").set_index("strain")["aa_sequence"].to_dict()
    weighted_rows = []

    for current_time, current in grouped.items():
        future_time = pair_map.get(current_time)
        future = grouped.get(future_time) if future_time else None
        current_records = list(current[["strain", "frequency"]].itertuples(index=False, name=None))
        future_records = (
            list(future[["strain", "frequency"]].itertuples(index=False, name=None))
            if future is not None
            else []
        )

        for strain, _ in current_records:
            weighted_distance_to_present = 0.0
            for other_strain, other_frequency in current_records:
                weighted_distance_to_present += other_frequency * strain_distance(
                    strain,
                    other_strain,
                    distances,
                    sequences,
                )

            weighted_distance_to_future = np.nan
            if future is not None:
                weighted_distance_to_future = 0.0
                for future_strain, future_frequency in future_records:
                    weighted_distance_to_future += future_frequency * strain_distance(
                        strain,
                        future_strain,
                        distances,
                        sequences,
                    )

            weighted_rows.append(
                {
                    "timepoint": pd.to_datetime(current_time),
                    "strain": strain,
                    "weighted_distance_to_present": weighted_distance_to_present,
                    "weighted_distance_to_future": weighted_distance_to_future,
                }
            )

    weighted = pd.DataFrame(weighted_rows)
    weighted["log2_distance_effect"] = np.log2(
        weighted["weighted_distance_to_future"] / weighted["weighted_distance_to_present"]
    )

    return df.merge(weighted, how="left", on=["strain", "timepoint"])


def write_target_distances(df: pd.DataFrame, pair_map: dict[str, str], output_path: Path) -> None:
    grouped = {pd.to_datetime(t).strftime("%Y-%m-%d"): g.copy() for t, g in df.groupby("timepoint")}
    sequence_by_strain = df.drop_duplicates("strain").set_index("strain")["aa_sequence"].to_dict()
    seen_pairs: set[tuple[str, str]] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("sample\tother_sample\tdistance\n")
        for current_time, current in grouped.items():
            comparison_strains = set(current["strain"])
            future_time = pair_map.get(current_time)
            if future_time in grouped:
                comparison_strains.update(grouped[future_time]["strain"])

            for sample in current["strain"]:
                for other_sample in comparison_strains:
                    pair = tuple(sorted((sample, other_sample)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    distance = seq_distance(sequence_by_strain[sample], sequence_by_strain[other_sample])
                    handle.write(f"{sample}\t{other_sample}\t{distance}\n")


def add_weighted_distances(
    df: pd.DataFrame,
    pair_map: dict[str, str],
    distances: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    grouped = {pd.to_datetime(t).strftime("%Y-%m-%d"): g.copy() for t, g in df.groupby("timepoint")}
    sequences = None if distances is not None else df.drop_duplicates("strain").set_index("strain")["aa_sequence"].to_dict()
    rows = []
    for current_time, future_time in pair_map.items():
        if current_time not in grouped or future_time not in grouped:
            continue

        current = grouped[current_time].copy()
        future = grouped[future_time].copy()
        current_records = list(current[["strain", "frequency"]].itertuples(index=False, name=None))
        future_records = list(future[["strain", "frequency"]].itertuples(index=False, name=None))

        future_present = []
        for strain, _ in future_records:
            d = 0.0
            for other_strain, other_frequency in future_records:
                d += other_frequency * strain_distance(strain, other_strain, distances, sequences)
            future_present.append(d)
        future["weighted_distance_to_present"] = future_present

        current_present = []
        current_future = []
        for strain, _ in current_records:
            d_present = 0.0
            for other_strain, other_frequency in current_records:
                d_present += other_frequency * strain_distance(strain, other_strain, distances, sequences)
            current_present.append(d_present)

            d_future = 0.0
            for future_strain, future_frequency in future_records:
                d_future += future_frequency * strain_distance(strain, future_strain, distances, sequences)
            current_future.append(d_future)

        current["weighted_distance_to_present"] = current_present
        current["weighted_distance_to_future"] = current_future
        current["future_timepoint"] = future_time
        current["log2_distance_effect"] = np.log2(
            current["weighted_distance_to_future"] / current["weighted_distance_to_present"]
        )
        rows.append(current)

        future_targets = future.copy()
        future_targets["timepoint"] = pd.to_datetime(current_time)
        future_targets["future_timepoint"] = pd.to_datetime(future_time)
        rows.append(future_targets)

    return pd.concat(rows, ignore_index=True)


def build_pairs(df: pd.DataFrame, hemisphere: str) -> dict[str, str]:
    times = sorted(pd.to_datetime(df["timepoint"].unique()))
    pair_map = {}
    for t in times:
        if hemisphere == "south" and t.month == 9:
            future = t + pd.DateOffset(months=5)
        elif hemisphere == "north" and t.month == 2:
            future = t + pd.DateOffset(months=7)
        else:
            continue
        if future in times:
            pair_map[t.strftime("%Y-%m-%d")] = future.strftime("%Y-%m-%d")
    return pair_map


def build_distance_dict(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    seqs = dict(zip(df["strain"], df["aa_sequence"]))
    strains = list(seqs)
    dist = {s: {} for s in strains}
    for i, a in enumerate(strains):
        for b in strains[i:]:
            d = seq_distance(seqs[a], seqs[b])
            dist[a][b] = d
            dist[b][a] = d
    return dist


def fit_and_forecast(
    current_df: pd.DataFrame,
    target_df: pd.DataFrame,
    predictors: list[str],
    delta_months: int,
    target_forecast_years: set[int],
    hemisphere: str,
    model_name: str,
    out_dir: Path,
    distances: dict[str, dict[str, int]],
    resume_existing: bool,
) -> None:
    forecast_path = out_dir / f"forecasts_{hemisphere}_{model_name}.tsv"
    errors_path = out_dir / f"errors_{hemisphere}_{model_name}.tsv"
    coefficients_path = out_dir / f"coefficients_{hemisphere}_{model_name}.tsv"
    if resume_existing and forecast_path.exists() and errors_path.exists() and coefficients_path.exists():
        print(f"Skipping existing {hemisphere} {model_name}", flush=True)
        return

    current_times = sorted(pd.to_datetime(current_df["timepoint"].unique()))
    forecast_rows = []
    error_rows = []
    coef_rows = []

    model = DistanceExponentialGrowthModel(
        predictors=predictors,
        delta_time=delta_months / 12.0,
        l1_lambda=0.1,
        cost_function="diffsum",
        distances=distances,
    )

    for current_time in current_times:
        block = current_df[current_df["timepoint"] == current_time].copy()
        forecast_year = int(block["forecast_year"].iloc[0])
        if forecast_year not in target_forecast_years:
            continue

        train_times = [t for t in current_times if t < current_time]
        if len(train_times) < 2:
            continue

        train_X = current_df[current_df["timepoint"].isin(train_times)].copy()
        train_y = target_df[target_df["timepoint"].isin(train_times)].copy()

        if model_name == "naive":
            model.coef_ = np.zeros(len(predictors))
            model.mean_stds_ = model.calculate_mean_stds(train_X, model.predictors)
            training_error = model.score(train_X, train_y)
        else:
            training_error = model.fit(train_X, train_y)

        test_X = current_df[current_df["timepoint"] == current_time].copy()
        test_y = target_df[target_df["timepoint"] == current_time].copy()
        validation_error = model.score(test_X, test_y)
        null_validation_error = model._fit(np.zeros_like(model.coef_), test_X, test_y, calculate_optimal_distance=True)
        optimal_validation_error = model.optimal_model_emd

        preds = model.predict(test_X)
        preds["future_timepoint"] = pd.to_datetime(test_X["future_timepoint"].iloc[0])
        preds["model"] = model_name
        preds["hemisphere"] = hemisphere
        preds["forecast_year"] = forecast_year
        forecast_rows.append(preds)

        error_rows.append(
            {
                "predictors": "-".join(predictors),
                "hemisphere": hemisphere,
                "forecast_year": forecast_year,
                "validation_timepoint": current_time.strftime("%Y-%m-%d"),
                "validation_error": validation_error,
                "null_validation_error": null_validation_error,
                "optimal_validation_error": optimal_validation_error,
                "training_error": training_error,
                "validation_n": test_X["strain"].nunique(),
            }
        )

        for predictor, coef in zip(predictors, model.coef_):
            coef_rows.append(
                {
                    "predictors": "-".join(predictors),
                    "hemisphere": hemisphere,
                    "forecast_year": forecast_year,
                    "predictor": predictor,
                    "coefficient": float(coef),
                    "validation_timepoint": current_time.strftime("%Y-%m-%d"),
                }
            )

    if forecast_rows:
        pd.concat(forecast_rows, ignore_index=True).to_csv(
            forecast_path,
            sep="\t",
            index=False,
        )
    pd.DataFrame(error_rows).to_csv(errors_path, sep="\t", index=False)
    pd.DataFrame(coef_rows).to_csv(
        coefficients_path,
        sep="\t",
        index=False,
    )
    print(f"Wrote {hemisphere} {model_name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run explicit issue-date pair modeling with the distance forecast model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="run directory containing aggregated tip attributes")
    parser.add_argument(
        "--tip-attributes",
        help="optional tip attributes file; defaults to <run-dir>/tip_attributes_with_naive.tsv when present",
    )
    parser.add_argument(
        "--predictor",
        dest="predictors",
        action="append",
        help="single predictor to fit; repeat for multiple predictors",
    )
    parser.add_argument(
        "--forecast-years",
        nargs="+",
        type=int,
        default=[2013, 2014],
        help="forecast years to emit",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "skip model outputs that already have forecast/error/coefficient tables; "
            "use only for speed, not for bitwise reproducibility after interruption"
        ),
    )
    args = parser.parse_args()

    # English: Re-seed at command start so a full model command follows the same optimizer path every time.
    # 中文：在命令开始时重新固定随机种子，确保完整 model 命令每次使用相同优化路径。
    np.random.seed(MODEL_RANDOM_SEED)

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_tip_paths = []
    if args.tip_attributes:
        candidate_tip_paths.append(Path(args.tip_attributes))
    candidate_tip_paths.append(run_dir / "tip_attributes_with_naive.tsv")
    candidate_tip_paths.append(run_dir / "tip_attributes_issue_dates_with_naive.tsv")
    tip_path = next((path for path in candidate_tip_paths if path.exists()), None)
    if tip_path is None:
        raise FileNotFoundError(f"No tip attributes file found in {candidate_tip_paths}")

    df = pd.read_csv(tip_path, sep="\t", parse_dates=["timepoint"])

    predictor_sets = args.predictors or ["lbi", "naive"]
    parsed_predictor_sets = [predictor_set.split(",") for predictor_set in predictor_sets]
    target_forecast_years = set(args.forecast_years)

    requested_predictors = sorted({predictor for predictor_set in parsed_predictor_sets for predictor in predictor_set})
    missing_predictors = [predictor for predictor in requested_predictors if predictor not in df.columns]
    if missing_predictors:
        raise ValueError(f"Missing predictors in {tip_path}: {missing_predictors}")
    df.loc[:, requested_predictors] = df.loc[:, requested_predictors].fillna(0.0)

    south_pairs = build_pairs(df, "south")
    north_pairs = build_pairs(df, "north")
    all_pairs = {}
    all_pairs.update(south_pairs)
    all_pairs.update(north_pairs)

    target_distances_path = run_dir / "target_distances.tsv"
    weighted_tip_attributes_path = run_dir / "tip_attributes_with_weighted_distances.tsv"
    if not target_distances_path.exists():
        write_target_distances(df, all_pairs, target_distances_path)

    print("Building shared strain distance dictionary", flush=True)
    with target_distances_path.open("r", encoding="utf-8") as handle:
        distances = get_distances_by_sample_names(csv.DictReader(handle, delimiter="\t"))
    for sample, sample_distances in list(distances.items()):
        for other_sample, distance in list(sample_distances.items()):
            distances[other_sample][sample] = distance
    print(f"Built shared strain distance dictionary for {len(distances)} strains", flush=True)

    if not weighted_tip_attributes_path.exists():
        annotate_weighted_distances(df, all_pairs, distances).to_csv(
            weighted_tip_attributes_path,
            sep="\t",
            index=False,
            na_rep="N/A",
        )

    south_df = df[df["hemisphere"] == "south"].copy()
    north_df = df[df["hemisphere"] == "north"].copy()

    south_aug = add_weighted_distances(df, south_pairs, distances)
    north_aug = add_weighted_distances(df, north_pairs, distances)

    south_current = south_aug[south_aug["hemisphere"] == "south"].copy()
    south_targets = south_aug[south_aug["hemisphere"] != "south"].copy()
    north_current = north_aug[north_aug["hemisphere"] == "north"].copy()
    north_targets = north_aug[north_aug["hemisphere"] != "north"].copy()

    for predictors in parsed_predictor_sets:
        model_name = "-".join(predictors)
        fit_and_forecast(
            south_current,
            south_targets,
            predictors,
            5,
            target_forecast_years,
            "south",
            model_name,
            out_dir,
            distances,
            args.resume_existing,
        )
        fit_and_forecast(
            north_current,
            north_targets,
            predictors,
            7,
            target_forecast_years,
            "north",
            model_name,
            out_dir,
            distances,
            args.resume_existing,
        )

    manifest = {
        "south_pairs": south_pairs,
        "north_pairs": north_pairs,
        "tip_attributes": public_path(tip_path),
        "target_distances": public_path(target_distances_path),
        "weighted_tip_attributes": public_path(weighted_tip_attributes_path),
        "predictors": parsed_predictor_sets,
        "random_seed": MODEL_RANDOM_SEED,
        "resume_existing": args.resume_existing,
    }
    with open(out_dir / "pair_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote forecasts under {out_dir}")


if __name__ == "__main__":
    main()
