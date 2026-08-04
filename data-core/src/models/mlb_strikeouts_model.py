"""Leakage-aware starter strikeout regression from the MLB v2 feature store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "starter_k9_last5",
    "starter_bb9_last5",
    "starter_era_proxy_last5",
    "starter_pitches_last3_avg",
    "starter_outs_per_start_last5",
    "starter_starts_last365",
    "starter_days_since_last_start",
    "starter_career_starts_prior",
    "starter_throws_r",
    "opponent_team_bat_k_pg_15",
    "venue_total_runs_per_game",
    "temp_f",
    "wind_mph",
    "wind_out",
    "wind_in",
    "wind_cross",
    "is_dome_or_closed",
    "is_day_game",
    "elevation",
    "is_home",
]

REQUIRED_STORE_COLUMNS = {
    "game_pk",
    "game_date",
    "venue_total_runs_per_game",
    "temp_f",
    "wind_mph",
    "wind_out",
    "wind_in",
    "wind_cross",
    "is_dome_or_closed",
    "is_day_game",
    "elevation",
}

SIDE_FEATURES = [
    "starter_k9_last5",
    "starter_bb9_last5",
    "starter_era_proxy_last5",
    "starter_pitches_last3_avg",
    "starter_outs_per_start_last5",
    "starter_starts_last365",
    "starter_days_since_last_start",
    "starter_career_starts_prior",
    "starter_has_history",
    "starter_throws_r",
]


@dataclass
class StrikeoutsTrainingResult:
    """Outputs needed by the CLI without coupling artifact IO to the model."""

    model: Pipeline
    metrics: dict[str, Any]
    predictions: pd.DataFrame


def reshape_starter_sides(feature_store: pd.DataFrame) -> pd.DataFrame:
    """Convert game rows to one pregame feature row per probable starter side."""
    required = set(REQUIRED_STORE_COLUMNS)
    for side in ("home", "away"):
        required.update(
            {
                f"{side}_probable_pitcher_id",
                f"{side}_probable_pitcher",
                f"{side}_starter_ks_label",
                f"{side}_probable_matches_actual",
                f"{side}_team_bat_k_pg_15",
            }
        )
        required.update(f"{side}_{feature}" for feature in SIDE_FEATURES)
    missing = sorted(required.difference(feature_store.columns))
    if missing:
        raise ValueError(f"Feature store is missing required columns: {missing}")

    common = [
        "game_pk",
        "game_date",
        "venue_total_runs_per_game",
        "temp_f",
        "wind_mph",
        "wind_out",
        "wind_in",
        "wind_cross",
        "is_dome_or_closed",
        "is_day_game",
        "elevation",
    ]
    frames: list[pd.DataFrame] = []
    for side, opponent in (("home", "away"), ("away", "home")):
        out = feature_store[common].copy()
        out["side"] = side
        out["is_home"] = float(side == "home")
        out["pitcher_id"] = feature_store[f"{side}_probable_pitcher_id"]
        out["pitcher_name"] = feature_store[f"{side}_probable_pitcher"]
        out["actual"] = feature_store[f"{side}_starter_ks_label"]
        out["probable_matches_actual"] = feature_store[f"{side}_probable_matches_actual"].eq(True)
        for feature in SIDE_FEATURES:
            out[feature] = feature_store[f"{side}_{feature}"]
        out["opponent_team_bat_k_pg_15"] = feature_store[f"{opponent}_team_bat_k_pg_15"]
        frames.append(out)

    sides = pd.concat(frames, ignore_index=True)
    sides["game_date"] = pd.to_datetime(sides["game_date"])
    sides["season"] = sides["game_date"].dt.year
    sides["has_clean_label"] = (
        sides["probable_matches_actual"]
        & sides["actual"].notna()
        & sides["starter_has_history"].eq(True)
        & sides["starter_k9_last5"].notna()
    )
    return sides.sort_values(["game_date", "game_pk", "side"]).reset_index(drop=True)


def coverage_metrics(sides: pd.DataFrame, season: int = 2026) -> dict[str, Any]:
    """Summarize probable-starter and clean-label coverage for a season."""
    sample = sides.loc[sides["season"].eq(season)]
    total = int(len(sample))
    matched = int(sample["probable_matches_actual"].sum())
    labeled = int(sample["actual"].notna().sum())
    history = int(sample["starter_has_history"].eq(True).sum())
    clean = int(sample["has_clean_label"].sum())
    by_side: dict[str, Any] = {}
    for side in ("home", "away"):
        part = sample.loc[sample["side"].eq(side)]
        side_total = int(len(part))
        side_clean = int(part["has_clean_label"].sum())
        by_side[side] = {
            "total_sides": side_total,
            "clean_sides": side_clean,
            "clean_coverage": side_clean / side_total if side_total else None,
        }
    return {
        "season": season,
        "total_sides": total,
        "probable_matches_actual_sides": matched,
        "probable_matches_actual_fraction": matched / total if total else None,
        "probable_mismatch_discarded_fraction": (total - matched) / total if total else None,
        "non_null_label_sides": labeled,
        "starter_history_sides": history,
        "clean_sides": clean,
        "clean_coverage": clean / total if total else None,
        "by_side": by_side,
    }


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
    }


def _baseline_predictions(rows: pd.DataFrame) -> np.ndarray:
    # K/9 * innings, where expected innings = trailing outs / 3.
    values = (
        pd.to_numeric(rows["starter_k9_last5"], errors="coerce")
        * pd.to_numeric(rows["starter_outs_per_start_last5"], errors="coerce")
        / 27.0
    )
    fallback = float(values.median()) if values.notna().any() else 5.0
    return values.fillna(fallback).clip(lower=0.01).to_numpy(dtype=float)


def _make_model(params: dict[str, Any], *, loss: str, random_state: int) -> Pipeline:
    regressor = HistGradientBoostingRegressor(
        loss=loss,
        random_state=random_state,
        early_stopping=False,
        **params,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("regressor", regressor),
        ]
    )


def _poisson_loss_available() -> bool:
    # Parameter validation happens during fit, so use a tiny nonnegative probe.
    try:
        HistGradientBoostingRegressor(loss="poisson", max_iter=1).fit(
            np.asarray([[0.0], [1.0]]), np.asarray([0.0, 1.0])
        )
        return True
    except (TypeError, ValueError):
        return False


def _threshold_calibration(
    actual: np.ndarray,
    expected_ks: np.ndarray,
    reference_actual: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    means = np.clip(expected_ks, 0.01, None)
    for threshold, cutoff in (("5.5", 6), ("6.5", 7)):
        probabilities = poisson.sf(cutoff - 1, means)
        outcomes = (actual >= cutoff).astype(float)
        base_rate = float(np.mean(reference_actual >= cutoff))
        output[f"over_{threshold.replace('.', '_')}"] = {
            "threshold": float(threshold),
            "empirical_hit_rate": float(outcomes.mean()),
            "mean_predicted_probability": float(probabilities.mean()),
            "brier": float(np.mean((probabilities - outcomes) ** 2)),
            "reference_base_rate": base_rate,
            "base_rate_brier": float(np.mean((base_rate - outcomes) ** 2)),
        }
    return output


def train_and_evaluate_strikeouts(
    sides: pd.DataFrame,
    *,
    train_start_season: int = 2021,
    train_end_season: int = 2024,
    validation_season: int = 2025,
    test_season: int = 2026,
    random_state: int = 42,
) -> StrikeoutsTrainingResult:
    """Select on 2025, refit through 2025, and evaluate once on 2026 YTD."""
    clean = sides.loc[sides["has_clean_label"]].copy()
    train = clean.loc[clean["season"].between(train_start_season, train_end_season)]
    validation = clean.loc[clean["season"].eq(validation_season)]
    test = clean.loc[clean["season"].eq(test_season)]
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Time split produced an empty partition: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )

    loss = "poisson" if _poisson_loss_available() else "squared_error"
    candidates = {
        "poisson_shallow": {"learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        "poisson_medium": {"learning_rate": 0.04, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 2.0},
        "poisson_conservative": {"learning_rate": 0.03, "max_iter": 250, "max_leaf_nodes": 7, "l2_regularization": 4.0},
    }
    validation_results: dict[str, Any] = {}
    fitted_candidates: dict[str, Pipeline] = {}
    for name, params in candidates.items():
        model = _make_model(params, loss=loss, random_state=random_state)
        model.fit(train[FEATURE_COLUMNS], train["actual"])
        predictions = np.clip(model.predict(validation[FEATURE_COLUMNS]), 0.01, None)
        validation_results[name] = _regression_metrics(validation["actual"].to_numpy(), predictions)
        fitted_candidates[name] = model

    validation_baseline = _baseline_predictions(validation)
    validation_results["k9_expected_outs_baseline"] = _regression_metrics(
        validation["actual"].to_numpy(), validation_baseline
    )
    selected_name = min(candidates, key=lambda name: validation_results[name]["mae"])

    refit = clean.loc[clean["season"].between(train_start_season, validation_season)]
    selected_model = _make_model(candidates[selected_name], loss=loss, random_state=random_state)
    selected_model.fit(refit[FEATURE_COLUMNS], refit["actual"])
    expected = np.clip(selected_model.predict(test[FEATURE_COLUMNS]), 0.01, None)
    baseline = _baseline_predictions(test)
    actual = test["actual"].to_numpy(dtype=float)
    p_over_5_5 = poisson.sf(5, expected)
    p_over_6_5 = poisson.sf(6, expected)

    predictions = test[
        ["game_pk", "game_date", "side", "pitcher_name", "pitcher_id", "actual"]
    ].copy()
    predictions = predictions.rename(columns={"game_date": "date"})
    predictions["pitcher_id"] = predictions["pitcher_id"].astype("Int64")
    predictions["expected_ks"] = expected
    predictions["p_over_5_5"] = p_over_5_5
    predictions["p_over_6_5"] = p_over_6_5
    predictions = predictions[
        [
            "game_pk",
            "date",
            "side",
            "pitcher_name",
            "pitcher_id",
            "expected_ks",
            "p_over_5_5",
            "p_over_6_5",
            "actual",
        ]
    ]

    coverage = coverage_metrics(sides, season=test_season)
    model_test_metrics = _regression_metrics(actual, expected)
    baseline_test_metrics = _regression_metrics(actual, baseline)
    metrics: dict[str, Any] = {
        "model": "HistGradientBoostingRegressor",
        "loss": loss,
        "selected_candidate": selected_name,
        "feature_columns": FEATURE_COLUMNS,
        "split": {
            "train_seasons": f"{train_start_season}-{train_end_season}",
            "validation_season": validation_season,
            "test_season": test_season,
            "refit_seasons": f"{train_start_season}-{validation_season}",
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "refit_rows": int(len(refit)),
            "test_rows": int(len(test)),
        },
        "coverage": coverage,
        "all_seasons_probable_mismatch_discarded_fraction": float(
            (~sides["probable_matches_actual"]).mean()
        ),
        "validation": validation_results,
        "test": {
            "model": model_test_metrics,
            "k9_expected_outs_baseline": baseline_test_metrics,
            "model_minus_baseline": {
                "mae": model_test_metrics["mae"] - baseline_test_metrics["mae"],
                "rmse": model_test_metrics["rmse"] - baseline_test_metrics["rmse"],
            },
            "threshold_calibration": _threshold_calibration(
                actual, expected, refit["actual"].to_numpy(dtype=float)
            ),
        },
        "roi": None,
        "reason": "no strikeout odds source",
    }
    return StrikeoutsTrainingResult(model=selected_model, metrics=metrics, predictions=predictions)
