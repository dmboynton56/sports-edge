"""Leakage-aware MLB total-runs model training and evaluation utilities."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import math
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.mlb_winner_model import default_feature_columns, expected_calibration_error


TOTAL_LINES = (8.5, 9.5)
WEATHER_PARK_COLUMNS = {
    "temp_f",
    "wind_mph",
    "wind_out",
    "wind_in",
    "wind_cross",
    "is_dome_or_closed",
    "is_day_game",
    "elevation",
    "venue_prior_games",
    "venue_home_win_pct",
    "venue_total_runs_per_game",
}


def over_label(total_runs: pd.Series | np.ndarray, line: float) -> np.ndarray:
    """Return 1 for an over and 0 for an under; reject pushes on integer lines."""
    totals = np.asarray(total_runs, dtype=float)
    if float(line).is_integer() and np.any(totals == float(line)):
        raise ValueError("Integer total lines can push; remove pushes before binary labeling.")
    return (totals > float(line)).astype(int)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_prediction": float(np.mean(y_pred)),
        "mean_actual": float(np.mean(y_true)),
    }


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    probabilities = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    metrics = {
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "ece_10": expected_calibration_error(y_true, probabilities, n_bins=10),
        "mean_probability": float(np.mean(probabilities)),
        "actual_over_rate": float(np.mean(y_true)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    )
    return metrics


def _normal_over_probability(predicted_total: np.ndarray, line: float, sigma: float) -> np.ndarray:
    """Convert a total estimate to P(total > line) using a Gaussian residual model."""
    safe_sigma = max(float(sigma), 0.25)
    z = (float(line) - np.asarray(predicted_total, dtype=float)) / (safe_sigma * math.sqrt(2.0))
    return np.fromiter((0.5 * math.erfc(float(value)) for value in z), dtype=float, count=len(z))


def _candidate_regressors(random_state: int) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.05,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=15,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _league_rolling_mean_predictions(
    history: pd.DataFrame, test: pd.DataFrame, fallback: float, days: int = 30
) -> np.ndarray:
    """Predict with totals from strictly earlier game dates in a trailing window."""
    history_by_date = (
        history.assign(_date=pd.to_datetime(history["game_date"]).dt.normalize())
        .groupby("_date")["total_runs"]
        .agg(["sum", "count"])
        .sort_index()
    )
    dates = list(history_by_date.index)
    sums = history_by_date["sum"].to_numpy(dtype=float)
    counts = history_by_date["count"].to_numpy(dtype=int)
    window: deque[int] = deque()
    right = 0
    running_sum = 0.0
    running_count = 0
    predictions: list[float] = []

    for test_date in pd.to_datetime(test["game_date"]).dt.normalize():
        while right < len(dates) and dates[right] < test_date:
            window.append(right)
            running_sum += sums[right]
            running_count += counts[right]
            right += 1
        cutoff = test_date - pd.Timedelta(days=days)
        while window and dates[window[0]] < cutoff:
            idx = window.popleft()
            running_sum -= sums[idx]
            running_count -= counts[idx]
        predictions.append(running_sum / running_count if running_count else fallback)
    return np.asarray(predictions, dtype=float)


def _finite_correlation(left: pd.Series | np.ndarray, right: np.ndarray) -> Optional[float]:
    left_values = np.atleast_1d(np.asarray(left, dtype=float))
    right_values = np.atleast_1d(np.asarray(right, dtype=float))
    if left_values.size != right_values.size:
        return None
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if mask.sum() < 2 or np.std(left_values[mask]) == 0 or np.std(right_values[mask]) == 0:
        return None
    return float(np.corrcoef(left_values[mask], right_values[mask])[0, 1])


def _american_profit(price: float) -> float:
    return float(price) / 100.0 if price > 0 else 100.0 / abs(float(price))


def evaluate_totals_roi(predictions: pd.DataFrame, odds: Optional[pd.DataFrame]) -> dict:
    """Evaluate flat-stake ROI for 8.5/9.5 odds, or return an explicit no-odds result."""
    if odds is None:
        return {"roi": None, "reason": "no totals odds source"}
    required = {"game_pk", "total_line", "over_price", "under_price"}
    missing = sorted(required.difference(odds.columns))
    if missing:
        raise ValueError(f"Odds file is missing required columns: {missing}")

    joined = predictions.merge(odds[list(required)], on="game_pk", how="inner")
    joined = joined[joined["total_line"].isin(TOTAL_LINES)].copy()
    if joined.empty:
        return {"roi": None, "reason": "no supported 8.5/9.5 totals odds joined"}

    profits: list[float] = []
    over_bets = 0
    under_bets = 0
    for row in joined.itertuples(index=False):
        suffix = str(float(row.total_line)).replace(".", "_")
        p_over = float(getattr(row, f"p_over_{suffix}"))
        over_profit = _american_profit(float(row.over_price))
        under_profit = _american_profit(float(row.under_price))
        over_ev = p_over * over_profit - (1.0 - p_over)
        under_ev = (1.0 - p_over) * under_profit - p_over
        if max(over_ev, under_ev) <= 0:
            continue
        actual_over = float(row.total_runs) > float(row.total_line)
        if over_ev >= under_ev:
            profits.append(over_profit if actual_over else -1.0)
            over_bets += 1
        else:
            profits.append(-1.0 if actual_over else under_profit)
            under_bets += 1
    if not profits:
        return {"roi": None, "reason": "no positive-EV bets on joined totals odds", "joined_rows": int(len(joined))}
    return {
        "roi": float(np.mean(profits)),
        "profit_units": float(np.sum(profits)),
        "bets": len(profits),
        "over_bets": over_bets,
        "under_bets": under_bets,
        "joined_rows": int(len(joined)),
        "reason": None,
    }


def train_and_evaluate_mlb_totals(
    features: pd.DataFrame,
    *,
    validation_season: int = 2025,
    test_season: int = 2026,
    random_state: int = 42,
    odds: Optional[pd.DataFrame] = None,
) -> dict:
    """Select a total-runs regressor on validation MAE and evaluate 2026 YTD."""
    if features.empty:
        raise ValueError("No MLB features provided.")
    required = {
        "game_pk",
        "season",
        "game_date",
        "game_datetime",
        "total_runs",
        "venue_total_runs_per_game",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Feature frame is missing required columns: {missing}")
    if validation_season >= test_season:
        raise ValueError("validation_season must be earlier than test_season.")

    frame = features.dropna(subset=["total_runs"]).copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame["game_datetime"] = pd.to_datetime(frame["game_datetime"])
    frame = frame.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)
    feature_cols = default_feature_columns(frame)
    forbidden = {"total_runs", "home_score", "away_score"}.intersection(feature_cols)
    if forbidden:
        raise AssertionError(f"Outcome leakage in resolved features: {sorted(forbidden)}")

    train_df = frame[frame["season"] < validation_season].copy()
    val_df = frame[frame["season"] == validation_season].copy()
    test_df = frame[frame["season"] == test_season].copy()
    final_train_df = frame[frame["season"] < test_season].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Split produced empty data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    candidates = _candidate_regressors(random_state)
    y_train = train_df["total_runs"].to_numpy(dtype=float)
    y_val = val_df["total_runs"].to_numpy(dtype=float)
    y_test = test_df["total_runs"].to_numpy(dtype=float)
    candidate_metrics: dict[str, dict] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    for name, candidate in candidates.items():
        candidate.fit(train_df[feature_cols], y_train)
        val_prediction = candidate.predict(val_df[feature_cols])
        test_prediction = candidate.predict(test_df[feature_cols])
        validation_predictions[name] = val_prediction
        candidate_metrics[name] = {
            "validation": _regression_metrics(y_val, val_prediction),
            "test": _regression_metrics(y_test, test_prediction),
        }

    selected_name = min(candidate_metrics, key=lambda name: candidate_metrics[name]["validation"]["mae"])
    residual_sigma = float(np.sqrt(np.mean(np.square(y_val - validation_predictions[selected_name]))))
    selected_model = clone(candidates[selected_name])
    selected_model.fit(final_train_df[feature_cols], final_train_df["total_runs"].to_numpy(dtype=float))
    selected_test_prediction = selected_model.predict(test_df[feature_cols])

    constant_mean = float(final_train_df["total_runs"].mean())
    rolling_prediction = _league_rolling_mean_predictions(
        frame[frame["season"] <= test_season], test_df, constant_mean
    )
    venue_prediction = pd.to_numeric(test_df["venue_total_runs_per_game"], errors="coerce").fillna(
        float(final_train_df["venue_total_runs_per_game"].median())
    ).to_numpy(dtype=float)
    baseline_metrics = {
        "league_trailing_30_day_mean": _regression_metrics(y_test, rolling_prediction),
        "venue_total_runs_per_game": _regression_metrics(y_test, venue_prediction),
        "constant_train_mean": {
            **_regression_metrics(y_test, np.full(len(test_df), constant_mean)),
            "value": constant_mean,
        },
    }

    binary_heads: dict[str, dict] = {}
    probability_columns: dict[float, np.ndarray] = {}
    for line in TOTAL_LINES:
        label = over_label(y_test, line)
        probability = _normal_over_probability(selected_test_prediction, line, residual_sigma)
        base_rate = float(over_label(final_train_df["total_runs"], line).mean())
        probability_columns[line] = probability
        binary_heads[str(line)] = {
            "line": line,
            "model": _binary_metrics(label, probability),
            "constant_base_rate": {
                "probability": base_rate,
                **_binary_metrics(label, np.full(len(label), base_rate)),
            },
        }

    ablated_cols = [column for column in feature_cols if column not in WEATHER_PARK_COLUMNS]
    removed_cols = [column for column in feature_cols if column in WEATHER_PARK_COLUMNS]
    ablation_validation_model = clone(candidates[selected_name])
    ablation_validation_model.fit(train_df[ablated_cols], y_train)
    ablation_val_prediction = ablation_validation_model.predict(val_df[ablated_cols])
    ablation_sigma = float(np.sqrt(np.mean(np.square(y_val - ablation_val_prediction))))
    ablation_model = clone(candidates[selected_name])
    ablation_model.fit(final_train_df[ablated_cols], final_train_df["total_runs"].to_numpy(dtype=float))
    ablation_test_prediction = ablation_model.predict(test_df[ablated_cols])
    full_test_metrics = _regression_metrics(y_test, selected_test_prediction)
    ablation_test_metrics = _regression_metrics(y_test, ablation_test_prediction)
    full_85_auc = _binary_metrics(over_label(y_test, 8.5), probability_columns[8.5])["roc_auc"]
    ablation_85_probability = _normal_over_probability(ablation_test_prediction, 8.5, ablation_sigma)
    ablation_85_auc = _binary_metrics(over_label(y_test, 8.5), ablation_85_probability)["roc_auc"]
    net_wind = pd.to_numeric(test_df.get("wind_out", np.nan), errors="coerce") - pd.to_numeric(
        test_df.get("wind_in", np.nan), errors="coerce"
    )
    weather_ablation = {
        "group": "weather_and_park",
        "removed_features": removed_cols,
        "full": {"regression": full_test_metrics, "over_8_5_auc": full_85_auc},
        "without_weather_and_park": {
            "regression": ablation_test_metrics,
            "over_8_5_auc": ablation_85_auc,
        },
        "delta_mae_ablation_minus_full": float(ablation_test_metrics["mae"] - full_test_metrics["mae"]),
        "delta_auc_full_minus_ablation_8_5": (
            float(full_85_auc - ablation_85_auc)
            if full_85_auc is not None and ablation_85_auc is not None
            else None
        ),
        "sanity_correlations": {
            "predicted_total_vs_temp_f": _finite_correlation(test_df.get("temp_f", np.nan), selected_test_prediction),
            "predicted_total_vs_wind_out_minus_wind_in": _finite_correlation(net_wind, selected_test_prediction),
            "expected_direction": "positive for both (warmer temperatures and wind out imply higher totals)",
        },
    }

    predictions = pd.DataFrame(
        {
            "game_pk": test_df["game_pk"].astype(int).to_numpy(),
            "date": test_df["game_date"].dt.date.astype(str).to_numpy(),
            "home_team": test_df.get("home_team", pd.Series([None] * len(test_df))).to_numpy(),
            "away_team": test_df.get("away_team", pd.Series([None] * len(test_df))).to_numpy(),
            "predicted_total": selected_test_prediction,
            "p_over_8_5": probability_columns[8.5],
            "p_over_9_5": probability_columns[9.5],
            "total_runs": y_test,
        }
    )
    roi = evaluate_totals_roi(predictions, odds)

    return {
        "selected_model_name": selected_name,
        "selected_model": selected_model,
        "feature_columns": feature_cols,
        "candidate_metrics": candidate_metrics,
        "selected_refit_test": full_test_metrics,
        "probability_method": {
            "name": "gaussian_validation_residual",
            "description": "Over probabilities use the selected regressor mean and Gaussian residual sigma estimated out-of-sample on 2025 validation rows.",
            "validation_residual_rmse_sigma": residual_sigma,
        },
        "binary_heads": binary_heads,
        "baselines": baseline_metrics,
        "weather_ablation": weather_ablation,
        "roi": roi["roi"],
        "reason": roi["reason"],
        "roi_evaluation": roi,
        "splits": {
            "train_seasons": sorted(train_df["season"].astype(int).unique().tolist()),
            "validation_season": int(validation_season),
            "test_season": int(test_season),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "final_train_seasons": sorted(final_train_df["season"].astype(int).unique().tolist()),
            "final_train_rows": int(len(final_train_df)),
        },
        "data_summary": {
            "rows": int(len(frame)),
            "min_game_date": str(frame["game_date"].min().date()),
            "max_game_date": str(frame["game_date"].max().date()),
            "seasons": sorted(frame["season"].astype(int).unique().tolist()),
            "mean_total_runs": float(frame["total_runs"].mean()),
        },
        "predictions": predictions,
    }


def metrics_payload(result: dict) -> dict:
    """Return the JSON-safe, non-estimator portion of a training result."""
    keys = (
        "selected_model_name",
        "feature_columns",
        "candidate_metrics",
        "selected_refit_test",
        "probability_method",
        "binary_heads",
        "baselines",
        "weather_ablation",
        "roi",
        "reason",
        "roi_evaluation",
        "splits",
        "data_summary",
    )
    return {key: result[key] for key in keys}


def save_mlb_totals_outputs(
    result: dict,
    *,
    metrics_path: str,
    predictions_path: str,
    model_path: Optional[str] = None,
    model_version: str = "v1",
) -> None:
    """Save metrics, test predictions, and optionally a reusable model artifact."""
    for path in (metrics_path, predictions_path, model_path):
        if path:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload(result), handle, indent=2, sort_keys=True, allow_nan=False)
    result["predictions"].to_csv(predictions_path, index=False)
    if model_path:
        artifact = {
            "model": result["selected_model"],
            "model_name": result["selected_model_name"],
            "feature_columns": result["feature_columns"],
            "model_version": model_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "probability_method": result["probability_method"],
            "splits": result["splits"],
        }
        with open(model_path, "wb") as handle:
            pickle.dump(artifact, handle)
