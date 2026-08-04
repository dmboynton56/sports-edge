"""Leakage-aware MLB home -1.5 run-line modeling utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.mlb_winner_model import default_feature_columns


def resolved_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Use the shared MLB selector and defensively exclude run-line outcomes."""
    columns = default_feature_columns(frame)
    forbidden = {"home_cover_15", "run_diff", "home_score", "away_score", "total_runs"}
    leaked = forbidden.intersection(columns)
    if leaked:
        raise ValueError(f"Outcome columns selected as model features: {sorted(leaked)}")
    return columns


def split_runline_frame(
    frame: pd.DataFrame, validation_season: int, test_season: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return selection train, validation, test, and pre-test refit frames."""
    if validation_season >= test_season:
        raise ValueError("validation_season must be earlier than test_season")
    train = frame[frame["season"] < validation_season].copy()
    validation = frame[frame["season"] == validation_season].copy()
    test = frame[frame["season"] == test_season].copy()
    refit = frame[frame["season"] < test_season].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Split produced empty data: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )
    return train, validation, test, refit


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Return equal-width reliability bins, including empty bins."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows: list[dict] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (y_prob >= lower) & (y_prob <= upper if index == n_bins - 1 else y_prob < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin": index + 1,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": float(y_prob[mask].mean()) if count else None,
                "actual_rate": float(y_true[mask].mean()) if count else None,
                "absolute_gap": (
                    float(abs(y_prob[mask].mean() - y_true[mask].mean())) if count else None
                ),
            }
        )
    return rows


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    rows = calibration_table(y_true, y_prob, n_bins=n_bins)
    total = len(y_true)
    return float(
        sum((row["count"] / total) * row["absolute_gap"] for row in rows if row["count"])
    )


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_prob >= 0.5)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece_10": expected_calibration_error(y_true, y_prob),
        "average_probability": float(y_prob.mean()),
        "actual_cover_rate": float(y_true.mean()),
    }
    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mean_predicted_margin": float(np.mean(prediction)),
        "mean_actual_margin": float(np.mean(y_true)),
    }


def cover_probability_from_residuals(
    predicted_margin: np.ndarray, residuals: np.ndarray, threshold: float = 2.0
) -> np.ndarray:
    """Estimate P(margin >= threshold) with the empirical training residual CDF."""
    residuals = np.sort(np.asarray(residuals, dtype=float))
    if residuals.size == 0:
        raise ValueError("At least one training residual is required")
    cutoffs = threshold - np.asarray(predicted_margin, dtype=float)
    covered = residuals.size - np.searchsorted(residuals, cutoffs, side="left")
    # Half-count smoothing avoids exact zero/one probabilities on finite samples.
    return (covered + 0.5) / (residuals.size + 1.0)


def _classification_candidates(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.02,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=20,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _regression_candidates(random_state: int) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=15,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate_mlb_runline(
    features: pd.DataFrame,
    *,
    validation_season: Optional[int] = None,
    test_season: Optional[int] = None,
    random_state: int = 42,
) -> dict:
    """Fit classifier and margin heads with season-based selection and refit."""
    if features.empty:
        raise ValueError("No MLB features provided")
    required = {"season", "game_datetime", "game_pk", "run_diff", "home_cover_15"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = features.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True).copy()
    expected_label = (frame["run_diff"] >= 2).astype(int)
    if not np.array_equal(frame["home_cover_15"].astype(int).to_numpy(), expected_label.to_numpy()):
        raise ValueError("home_cover_15 must equal (run_diff >= 2)")
    seasons = sorted(frame["season"].astype(int).unique().tolist())
    if len(seasons) < 3 and (validation_season is None or test_season is None):
        raise ValueError("Need at least three seasons for the default split")
    test_season = int(test_season or seasons[-1])
    validation_season = int(validation_season or seasons[-2])
    train, validation, test, refit = split_runline_frame(
        frame, validation_season, test_season
    )
    columns = resolved_feature_columns(frame)
    X_train, X_val, X_test = train[columns], validation[columns], test[columns]
    y_train = train["home_cover_15"].astype(int).to_numpy()
    y_val = validation["home_cover_15"].astype(int).to_numpy()
    y_test = test["home_cover_15"].astype(int).to_numpy()

    classifiers = _classification_candidates(random_state)
    classifier_metrics = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        classifier_metrics[name] = {
            "validation": classification_metrics(y_val, model.predict_proba(X_val)[:, 1]),
            "test": classification_metrics(y_test, model.predict_proba(X_test)[:, 1]),
        }
    selected_classifier_name = min(
        classifier_metrics, key=lambda name: classifier_metrics[name]["validation"]["brier"]
    )
    classifier = classifiers[selected_classifier_name]
    classifier.fit(refit[columns], refit["home_cover_15"].astype(int))
    classifier_probability = classifier.predict_proba(X_test)[:, 1]

    regressors = _regression_candidates(random_state)
    margin_metrics = {}
    y_margin_train = train["run_diff"].astype(float).to_numpy()
    y_margin_val = validation["run_diff"].astype(float).to_numpy()
    y_margin_test = test["run_diff"].astype(float).to_numpy()
    for name, model in regressors.items():
        model.fit(X_train, y_margin_train)
        train_prediction = model.predict(X_train)
        residuals = y_margin_train - train_prediction
        val_prediction = model.predict(X_val)
        test_prediction = model.predict(X_test)
        margin_metrics[name] = {
            "validation": regression_metrics(y_margin_val, val_prediction),
            "test": regression_metrics(y_margin_test, test_prediction),
            "cover_probability_validation": classification_metrics(
                y_val, cover_probability_from_residuals(val_prediction, residuals)
            ),
            "cover_probability_test": classification_metrics(
                y_test, cover_probability_from_residuals(test_prediction, residuals)
            ),
        }
    selected_regressor_name = min(
        margin_metrics, key=lambda name: margin_metrics[name]["validation"]["mae"]
    )
    regressor = regressors[selected_regressor_name]
    refit_margin = refit["run_diff"].astype(float).to_numpy()
    regressor.fit(refit[columns], refit_margin)
    refit_residuals = refit_margin - regressor.predict(refit[columns])
    predicted_margin = regressor.predict(X_test)
    margin_probability = cover_probability_from_residuals(predicted_margin, refit_residuals)

    classifier_test_metrics = classification_metrics(y_test, classifier_probability)
    margin_probability_test_metrics = classification_metrics(y_test, margin_probability)
    base_probability = float(refit["home_cover_15"].mean())
    baseline_probability = np.full(len(test), base_probability)
    classifier_wins = classifier_test_metrics["brier"] <= margin_probability_test_metrics["brier"]

    predictions = test[["game_pk", "game_date", "home_team", "away_team", "home_cover_15"]].copy()
    predictions = predictions.rename(columns={"game_date": "date"})
    predictions["p_home_cover_15"] = classifier_probability
    predictions["p_away_cover_plus_15"] = 1.0 - classifier_probability
    predictions["p_home_cover_15_margin_head"] = margin_probability
    predictions["predicted_margin"] = predicted_margin
    predictions = predictions[
        [
            "game_pk", "date", "home_team", "away_team", "p_home_cover_15",
            "p_away_cover_plus_15", "p_home_cover_15_margin_head", "predicted_margin",
            "home_cover_15",
        ]
    ]

    return {
        "selected_classifier_name": selected_classifier_name,
        "selected_classifier": classifier,
        "selected_regressor_name": selected_regressor_name,
        "selected_regressor": regressor,
        "feature_columns": columns,
        "classifier_candidates": classifier_metrics,
        "margin_candidates": margin_metrics,
        "selected_refit_test": classifier_test_metrics,
        "selected_margin_refit_test": {
            "margin": regression_metrics(y_margin_test, predicted_margin),
            "cover_probability": margin_probability_test_metrics,
        },
        "head_comparison": {
            "selection_basis": "classifier candidates: validation Brier; margin candidates: validation MAE",
            "better_calibrated_on_test_brier": "classifier" if classifier_wins else "margin_residual",
            "classifier_brier": classifier_test_metrics["brier"],
            "margin_residual_brier": margin_probability_test_metrics["brier"],
        },
        "baseline": {
            "probability": base_probability,
            "source": "pre-test (2021-2025) home -1.5 cover rate",
            "test": classification_metrics(y_test, baseline_probability),
            "selected_classifier_brier_delta_vs_baseline": (
                classifier_test_metrics["brier"]
                - classification_metrics(y_test, baseline_probability)["brier"]
            ),
        },
        "calibration_table": {
            "classifier_test": calibration_table(y_test, classifier_probability),
            "margin_residual_test": calibration_table(y_test, margin_probability),
        },
        "mirrored_market_sanity": {
            "home_minus_1_5_plus_away_plus_1_5_probability_max_error": float(
                np.max(np.abs(classifier_probability + (1.0 - classifier_probability) - 1.0))
            ),
            "average_p_home_minus_1_5": float(classifier_probability.mean()),
            "average_p_away_plus_1_5": float((1.0 - classifier_probability).mean()),
            "push_possible": False,
            "reason": "A half-run line cannot push.",
        },
        "splits": {
            "train_seasons": sorted(train["season"].astype(int).unique().tolist()),
            "validation_season": validation_season,
            "test_season": test_season,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "final_train_rows": int(len(refit)),
        },
        "data_summary": {
            "rows": int(len(frame)),
            "seasons": seasons,
            "min_game_date": str(pd.to_datetime(frame["game_date"]).min().date()),
            "max_game_date": str(pd.to_datetime(frame["game_date"]).max().date()),
            "cover_rate_by_season": {
                str(int(season)): float(rate)
                for season, rate in frame.groupby("season")["home_cover_15"].mean().items()
            },
        },
        "roi": None,
        "roi_reason": "no run-line odds source",
        "predictions": predictions,
        "_selected_margin_residuals": refit_residuals,
    }


def save_mlb_runline_artifact(result: dict, output_path: str, model_version: str = "v1") -> None:
    """Persist both fitted heads and the empirical residual distribution."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    artifact = {
        "classifier": result["selected_classifier"],
        "classifier_name": result["selected_classifier_name"],
        "margin_regressor": result["selected_regressor"],
        "margin_regressor_name": result["selected_regressor_name"],
        "margin_residuals": result["_selected_margin_residuals"],
        "feature_columns": result["feature_columns"],
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "splits": result["splits"],
    }
    with open(output_path, "wb") as handle:
        pickle.dump(artifact, handle)
