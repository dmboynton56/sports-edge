"""Walk-forward post-hoc calibration for NFL v2 win probability and margin.

Fits never see the games they score. Methods are applied on earlier weeks only,
then evaluated on later weeks. Shipping a live artifact requires an out-of-sample
Brier and log-loss improvement without a MAE regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from src.models.nfl_v2 import margin_metrics, probability_metrics


MIN_TRAIN_ROWS = 40
MIN_OOS_ROWS = 40
BRIER_GAIN_FLOOR = 0.002
MAE_TOLERANCE = 0.05


def _clip_prob(probability: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)


def _logit(probability: Sequence[float]) -> np.ndarray:
    p = _clip_prob(probability)
    return np.log(p / (1.0 - p))


def _sigmoid(logit: Sequence[float]) -> np.ndarray:
    z = np.clip(np.asarray(logit, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class ProbabilityCalibrator:
    method: str
    params: dict[str, Any]

    def transform(self, probability: Sequence[float], market_probability: Sequence[float] | None = None) -> np.ndarray:
        raw = _clip_prob(probability)
        if self.method == "identity":
            return raw
        if self.method == "temperature":
            temperature = max(float(self.params["temperature"]), 1e-3)
            return _sigmoid(_logit(raw) / temperature)
        if self.method == "platt":
            intercept = float(self.params["intercept"])
            coef = float(self.params["coef"])
            return _sigmoid(coef * _logit(raw) + intercept)
        if self.method == "isotonic":
            x = np.asarray(self.params["x"], dtype=float)
            y = np.asarray(self.params["y"], dtype=float)
            return _clip_prob(np.interp(raw, x, y, left=y[0], right=y[-1]))
        if self.method == "market_shrink":
            if market_probability is None:
                raise ValueError("market_shrink requires market probabilities.")
            weight = float(np.clip(self.params["weight"], 0.0, 1.0))
            market = _clip_prob(market_probability)
            return _clip_prob((1.0 - weight) * raw + weight * market)
        raise ValueError(f"Unknown probability calibrator: {self.method}")


@dataclass(frozen=True)
class MarginCalibrator:
    method: str
    params: dict[str, Any]

    def transform(self, margin: Sequence[float]) -> np.ndarray:
        values = np.asarray(margin, dtype=float)
        if self.method == "identity":
            return values
        if self.method == "scale":
            return values * float(self.params["scale"])
        if self.method == "affine":
            return values * float(self.params["scale"]) + float(self.params["intercept"])
        raise ValueError(f"Unknown margin calibrator: {self.method}")


def fit_temperature(y: Sequence[int], probability: Sequence[float]) -> ProbabilityCalibrator:
    target = np.asarray(y, dtype=int)
    raw = _clip_prob(probability)

    def nll(temperature: float) -> float:
        calibrated = _sigmoid(_logit(raw) / max(temperature, 1e-3))
        return float(log_loss(target, calibrated, labels=[0, 1]))

    fitted = minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded")
    temperature = float(fitted.x) if fitted.success else 1.0
    return ProbabilityCalibrator("temperature", {"temperature": temperature})


def fit_platt(y: Sequence[int], probability: Sequence[float]) -> ProbabilityCalibrator:
    target = np.asarray(y, dtype=int)
    estimator = LogisticRegression(C=1e6, solver="lbfgs")
    estimator.fit(_logit(probability).reshape(-1, 1), target)
    return ProbabilityCalibrator(
        "platt",
        {"intercept": float(estimator.intercept_[0]), "coef": float(estimator.coef_[0][0])},
    )


def fit_isotonic(y: Sequence[int], probability: Sequence[float]) -> ProbabilityCalibrator:
    estimator = IsotonicRegression(out_of_bounds="clip")
    estimator.fit(_clip_prob(probability), np.asarray(y, dtype=int))
    x = np.asarray(estimator.X_thresholds_, dtype=float)
    y_hat = np.clip(np.asarray(estimator.y_thresholds_, dtype=float), 1e-6, 1 - 1e-6)
    return ProbabilityCalibrator("isotonic", {"x": x.tolist(), "y": y_hat.tolist()})


def fit_market_shrink(
    y: Sequence[int],
    probability: Sequence[float],
    market_probability: Sequence[float],
) -> ProbabilityCalibrator:
    target = np.asarray(y, dtype=int)
    raw = _clip_prob(probability)
    market = _clip_prob(market_probability)

    def nll(weight: float) -> float:
        mixed = _clip_prob((1.0 - weight) * raw + weight * market)
        return float(log_loss(target, mixed, labels=[0, 1]))

    fitted = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    weight = float(fitted.x) if fitted.success else 0.0
    return ProbabilityCalibrator("market_shrink", {"weight": weight})


def fit_margin_scale(actual: Sequence[float], predicted: Sequence[float]) -> MarginCalibrator:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)

    def mae(scale: float) -> float:
        return float(np.mean(np.abs(scale * p - y)))

    fitted = minimize_scalar(mae, bounds=(0.25, 2.5), method="bounded")
    scale = float(fitted.x) if fitted.success else 1.0
    return MarginCalibrator("scale", {"scale": scale})


def fit_margin_affine(actual: Sequence[float], predicted: Sequence[float]) -> MarginCalibrator:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if len(p) < 2 or float(np.std(p)) < 1e-6:
        return MarginCalibrator("affine", {"scale": 1.0, "intercept": 0.0})
    scale, intercept = np.polyfit(p, y, 1)
    return MarginCalibrator("affine", {"scale": float(scale), "intercept": float(intercept)})


IDENTITY_PROB = ProbabilityCalibrator("identity", {})
IDENTITY_MARGIN = MarginCalibrator("identity", {})


def _week_order(frame: pd.DataFrame) -> list[int]:
    weeks = pd.to_numeric(frame["week"], errors="coerce").dropna().astype(int)
    return sorted(weeks.unique().tolist())


def walk_forward_predictions(
    frame: pd.DataFrame,
    *,
    fit_fn: Callable[[pd.DataFrame], Any],
    apply_fn: Callable[[Any, pd.DataFrame], np.ndarray],
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Expanding-week walk-forward. Returns (oos_mask, oos_values aligned to frame)."""
    ordered = frame.sort_values(["week", "game_date", "game_id"], kind="mergesort")
    values = np.full(len(frame), np.nan, dtype=float)
    mask = np.zeros(len(frame), dtype=bool)
    folds: list[dict[str, Any]] = []
    weeks = _week_order(ordered)
    index_map = {idx: position for position, idx in enumerate(frame.index)}
    for i, week in enumerate(weeks):
        train = ordered[pd.to_numeric(ordered["week"], errors="coerce") < week]
        test = ordered[pd.to_numeric(ordered["week"], errors="coerce") == week]
        if len(train) < min_train_rows or test.empty:
            continue
        try:
            fitted = fit_fn(train)
            predicted = np.asarray(apply_fn(fitted, test), dtype=float)
        except Exception as exc:  # noqa: BLE001
            folds.append({"test_week": int(week), "train_rows": int(len(train)), "error": str(exc)})
            continue
        if len(predicted) != len(test):
            raise ValueError("Calibrator returned the wrong number of rows.")
        for loc, value in zip(test.index, predicted):
            values[index_map[loc]] = float(value)
            mask[index_map[loc]] = True
        folds.append(
            {
                "test_week": int(week),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_weeks": [int(w) for w in weeks[:i]],
            }
        )
    return mask, values, folds


def _probability_quality(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p = _clip_prob(probability)
    return {
        "rows": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        **{key: value for key, value in probability_metrics(y, p).items() if key in {"ece_10", "home_probability_bias", "auc"}},
    }


def evaluate_probability_method(
    frame: pd.DataFrame,
    method: str,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> dict[str, Any]:
    graded = frame.dropna(subset=["home_win", "my_home_win_prob"]).copy()

    def fit_fn(train: pd.DataFrame) -> ProbabilityCalibrator:
        y = train["home_win"].astype(int)
        p = train["my_home_win_prob"]
        if method == "temperature":
            return fit_temperature(y, p)
        if method == "platt":
            return fit_platt(y, p)
        if method == "isotonic":
            return fit_isotonic(y, p)
        if method == "market_shrink":
            usable = train.dropna(subset=["market_home_prob"])
            if len(usable) < min_train_rows:
                raise ValueError("Not enough market rows to fit shrink.")
            return fit_market_shrink(usable["home_win"].astype(int), usable["my_home_win_prob"], usable["market_home_prob"])
        raise ValueError(method)

    def apply_fn(calibrator: ProbabilityCalibrator, test: pd.DataFrame) -> np.ndarray:
        market = test["market_home_prob"] if "market_home_prob" in test.columns else None
        if method == "market_shrink":
            if market is None or market.isna().any():
                # Fail closed on missing market rather than silently mixing zeros.
                raise ValueError("market_shrink test rows are missing market probabilities.")
        return calibrator.transform(test["my_home_win_prob"], market)

    mask, values, folds = walk_forward_predictions(graded, fit_fn=fit_fn, apply_fn=apply_fn, min_train_rows=min_train_rows)
    oos = graded.loc[mask]
    if oos.empty:
        return {"method": method, "status": "insufficient_oos", "folds": folds, "oos_rows": 0}
    y = oos["home_win"].astype(int).to_numpy()
    baseline = _clip_prob(oos["my_home_win_prob"])
    candidate = _clip_prob(values[mask])
    return {
        "method": method,
        "status": "ok",
        "folds": folds,
        "oos_rows": int(mask.sum()),
        "baseline": _probability_quality(y, baseline),
        "candidate": _probability_quality(y, candidate),
        "delta": {
            "brier": float(_probability_quality(y, candidate)["brier"] - _probability_quality(y, baseline)["brier"]),
            "log_loss": float(_probability_quality(y, candidate)["log_loss"] - _probability_quality(y, baseline)["log_loss"]),
        },
    }


def evaluate_margin_method(
    frame: pd.DataFrame,
    method: str,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> dict[str, Any]:
    graded = frame.dropna(subset=["actual_margin", "predicted_margin"]).copy()

    def fit_fn(train: pd.DataFrame) -> MarginCalibrator:
        if method == "scale":
            return fit_margin_scale(train["actual_margin"], train["predicted_margin"])
        if method == "affine":
            return fit_margin_affine(train["actual_margin"], train["predicted_margin"])
        raise ValueError(method)

    def apply_fn(calibrator: MarginCalibrator, test: pd.DataFrame) -> np.ndarray:
        return calibrator.transform(test["predicted_margin"])

    mask, values, folds = walk_forward_predictions(graded, fit_fn=fit_fn, apply_fn=apply_fn, min_train_rows=min_train_rows)
    oos = graded.loc[mask]
    if oos.empty:
        return {"method": method, "status": "insufficient_oos", "folds": folds, "oos_rows": 0}
    actual = oos["actual_margin"].to_numpy(dtype=float)
    baseline = oos["predicted_margin"].to_numpy(dtype=float)
    candidate = values[mask]
    return {
        "method": method,
        "status": "ok",
        "folds": folds,
        "oos_rows": int(mask.sum()),
        "baseline": margin_metrics(actual, baseline),
        "candidate": margin_metrics(actual, candidate),
        "delta": {
            "mae": float(mean_absolute_error(actual, candidate) - mean_absolute_error(actual, baseline)),
            "rmse": float(margin_metrics(actual, candidate)["rmse"] - margin_metrics(actual, baseline)["rmse"]),
        },
    }


def ship_decision(
    probability_results: Sequence[Mapping[str, Any]],
    margin_results: Sequence[Mapping[str, Any]],
    *,
    min_oos_rows: int = MIN_OOS_ROWS,
    brier_gain_floor: float = BRIER_GAIN_FLOOR,
    mae_tolerance: float = MAE_TOLERANCE,
) -> dict[str, Any]:
    eligible = []
    for result in probability_results:
        if result.get("status") != "ok":
            continue
        if int(result.get("oos_rows") or 0) < min_oos_rows:
            continue
        delta = result.get("delta") or {}
        if float(delta.get("brier", 0.0)) >= -brier_gain_floor:
            continue
        if float(delta.get("log_loss", 0.0)) >= 0.0:
            continue
        eligible.append(result)

    margin_ok = {
        result["method"]: result
        for result in margin_results
        if result.get("status") == "ok" and int(result.get("oos_rows") or 0) >= min_oos_rows
    }
    raw_mae_delta = 0.0
    chosen_margin = "identity"
    if "scale" in margin_ok and float(margin_ok["scale"]["delta"]["mae"]) <= mae_tolerance:
        chosen_margin = "scale"
        raw_mae_delta = float(margin_ok["scale"]["delta"]["mae"])
    if "affine" in margin_ok and float(margin_ok["affine"]["delta"]["mae"]) <= min(raw_mae_delta, mae_tolerance):
        if float(margin_ok["affine"]["delta"]["mae"]) < raw_mae_delta:
            chosen_margin = "affine"
            raw_mae_delta = float(margin_ok["affine"]["delta"]["mae"])

    if not eligible:
        return {
            "ship": False,
            "reason": "No walk-forward probability method improved OOS Brier and log loss enough without using scored games.",
            "probability_method": "identity",
            "margin_method": "identity",
            "eligible_probability_methods": [],
        }

    best = min(eligible, key=lambda row: row["candidate"]["brier"])
    margin_hurts = False
    if chosen_margin != "identity":
        margin_hurts = float(margin_ok[chosen_margin]["delta"]["mae"]) > mae_tolerance
    if margin_hurts:
        chosen_margin = "identity"

    return {
        "ship": True,
        "reason": (
            f"{best['method']} improved walk-forward Brier by "
            f"{-best['delta']['brier']:.4f} and log loss by {-best['delta']['log_loss']:.4f}."
        ),
        "probability_method": best["method"],
        "margin_method": chosen_margin,
        "probability_delta": best["delta"],
        "margin_delta": None if chosen_margin == "identity" else margin_ok[chosen_margin]["delta"],
        "eligible_probability_methods": [row["method"] for row in eligible],
        "oos_rows": best["oos_rows"],
    }


def fit_production_calibrators(
    frame: pd.DataFrame,
    probability_method: str,
    margin_method: str,
) -> dict[str, Any]:
    """Fit on all completed rows for *future* games. Do not use these params on the scored sample."""
    graded = frame.dropna(subset=["home_win", "my_home_win_prob", "actual_margin", "predicted_margin"]).copy()
    y = graded["home_win"].astype(int)
    p = graded["my_home_win_prob"]
    if probability_method == "temperature":
        probability = fit_temperature(y, p)
    elif probability_method == "platt":
        probability = fit_platt(y, p)
    elif probability_method == "isotonic":
        probability = fit_isotonic(y, p)
    elif probability_method == "market_shrink":
        usable = graded.dropna(subset=["market_home_prob"])
        probability = fit_market_shrink(usable["home_win"].astype(int), usable["my_home_win_prob"], usable["market_home_prob"])
    else:
        probability = IDENTITY_PROB

    if margin_method == "scale":
        margin = fit_margin_scale(graded["actual_margin"], graded["predicted_margin"])
    elif margin_method == "affine":
        margin = fit_margin_affine(graded["actual_margin"], graded["predicted_margin"])
    else:
        margin = IDENTITY_MARGIN

    return {
        "probability": {"method": probability.method, "params": probability.params},
        "margin": {"method": margin.method, "params": margin.params},
        "trained_rows": int(len(graded)),
        "trained_weeks": sorted(pd.to_numeric(graded["week"], errors="coerce").dropna().astype(int).unique().tolist()),
    }


def calibrators_from_artifact(payload: Mapping[str, Any]) -> tuple[ProbabilityCalibrator, MarginCalibrator]:
    probability = payload.get("probability") or {"method": "identity", "params": {}}
    margin = payload.get("margin") or {"method": "identity", "params": {}}
    return (
        ProbabilityCalibrator(str(probability.get("method", "identity")), dict(probability.get("params") or {})),
        MarginCalibrator(str(margin.get("method", "identity")), dict(margin.get("params") or {})),
    )
