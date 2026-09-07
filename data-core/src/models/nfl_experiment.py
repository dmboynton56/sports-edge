"""Config-driven, chronological NFL v2 experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.optimize import minimize

from src.models.nfl_v2 import (
    FEATURE_COLUMNS,
    audit_nfl_feature_store,
    dataframe_fingerprint,
    margin_metrics,
    probability_metrics,
    write_json,
)


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class LinearPreprocessor:
    feature_columns: list[str]
    medians: pd.Series | None = None
    scales: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "LinearPreprocessor":
        numeric = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        self.medians = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.medians)
        self.scales = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        self.scales.loc["home_field"] = 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.scales is None:
            raise RuntimeError("Preprocessor has not been fit.")
        numeric = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(self.medians)
        # Do not center directional values: zero must remain the neutral/symmetric state.
        return (numeric / self.scales).to_numpy(dtype=float)


@dataclass
class FittedModel:
    kind: str
    estimator: Any
    features: list[str]
    preprocessor: LinearPreprocessor | None = None

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is not None:
            return self.preprocessor.transform(frame)
        return frame[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(self.estimator.predict_proba(self._matrix(frame))[:, 1], dtype=float)
        if "home_field" in frame:
            neutral = pd.to_numeric(frame["home_field"], errors="coerce").fillna(0).to_numpy() == 0
            if neutral.any():
                mirrored = frame.loc[neutral, self.features].copy()
                for column in self.features:
                    if column != "home_field":
                        mirrored[column] = -pd.to_numeric(mirrored[column], errors="coerce")
                mirror_probability = self.estimator.predict_proba(self._matrix(mirrored))[:, 1]
                probability[neutral] = (probability[neutral] + 1.0 - mirror_probability) / 2.0
        return probability

    def predict_margin(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(self._matrix(frame)), dtype=float)


@dataclass
class ProbabilityCalibrator:
    method: str
    estimator: Any = None

    def predict(self, raw_probability: Sequence[float]) -> np.ndarray:
        raw = np.asarray(raw_probability, dtype=float)
        if self.method == "none":
            return np.clip(raw, 1e-6, 1 - 1e-6)
        if self.method == "sigmoid":
            calibrated = self.estimator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        else:
            calibrated = np.asarray(self.estimator.predict(raw), dtype=float)
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


@dataclass
class CalibratedProbabilityModel:
    model: FittedModel
    calibrator: ProbabilityCalibrator

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_probability(frame)
        probability = self.calibrator.predict(raw)
        if "home_field" in frame:
            neutral = pd.to_numeric(frame["home_field"], errors="coerce").fillna(0).to_numpy() == 0
            if neutral.any():
                inverse = self.calibrator.predict(1.0 - raw[neutral])
                probability[neutral] = (probability[neutral] + 1.0 - inverse) / 2.0
        return probability


@dataclass
class MarketResidualModel:
    """Regularized correction with consensus log-odds held as a fixed offset."""

    preprocessor: LinearPreprocessor
    coefficients: np.ndarray

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        offset = _logit(frame["market_home_prob"].to_numpy())
        correction = self.preprocessor.transform(frame) @ self.coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(offset + correction, -30, 30)))


def fit_calibrator(method: str, y: Sequence[int], raw_probability: Sequence[float]) -> ProbabilityCalibrator:
    target = np.asarray(y, dtype=int)
    raw = np.asarray(raw_probability, dtype=float)
    if method == "none":
        return ProbabilityCalibrator(method="none")
    if method == "sigmoid":
        estimator = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(raw).reshape(-1, 1), target)
    elif method == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip").fit(raw, target)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return ProbabilityCalibrator(method=method, estimator=estimator)


def candidate_specs(config: Mapping[str, Any], task: str) -> list[dict[str, Any]]:
    specs = [dict(spec) for spec in config["candidates"][task]]
    for spec in specs:
        spec.setdefault("features", list(config.get("feature_columns", FEATURE_COLUMNS)))
    return specs


def fit_candidate(spec: Mapping[str, Any], frame: pd.DataFrame, *, task: str) -> FittedModel:
    features = list(spec.get("features", FEATURE_COLUMNS))
    kind = str(spec["kind"])
    y_column = "home_win" if task == "probability" else "home_margin"
    if kind in {"logistic", "ridge"}:
        prep = LinearPreprocessor(features).fit(frame)
        matrix = prep.transform(frame)
        if kind == "logistic":
            estimator = LogisticRegression(
                C=float(spec["C"]), penalty="l2", solver="lbfgs", fit_intercept=False, max_iter=3000
            )
        else:
            estimator = Ridge(alpha=float(spec["alpha"]), fit_intercept=False)
        estimator.fit(matrix, frame[y_column])
        return FittedModel(kind=kind, estimator=estimator, features=features, preprocessor=prep)
    if kind == "hist_gradient_boosting":
        kwargs = {
            "learning_rate": float(spec["learning_rate"]),
            "max_leaf_nodes": int(spec["max_leaf_nodes"]),
            "l2_regularization": float(spec.get("l2_regularization", 0.0)),
            "random_state": int(spec.get("random_state", 20260906)),
        }
        estimator = HistGradientBoostingClassifier(**kwargs) if task == "probability" else HistGradientBoostingRegressor(**kwargs)
    elif kind == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("LightGBM candidate requested but lightgbm is not installed.") from exc
        cls = LGBMClassifier if task == "probability" else LGBMRegressor
        estimator = cls(
            n_estimators=int(spec["n_estimators"]),
            learning_rate=float(spec["learning_rate"]),
            num_leaves=int(spec["num_leaves"]),
            min_child_samples=int(spec.get("min_child_samples", 30)),
            reg_lambda=float(spec.get("reg_lambda", 1.0)),
            verbosity=-1,
            random_state=int(spec.get("random_state", 20260906)),
        )
    else:
        raise ValueError(f"Unknown candidate kind: {kind}")
    matrix = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    estimator.fit(matrix, frame[y_column])
    return FittedModel(kind=kind, estimator=estimator, features=features)


def rolling_candidate_selection(
    data: pd.DataFrame, specs: Sequence[Mapping[str, Any]], folds: Sequence[Mapping[str, int]], *, task: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    metric = "brier" if task == "probability" else "mae"
    for spec in specs:
        fold_rows = []
        for fold in folds:
            train = data[data["season"] <= int(fold["train_through"])]
            test = data[data["season"] == int(fold["test_season"])]
            if train.empty or test.empty:
                raise ValueError(f"Empty rolling fold: {fold}")
            fitted = fit_candidate(spec, train, task=task)
            if task == "probability":
                fold_metrics = probability_metrics(test["home_win"], fitted.predict_probability(test))
                score = fold_metrics[metric]
            else:
                score = margin_metrics(test["home_margin"], fitted.predict_margin(test))[metric]
                fold_metrics = {}
            fold_rows.append(
                {
                    **fold,
                    metric: score,
                    **(
                        {
                            "auc": fold_metrics.get("auc"),
                            "log_loss": fold_metrics["log_loss"],
                            "ece_10": fold_metrics["ece_10"],
                            "home_probability_bias": fold_metrics["home_probability_bias"],
                            "sharpness_std": fold_metrics["sharpness_std"],
                        }
                        if task == "probability"
                        else {}
                    ),
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
        summary = {
            "spec": dict(spec),
            "folds": fold_rows,
            f"mean_{metric}": float(np.mean([x[metric] for x in fold_rows])),
        }
        if task == "probability":
            for diagnostic in ("log_loss", "ece_10", "home_probability_bias", "sharpness_std", "auc"):
                summary[f"mean_{diagnostic}"] = float(np.mean([x[diagnostic] for x in fold_rows]))
        results.append(summary)
    best = min(results, key=lambda row: row[f"mean_{metric}"])
    return dict(best["spec"]), results


def bootstrap_improvement_probability(
    y: Sequence[int], candidate: Sequence[float], baseline: Sequence[float], *, samples: int, seed: int
) -> float:
    target = np.asarray(y, dtype=float)
    candidate_loss = (np.asarray(candidate) - target) ** 2
    baseline_loss = (np.asarray(baseline) - target) ** 2
    rng = np.random.default_rng(seed)
    improvements = 0
    for _ in range(samples):
        index = rng.integers(0, len(target), len(target))
        improvements += float(candidate_loss[index].mean() < baseline_loss[index].mean())
    return float(improvements / samples)


def bootstrap_mean_interval(values: Sequence[float], *, samples: int, seed: int) -> dict[str, float]:
    observed = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        selection = rng.integers(0, len(observed), len(observed))
        means[index] = observed[selection].mean()
    return {
        "mean": float(observed.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _segments(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, np.ndarray]:
    week = pd.to_numeric(frame["week"], errors="coerce").fillna(0).to_numpy()
    game_type = frame["game_type"].astype(str).str.upper().to_numpy()
    neutral = frame["neutral_site"].fillna(False).astype(bool).to_numpy()
    return {
        "weeks_1_4": (week >= 1) & (week <= 4),
        "later_season": week >= 5,
        "playoffs": game_type != "REG",
        "favorites": probability >= 0.5,
        "underdogs": probability < 0.5,
        "neutral_sites": neutral,
    }


def segmented_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    output = {}
    for name, mask in _segments(frame, probability).items():
        if mask.sum():
            output[name] = probability_metrics(frame.loc[mask, "home_win"], probability[mask])
    return output


def home_field_diagnostics(model: FittedModel, frame: pd.DataFrame) -> dict[str, float]:
    home = frame.copy()
    neutral = frame.copy()
    home["home_field"] = 1.0
    neutral["home_field"] = 0.0
    lift = model.predict_probability(home) - model.predict_probability(neutral)
    return {
        "mean_counterfactual_home_lift": float(lift.mean()),
        "median_counterfactual_home_lift": float(np.median(lift)),
        "maximum_absolute_counterfactual_home_lift": float(np.max(np.abs(lift))),
    }


def market_track_status(data: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    timestamp_safe = bool(config["market_track"].get("timestamp_safe", False))
    required = float(config["market_track"].get("minimum_coverage", 0.9))
    required_columns = {"market_home_prob", "market_spread", "market_as_of_ts", "kickoff_ts"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        coverage = 0.0
        timestamp_violations = None
    else:
        complete = data[list(required_columns)].notna().all(axis=1)
        coverage = float(complete.mean())
        as_of = pd.to_datetime(data.loc[complete, "market_as_of_ts"], utc=True, errors="coerce")
        kickoff = pd.to_datetime(data.loc[complete, "kickoff_ts"], utc=True, errors="coerce")
        timestamp_violations = int((as_of.isna() | kickoff.isna() | (as_of >= kickoff)).sum())
    enabled = timestamp_safe and not missing_columns and coverage >= required and timestamp_violations == 0
    return {
        "enabled": enabled,
        "timestamp_safe": timestamp_safe,
        "coverage": coverage,
        "required_coverage": required,
        "missing_columns": missing_columns,
        "timestamp_violations": timestamp_violations,
        "reason": None if enabled else "Market inputs are benchmark-only until timestamp safety and coverage pass.",
    }


def evaluate_market_anchor(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, MarketResidualModel]:
    usable_train = train[train["market_home_prob"].notna() & train["market_spread"].notna()].copy()
    if usable_train.empty:
        raise ValueError("No timestamp-safe market training rows.")
    residual_features = ["market_spread", *features]
    preprocessor = LinearPreprocessor(residual_features).fit(usable_train)
    matrix = preprocessor.transform(usable_train)
    offset = _logit(usable_train["market_home_prob"].to_numpy())
    target = usable_train["home_win"].to_numpy(dtype=float)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        score = offset + matrix @ coefficients
        probability = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
        penalty = 2.0
        loss = float(np.mean(np.logaddexp(0.0, score) - target * score) + penalty * np.sum(coefficients**2))
        gradient = matrix.T @ (probability - target) / len(target) + 2 * penalty * coefficients
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(matrix.shape[1]),
        method="L-BFGS-B",
        jac=True,
    )
    if not fitted.success:
        raise RuntimeError(f"Market residual fit failed: {fitted.message}")
    model = MarketResidualModel(preprocessor=preprocessor, coefficients=np.asarray(fitted.x))
    return model.predict_probability(test), model


def _baseline_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "constant_50": np.full(len(test), 0.5),
        "historical_home_rate": np.full(len(test), float(train["home_win"].mean())),
    }


def _v1_predictions(test: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=test.index)
    v1 = pd.read_csv(path)
    probability_column = next(
        (c for c in ("final_probability", "prob_home", "home_win_probability") if c in v1), None
    )
    margin_column = next((c for c in ("predicted_margin", "home_margin_pred") if c in v1), None)
    keep = ["game_id", *([probability_column] if probability_column else []), *([margin_column] if margin_column else [])]
    joined = test[["game_id"]].merge(v1[keep], on="game_id", how="left")
    joined.index = test.index
    if probability_column:
        joined = joined.rename(columns={probability_column: "v1_probability"})
    if margin_column:
        joined = joined.rename(columns={margin_column: "v1_margin"})
    return joined


def fit_production_shadow_bundle(
    data: pd.DataFrame,
    probability_spec: Mapping[str, Any],
    margin_spec: Mapping[str, Any],
    calibration_method: str,
    *,
    first_oof_season: int = 2022,
) -> dict[str, Any]:
    """Refit through the latest season with rolling out-of-fold calibration."""
    raw_parts = []
    target_parts = []
    oof_seasons = []
    for season in sorted(data["season"].unique()):
        if int(season) < first_oof_season:
            continue
        train = data[data["season"] < int(season)]
        validation = data[data["season"] == int(season)]
        if train.empty or validation.empty:
            continue
        model = fit_candidate(probability_spec, train, task="probability")
        raw_parts.append(model.predict_probability(validation))
        target_parts.append(validation["home_win"].to_numpy())
        oof_seasons.append(int(season))
    if not raw_parts:
        raise ValueError("No rolling out-of-fold seasons are available for production calibration.")
    calibrator = fit_calibrator(calibration_method, np.concatenate(target_parts), np.concatenate(raw_parts))
    probability_model = fit_candidate(probability_spec, data, task="probability")
    return {
        "probability_model": CalibratedProbabilityModel(probability_model, calibrator),
        "margin_model": fit_candidate(margin_spec, data, task="margin"),
        "calibrator": calibrator,
        "oof_seasons": oof_seasons,
        "trained_through": int(data["season"].max()),
    }


def _promotion_decision(
    test: pd.DataFrame, candidate_probability: np.ndarray, candidate_margin: np.ndarray,
    metrics: Mapping[str, Any], baselines: Mapping[str, np.ndarray], v1: pd.DataFrame,
    config: Mapping[str, Any], probability_selection: Sequence[Mapping[str, Any]],
    selected_probability_spec: Mapping[str, Any], ablation: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["promotion_gates"]
    candidate = metrics["candidate_probability"]
    best_baseline_name = min(metrics["baselines"], key=lambda name: metrics["baselines"][name]["brier"])
    best_baseline = baselines[best_baseline_name]
    bootstrap_probability = bootstrap_improvement_probability(
        test["home_win"],
        candidate_probability,
        best_baseline,
        samples=int(gates["bootstrap_samples"]),
        seed=int(gates["bootstrap_seed"]),
    )
    validation_auc = {
        str(row["spec"]["name"]): float(np.mean([fold["auc"] for fold in row["folds"]]))
        for row in probability_selection
    }
    selected_validation_auc = validation_auc[str(selected_probability_spec["name"])]
    best_validation_auc = max(validation_auc.values())
    checks: dict[str, Any] = {
        "brier_beats_best_constant": candidate["brier"] < metrics["baselines"][best_baseline_name]["brier"],
        "log_loss_beats_both_constants": all(candidate["log_loss"] < row["log_loss"] for row in metrics["baselines"].values()),
        "bootstrap_brier_improvement": bootstrap_probability >= float(gates["bootstrap_probability"]),
        "ece": candidate["ece_10"] <= float(gates["max_ece"]),
        "home_probability_bias": abs(candidate["home_probability_bias"]) <= float(gates["max_home_probability_bias"]),
        "ablation_support": bool(ablation["supports_hypothesis"]),
    }
    segment_failures = [
        name for name, row in metrics["segments"].items()
        if row["rows"] >= int(gates["home_bias_segment_min_rows"])
        and abs(row["home_probability_bias"]) > float(gates["max_segment_home_bias"])
    ]
    checks["home_bias_segments"] = not segment_failures
    if "v1_probability" in v1 and v1["v1_probability"].notna().all():
        checks["brier_beats_v1"] = candidate["brier"] < metrics["v1_probability"]["brier"]
        checks["log_loss_beats_v1"] = candidate["log_loss"] < metrics["v1_probability"]["log_loss"]
    else:
        checks["brier_beats_v1"] = False
        checks["log_loss_beats_v1"] = False
    checks["auc"] = selected_validation_auc >= best_validation_auc - float(gates["max_auc_gap"])
    margin = metrics["candidate_margin"]
    checks["margin_home_bias"] = abs(margin["home_margin_bias"]) <= float(gates["max_home_margin_bias"])
    checks["margin_beats_v1"] = bool(
        "v1_margin" in metrics and margin["mae"] < metrics["v1_margin"]["mae"]
    )
    return {
        "advance_to_2026_shadow": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "failed_home_bias_segments": segment_failures,
        "diagnostics": {
            "best_constant_baseline": best_baseline_name,
            "bootstrap_probability_of_brier_improvement": bootstrap_probability,
            "bootstrap_samples": int(gates["bootstrap_samples"]),
            "selected_validation_auc": selected_validation_auc,
            "best_validation_auc": best_validation_auc,
            "validation_auc_by_candidate": validation_auc,
        },
        "locked_test_consumed": True,
        "shadow_requirement": "At least four weeks or 64 graded games, whichever is later.",
    }


def run_experiment(
    config_path: Path,
    dataset_path: Path,
    output_root: Path,
    *,
    allow_consumed_locked_test: bool = False,
) -> Path:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    config_fingerprint = sha256(config_bytes).hexdigest()
    if config.get("experiment_state", {}).get("locked_test_consumed") and not allow_consumed_locked_test:
        raise ValueError(
            "This configuration's locked test has already been consumed. "
            "Use rolling folds through 2025 and reserve graded 2026 shadow predictions as new evidence."
        )
    data = pd.read_parquet(dataset_path).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    audit = audit_nfl_feature_store(data, raise_on_error=True)
    fingerprint = dataframe_fingerprint(data)
    locked_season = int(config["windows"]["locked_test"])
    calibration_season = int(config["windows"]["calibration"])
    selection_folds = config["windows"]["selection"]

    probability_spec, probability_selection = rolling_candidate_selection(
        data, candidate_specs(config, "probability"), selection_folds, task="probability"
    )
    margin_spec, margin_selection = rolling_candidate_selection(
        data, candidate_specs(config, "margin"), selection_folds, task="margin"
    )
    ablation_columns = set(config.get("ablation_feature_columns", []))
    if not ablation_columns:
        raise ValueError("Experiment config must register ablation_feature_columns.")
    ablation_spec = dict(probability_spec)
    ablation_spec["name"] = f"{probability_spec['name']}_ablation"
    ablation_spec["features"] = [
        column for column in probability_spec["features"] if column not in ablation_columns
    ]
    _, ablation_results = rolling_candidate_selection(
        data, [ablation_spec], selection_folds, task="probability"
    )
    selected_validation = next(
        row for row in probability_selection if row["spec"]["name"] == probability_spec["name"]
    )
    ablation = {
        "removed_features": sorted(ablation_columns),
        "full_mean_brier": selected_validation["mean_brier"],
        "ablated_mean_brier": ablation_results[0]["mean_brier"],
        "brier_improvement": ablation_results[0]["mean_brier"] - selected_validation["mean_brier"],
        "supports_hypothesis": selected_validation["mean_brier"] < ablation_results[0]["mean_brier"],
        "guardrails": {
            diagnostic: {
                "full": selected_validation[f"mean_{diagnostic}"],
                "ablated": ablation_results[0][f"mean_{diagnostic}"],
            }
            for diagnostic in ("log_loss", "ece_10", "home_probability_bias", "sharpness_std", "auc")
        },
        "folds": ablation_results[0]["folds"],
    }
    calibration_train = data[data["season"] <= calibration_season - 1]
    calibration_rows = data[data["season"] == calibration_season]
    locked_test = data[data["season"] == locked_season]
    if calibration_train.empty or calibration_rows.empty or locked_test.empty:
        raise ValueError("Calibration or locked-test window is empty.")

    pre_calibration_model = fit_candidate(probability_spec, calibration_train, task="probability")
    calibration_raw = pre_calibration_model.predict_probability(calibration_rows)
    calibrator_results = []
    calibrators = {}
    for method in config["calibration_methods"]:
        calibrator = fit_calibrator(method, calibration_rows["home_win"], calibration_raw)
        calibrated = calibrator.predict(calibration_raw)
        calibrators[method] = calibrator
        calibrator_results.append({"method": method, **probability_metrics(calibration_rows["home_win"], calibrated)})
    selected_calibration = min(calibrator_results, key=lambda row: row["brier"])["method"]
    calibrator = calibrators[selected_calibration]

    pre_margin_model = fit_candidate(margin_spec, calibration_train, task="margin")
    calibration_margin = pre_margin_model.predict_margin(calibration_rows)
    margin_link = LogisticRegression(C=1e6, solver="lbfgs").fit(
        calibration_margin.reshape(-1, 1), calibration_rows["home_win"]
    )
    margin_residual_std = float(np.std(calibration_rows["home_margin"].to_numpy() - calibration_margin, ddof=1))

    final_train = data[data["season"] <= calibration_season]
    probability_model = fit_candidate(probability_spec, final_train, task="probability")
    margin_model = fit_candidate(margin_spec, final_train, task="margin")
    locked_raw = probability_model.predict_probability(locked_test)
    calibrated_model = CalibratedProbabilityModel(probability_model, calibrator)
    locked_probability = calibrated_model.predict_probability(locked_test)
    locked_margin = margin_model.predict_margin(locked_test)
    margin_probability = margin_link.predict_proba(locked_margin.reshape(-1, 1))[:, 1]
    baselines = _baseline_probabilities(final_train, locked_test)
    v1_path = Path(config["baselines"]["nfl_v1_predictions"])
    if not v1_path.is_absolute() and not v1_path.exists():
        v1_path = config_path.resolve().parents[3] / v1_path
    v1 = _v1_predictions(locked_test, v1_path)

    metrics: dict[str, Any] = {
        "candidate_probability": probability_metrics(locked_test["home_win"], locked_probability),
        "candidate_margin": margin_metrics(locked_test["home_margin"], locked_margin),
        "baselines": {name: probability_metrics(locked_test["home_win"], values) for name, values in baselines.items()},
        "segments": segmented_metrics(locked_test, locked_probability),
        "home_field_diagnostics": home_field_diagnostics(probability_model, locked_test),
        "margin_derived_probability": probability_metrics(locked_test["home_win"], margin_probability),
        "margin_residual_uncertainty_points": margin_residual_std,
    }
    if "v1_probability" in v1 and v1["v1_probability"].notna().all():
        metrics["v1_probability"] = probability_metrics(locked_test["home_win"], v1["v1_probability"])
    if "v1_margin" in v1 and v1["v1_margin"].notna().all():
        metrics["v1_margin"] = margin_metrics(locked_test["home_margin"], v1["v1_margin"])
    market_status = market_track_status(data, config)
    market_model = None
    if locked_test["closing_market_home_prob"].notna().all():
        metrics["market_only_benchmark"] = probability_metrics(
            locked_test["home_win"], locked_test["closing_market_home_prob"]
        )
    if market_status["enabled"] and locked_test[["market_home_prob", "market_spread"]].notna().all(axis=1).all():
        market_probability, market_model = evaluate_market_anchor(
            final_train, locked_test, list(config.get("feature_columns", FEATURE_COLUMNS))
        )
        market_only = locked_test["market_home_prob"].to_numpy(dtype=float)
        market_only_metrics = probability_metrics(locked_test["home_win"], market_only)
        metrics["market_only_timestamp_safe"] = market_only_metrics
        metrics["market_anchored"] = probability_metrics(locked_test["home_win"], market_probability)
        target = locked_test["home_win"].to_numpy(dtype=float)
        loss_delta = (market_probability - target) ** 2 - (market_only - target) ** 2
        market_relative = {
            "brier_delta_anchored_minus_market": bootstrap_mean_interval(
                loss_delta,
                samples=int(config["promotion_gates"]["bootstrap_samples"]),
                seed=int(config["promotion_gates"]["bootstrap_seed"]),
            )
        }
        if "closing_market_home_prob" in locked_test and locked_test["closing_market_home_prob"].notna().all():
            side = np.where(market_probability >= 0.5, 1.0, -1.0)
            clv = side * (
                locked_test["closing_market_home_prob"].to_numpy(dtype=float) - market_only
            )
            market_relative["signed_probability_clv"] = bootstrap_mean_interval(
                clv,
                samples=int(config["promotion_gates"]["bootstrap_samples"]),
                seed=int(config["promotion_gates"]["bootstrap_seed"]),
            )
        else:
            market_relative["signed_probability_clv"] = None
        metrics["market_relative"] = market_relative

    margin_probability_metrics = metrics["margin_derived_probability"]
    direct_metrics = metrics["candidate_probability"]
    metrics["margin_probability_ensemble_eligible"] = all(
        [
            margin_probability_metrics["brier"] < direct_metrics["brier"],
            margin_probability_metrics["log_loss"] < direct_metrics["log_loss"],
            margin_probability_metrics["ece_10"] < direct_metrics["ece_10"],
            abs(margin_probability_metrics["home_probability_bias"])
            < abs(direct_metrics["home_probability_bias"]),
        ]
    )

    decision = _promotion_decision(
        locked_test,
        locked_probability,
        locked_margin,
        metrics,
        baselines,
        v1,
        config,
        probability_selection,
        probability_spec,
        ablation,
    )
    created_at = datetime.now(timezone.utc)
    run_id = (
        f"{created_at.strftime('%Y%m%dT%H%M%SZ')}_"
        f"{fingerprint[:8]}_{config_fingerprint[:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    write_json(
        run_dir / "data_manifest.json",
        {
            "path": str(dataset_path.resolve()),
            "fingerprint": fingerprint,
            "config_fingerprint": config_fingerprint,
            "created_at": created_at.isoformat(),
            "audit": audit,
        },
    )
    write_json(
        run_dir / "feature_manifest.json",
        {"features": probability_model.features, "missingness": audit["feature_missingness"]},
    )
    write_json(run_dir / "selection_metrics.json", {"probability": probability_selection, "margin": margin_selection})
    write_json(run_dir / "ablation_metrics.json", ablation)
    write_json(run_dir / "calibration_metrics.json", {"selected": selected_calibration, "candidates": calibrator_results})
    write_json(run_dir / "locked_test_metrics.json", metrics)
    write_json(run_dir / "market_track.json", market_status)
    write_json(run_dir / "promotion_decision.json", decision)
    selection_ids = []
    for fold in selection_folds:
        selection_ids.append(
            {
                **fold,
                "train": data.loc[data["season"] <= int(fold["train_through"]), "game_id"].astype(str).tolist(),
                "test": data.loc[data["season"] == int(fold["test_season"]), "game_id"].astype(str).tolist(),
            }
        )
    split_ids = {
        "selection": selection_ids,
        "calibration_train": final_train[final_train["season"] < calibration_season]["game_id"].astype(str).tolist(),
        "calibration": calibration_rows["game_id"].astype(str).tolist(),
        "locked_test": locked_test["game_id"].astype(str).tolist(),
    }
    write_json(run_dir / "row_identities.json", split_ids)
    predictions = locked_test[["game_id", "season", "week", "game_date", "home_team", "away_team", "home_win", "home_margin"]].copy()
    predictions["raw_probability"] = locked_raw
    predictions["calibrated_probability"] = locked_probability
    predictions["predicted_margin"] = locked_margin
    predictions["margin_residual_uncertainty_points"] = margin_residual_std
    predictions["experimental_margin_probability"] = margin_probability
    for name, values in baselines.items():
        predictions[f"baseline_{name}"] = values
    predictions = pd.concat([predictions, v1.drop(columns=["game_id"], errors="ignore")], axis=1)
    predictions.to_csv(run_dir / "locked_test_predictions.csv", index=False)
    joblib.dump(calibrated_model, run_dir / "probability_model.joblib")
    joblib.dump(margin_model, run_dir / "margin_model.joblib")
    if market_model is not None:
        joblib.dump(market_model, run_dir / "market_model.joblib")
    if decision["advance_to_2026_shadow"]:
        shadow_bundle = fit_production_shadow_bundle(
            data[data["season"] <= locked_season],
            probability_spec,
            margin_spec,
            selected_calibration,
        )
        joblib.dump(shadow_bundle, run_dir / "production_shadow_bundle.joblib")
        write_json(
            run_dir / "production_shadow_manifest.json",
            {"oof_seasons": shadow_bundle["oof_seasons"], "trained_through": shadow_bundle["trained_through"]},
        )
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source_paths = [Path(__file__), Path(__file__).with_name("nfl_v2.py"), config_path]
    write_json(
        run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
            "source_fingerprints": {
                str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in source_paths
            },
        },
    )
    return run_dir
