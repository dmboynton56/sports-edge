"""Margin-consistent NFL win probability.

The frozen HGB classifier and ridge margin model stay untouched. Live win
probability can be derived from the ridge spread via Phi(-spread / sigma),
optionally stacked with the classifier logit. Sigma and the stack are fit
walk-forward on prior-season OOF residuals only.

This module also owns the serving guard: a published row may not have a
win-prob side that disagrees with the spread side outside the toss-up band.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression

from src.models.nfl_bucket_backtest import (
    _oof_probability_frame,
    json_safe,
    load_live_specs,
    predict_bundle,
)
from src.models.nfl_calibrator_promotion import (
    compare_predictions,
    score_probability_and_margin,
)
from src.models.nfl_experiment import fit_calibrator, fit_candidate
from src.models.nfl_toss_up import NFL_TOSS_UP_BAND, is_nfl_toss_up
from src.models.nfl_v2 import FEATURE_COLUMNS

DEFAULT_WINPROB_SIDECAR = Path(__file__).resolve().parents[2] / "models" / "nfl_v2_winprob.json"


LOGGER = logging.getLogger("nfl_margin_probability")

WIN_PROB_METHODS = ("isotonic", "margin_normal", "blend")
CANDIDATE_LIVE_MODEL_VERSION = "nfl-v2-live-20261020"
MIN_SIGMA = 8.0
MAX_SIGMA = 20.0
# Conservative fallback: 2021–2025 walk-forward ridge RMSE from the frozen
# live candidate (~13.21). Eval overwrites this via the artifact sidecar.
FALLBACK_RESIDUAL_SIGMA = 13.21
MIN_BLEND_ROWS = 200
BLEND_C = 1e6
PROB_CLIP = 1e-6
SPREAD_PICKEM_EPS = 1e-9


def _clip_prob(probability: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), PROB_CLIP, 1.0 - PROB_CLIP)


def logit(probability: Sequence[float]) -> np.ndarray:
    p = _clip_prob(probability)
    return np.log(p / (1.0 - p))


def sigmoid(values: Sequence[float]) -> np.ndarray:
    z = np.clip(np.asarray(values, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def residual_sigma(residuals: Sequence[float]) -> float:
    """RMSE of mean-zero Gaussian residuals, floored/capped for serving."""
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return FALLBACK_RESIDUAL_SIGMA
    rmse = float(np.sqrt(np.mean(np.square(values))))
    if not np.isfinite(rmse) or rmse <= 0:
        return FALLBACK_RESIDUAL_SIGMA
    return float(np.clip(rmse, MIN_SIGMA, MAX_SIGMA))


def margin_implied_home_win_prob(
    margin: Sequence[float] | float,
    sigma: float,
) -> np.ndarray:
    """P(home win) = Phi(margin / sigma) = Phi(-spread / sigma)."""
    scale = float(sigma) if np.isfinite(sigma) and sigma > 0 else FALLBACK_RESIDUAL_SIGMA
    scale = float(np.clip(scale, MIN_SIGMA, MAX_SIGMA))
    z = np.asarray(margin, dtype=float) / scale
    return _clip_prob(ndtr(z))


def spread_side(spread: object) -> str | None:
    try:
        value = float(spread)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or abs(value) < SPREAD_PICKEM_EPS:
        return None
    return "home" if value < 0 else "away"


def probability_side(home_win_prob: object, band: float = NFL_TOSS_UP_BAND) -> str | None:
    try:
        value = float(home_win_prob)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    if is_nfl_toss_up(value, band=band):
        return None
    if value > 0.5:
        return "home"
    if value < 0.5:
        return "away"
    return None


def is_sign_contradiction(
    home_win_prob: object,
    predicted_spread: object,
    *,
    band: float = NFL_TOSS_UP_BAND,
) -> bool:
    """True when p and spread disagree on the side outside the toss-up band."""
    prob = probability_side(home_win_prob, band=band)
    spread = spread_side(predicted_spread)
    if prob is None or spread is None:
        return False
    return prob != spread


def count_sign_contradictions(
    home_win_prob: Sequence[float],
    predicted_spread: Sequence[float],
    *,
    band: float = NFL_TOSS_UP_BAND,
) -> int:
    return int(
        sum(
            is_sign_contradiction(prob, spread, band=band)
            for prob, spread in zip(home_win_prob, predicted_spread, strict=False)
        )
    )


@dataclass(frozen=True)
class BlendStack:
    intercept: float
    coef_margin_p: float
    coef_classifier_logit: float

    def predict(self, margin_p: Sequence[float], classifier_logit: Sequence[float]) -> np.ndarray:
        linear = (
            self.intercept
            + self.coef_margin_p * np.asarray(margin_p, dtype=float)
            + self.coef_classifier_logit * np.asarray(classifier_logit, dtype=float)
        )
        return _clip_prob(sigmoid(linear))

    def to_dict(self) -> dict[str, float]:
        return {
            "intercept": float(self.intercept),
            "coef_margin_p": float(self.coef_margin_p),
            "coef_classifier_logit": float(self.coef_classifier_logit),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "BlendStack | None":
        if not payload:
            return None
        try:
            return cls(
                intercept=float(payload["intercept"]),
                coef_margin_p=float(payload["coef_margin_p"]),
                coef_classifier_logit=float(payload["coef_classifier_logit"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def fit_blend_stack(
    y: Sequence[int],
    margin_p: Sequence[float],
    classifier_logit: Sequence[float],
) -> BlendStack:
    target = np.asarray(y, dtype=int)
    features = np.column_stack(
        [_clip_prob(margin_p), np.asarray(classifier_logit, dtype=float)]
    )
    estimator = LogisticRegression(C=BLEND_C, solver="lbfgs")
    estimator.fit(features, target)
    return BlendStack(
        intercept=float(estimator.intercept_[0]),
        coef_margin_p=float(estimator.coef_[0][0]),
        coef_classifier_logit=float(estimator.coef_[0][1]),
    )


def attach_win_prob_config(
    artifact: Mapping[str, Any],
    *,
    method: str,
    residual_sigma: float,
    blend: BlendStack | None,
    model_version: str,
    parent_model_version: str,
    gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if method not in WIN_PROB_METHODS:
        raise ValueError(f"Unknown win-prob method: {method}")
    out = dict(artifact)
    out["model_version"] = model_version
    out["parent_model_version"] = parent_model_version
    selection = dict(out.get("selection") or {})
    selection["model_version"] = model_version
    selection["win_prob_method"] = method
    selection["parent_model_version"] = parent_model_version
    selection["activation_reason"] = (
        "margin_consistent_win_prob" if method != "isotonic" else "frozen_isotonic_with_spread_guard"
    )
    out["selection"] = selection
    out["win_prob"] = {
        "method": method,
        "residual_sigma": float(residual_sigma),
        "blend": None if blend is None else blend.to_dict(),
        "guard": "fallback_to_margin_implied_on_sign_contradiction",
        "gate": dict(gate or {}),
    }
    return out


def load_win_prob_sidecar(path: Path = DEFAULT_WINPROB_SIDECAR) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def win_prob_config_from_artifact(
    artifact: Mapping[str, Any] | None,
    *,
    sidecar: bool = True,
) -> dict[str, Any]:
    payload = dict((artifact or {}).get("win_prob") or {})
    if not payload and sidecar:
        payload = load_win_prob_sidecar()
    method = str(payload.get("method") or "isotonic")
    if method not in WIN_PROB_METHODS:
        method = "isotonic"
    sigma = payload.get("residual_sigma", FALLBACK_RESIDUAL_SIGMA)
    try:
        sigma_value = float(sigma)
    except (TypeError, ValueError):
        sigma_value = FALLBACK_RESIDUAL_SIGMA
    if not np.isfinite(sigma_value) or sigma_value <= 0:
        sigma_value = FALLBACK_RESIDUAL_SIGMA
    return {
        "method": method,
        "residual_sigma": float(np.clip(sigma_value, MIN_SIGMA, MAX_SIGMA)),
        "blend": BlendStack.from_dict(payload.get("blend")),
    }


def classifier_raw_probability(artifact: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    model = artifact["probability_model"]
    inner = getattr(model, "model", model)
    return np.asarray(inner.predict_probability(frame), dtype=float)


def choose_published_probability(
    *,
    classifier_p: Sequence[float],
    margin: Sequence[float],
    method: str,
    sigma: float,
    blend: BlendStack | None = None,
    classifier_logit: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    margin_p = margin_implied_home_win_prob(margin, sigma)
    classifier = _clip_prob(classifier_p)
    if method == "margin_normal":
        published = margin_p
    elif method == "blend":
        if blend is None:
            LOGGER.warning("Blend win probability is missing stack coefficients; using margin-implied p.")
            published = margin_p
        else:
            logits = (
                np.asarray(classifier_logit, dtype=float)
                if classifier_logit is not None
                else logit(classifier)
            )
            published = blend.predict(margin_p, logits)
    else:
        published = classifier
    return np.asarray(published, dtype=float), np.asarray(margin_p, dtype=float)


def reconcile_published_probability(
    published: Sequence[float],
    margin: Sequence[float],
    margin_p: Sequence[float],
    *,
    game_ids: Sequence[object] | None = None,
    log: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace sign-contradicting rows with the margin-implied probability."""
    published_arr = _clip_prob(published)
    margin_arr = np.asarray(margin, dtype=float)
    margin_p_arr = _clip_prob(margin_p)
    spread = -margin_arr
    reconciled = np.zeros(len(published_arr), dtype=bool)
    ids = list(game_ids) if game_ids is not None else [None] * len(published_arr)
    for index, (prob, line, fallback, game_id) in enumerate(
        zip(published_arr, spread, margin_p_arr, ids, strict=False)
    ):
        if not is_sign_contradiction(prob, line):
            continue
        reconciled[index] = True
        published_arr[index] = float(fallback)
        if log:
            LOGGER.warning(
                "NFL win-prob/spread contradiction game_id=%s p=%.4f spread=%.2f; "
                "falling back to margin-implied p=%.4f",
                game_id,
                float(prob),
                float(line),
                float(fallback),
            )
    return published_arr, reconciled


def score_win_prob(
    frame: pd.DataFrame,
    probability: Sequence[float],
    margin: Sequence[float],
) -> dict[str, Any]:
    metrics = score_probability_and_margin(frame, probability, margin)
    spread = -np.asarray(margin, dtype=float)
    p = np.asarray(probability, dtype=float)
    metrics["sign_contradictions"] = count_sign_contradictions(p, spread)
    return metrics


def compare_win_prob(
    frame: pd.DataFrame,
    baseline_probability: Sequence[float],
    candidate_probability: Sequence[float],
    margin: Sequence[float],
) -> dict[str, Any]:
    comparison = compare_predictions(
        frame,
        baseline_probability,
        candidate_probability,
        margin,
        margin,
    )
    spread = -np.asarray(margin, dtype=float)
    comparison["baseline"]["sign_contradictions"] = count_sign_contradictions(
        baseline_probability, spread
    )
    comparison["candidate"]["sign_contradictions"] = count_sign_contradictions(
        candidate_probability, spread
    )
    comparison["delta"]["sign_contradictions"] = (
        comparison["candidate"]["sign_contradictions"] - comparison["baseline"]["sign_contradictions"]
    )
    return comparison


def _ci_not_clearly_worse(interval: Mapping[str, Any]) -> bool:
    return float(interval["lower_95"]) <= 0.0


def margin_winprob_gate(
    *,
    walk_margin: Mapping[str, Any],
    walk_blend: Mapping[str, Any],
) -> dict[str, Any]:
    """Ship if a candidate is no worse than frozen on WF Brier and log loss (within CI) and has 0 contradictions."""

    def _eligible(side: Mapping[str, Any], name: str) -> dict[str, Any]:
        contradictions = int(side["candidate"].get("sign_contradictions") or 0)
        brier_ok = _ci_not_clearly_worse(side["delta_brier_ci"])
        log_ok = _ci_not_clearly_worse(side["delta_log_loss_ci"])
        eligible = brier_ok and log_ok and contradictions == 0
        return {
            "method": name,
            "eligible": eligible,
            "contradictions_removed": contradictions == 0,
            "walk_brier_not_clearly_worse": brier_ok,
            "walk_log_loss_not_clearly_worse": log_ok,
            "walk_brier": float(side["candidate"]["brier"]),
            "walk_log_loss": float(side["candidate"]["log_loss"]),
            "sign_contradictions": contradictions,
        }

    margin_row = _eligible(walk_margin, "margin_normal")
    blend_row = _eligible(walk_blend, "blend")
    eligible = [row for row in (margin_row, blend_row) if row["eligible"]]
    selected = None
    if eligible:
        selected = min(eligible, key=lambda row: (row["walk_brier"], row["walk_log_loss"]))
    reasons: list[str] = []
    if selected is None:
        reasons.append(
            "neither margin-normal nor blend is no-worse than frozen on walk-forward "
            "Brier and log loss (within CI) while removing sign contradictions"
        )
        for row in (margin_row, blend_row):
            reasons.append(
                f"{row['method']}: eligible={row['eligible']} contradictions={row['sign_contradictions']} "
                f"brier_ci_ok={row['walk_brier_not_clearly_worse']} "
                f"logloss_ci_ok={row['walk_log_loss_not_clearly_worse']}"
            )
    else:
        reasons.append(
            f"ship {selected['method']}: walk-forward Brier and log loss not clearly worse than "
            "frozen isotonic, and sign contradictions outside the toss-up band are 0"
        )
    return {
        "ship": selected is not None,
        "selected_method": None if selected is None else selected["method"],
        "reason": "; ".join(reasons),
        "candidates": [margin_row, blend_row],
    }


def _actual_margin(frame: pd.DataFrame) -> pd.Series:
    if "home_margin" in frame.columns:
        values = pd.to_numeric(frame["home_margin"], errors="coerce")
        if values.notna().any():
            return values
    return pd.to_numeric(frame.get("home_score"), errors="coerce") - pd.to_numeric(
        frame.get("away_score"), errors="coerce"
    )


def _ensure_home_win(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "home_win" not in out.columns or pd.to_numeric(out["home_win"], errors="coerce").isna().all():
        out["home_win"] = (
            pd.to_numeric(out["home_score"], errors="coerce") > pd.to_numeric(out["away_score"], errors="coerce")
        ).astype(int)
    if "home_margin" not in out.columns or pd.to_numeric(out.get("home_margin"), errors="coerce").isna().all():
        out["home_margin"] = _actual_margin(out)
    return out


def _completed_regular_season(features: pd.DataFrame) -> pd.DataFrame:
    completed = features[features["home_score"].notna() & features["away_score"].notna()].copy()
    if "game_type" in completed.columns:
        completed = completed[completed["game_type"].astype(str).str.upper().eq("REG")].copy()
    completed = completed[completed[FEATURE_COLUMNS].notna().all(axis=1)].copy()
    return _ensure_home_win(completed)


def walk_forward_margin_winprob(
    features: pd.DataFrame,
    specs: Mapping[str, Any],
    *,
    test_seasons: Sequence[int] = (2021, 2022, 2023, 2024, 2025),
) -> dict[str, Any]:
    """One HGB+ridge walk-forward; score isotonic, sigmoid, Phi, and blend."""
    completed = _completed_regular_season(features)
    rows: list[pd.DataFrame] = []
    oof_history: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []

    for season in test_seasons:
        train = completed[pd.to_numeric(completed["season"], errors="coerce") < int(season)].copy()
        test = completed[pd.to_numeric(completed["season"], errors="coerce") == int(season)].copy()
        if train.empty or test.empty:
            continue
        train_seasons = sorted(
            int(value) for value in pd.to_numeric(train["season"], errors="coerce").dropna().unique()
        )
        probability_model = fit_candidate(specs["probability_spec"], train, task="probability")
        margin_model = fit_candidate(specs["margin_spec"], train, task="margin")
        raw_p = np.asarray(probability_model.predict_probability(test), dtype=float)
        pred_margin = np.asarray(margin_model.predict_margin(test), dtype=float)
        actual = _actual_margin(test).to_numpy(dtype=float)

        if len(train_seasons) < 2:
            iso_p = raw_p
            sig_p = raw_p
            iso_method = "none"
            sig_method = "none"
            oof_rows = 0
            train_pred = np.asarray(margin_model.predict_margin(train), dtype=float)
            train_actual = _actual_margin(train).to_numpy(dtype=float)
            sigma = residual_sigma(train_actual - train_pred)
            sigma_source = "in_sample_prior_season"
        else:
            oof = _oof_probability_frame(train, specs["probability_spec"], int(train_seasons[1]))
            oof_rows = int(len(oof))
            iso_cal = fit_calibrator("isotonic", oof["home_win"], oof["raw_probability"])
            sig_cal = fit_calibrator("sigmoid", oof["home_win"], oof["raw_probability"])
            iso_p = iso_cal.predict(raw_p)
            sig_p = sig_cal.predict(raw_p)
            iso_method = "isotonic"
            sig_method = "sigmoid"
            if oof_history:
                prior = pd.concat(oof_history, ignore_index=True)
                sigma = residual_sigma(prior["residual"])
                sigma_source = "oof_prior_seasons"
            else:
                train_pred = np.asarray(margin_model.predict_margin(train), dtype=float)
                train_actual = _actual_margin(train).to_numpy(dtype=float)
                sigma = residual_sigma(train_actual - train_pred)
                sigma_source = "in_sample_prior_seasons"

        margin_p = margin_implied_home_win_prob(pred_margin, sigma)
        raw_logit = logit(raw_p)
        if oof_history and sum(len(chunk) for chunk in oof_history) >= MIN_BLEND_ROWS:
            prior = pd.concat(oof_history, ignore_index=True)
            blend = fit_blend_stack(prior["home_win"], prior["margin_p"], prior["raw_logit"])
            blend_p = blend.predict(margin_p, raw_logit)
            blend_source = "oof_prior_seasons"
            blend_params = blend.to_dict()
        else:
            blend_p = margin_p
            blend_source = "margin_normal_cold_start"
            blend_params = None

        chunk = test.copy()
        chunk["raw_probability"] = raw_p
        chunk["raw_logit"] = raw_logit
        chunk["isotonic_p"] = iso_p
        chunk["sigmoid_p"] = sig_p
        chunk["margin_p"] = margin_p
        chunk["blend_p"] = blend_p
        chunk["predicted_margin"] = pred_margin
        chunk["my_spread"] = -pred_margin
        chunk["residual"] = actual - pred_margin
        chunk["sigma"] = sigma
        chunk["test_season"] = int(season)
        rows.append(chunk)
        oof_history.append(chunk)
        fits.append(
            {
                "test_season": int(season),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_seasons": train_seasons,
                "oof_calibrator_rows": oof_rows,
                "isotonic_method": iso_method,
                "sigmoid_method": sig_method,
                "sigma": sigma,
                "sigma_source": sigma_source,
                "blend_source": blend_source,
                "blend": blend_params,
            }
        )

    if not rows:
        raise ValueError("Walk-forward margin win-prob produced an empty frame.")
    predicted = pd.concat(rows, ignore_index=True)
    live_sigma = residual_sigma(predicted["residual"])
    live_blend = fit_blend_stack(predicted["home_win"], predicted["margin_p"], predicted["raw_logit"])
    return {
        "frame": predicted,
        "fits": fits,
        "live_sigma": live_sigma,
        "live_blend": live_blend,
        "n": int(len(predicted)),
    }


def evaluate_margin_winprob(
    features: pd.DataFrame,
    *,
    frozen: Mapping[str, Any] | None = None,
    specs: Mapping[str, Any] | None = None,
    holdout_season: int = 2026,
) -> dict[str, Any]:
    from src.models.nfl_live import load_live_artifact

    specs = specs or load_live_specs()
    frozen = frozen or load_live_artifact()
    completed = _completed_regular_season(features)
    train = completed[pd.to_numeric(completed["season"], errors="coerce") <= 2025].copy()
    holdout = completed[pd.to_numeric(completed["season"], errors="coerce") == int(holdout_season)].copy()
    if train.empty or holdout.empty:
        raise ValueError("Need 2020–2025 train rows and a 2026 holdout.")

    walk = walk_forward_margin_winprob(completed, specs)
    walk_frame = walk["frame"]
    walk_isotonic = compare_win_prob(
        walk_frame, walk_frame["isotonic_p"], walk_frame["isotonic_p"], walk_frame["predicted_margin"]
    )
    walk_sigmoid = compare_win_prob(
        walk_frame, walk_frame["isotonic_p"], walk_frame["sigmoid_p"], walk_frame["predicted_margin"]
    )
    walk_margin = compare_win_prob(
        walk_frame, walk_frame["isotonic_p"], walk_frame["margin_p"], walk_frame["predicted_margin"]
    )
    walk_blend = compare_win_prob(
        walk_frame, walk_frame["isotonic_p"], walk_frame["blend_p"], walk_frame["predicted_margin"]
    )

    frozen_holdout = predict_bundle(holdout, frozen)
    frozen_p = np.asarray(frozen_holdout["my_home_win_prob"], dtype=float)
    frozen_margin = np.asarray(frozen_holdout["predicted_margin"], dtype=float)
    frozen_raw = classifier_raw_probability(frozen, holdout)
    holdout_sigma = float(walk["live_sigma"])
    holdout_margin_p = margin_implied_home_win_prob(frozen_margin, holdout_sigma)
    holdout_blend_p = walk["live_blend"].predict(holdout_margin_p, logit(frozen_raw))
    sig_cal = fit_calibrator("sigmoid", walk_frame["home_win"], walk_frame["raw_probability"])
    holdout_sigmoid_p = sig_cal.predict(frozen_raw)

    holdout_sigmoid = compare_win_prob(holdout, frozen_p, holdout_sigmoid_p, frozen_margin)
    holdout_margin = compare_win_prob(holdout, frozen_p, holdout_margin_p, frozen_margin)
    holdout_blend = compare_win_prob(holdout, frozen_p, holdout_blend_p, frozen_margin)

    gate = margin_winprob_gate(walk_margin=walk_margin, walk_blend=walk_blend)
    method = gate["selected_method"] if gate["ship"] else "isotonic"
    payload = json_safe(
        {
            "parent_model_version": frozen.get("model_version"),
            "candidate_model_version": CANDIDATE_LIVE_MODEL_VERSION,
            "gate": gate,
            "live_sigma": holdout_sigma,
            "live_blend": walk["live_blend"].to_dict(),
            "win_prob_method_if_shipped": method,
            "walk_forward_fits": walk["fits"],
            "walk_forward_2021_2025": {
                "frozen_isotonic": walk_isotonic["baseline"],
                "pr45_sigmoid": walk_sigmoid["candidate"],
                "margin_normal": walk_margin["candidate"],
                "blend": walk_blend["candidate"],
                "vs_frozen": {
                    "pr45_sigmoid": {key: walk_sigmoid[key] for key in ("delta", "delta_brier_ci", "delta_log_loss_ci")},
                    "margin_normal": {
                        key: walk_margin[key] for key in ("delta", "delta_brier_ci", "delta_log_loss_ci")
                    },
                    "blend": {key: walk_blend[key] for key in ("delta", "delta_brier_ci", "delta_log_loss_ci")},
                },
            },
            "holdout_2026": {
                "frozen_isotonic": holdout_sigmoid["baseline"],
                "pr45_sigmoid": holdout_sigmoid["candidate"],
                "margin_normal": holdout_margin["candidate"],
                "blend": holdout_blend["candidate"],
                "vs_frozen": {
                    "pr45_sigmoid": {
                        key: holdout_sigmoid[key] for key in ("delta", "delta_brier_ci", "delta_log_loss_ci")
                    },
                    "margin_normal": {
                        key: holdout_margin[key] for key in ("delta", "delta_brier_ci", "delta_log_loss_ci")
                    },
                    "blend": {key: holdout_blend[key] for key in ("delta", "delta_brier_ci", "delta_log_loss_ci")},
                },
            },
        }
    )
    return {
        "payload": payload,
        "walk_frame": walk_frame,
        "holdout": {
            "game_id": holdout["game_id"].astype(str).tolist(),
            "frozen_p": frozen_p.tolist(),
            "sigmoid_p": np.asarray(holdout_sigmoid_p, dtype=float).tolist(),
            "margin_p": holdout_margin_p.tolist(),
            "blend_p": holdout_blend_p.tolist(),
            "predicted_margin": frozen_margin.tolist(),
        },
        "live_sigma": holdout_sigma,
        "live_blend": walk["live_blend"],
        "gate": gate,
    }


__all__ = [
    "CANDIDATE_LIVE_MODEL_VERSION",
    "FALLBACK_RESIDUAL_SIGMA",
    "attach_win_prob_config",
    "BlendStack",
    "choose_published_probability",
    "classifier_raw_probability",
    "compare_win_prob",
    "count_sign_contradictions",
    "evaluate_margin_winprob",
    "fit_blend_stack",
    "is_sign_contradiction",
    "logit",
    "margin_implied_home_win_prob",
    "margin_winprob_gate",
    "probability_side",
    "reconcile_published_probability",
    "residual_sigma",
    "score_win_prob",
    "spread_side",
    "walk_forward_margin_winprob",
    "win_prob_config_from_artifact",
]
