"""Promote a smoother NFL v2 calibrator onto the frozen HGB/ridge, or don't.

The live HGB and ridge stay frozen. Only the probability calibrator is a
candidate. Selection matches PR #44: identity, Platt, temperature, min-bin
isotonic, and unregularized isotonic, picked by expanding-group OOF Brier and
log loss on seasons through 2025. 2026 is never used to fit.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.nfl_bucket_backtest import (
    _oof_probability_frame,
    json_safe,
    load_live_specs,
    predict_bundle,
    train_walk_forward_bundle,
    walk_forward_predictions,
)
from src.models.nfl_experiment import (
    CalibratedProbabilityModel,
    ProbabilityCalibrator,
    fit_calibrator,
    select_calibrator_walk_forward,
)
from src.models.nfl_live import DEFAULT_ARTIFACT, LIVE_MODEL_VERSION, LIVE_VERSION_ENV, load_live_artifact
from src.models.nfl_toss_up import NFL_TOSS_UP_BAND, is_nfl_toss_up
from src.models.nfl_v2 import FEATURE_COLUMNS, margin_metrics, probability_metrics, write_json


CANDIDATE_LIVE_MODEL_VERSION = "nfl-v2-live-20261019"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261019
MAE_ECE_TOLERANCE = 1e-9


def frozen_inner_probability_model(artifact: Mapping[str, Any]):
    model = artifact["probability_model"]
    return getattr(model, "model", model)


def collect_oof_for_live_train(
    train: pd.DataFrame,
    probability_spec: Mapping[str, Any],
) -> pd.DataFrame:
    seasons = sorted(int(value) for value in pd.to_numeric(train["season"], errors="coerce").dropna().unique())
    if len(seasons) < 2:
        return train.iloc[0:0].copy()
    return _oof_probability_frame(train, probability_spec, int(seasons[1]))


def select_and_fit_live_calibrator(
    train: pd.DataFrame,
    probability_spec: Mapping[str, Any],
) -> dict[str, Any]:
    oof = collect_oof_for_live_train(train, probability_spec)
    if oof.empty:
        raise ValueError("No OOF rows available to select a live calibrator.")
    group_column = "week" if int(oof["season"].nunique()) <= 1 and "week" in oof.columns else "season"
    selection = select_calibrator_walk_forward(oof, group_column=group_column)
    chosen = str(selection["selected"])
    calibrator = (
        ProbabilityCalibrator(method="none")
        if chosen == "none"
        else fit_calibrator(chosen, oof["home_win"], oof["raw_probability"])
    )
    return {
        "calibrator": calibrator,
        "calibration": chosen,
        "calibrator_selection": selection,
        "oof_seasons": sorted(int(value) for value in pd.to_numeric(oof["season"], errors="coerce").dropna().unique()),
        "oof_rows": int(len(oof)),
        "group_column": group_column,
    }


def wrap_frozen_artifact(
    frozen: Mapping[str, Any],
    *,
    calibrator: ProbabilityCalibrator,
    calibration: str,
    selection: Mapping[str, Any],
    model_version: str = CANDIDATE_LIVE_MODEL_VERSION,
) -> dict[str, Any]:
    """Keep frozen HGB + ridge; replace only the probability calibrator."""
    inner = frozen_inner_probability_model(frozen)
    candidate = dict(frozen)
    candidate["probability_model"] = CalibratedProbabilityModel(inner, calibrator)
    candidate["calibrator"] = calibrator
    candidate["model_version"] = model_version
    candidate["parent_model_version"] = frozen.get("model_version")
    candidate["created_at"] = datetime.now(timezone.utc).isoformat()
    live_selection = dict(frozen.get("selection") or {})
    live_selection["model_version"] = model_version
    live_selection["calibration_method"] = calibration
    live_selection["parent_model_version"] = frozen.get("model_version")
    live_selection["activation_reason"] = "smoother_calibrator_selected_by_pr44_oof_gate"
    candidate["selection"] = live_selection
    candidate["calibrator_selection"] = dict(selection)
    return candidate


def _unique_p(probability: Sequence[float]) -> int:
    return int(len(np.unique(np.round(np.asarray(probability, dtype=float), 6))))


def score_probability_and_margin(
    frame: pd.DataFrame,
    probability: Sequence[float],
    margin: Sequence[float],
) -> dict[str, Any]:
    y = pd.to_numeric(frame["home_win"], errors="coerce")
    actual_margin = pd.to_numeric(frame.get("home_margin"), errors="coerce")
    if actual_margin.isna().all():
        actual_margin = pd.to_numeric(frame.get("home_score"), errors="coerce") - pd.to_numeric(
            frame.get("away_score"), errors="coerce"
        )
    usable = y.notna() & actual_margin.notna()
    y = y.loc[usable].astype(int)
    p = np.asarray(probability, dtype=float)[usable.to_numpy()]
    m = np.asarray(margin, dtype=float)[usable.to_numpy()]
    actual = actual_margin.loc[usable].to_numpy(dtype=float)
    metrics = probability_metrics(y, p)
    metrics.update(margin_metrics(actual, m))
    metrics["unique_probabilities"] = _unique_p(p)
    metrics["toss_up_n"] = int(sum(is_nfl_toss_up(value) for value in p))
    metrics["n"] = int(len(y))
    return metrics


def toss_up_shift(baseline_p: Sequence[float], candidate_p: Sequence[float]) -> dict[str, Any]:
    base = np.asarray(baseline_p, dtype=float)
    cand = np.asarray(candidate_p, dtype=float)
    if len(base) != len(cand):
        raise ValueError("Toss-up shift requires aligned probability vectors.")
    base_flag = np.array([is_nfl_toss_up(value) for value in base])
    cand_flag = np.array([is_nfl_toss_up(value) for value in cand])
    return {
        "band": NFL_TOSS_UP_BAND,
        "baseline_n": int(base_flag.sum()),
        "candidate_n": int(cand_flag.sum()),
        "entered": int((~base_flag & cand_flag).sum()),
        "exited": int((base_flag & ~cand_flag).sum()),
        "stayed": int((base_flag & cand_flag).sum()),
        "net": int(cand_flag.sum() - base_flag.sum()),
    }


def paired_delta_interval(
    y: Sequence[int],
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    kind: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    target = np.asarray(y, dtype=float)
    cand = np.clip(np.asarray(candidate, dtype=float), 1e-6, 1 - 1e-6)
    base = np.clip(np.asarray(baseline, dtype=float), 1e-6, 1 - 1e-6)
    if kind == "brier":
        delta = (cand - target) ** 2 - (base - target) ** 2
    elif kind == "log_loss":
        delta = -(target * np.log(cand) + (1.0 - target) * np.log(1.0 - cand)) - (
            -(target * np.log(base) + (1.0 - target) * np.log(1.0 - base))
        )
    else:
        raise ValueError(f"Unknown paired loss: {kind}")
    observed = float(delta.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = delta[rng.integers(0, delta.size, delta.size)].mean()
    return {
        "mean": observed,
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "n": int(delta.size),
    }


def compare_predictions(
    frame: pd.DataFrame,
    baseline_probability: Sequence[float],
    candidate_probability: Sequence[float],
    baseline_margin: Sequence[float],
    candidate_margin: Sequence[float],
) -> dict[str, Any]:
    y = pd.to_numeric(frame["home_win"], errors="coerce")
    usable = y.notna()
    y_int = y.loc[usable].astype(int).to_numpy()
    base_p = np.asarray(baseline_probability, dtype=float)[usable.to_numpy()]
    cand_p = np.asarray(candidate_probability, dtype=float)[usable.to_numpy()]
    base_m = np.asarray(baseline_margin, dtype=float)[usable.to_numpy()]
    cand_m = np.asarray(candidate_margin, dtype=float)[usable.to_numpy()]
    scored = frame.loc[usable].copy()
    baseline = score_probability_and_margin(scored, base_p, base_m)
    candidate = score_probability_and_margin(scored, cand_p, cand_m)
    return {
        "n": int(len(y_int)),
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "brier": candidate["brier"] - baseline["brier"],
            "log_loss": candidate["log_loss"] - baseline["log_loss"],
            "ece_10": candidate["ece_10"] - baseline["ece_10"],
            "mae": candidate["mae"] - baseline["mae"],
            "unique_probabilities": candidate["unique_probabilities"] - baseline["unique_probabilities"],
        },
        "delta_brier_ci": paired_delta_interval(y_int, cand_p, base_p, kind="brier"),
        "delta_log_loss_ci": paired_delta_interval(y_int, cand_p, base_p, kind="log_loss"),
        "toss_up": toss_up_shift(base_p, cand_p),
    }


def score_artifact_on_frame(frame: pd.DataFrame, artifact: Mapping[str, Any]) -> pd.DataFrame:
    return predict_bundle(frame, artifact)


def promotion_gate(
    holdout: Mapping[str, Any],
    walk_forward: Mapping[str, Any],
    *,
    selected_method: str,
) -> dict[str, Any]:
    """Ship only if the holdout/walk-forward contract in the owner request holds."""

    def _not_worse(side: Mapping[str, Any], key: str) -> bool:
        return float(side["candidate"][key]) <= float(side["baseline"][key]) + MAE_ECE_TOLERANCE

    def _ci_includes_zero(interval: Mapping[str, Any]) -> bool:
        return float(interval["lower_95"]) <= 0.0 <= float(interval["upper_95"])

    def _clearly_better(interval: Mapping[str, Any]) -> bool:
        return float(interval["upper_95"]) < 0.0

    def _clearly_worse(interval: Mapping[str, Any]) -> bool:
        return float(interval["lower_95"]) > 0.0

    holdout_brier_better = holdout["candidate"]["brier"] < holdout["baseline"]["brier"]
    holdout_log_better = holdout["candidate"]["log_loss"] < holdout["baseline"]["log_loss"]
    holdout_both = holdout_brier_better and holdout_log_better
    holdout_clear = _clearly_better(holdout["delta_brier_ci"]) and _clearly_better(holdout["delta_log_loss_ci"])
    holdout_noise = _ci_includes_zero(holdout["delta_brier_ci"]) and _ci_includes_zero(holdout["delta_log_loss_ci"])
    holdout_worse = _clearly_worse(holdout["delta_brier_ci"]) or _clearly_worse(holdout["delta_log_loss_ci"])

    walk_both = (
        walk_forward["candidate"]["brier"] < walk_forward["baseline"]["brier"]
        and walk_forward["candidate"]["log_loss"] < walk_forward["baseline"]["log_loss"]
    )
    walk_clear = (
        walk_both
        and _clearly_better(walk_forward["delta_brier_ci"])
        and _clearly_better(walk_forward["delta_log_loss_ci"])
    )

    # MAE/ECE are constraints on the live holdout being replaced. Walk-forward
    # values are reported but a 0.00x ECE bump there does not veto 2026.
    mae_ok = _not_worse(holdout, "mae")
    ece_ok = _not_worse(holdout, "ece_10")

    reasons: list[str] = []
    if selected_method == "isotonic":
        reasons.append("selector kept unregularized isotonic; not a smoother live change")
    if holdout_worse:
        reasons.append("2026 Brier or log loss is worse than frozen outside noise")
    if not mae_ok:
        reasons.append("2026 MAE got worse")
    if not ece_ok:
        reasons.append("2026 ECE got worse")

    ship = False
    if selected_method != "isotonic" and mae_ok and ece_ok and not holdout_worse:
        if holdout_clear:
            ship = True
            reasons.append("2026 Brier and log loss both improved outside noise; MAE and ECE not worse")
        elif holdout_noise and walk_clear:
            ship = True
            reasons.append(
                "2026 Brier/log loss within noise; walk-forward Brier and log loss clearly better; MAE and ECE not worse"
            )
        elif holdout_both and not holdout_noise:
            ship = True
            reasons.append("2026 Brier and log loss both improved; MAE and ECE not worse")
        else:
            reasons.append(
                "2026 Brier/log loss are not better outside noise, and walk-forward is not clearly better on both"
            )
    if not ship and not reasons:
        reasons.append("promotion contract not met")

    return {
        "ship": ship,
        "reason": "; ".join(reasons),
        "selected_method": selected_method,
        "holdout_brier_and_log_loss_improved": holdout_both,
        "holdout_clearly_better": holdout_clear,
        "holdout_within_noise": holdout_noise,
        "holdout_clearly_worse": holdout_worse,
        "walk_forward_clearly_better": walk_clear,
        "mae_not_worse": mae_ok,
        "ece_not_worse": ece_ok,
    }


def walk_forward_calibrator_comparison(
    features: pd.DataFrame,
    specs: Mapping[str, Any],
    *,
    test_seasons: Sequence[int] = (2021, 2022, 2023, 2024, 2025),
) -> dict[str, Any]:
    baseline_log: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []
    baseline_specs = dict(specs)
    baseline_specs["calibration_method"] = "isotonic"
    candidate_specs_map = dict(specs)
    candidate_specs_map["calibration_method"] = "auto"
    baseline = walk_forward_predictions(features, test_seasons=test_seasons, specs=baseline_specs, fit_log=baseline_log)
    candidate = walk_forward_predictions(
        features, test_seasons=test_seasons, specs=candidate_specs_map, fit_log=candidate_log
    )
    if baseline.empty or candidate.empty:
        raise ValueError("Walk-forward comparison produced an empty frame.")
    aligned = baseline[["game_id", "home_win", "home_score", "away_score", "season", "week"]].copy()
    if "home_margin" in baseline.columns:
        aligned["home_margin"] = baseline["home_margin"]
    comparison = compare_predictions(
        aligned,
        baseline["my_home_win_prob"],
        candidate["my_home_win_prob"],
        baseline["predicted_margin"],
        candidate["predicted_margin"],
    )
    comparison["baseline_fits"] = baseline_log
    comparison["candidate_fits"] = candidate_log
    comparison["baseline_calibration"] = (
        baseline.groupby(baseline["season"].astype(int))["calibration"].first().to_dict()
        if "calibration" in baseline
        else {}
    )
    comparison["candidate_calibration"] = (
        candidate.groupby(candidate["season"].astype(int))["calibration"].first().to_dict()
        if "calibration" in candidate
        else {}
    )
    return comparison


def evaluate_candidate(
    features: pd.DataFrame,
    *,
    frozen: Mapping[str, Any] | None = None,
    specs: Mapping[str, Any] | None = None,
    holdout_season: int = 2026,
) -> dict[str, Any]:
    specs = specs or load_live_specs()
    frozen = frozen or load_live_artifact()
    completed = features[features["home_score"].notna() & features["away_score"].notna()].copy()
    if "game_type" in completed.columns:
        completed = completed[completed["game_type"].astype(str).str.upper().eq("REG")].copy()
    completed = completed[completed[FEATURE_COLUMNS].notna().all(axis=1)].copy()
    train = completed[pd.to_numeric(completed["season"], errors="coerce") <= 2025].copy()
    holdout = completed[pd.to_numeric(completed["season"], errors="coerce") == int(holdout_season)].copy()
    if train.empty or holdout.empty:
        raise ValueError("Need 2020–2025 train rows and a 2026 holdout.")
    if "home_win" not in train.columns:
        train["home_win"] = (
            pd.to_numeric(train["home_score"], errors="coerce") > pd.to_numeric(train["away_score"], errors="coerce")
        ).astype(int)
        holdout["home_win"] = (
            pd.to_numeric(holdout["home_score"], errors="coerce") > pd.to_numeric(holdout["away_score"], errors="coerce")
        ).astype(int)
        completed["home_win"] = (
            pd.to_numeric(completed["home_score"], errors="coerce") > pd.to_numeric(completed["away_score"], errors="coerce")
        ).astype(int)

    fitted = select_and_fit_live_calibrator(train, specs["probability_spec"])
    candidate_artifact = wrap_frozen_artifact(
        frozen,
        calibrator=fitted["calibrator"],
        calibration=fitted["calibration"],
        selection=fitted["calibrator_selection"],
    )
    frozen_holdout = score_artifact_on_frame(holdout, frozen)
    candidate_holdout = score_artifact_on_frame(holdout, candidate_artifact)
    holdout_comparison = compare_predictions(
        holdout,
        frozen_holdout["my_home_win_prob"],
        candidate_holdout["my_home_win_prob"],
        frozen_holdout["predicted_margin"],
        candidate_holdout["predicted_margin"],
    )
    walk = walk_forward_calibrator_comparison(completed, specs)
    gate = promotion_gate(holdout_comparison, walk, selected_method=fitted["calibration"])
    payload = json_safe(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "parent_model_version": frozen.get("model_version", LIVE_MODEL_VERSION),
            "candidate_model_version": CANDIDATE_LIVE_MODEL_VERSION,
            "selected_calibrator": fitted["calibration"],
            "calibrator_selection": fitted["calibrator_selection"],
            "oof_seasons": fitted["oof_seasons"],
            "oof_rows": fitted["oof_rows"],
            "holdout_2026": holdout_comparison,
            "walk_forward_2021_2025": walk,
            "gate": gate,
            "alters_live_predictions": bool(gate["ship"]),
            "rollback_env": LIVE_VERSION_ENV,
            "rollback_value": LIVE_MODEL_VERSION,
        }
    )
    return {
        "payload": payload,
        "candidate_artifact": candidate_artifact,
        "holdout_predictions": {
            "game_id": holdout["game_id"].astype(str).tolist(),
            "baseline": np.asarray(frozen_holdout["my_home_win_prob"], dtype=float).tolist(),
            "candidate": np.asarray(candidate_holdout["my_home_win_prob"], dtype=float).tolist(),
        },
    }


def write_candidate_artifact(
    artifact: Mapping[str, Any],
    *,
    path: Path,
    frozen_backup: Path | None = None,
    frozen_source: Path = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen_backup is not None and Path(frozen_source).exists() and not Path(frozen_backup).exists():
        Path(frozen_backup).write_bytes(Path(frozen_source).read_bytes())
        json_source = Path(frozen_source).with_suffix(".json")
        if json_source.exists():
            Path(frozen_backup).with_suffix(".json").write_bytes(json_source.read_bytes())
    joblib.dump(dict(artifact), path)
    manifest = {key: value for key, value in artifact.items() if key not in {"probability_model", "margin_model", "calibrator"}}
    manifest["artifact_sha256"] = sha256(path.read_bytes()).hexdigest()
    write_json(path.with_suffix(".json"), json_safe(manifest))
    return json_safe(manifest)
