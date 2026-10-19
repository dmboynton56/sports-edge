"""Fail-closed live inference adapter for the NFL v2 football-only model."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.models.nfl_toss_up import is_nfl_toss_up, nfl_moneyline_pick
from src.models.nfl_v2 import FEATURE_COLUMNS, audit_nfl_feature_store, build_nfl_feature_store


LIVE_MODEL_VERSION = "nfl-v2-live-20260906"
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "nfl_v2_live.joblib"
DEFAULT_POSTHOC_ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "nfl_v2_posthoc_calibration.json"
POSTHOC_CALIBRATION_ENV = "NFL_V2_POSTHOC_CALIBRATION"
# Default-off rollback: the frozen 20260906 path stays live unless this is
# explicitly enabled after a walk-forward health check ships a calibrator.
POSTHOC_DISABLE_ENV = "NFL_V2_DISABLE_POSTHOC_CALIBRATION"


def load_live_artifact(path: Path = DEFAULT_ARTIFACT) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"NFL v2 live artifact is missing: {path}")
    artifact = joblib.load(path)
    required = {"model_version", "feature_columns", "probability_model", "margin_model", "trained_through"}
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"NFL v2 live artifact is missing fields: {missing}")
    if artifact["model_version"] != LIVE_MODEL_VERSION:
        raise ValueError(
            f"NFL v2 artifact version {artifact['model_version']!r} does not match {LIVE_MODEL_VERSION!r}."
        )
    if list(artifact["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("NFL v2 artifact feature contract does not match runtime FEATURE_COLUMNS.")
    return artifact


def posthoc_calibration_enabled(env: dict[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    disable = str(source.get(POSTHOC_DISABLE_ENV, "")).strip().lower()
    if disable in {"1", "true", "on", "yes"}:
        return False
    enabled = str(source.get(POSTHOC_CALIBRATION_ENV, "")).strip().lower()
    return enabled in {"1", "true", "on", "yes"}


def load_posthoc_artifact(path: Path = DEFAULT_POSTHOC_ARTIFACT) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"NFL v2 post-hoc calibration artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("enabled"):
        raise ValueError("NFL v2 post-hoc calibration artifact is missing or not enabled.")
    if payload.get("base_model_version") != LIVE_MODEL_VERSION:
        raise ValueError(
            f"Post-hoc artifact base version {payload.get('base_model_version')!r} "
            f"does not match {LIVE_MODEL_VERSION!r}."
        )
    return payload


def apply_posthoc_calibration(
    probability: np.ndarray,
    margin: np.ndarray,
    payload: dict[str, Any],
    *,
    market_probability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from src.models.nfl_recalibration import calibrators_from_artifact

    prob_calibrator, margin_calibrator = calibrators_from_artifact(payload)
    calibrated_probability = prob_calibrator.transform(probability, market_probability)
    calibrated_margin = margin_calibrator.transform(margin)
    if not np.isfinite(calibrated_probability).all() or not np.isfinite(calibrated_margin).all():
        raise ValueError("NFL v2 post-hoc calibration produced non-finite predictions.")
    if ((calibrated_probability <= 0) | (calibrated_probability >= 1)).any():
        raise ValueError("NFL v2 post-hoc calibration produced probabilities outside the open unit interval.")
    return np.asarray(calibrated_probability, dtype=float), np.asarray(calibrated_margin, dtype=float)


def _row_hash(row: pd.Series, artifact: dict[str, Any]) -> str:
    values = []
    for column in FEATURE_COLUMNS:
        value = row[column]
        values.append(None if pd.isna(value) else float(value))
    payload = {
        "model_version": artifact["model_version"],
        "dataset_fingerprint": artifact.get("dataset_fingerprint"),
        "game_id": str(row["game_id"]),
        "features": values,
    }
    return sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def predict_live_games(
    schedules: pd.DataFrame,
    team_game_stats: pd.DataFrame,
    target_game_ids: list[str],
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    posted_starters: Mapping[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Score target games from a chronological schedule containing prior results."""
    artifact = load_live_artifact(artifact_path)
    completed = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()]
    expected_team_games = int(len(completed) * 2)
    completed_ids = set(completed["game_id"].astype(str))
    covered_team_games = int(team_game_stats["game_id"].astype(str).isin(completed_ids).sum()) if not team_game_stats.empty else 0
    coverage = covered_team_games / expected_team_games if expected_team_games else 1.0
    if coverage < 0.99:
        raise ValueError(
            f"NFL v2 completed-game PBP coverage is {coverage:.1%}; at least 99.0% is required."
        )
    features = build_nfl_feature_store(
        schedules,
        team_game_stats,
        include_unplayed=True,
        posted_starters=posted_starters,
    )
    audit = audit_nfl_feature_store(features, raise_on_error=True)
    if not audit["ready"]:
        raise ValueError(f"NFL v2 live feature audit failed: {audit['issues']}")

    targets = features[features["game_id"].astype(str).isin({str(value) for value in target_game_ids})].copy()
    if len(targets) != len(set(map(str, target_game_ids))):
        present = set(targets["game_id"].astype(str))
        missing = sorted(set(map(str, target_game_ids)) - present)
        raise ValueError(f"NFL v2 could not build every target game: {missing}")
    if targets[FEATURE_COLUMNS].isna().any().any():
        missing = targets[FEATURE_COLUMNS].isna().sum()
        raise ValueError(f"NFL v2 live features contain missing values: {missing[missing > 0].to_dict()}")

    probability = np.asarray(artifact["probability_model"].predict_probability(targets), dtype=float)
    margin = np.asarray(artifact["margin_model"].predict_margin(targets), dtype=float)
    if not np.isfinite(probability).all() or not np.isfinite(margin).all():
        raise ValueError("NFL v2 produced non-finite predictions.")
    if ((probability <= 0) | (probability >= 1)).any():
        raise ValueError("NFL v2 produced probabilities outside the open unit interval.")
    if posthoc_calibration_enabled():
        posthoc = load_posthoc_artifact()
        market = None
        if "closing_market_home_prob" in targets.columns:
            market_series = pd.to_numeric(targets["closing_market_home_prob"], errors="coerce")
            if market_series.notna().all():
                market = market_series.to_numpy(dtype=float)
        probability, margin = apply_posthoc_calibration(probability, margin, posthoc, market_probability=market)

    output = targets[["game_id", "season", "week", "game_date", "home_team", "away_team"]].copy()
    output = output.rename(columns={"week": "season_week"})
    output["home_win_probability"] = probability
    output["is_toss_up"] = [bool(is_nfl_toss_up(value)) for value in probability]
    output["moneyline_pick"] = [
        nfl_moneyline_pick(str(row.home_team), str(row.away_team), float(row.home_win_probability))
        for row in output.itertuples(index=False)
    ]
    output["predicted_margin"] = margin
    # Existing serving tables use betting-spread convention: negative means
    # the home team is favored, the inverse of predicted home margin.
    output["predicted_spread"] = -margin
    output["input_hash"] = targets.apply(lambda row: _row_hash(row, artifact), axis=1).to_numpy()
    return output
