"""Validation gate for graded NFL v2 shadow predictions."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.models.nfl_v2 import margin_metrics, probability_metrics


REQUIRED_COLUMNS = {
    "game_id",
    "week",
    "kickoff_ts",
    "prediction_ts",
    "data_fingerprint",
    "calibrated_probability",
    "predicted_margin",
    "home_win",
    "home_margin",
    "freshness_ok",
    "critical_integrity_failure",
}


def evaluate_shadow_predictions(
    frame: pd.DataFrame,
    gates: Mapping[str, Any],
    *,
    minimum_weeks: int = 4,
    minimum_games: int = 64,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Shadow predictions missing required columns: {missing}")
    if frame["game_id"].duplicated().any():
        raise ValueError("Shadow predictions are not unique by game_id.")

    prediction_ts = pd.to_datetime(frame["prediction_ts"], utc=True, errors="coerce")
    kickoff_ts = pd.to_datetime(frame["kickoff_ts"], utc=True, errors="coerce")
    issues = []
    if prediction_ts.isna().any() or kickoff_ts.isna().any() or (prediction_ts >= kickoff_ts).any():
        issues.append("prediction timestamp is missing or not pre-kickoff")
    if frame["data_fingerprint"].astype("string").str.strip().eq("").any() or frame["data_fingerprint"].isna().any():
        issues.append("feature lineage is incomplete")
    if (~frame["freshness_ok"].fillna(False).astype(bool)).any():
        issues.append("one or more predictions failed freshness checks")
    if frame["critical_integrity_failure"].fillna(True).astype(bool).any():
        issues.append("one or more predictions had a critical integrity failure")

    graded = frame[
        frame["home_win"].notna()
        & frame["home_margin"].notna()
        & frame["calibrated_probability"].notna()
        & frame["predicted_margin"].notna()
    ].copy()
    weeks = int(pd.to_numeric(graded["week"], errors="coerce").dropna().nunique())
    enough_evidence = len(graded) >= minimum_games and weeks >= minimum_weeks
    if not enough_evidence:
        issues.append(
            f"insufficient shadow evidence: {len(graded)} games/{weeks} weeks; "
            f"requires at least {minimum_games} games and {minimum_weeks} weeks"
        )

    probability = probability_metrics(graded["home_win"], graded["calibrated_probability"]) if len(graded) else None
    margin = margin_metrics(graded["home_margin"], graded["predicted_margin"]) if len(graded) else None
    quality_checks = {
        "ece": bool(probability and probability["ece_10"] <= float(gates["max_ece"])),
        "home_probability_bias": bool(
            probability and abs(probability["home_probability_bias"]) <= float(gates["max_home_probability_bias"])
        ),
        "home_margin_bias": bool(
            margin and abs(margin["home_margin_bias"]) <= float(gates["max_home_margin_bias"])
        ),
    }
    return {
        "promote": not issues and all(quality_checks.values()),
        "graded_games": int(len(graded)),
        "graded_weeks": weeks,
        "minimum_games": minimum_games,
        "minimum_weeks": minimum_weeks,
        "integrity_issues": issues,
        "quality_checks": quality_checks,
        "probability_metrics": probability,
        "margin_metrics": margin,
    }
