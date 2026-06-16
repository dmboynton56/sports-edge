"""Shared helpers for MLB batter home-run probability models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_VERSION = "mlb-hr-v1"

FEATURE_COLUMNS = [
    "batter_pa_lag",
    "batter_games_lag",
    "batter_hr_per_pa",
    "batter_recent_pa_28",
    "batter_recent_hr_per_pa_28",
    "batter_rate_over_league",
    "recent_rate_over_league",
    "pitcher_bf_lag",
    "pitcher_hr_factor",
    "venue_factor",
    "league_hr_per_pa",
    "lineup_slot",
    "expected_pa",
    "is_home",
]


def expected_pa_for_slot(lineup_slot: int | float | None) -> float:
    """Approximate pregame plate appearances from batting order slot."""
    try:
        slot = int(lineup_slot or 9)
    except (TypeError, ValueError):
        slot = 9
    slot = max(1, min(9, slot))
    return float(np.clip(4.75 - (slot - 1) * 0.105, 3.65, 4.85))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def build_hr_feature_values(
    *,
    batter_pa: Any,
    batter_hr: Any,
    batter_games: Any,
    batter_recent_pa: Any,
    batter_recent_hr: Any,
    pitcher_bf: Any,
    pitcher_hr_allowed: Any,
    venue_factor: Any,
    league_hr_pa: Any,
    lineup_slot: Any,
    is_home: bool,
) -> dict[str, float]:
    """Build one pregame feature row using only lagged inputs."""
    league_rate = float(np.clip(_safe_float(league_hr_pa, 0.03), 0.001, 0.08))
    batter_pa_value = _safe_float(batter_pa)
    batter_hr_value = _safe_float(batter_hr)
    batter_games_value = _safe_float(batter_games)
    recent_pa_value = _safe_float(batter_recent_pa)
    recent_hr_value = _safe_float(batter_recent_hr)
    pitcher_bf_value = _safe_float(pitcher_bf)
    pitcher_hr_value = _safe_float(pitcher_hr_allowed)

    batter_rate = (batter_hr_value + league_rate * 35.0) / max(batter_pa_value + 35.0, 1.0)
    recent_rate = (recent_hr_value + league_rate * 18.0) / max(recent_pa_value + 18.0, 1.0)
    pitcher_rate = (pitcher_hr_value + league_rate * 45.0) / max(pitcher_bf_value + 45.0, 1.0)
    pitcher_factor = float(np.clip(pitcher_rate / league_rate, 0.72, 1.35))
    slot = max(1, min(9, int(_safe_float(lineup_slot, 9.0) or 9)))

    return {
        "batter_pa_lag": batter_pa_value,
        "batter_games_lag": batter_games_value,
        "batter_hr_per_pa": float(batter_rate),
        "batter_recent_pa_28": recent_pa_value,
        "batter_recent_hr_per_pa_28": float(recent_rate),
        "batter_rate_over_league": float(np.clip(batter_rate / league_rate, 0.35, 3.25)),
        "recent_rate_over_league": float(np.clip(recent_rate / league_rate, 0.45, 2.75)),
        "pitcher_bf_lag": pitcher_bf_value,
        "pitcher_hr_factor": pitcher_factor,
        "venue_factor": float(np.clip(_safe_float(venue_factor, 1.0), 0.7, 1.35)),
        "league_hr_per_pa": league_rate,
        "lineup_slot": float(slot),
        "expected_pa": expected_pa_for_slot(slot),
        "is_home": 1.0 if is_home else 0.0,
    }


def heuristic_hr_probability(features: dict[str, float]) -> tuple[float, float]:
    """Fallback model probability from interpretable rate factors."""
    league_hr_pa = float(np.clip(features.get("league_hr_per_pa", 0.03), 0.001, 0.08))
    expected_pa = float(np.clip(features.get("expected_pa", 4.0), 3.4, 5.0))
    per_pa = (
        league_hr_pa
        * (0.60 * features.get("batter_rate_over_league", 1.0) + 0.25 * features.get("recent_rate_over_league", 1.0) + 0.15)
        * features.get("pitcher_hr_factor", 1.0)
        * features.get("venue_factor", 1.0)
    )
    per_pa = float(np.clip(per_pa, 0.002, 0.095))
    probability = float(np.clip(1.0 - (1.0 - per_pa) ** expected_pa, 0.002, 0.38))
    baseline = float(np.clip(1.0 - (1.0 - league_hr_pa) ** expected_pa, 0.002, 0.20))
    return probability, baseline


def load_hr_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import joblib

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Invalid MLB HR model artifact: {path}")
    missing = [col for col in FEATURE_COLUMNS if col not in artifact.get("feature_columns", FEATURE_COLUMNS)]
    if missing:
        raise ValueError(f"MLB HR model artifact missing features: {missing}")
    return artifact


def predict_hr_probability(artifact: dict[str, Any] | None, features: dict[str, float]) -> float | None:
    if artifact is None:
        return None
    feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
    frame = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_columns}])
    model = artifact["model"]
    probability = float(model.predict_proba(frame)[0, 1])
    return float(np.clip(probability, 0.001, 0.45))


def quality_flags_for_features(features: dict[str, float], *, probable_pitcher_known: bool) -> list[str]:
    flags: list[str] = []
    if features.get("batter_pa_lag", 0.0) < 50:
        flags.append("low_batter_history")
    if features.get("batter_recent_pa_28", 0.0) < 10:
        flags.append("low_recent_pa")
    if not probable_pitcher_known:
        flags.append("missing_probable_pitcher")
    if features.get("pitcher_bf_lag", 0.0) < 50:
        flags.append("low_pitcher_history")
    return flags


def top_feature_payload(features: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"feature": "batter_hr_per_pa", "value": round(float(features.get("batter_hr_per_pa", 0.0)), 4)},
        {"feature": "recent_hr_per_pa", "value": round(float(features.get("batter_recent_hr_per_pa_28", 0.0)), 4)},
        {"feature": "pitcher_hr_factor", "value": round(float(features.get("pitcher_hr_factor", 1.0)), 3)},
        {"feature": "venue_factor", "value": round(float(features.get("venue_factor", 1.0)), 3)},
        {"feature": "expected_pa", "value": round(float(features.get("expected_pa", 4.0)), 2)},
    ]
