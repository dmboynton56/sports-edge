"""Write explanation cache consumed by sync_explanations_to_supabase.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.injury_loader import extract_injury_metadata


def build_explanation_rows(
    predictions: pd.DataFrame,
    *,
    league: str,
    model_version: str,
    features_by_game: dict[str, pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_ts = datetime.now(timezone.utc).isoformat()

    for _, pred in predictions.iterrows():
        game_id = pred.get("game_id")
        if not game_id or pd.isna(game_id):
            continue

        injury_meta = {
            "injury_adjusted": False,
            "home_injury_delta": 0.0,
            "away_injury_delta": 0.0,
        }
        if features_by_game and str(game_id) in features_by_game:
            injury_meta = extract_injury_metadata(features_by_game[str(game_id)], league)

        top_features = pred.get("top_features") or []
        if isinstance(top_features, str):
            try:
                top_features = json.loads(top_features)
            except json.JSONDecodeError:
                top_features = []

        rows.append(
            {
                "game_id": str(game_id),
                "model_version": model_version,
                "prediction_ts": prediction_ts,
                "top_features": top_features,
                **injury_meta,
                "base_vs_adjusted": pred.get("base_vs_adjusted"),
            }
        )
    return rows


def write_explanation_cache(
    rows: list[dict[str, Any]],
    *,
    league: str,
    cache_dir: Path,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{league.lower()}_explanations_latest.json"
    payload = {
        "league": league,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
