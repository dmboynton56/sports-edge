#!/usr/bin/env python3
"""Validate the candidate payload before it can become a trusted board run."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def _predictions(payload: dict[str, Any], model_version: str) -> list[dict[str, Any]]:
    models = payload.get("models")
    if isinstance(models, dict):
        selected = models.get(model_version)
        if isinstance(selected, dict):
            return [row for row in selected.get("predictions", []) if isinstance(row, dict)]
        return [
            row
            for key, value in models.items()
            if str(key).startswith(model_version)
            and isinstance(value, dict)
            for row in value.get("predictions", [])
            if isinstance(row, dict)
        ]
    return [row for row in payload.get("predictions", []) if isinstance(row, dict)]


def validate(path: Path, slate_date: str, model_version: str = "mlb-hr-v1", allow_empty: bool = False) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ["payload must be an object"]
    rows = _predictions(payload, model_version)
    if not rows and not allow_empty:
        return [f"no {model_version} predictions were published"]
    failures: list[str] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        game_id = str(row.get("gameId") or row.get("game_id") or "")
        player_id = str(row.get("playerId") or row.get("player_id") or "")
        key = (game_id, player_id)
        if not game_id or not player_id:
            failures.append(f"row {index}: gameId and playerId are required")
        if key in seen:
            failures.append(f"row {index}: duplicate candidate key {key}")
        seen.add(key)
        probability = row.get("modelProbability") if row.get("modelProbability") is not None else row.get("model_probability")
        try:
            probability_value = float(probability)
        except (TypeError, ValueError):
            probability_value = math.nan
        if not math.isfinite(probability_value) or not 0 <= probability_value <= 1:
            failures.append(f"row {index}: model probability must be in [0, 1]")
        game_date = str(row.get("gameDate") or row.get("game_date") or "")[:10]
        if game_date != slate_date:
            failures.append(f"row {index}: expected game date {slate_date}, got {game_date or 'missing'}")
        event_time = row.get("eventTime") or row.get("event_time")
        try:
            datetime.fromisoformat(str(event_time).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            failures.append(f"row {index}: eventTime must be an ISO timestamp")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument("--model-version", default="mlb-hr-v1")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    failures = validate(args.path, args.date, args.model_version, args.allow_empty)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {args.path}: trusted MLB HR input is structurally sound")


if __name__ == "__main__":
    main()
