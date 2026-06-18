#!/usr/bin/env python3
"""Strictly validate public JSON artifacts used by the web app."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC_DATA = ROOT / "web" / "public" / "data"


def _walk_numbers(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_walk_numbers(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_numbers(child, f"{path}[{index}]"))
    return errors


def _prediction_game_date(prediction: dict[str, Any]) -> str | None:
    game_date = prediction.get("gameDate") or prediction.get("game_date")
    if game_date:
        return str(game_date)[:10]
    event_time = prediction.get("eventTime") or prediction.get("event_time")
    if event_time:
        return str(event_time)[:10]
    return None


def validate_payload(path: Path, payload: Any, *, mlb_hr_date: str | None = None) -> list[str]:
    failures: list[str] = []
    bad_numbers = _walk_numbers(payload)
    if bad_numbers:
        failures.append(f"{path}: non-finite numbers at {bad_numbers[:5]}")
    if isinstance(payload, dict) and path.name not in {"cbb_prob_matrix.json"}:
        if not (payload.get("generatedAt") or payload.get("generated_at")):
            failures.append(f"{path}: missing generatedAt/generated_at")
    if mlb_hr_date and path.name == "mlb_home_runs.json" and isinstance(payload, dict):
        predictions = payload.get("predictions") or []
        stale_dates = sorted(
            {
                game_date
                for prediction in predictions
                if isinstance(prediction, dict)
                for game_date in [_prediction_game_date(prediction)]
                if game_date and game_date != mlb_hr_date
            }
        )
        missing_dates = sum(
            1
            for prediction in predictions
            if isinstance(prediction, dict) and not prediction.get("gameDate")
        )
        if stale_dates:
            failures.append(
                f"{path}: MLB HR predictions include dates {stale_dates}; expected only {mlb_hr_date}"
            )
        if missing_dates:
            failures.append(f"{path}: {missing_dates} MLB HR predictions are missing gameDate")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate web/public/data JSON files.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--public-data", type=Path, default=DEFAULT_PUBLIC_DATA)
    parser.add_argument("--mlb-hr-date", help="Require mlb_home_runs.json predictions to match this YYYY-MM-DD date.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.paths or sorted(args.public_data.rglob("*.json"))
    failures = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant {value}")))
            failures.extend(validate_payload(path, payload, mlb_hr_date=args.mlb_hr_date))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(paths)} JSON files")


if __name__ == "__main__":
    main()
