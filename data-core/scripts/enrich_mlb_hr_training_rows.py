#!/usr/bin/env python3
"""Add pregame-available MLB player handedness metadata to HR training rows."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows.csv"
DEFAULT_OUTPUT = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows_enriched.csv"
DEFAULT_CACHE = ROOT / "notebooks" / "cache" / "mlb_player_handedness_cache.json"
DEFAULT_AUDIT = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows_enriched_audit.json"
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _fetch_people(ids: list[int], *, timeout: int) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    response = requests.get(
        PEOPLE_URL,
        params={"personIds": ",".join(str(value) for value in ids)},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    out: dict[str, dict[str, Any]] = {}
    for person in payload.get("people", []):
        player_id = str(person.get("id"))
        out[player_id] = {
            "player_id": int(person.get("id")),
            "full_name": person.get("fullName"),
            "bat_side": (person.get("batSide") or {}).get("code"),
            "pitch_hand": (person.get("pitchHand") or {}).get("code"),
            "primary_position": (person.get("primaryPosition") or {}).get("abbreviation"),
            "active": person.get("active"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    return out


def _normalize_hand(value: Any) -> str:
    text = str(value or "U").upper().strip()
    if text in {"L", "R", "S"}:
        return text
    return "U"


def _meta_field(row: Any, field: str) -> Any:
    return row.get(field) if isinstance(row, dict) else None


def _batter_platoon_advantage(batter_hand: str, pitcher_hand: str) -> float:
    if batter_hand == "U" or pitcher_hand == "U":
        return 0.0
    if batter_hand == "S":
        return 1.0
    return 1.0 if batter_hand != pitcher_hand else 0.0


def enrich_rows(rows: pd.DataFrame, people: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = rows.copy()
    batter_meta = out["player_id"].astype("Int64").astype(str).map(people)
    pitcher_meta = out["opposing_starter_id"].astype("Int64").astype(str).map(people)

    out["batter_bat_side"] = batter_meta.map(lambda row: _normalize_hand(_meta_field(row, "bat_side")))
    out["pitcher_throw_hand"] = pitcher_meta.map(lambda row: _normalize_hand(_meta_field(row, "pitch_hand")))
    out["batter_bats_left"] = out["batter_bat_side"].isin(["L", "S"]).astype(float)
    out["batter_bats_right"] = out["batter_bat_side"].isin(["R", "S"]).astype(float)
    out["batter_switch_hitter"] = (out["batter_bat_side"] == "S").astype(float)
    out["pitcher_throws_left"] = (out["pitcher_throw_hand"] == "L").astype(float)
    out["pitcher_throws_right"] = (out["pitcher_throw_hand"] == "R").astype(float)
    out["known_handedness_matchup"] = (
        (out["batter_bat_side"] != "U") & (out["pitcher_throw_hand"] != "U")
    ).astype(float)
    out["batter_platoon_advantage"] = [
        _batter_platoon_advantage(batter, pitcher)
        for batter, pitcher in zip(out["batter_bat_side"], out["pitcher_throw_hand"], strict=False)
    ]
    out["same_side_matchup"] = (
        (out["known_handedness_matchup"] == 1.0)
        & (out["batter_bat_side"] != "S")
        & (out["batter_bat_side"] == out["pitcher_throw_hand"])
    ).astype(float)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich MLB HR training rows with player handedness.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.input)
    ids = pd.concat([rows["player_id"], rows["opposing_starter_id"]], ignore_index=True)
    unique_ids = sorted(int(value) for value in ids.dropna().unique())
    cache = _read_cache(args.cache)
    missing = [player_id for player_id in unique_ids if str(player_id) not in cache]

    for batch in _chunks(missing, args.batch_size):
        cache.update(_fetch_people(batch, timeout=args.timeout))
        _write_cache(args.cache, cache)
        time.sleep(args.sleep)

    enriched = enrich_rows(rows, cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    audit = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "rows": int(len(enriched)),
        "unique_player_ids": int(len(unique_ids)),
        "cache_rows": int(len(cache)),
        "fetched_player_ids": int(len(missing)),
        "batter_unknown_hand_rate": float((enriched["batter_bat_side"] == "U").mean()),
        "pitcher_unknown_hand_rate": float((enriched["pitcher_throw_hand"] == "U").mean()),
        "known_matchup_rate": float(enriched["known_handedness_matchup"].mean()),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(enriched)} rows to {args.output}")
    print(f"Wrote audit to {args.audit_output}")


if __name__ == "__main__":
    main()
