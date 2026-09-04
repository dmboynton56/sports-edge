#!/usr/bin/env python3
"""Refresh current NFL roster and injury context in the fantasy artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "data-core") not in sys.path:
    sys.path.insert(0, str(ROOT / "data-core"))

from src.fantasy.sleeper import load_nflverse_rosters, load_sleeper_players, merge_sleeper_context  # noqa: E402


DEFAULT_ARTIFACT = ROOT / "web" / "public" / "data" / "fantasy_projections.json"
CONTEXT_FIELDS = (
    "team",
    "availability",
    "roster_status",
    "injury_status",
    "injury_body_part",
    "practice_participation",
    "depth_chart_order",
    "availability_updated_at",
    "official_roster_status",
)


def propagate_weekly_context(payload: dict) -> int:
    """Copy refreshed preseason context onto compact weekly rows."""

    base_by_id = {str(row.get("player_id")): row for row in (payload.get("projections") or [])}
    updated = 0
    for weekly in (payload.get("weekly") or {}).values():
        for row in weekly or []:
            base = base_by_id.get(str(row.get("player_id")))
            if not base:
                continue
            for field in CONTEXT_FIELDS:
                if field in base:
                    row[field] = base[field]
            updated += 1
    return updated


def summarize_existing_context(payload: dict) -> dict[str, int]:
    rows = list(payload.get("projections") or [])
    matched = [row for row in rows if "roster_status" in row]
    return {
        "matched": len(matched),
        "unmatched": len(rows) - len(matched),
        "questionable": sum(row.get("availability") in {"questionable", "doubtful"} for row in matched),
        "unavailable": sum(row.get("availability") in {"out", "inactive"} for row in matched),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--propagate-only", action="store_true", help="Reuse context already present in preseason rows.")
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    summary = summarize_existing_context(payload) if args.propagate_only else {}
    if not args.propagate_only:
        season = args.season or int(payload.get("season") or datetime.now(timezone.utc).year)
        players = load_sleeper_players()
        summary = merge_sleeper_context(
            payload.get("projections") or [],
            players,
            load_nflverse_rosters(season),
        )
        payload["contextUpdatedAt"] = datetime.now(timezone.utc).isoformat()
        payload["context"] = {**(payload.get("context") or {}), "sleeper": summary}
        payload["sources"] = [
            source
            for source in (payload.get("sources") or [])
            if not str(source).startswith("Sleeper")
            and not str(source).startswith("nflverse season rosters + Sleeper")
        ] + ["nflverse season rosters + Sleeper public NFL player directory (daily context)"]
        payload["gaps"] = [
            gap for gap in (payload.get("gaps") or []) if not str(gap).startswith("Sleeper roster/injury")
        ]
    weekly_rows = propagate_weekly_context(payload)
    payload["context"] = {**(payload.get("context") or {}), "sleeper": summary}
    args.artifact.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(args.artifact), **summary, "weekly_rows_updated": weekly_rows}, indent=2))


if __name__ == "__main__":
    main()
