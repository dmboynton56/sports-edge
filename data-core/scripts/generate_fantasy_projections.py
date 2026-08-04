#!/usr/bin/env python3
"""Generate the public fantasy projection artifact.

Examples:
  python scripts/generate_fantasy_projections.py --season 2026
  python scripts/generate_fantasy_projections.py --season 2026 --no-adp

The model uses nflverse data; FantasyPros is queried only for the separately
labeled market ADP signal when ``FANTASYPROS_API_KEY`` is configured.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
if str(ROOT / "data-core") not in sys.path:
    sys.path.insert(0, str(ROOT / "data-core"))

from src.fantasy.fantasypros import FantasyProsClient, FantasyProsError, normalize_consensus_rows  # noqa: E402
from src.fantasy.projection_model import build_preseason_projections, load_nflverse_weekly  # noqa: E402
from src.fantasy.scoring import FULL_PPR_SCORING  # noqa: E402


DEFAULT_OUTPUT = ROOT / "web" / "public" / "data" / "fantasy_projections.json"
DEFAULT_METRICS = ROOT / "data-core" / "models" / "fantasy_projection_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--history-start", type=int, default=None)
    parser.add_argument("--history-seasons", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--no-adp", action="store_true", help="Do not call FantasyPros.")
    parser.add_argument("--skip-weekly", action="store_true")
    return parser.parse_args()


def _parse_name(value: Any) -> str:
    return " ".join(str(value or "").lower().replace(".", "").split())


def _load_adp(season: int) -> list[dict[str, Any]]:
    client = FantasyProsClient.from_env()
    rows: list[dict[str, Any]] = []
    for position in ("QB", "RB", "WR", "TE", "K", "DST"):
        payload = client.consensus_rankings(
            season=season,
            position=position,
            scoring="PPR",
            ranking_type="ADP",
            week=0,
        )
        rows.extend(normalize_consensus_rows(payload))
    return rows


def _join_adp(projections: list[dict[str, Any]], adp_rows: list[dict[str, Any]]) -> None:
    by_key = {
        (_parse_name(row.get("player_name")), str(row.get("position") or "").upper(), str(row.get("team") or "")): row
        for row in adp_rows
    }
    by_name = {_parse_name(row.get("player_name")): row for row in adp_rows}
    for projection in projections:
        key = (_parse_name(projection["player_name"]), projection["position"], str(projection.get("team") or ""))
        row = by_key.get(key) or by_name.get(_parse_name(projection["player_name"]))
        projection["adp"] = row.get("adp") if row else None
        projection["adp_rank"] = row.get("consensus_rank") if row else None
        projection["adp_tier"] = row.get("tier") if row else None
        projection["adp_updated_at"] = row.get("last_updated") if row else None
        projection["adp_source"] = "FantasyPros ADP" if row else None


def _rank(projections: list[dict[str, Any]]) -> None:
    overall = sorted(projections, key=lambda row: row.get("points", 0), reverse=True)
    for index, row in enumerate(overall, start=1):
        row["overall_rank"] = index
    for position in {row["position"] for row in projections}:
        position_rows = sorted(
            (row for row in projections if row["position"] == position),
            key=lambda row: row.get("points", 0),
            reverse=True,
        )
        for index, row in enumerate(position_rows, start=1):
            row["position_rank"] = index
            row["tier"] = max(1, (index - 1) // 8 + 1)


def _weekly_rows(projections: list[dict[str, Any]], season: int) -> dict[str, list[dict[str, Any]]]:
    try:
        import nflreadpy as nfl

        schedule_raw = nfl.load_schedules([season])
        schedule = schedule_raw.to_pandas() if hasattr(schedule_raw, "to_pandas") else pd.DataFrame(schedule_raw)
        schedule = schedule[schedule.get("game_type", "REG").eq("REG")] if "game_type" in schedule else schedule
    except Exception as exc:  # pragma: no cover - external data failure path
        schedule = pd.DataFrame()

    by_week: dict[str, list[dict[str, Any]]] = {}
    for week in range(1, 19):
        rows: list[dict[str, Any]] = []
        for source in projections:
            team = str(source.get("team") or "")
            has_game = True
            if not schedule.empty and team:
                has_game = bool(
                    ((schedule["week"] == week) & ((schedule["home_team"] == team) | (schedule["away_team"] == team))).any()
                )
            row = {"player_id": source["player_id"], "season": source["season"], "position": source["position"]}
            row["scope"] = "week"
            row["week"] = week
            row["projected_games"] = 0 if not has_game else 1
            multiplier = 0.0 if not has_game else 1.0 / max(1.0, float(source.get("projected_games") or 17.0))
            row["points"] = round(float(source.get("points") or 0.0) * multiplier, 2)
            row["floor_points"] = round(float(source.get("floor_points") or 0.0) * multiplier, 2)
            row["ceiling_points"] = round(float(source.get("ceiling_points") or 0.0) * multiplier, 2)
            row["points_per_game"] = row["points"]
            row["availability"] = "bye" if not has_game else source.get("availability", "expected")
            rows.append(row)
        _rank(rows)
        by_week[str(week)] = rows
    return by_week


def main() -> None:
    args = parse_args()
    history_seasons = args.history_seasons
    if not history_seasons:
        start = args.history_start or max(2018, args.season - 8)
        history_seasons = list(range(start, args.season))

    projections, model = build_preseason_projections(
        seasons=history_seasons,
        target_season=args.season,
        holdout_season=max(history_seasons) if history_seasons else None,
    )
    gaps: list[str] = []
    adp_rows: list[dict[str, Any]] = []
    if not args.no_adp:
        try:
            adp_rows = _load_adp(args.season)
            _join_adp(projections, adp_rows)
        except (FantasyProsError, Exception) as exc:  # noqa: BLE001
            gaps.append(f"FantasyPros ADP unavailable: {exc}")
    else:
        gaps.append("FantasyPros ADP disabled for this run.")

    _rank(projections)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "modelVersion": model.model_version,
        "productionStatus": "candidate",
        "defaultScoring": FULL_PPR_SCORING.to_dict(),
        "projections": projections,
        "weekly": {} if args.skip_weekly else _weekly_rows(projections, args.season),
        "adp": adp_rows,
        "metrics": model.metrics,
        "gaps": gaps,
        "sources": [
            "nflverse player stats, players, schedules, and rosters",
            "FantasyPros ADP (market signal only)" if adp_rows else "FantasyPros ADP not loaded",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False), encoding="utf-8")
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(model.metrics, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.output} with {len(projections)} preseason projections")
    print(f"Model metrics: {args.metrics}")
    if gaps:
        print("Gaps:")
        for gap in gaps:
            print(f"- {gap}")


if __name__ == "__main__":
    main()
