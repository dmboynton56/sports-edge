#!/usr/bin/env python3
"""Fetch ESPN results and train the college-football team-market artifact."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cfb_market import CfbMarketModel, build_feature_frames, parse_espn_scoreboard  # noqa: E402


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"


def fetch_games(start: date, end: date, *, chunk_days: int = 31) -> list[dict]:
    session = requests.Session()
    games: dict[str, dict] = {}
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        response = session.get(
            ESPN_SCOREBOARD,
            params={
                "dates": f"{cursor:%Y%m%d}-{chunk_end:%Y%m%d}",
                "limit": 1000,
                "groups": 80,
            },
            timeout=30,
        )
        response.raise_for_status()
        for game in parse_espn_scoreboard(response.json()):
            games[game["event_id"]] = game
        cursor = chunk_end + timedelta(days=1)
    return list(games.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the college-football market model.")
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--model-version", default="cfb-team-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "cfb_team_v1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games: list[dict] = []
    for season in range(args.start_season, args.end_season + 1):
        games.extend(fetch_games(date(season, 8, 1), date(season + 1, 1, 31)))
    historical, _ = build_feature_frames(games)
    model, metrics = CfbMarketModel.fit(
        historical,
        holdout_season=args.holdout_season,
        model_version=args.model_version,
    )
    model.save(args.output)
    print(json.dumps({**metrics, "artifact": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
