#!/usr/bin/env python3
"""Fetch ESPN results and train the college-football team-market artifact."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cfb_market import CfbMarketModel, build_feature_frames, parse_espn_scoreboard  # noqa: E402


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
# YYYYMMDD-YYYYMMDD ranges 400 with "Failed to get events endpoint."
# limit=1000 silently truncates the payload to 25 events; 200 returns the full month.
ESPN_SCOREBOARD_LIMIT = 200
ESPN_FBS_GROUP = 80
DENVER = ZoneInfo("America/Denver")


def month_tokens(start: date, end: date) -> list[str]:
    """Inclusive YYYYMM tokens covering [start, end]."""

    if end < start:
        return []
    tokens: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        tokens.append(f"{year:04d}{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return tokens


def scoreboard_params(month_token: str) -> dict[str, str | int]:
    return {
        "dates": month_token,
        "limit": ESPN_SCOREBOARD_LIMIT,
        "groups": ESPN_FBS_GROUP,
    }


def game_date_denver(game: dict) -> date | None:
    raw = str(game.get("game_time_utc") or "")
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(DENVER).date()


def fetch_games(start: date, end: date) -> list[dict]:
    """Fetch FBS scoreboard events in [start, end] (America/Denver game date).

    ESPN's college-football scoreboard rejects hyphenated date ranges
    (`dates=YYYYMMDD-YYYYMMDD` → HTTP 400). Query calendar months with
    `dates=YYYYMM` and filter locally.
    """

    session = requests.Session()
    games: dict[str, dict] = {}
    for token in month_tokens(start, end):
        response = session.get(
            ESPN_SCOREBOARD,
            params=scoreboard_params(token),
            timeout=30,
        )
        response.raise_for_status()
        for game in parse_espn_scoreboard(response.json()):
            game_day = game_date_denver(game)
            if game_day is None or game_day < start or game_day > end:
                continue
            games[game["event_id"]] = game
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
