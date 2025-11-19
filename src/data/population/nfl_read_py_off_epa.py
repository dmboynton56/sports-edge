#!/usr/bin/env python3
"""
Compute offensive EPA totals for the last N games of a team using nflreadpy.

Example:
    python nfl_read_py_off_epa.py --team NE --max-games 5 --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Sequence

import pandas as pd


def _default_seasons(years_back: int = 2) -> List[int]:
    current = dt.date.today().year
    start = max(1999, current - years_back)
    return list(range(start, current + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize offensive EPA per game for the last N games via nflreadpy."
    )
    parser.add_argument(
        "--team",
        default="NE",
        help="Team abbreviation to evaluate (default: NE).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=5,
        help="Number of most recent games to report (default: 5).",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Explicit season list to fetch (default: current year and prior year).",
    )
    return parser.parse_args()


def _load_pbp(seasons: Sequence[int]) -> pd.DataFrame:
    try:
        import nflreadpy as nfl
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "nflreadpy is not installed. Run `pip install nflreadpy` first."
        ) from exc

    print(f"nflreadpy version: {getattr(nfl, '__version__', 'unknown')}")
    print(f"Requesting seasons: {', '.join(map(str, seasons))}")

    try:
        pbp_rel = nfl.load_pbp(seasons=seasons)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Unable to load play-by-play data: {exc}") from exc

    if not hasattr(pbp_rel, "to_pandas"):
        raise SystemExit("Unexpected object from nflreadpy.load_pbp; expected Polars relation.")
    df = pbp_rel.to_pandas()
    if df.empty:
        raise SystemExit("No play-by-play data returned; check seasons or connectivity.")
    return df


def _prep_games(df: pd.DataFrame, team: str) -> pd.DataFrame:
    plays = df[df["posteam"] == team].copy()
    if plays.empty:
        raise SystemExit(f"No offensive plays found for team {team}.")

    plays["game_date"] = pd.to_datetime(plays.get("game_date", plays.get("gameday")), errors="coerce")
    plays = plays.sort_values(["game_date", "game_id"])

    grouped = (
        plays.groupby("game_id")
        .agg(
            season=("season", "first"),
            week=("week", "first"),
            game_date=("game_date", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            offensive_epa=("epa", "sum"),
        )
        .reset_index()
    )
    grouped["is_home"] = grouped["home_team"] == team
    grouped["opponent"] = grouped["away_team"].where(grouped["is_home"], grouped["home_team"])
    grouped["venue"] = grouped["is_home"].map({True: "home", False: "away"})

    return grouped.sort_values("game_date", ascending=False)


def main() -> None:
    args = parse_args()
    seasons = args.seasons or _default_seasons()

    pbp = _load_pbp(seasons)
    games = _prep_games(pbp, args.team.upper())

    recent = games.head(args.max_games).sort_values("game_date")
    if recent.empty:
        print("No games matched the criteria.")
        return

    print(
        f"\nLast {len(recent)} games of offensive EPA for {args.team.upper()} "
        f"(most recent first in the table below)."
    )
    result = recent[
        [
            "game_date",
            "season",
            "week",
            "venue",
            "opponent",
            "offensive_epa",
        ]
    ].copy()
    result["game_date"] = result["game_date"].dt.strftime("%Y-%m-%d")
    print(result.to_string(index=False, float_format=lambda x: f"{x:0.2f}"))


if __name__ == "__main__":
    main()
