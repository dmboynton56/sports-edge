#!/usr/bin/env python3
"""
Compute offensive EPA totals for the last N games of a team using nfl_data_py.

Example:
    python nfl_data_py_off_epa.py --team NE --max-games 5 --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Iterable, List, Sequence

import pandas as pd


PBP_COLUMNS = [
    "season",
    "week",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "epa",
]


def _default_seasons(years_back: int = 2) -> List[int]:
    """Return a simple rolling window ending at the current year."""
    current = dt.date.today().year
    start = max(1999, current - years_back)
    return list(range(start, current + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize offensive EPA per game for the last N games via nfl_data_py."
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
    parser.add_argument(
        "--include-participation",
        action="store_true",
        help="Pass through to nfl_data_py.import_pbp_data (defaults to False for speed).",
    )
    return parser.parse_args()


def _load_pbp(seasons: Sequence[int], include_participation: bool):
    try:
        import nfl_data_py as nfl
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "nfl_data_py is not installed. Run `pip install nfl-data-py` first."
        ) from exc

    print(f"nfl_data_py version: {getattr(nfl, '__version__', 'unknown')}")
    print(f"Requesting seasons: {', '.join(map(str, seasons))}")

    df = nfl.import_pbp_data(
        years=list(seasons),
        columns=PBP_COLUMNS,
        include_participation=include_participation,
        downcast=True,
        cache=False,
        thread_requests=False,
    )
    if df.empty:
        raise SystemExit("No play-by-play data returned; check seasons or connectivity.")
    return df


def _prep_games(df: pd.DataFrame, team: str) -> pd.DataFrame:
    plays = df[df["posteam"] == team].copy()
    if plays.empty:
        raise SystemExit(f"No offensive plays found for team {team}.")

    plays["game_date"] = pd.to_datetime(plays["game_date"], errors="coerce")
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

    pbp = _load_pbp(seasons, include_participation=args.include_participation)
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
