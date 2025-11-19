#!/usr/bin/env python3
"""
Utility script to inspect what `nfl_data_py` can currently provide.

It focuses on two checks:
1. Does play-by-play (PBP) data exist for a target season (e.g., 2025)?
2. Are core stats (PBP + weekly player data) available for each season in
   a configurable trailing window (default = last 5 seasons)?

Run:
    python nfl_data_py_inspect.py --target-season 2025 --years-back 5
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Sequence, Tuple

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
    "yardline_100",
    "play_type",
    "yards_gained",
    "epa",
]

WEEKLY_COLUMNS = [
    "season",
    "week",
    "recent_team",
    "player_display_name",
    "position",
    "team",
    "opponent",
    "rush_yards",
    "rec_yards",
    "pass_yards",
    "fantasy_points",
]


def _season_window(target_season: int, years_back: int) -> List[int]:
    """Return a sorted list of seasons ending at target_season."""
    start = max(1999, target_season - years_back + 1)
    return list(range(start, target_season + 1))


def _summarize_df(name: str, df: pd.DataFrame, sample_rows: int = 5) -> None:
    """Print a compact summary for a DataFrame."""
    if df is None or df.empty:
        print(f"{name}: no rows returned.")
        return

    preview_cols = ", ".join(df.columns[:12])
    date_cols = [col for col in ("game_date", "gameday") if col in df.columns]
    date_range: Tuple[str | float, str | float] | None = None
    if date_cols:
        date_series = pd.to_datetime(df[date_cols[0]], errors="coerce")
        if not date_series.isna().all():
            date_range = (
                date_series.min().strftime("%Y-%m-%d"),
                date_series.max().strftime("%Y-%m-%d"),
            )

    print(f"{name}: {len(df):,} rows x {len(df.columns)} cols")
    if date_range:
        print(f"  date range: {date_range[0]} → {date_range[1]}")
    print(f"  columns: {preview_cols}{' ...' if len(df.columns) > 12 else ''}")
    print(df.head(sample_rows).to_string(index=False))


def _load_module():
    """Import nfl_data_py with a friendly error if missing."""
    try:
        import nfl_data_py as nfl
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "nfl_data_py is not installed. Run `pip install nfl-data-py` first."
        ) from exc
    return nfl


def _check_pbp_for_season(nfl, season: int) -> pd.DataFrame:
    """Fetch a single season of PBP data (limited columns for speed)."""
    print(f"\nChecking play-by-play availability for season {season}...")
    pbp = nfl.import_pbp_data(
        years=[season],
        columns=PBP_COLUMNS,
        include_participation=False,
        downcast=True,
        cache=False,
        thread_requests=False,
    )
    if pbp.empty:
        print("  No data returned for that season.")
    else:
        print(f"  Retrieved {len(pbp):,} play rows.")
    return pbp


def _check_multi_year_stats(
    nfl, seasons: Sequence[int]
) -> List[Tuple[int, int, int]]:
    """Fetch week-level counts for each requested season."""
    print(f"\nChecking weekly player stats for seasons: {', '.join(map(str, seasons))}")
    stats: List[Tuple[int, int, int]] = []
    for season in seasons:
        weekly = nfl.import_weekly_data(
            years=[season],
            columns=WEEKLY_COLUMNS,
            downcast=True,
        )
        stats.append((season, len(weekly), weekly["week"].max() if not weekly.empty else 0))
        print(
            f"  {season}: {len(weekly):,} rows (max week "
            f"{weekly['week'].max() if not weekly.empty else 'n/a'})"
        )
    return stats


def _check_schedule_span(nfl, seasons: Sequence[int]) -> pd.DataFrame:
    """Pull combined schedules for the requested seasons."""
    print(f"\nLoading schedules for seasons: {', '.join(map(str, seasons))}")
    schedules = nfl.import_schedules(seasons)
    print(f"  Retrieved {len(schedules):,} scheduled or completed games.")
    return schedules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe nfl_data_py coverage for a target season and trailing history."
    )
    parser.add_argument(
        "--target-season",
        type=int,
        default=dt.date.today().year,
        help="Season to treat as 'current' (default: current calendar year).",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=5,
        help="How many seasons (including target) to inspect for weekly stats (default: 5).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Rows to display in previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nfl = _load_module()

    history_seasons = _season_window(args.target_season, args.years_back)

    print(f"nfl_data_py version: {getattr(nfl, '__version__', 'unknown')}")
    print(f"Import helpers available: {[name for name in dir(nfl) if name.startswith('import_')][:10]} ...")

    pbp_df = _check_pbp_for_season(nfl, args.target_season)
    if not pbp_df.empty:
        _summarize_df(f"PBP {args.target_season}", pbp_df, args.sample_rows)

    _check_multi_year_stats(nfl, history_seasons)
    schedules_df = _check_schedule_span(nfl, history_seasons)
    _summarize_df("Schedules sample", schedules_df, args.sample_rows)

    print("\nDone. Review the summaries above to determine coverage.")


if __name__ == "__main__":
    main()
