#!/usr/bin/env python3
"""
Utility script to inspect `nflreadpy` coverage for play-by-play, schedules,
and team stats across a configurable set of seasons.

Run:
    python nfl_read_py.py --target-season 2025 --years-back 5
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Sequence

import pandas as pd


def _season_window(target_season: int, years_back: int) -> List[int]:
    start = max(1999, target_season - years_back + 1)
    return list(range(start, target_season + 1))


def _load_module():
    try:
        import nflreadpy as nfl
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "nflreadpy is not installed. Run `pip install nflreadpy` (or uv add) first."
        ) from exc
    return nfl


def _to_pandas(obj):
    """Convert Polars relations/DataFrames to pandas for lightweight previews."""
    if obj is None:
        return None
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    # Polars DataFrames also expose .to_dicts(), but pandas keeps the format consistent.
    return pd.DataFrame(obj)


def _summarize_df(name: str, df: pd.DataFrame | None, sample_rows: int = 5) -> None:
    if df is None or df.empty:
        print(f"{name}: no rows returned.")
        return
    preview_cols = ", ".join(df.columns[:12])
    date_cols = [col for col in ("game_date", "gameday") if col in df.columns]
    date_range = None
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


def _check_pbp(nfl, season: int, sample_rows: int) -> None:
    print(f"\nChecking nflreadpy play-by-play for {season} ...")
    try:
        pbp = nfl.load_pbp(seasons=season)
    except Exception as exc:  # noqa: BLE001
        print(f"  Unable to load season {season}: {exc}")
        return
    pbp_pd = _to_pandas(pbp)
    _summarize_df(f"PBP {season}", pbp_pd, sample_rows)


def _check_team_stats(nfl, seasons: Sequence[int], sample_rows: int) -> None:
    print(f"\nChecking team weekly stats for seasons: {', '.join(map(str, seasons))}")
    try:
        stats = nfl.load_team_stats(seasons=seasons, summary_level="week")
    except Exception as exc:  # noqa: BLE001
        print(f"  Unable to load team stats: {exc}")
        return
    stats_pd = _to_pandas(stats)
    _summarize_df("Team weekly stats", stats_pd, sample_rows)


def _check_schedules(nfl, seasons: Sequence[int], sample_rows: int) -> None:
    print(f"\nChecking schedules for seasons: {', '.join(map(str, seasons))}")
    try:
        schedules = nfl.load_schedules(seasons=seasons)
    except Exception as exc:  # noqa: BLE001
        print(f"  Unable to load schedules: {exc}")
        return
    schedules_pd = _to_pandas(schedules)
    _summarize_df("Schedules", schedules_pd, sample_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe nflreadpy coverage (play-by-play, schedules, stats)."
    )
    parser.add_argument(
        "--target-season",
        type=int,
        default=dt.date.today().year,
        help="Season to treat as current (default: calendar year).",
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=5,
        help="Number of trailing seasons (including the target) to evaluate.",
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

    history = _season_window(args.target_season, args.years_back)

    print(f"nflreadpy version: {getattr(nfl, '__version__', 'unknown')}")
    print("Available loaders:", [name for name in dir(nfl) if name.startswith("load_")])

    _check_pbp(nfl, args.target_season, args.sample_rows)
    _check_team_stats(nfl, history, args.sample_rows)
    _check_schedules(nfl, history, args.sample_rows)

    print("\nDone. Review the summaries above to decide if coverage is sufficient.")


if __name__ == "__main__":
    main()
