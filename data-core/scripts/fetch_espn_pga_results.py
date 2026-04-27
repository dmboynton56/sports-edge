#!/usr/bin/env python3
"""
Pull completed PGA / LIV / DP World Tour results from ESPN and write
pga_results_espn_supplement.tsv.

Supports multi-season fetching and merge mode (deduplicates against existing
supplement so historical runs don't overwrite current-year data).

  cd data-core && .venv/bin/python scripts/fetch_espn_pga_results.py
  cd data-core && .venv/bin/python scripts/fetch_espn_pga_results.py --season 2022,2023,2024,2025,2026

Limitation: if two tournaments share the same week, the scoreboard ?dates= probe may
return the "wrong" event (e.g. Puerto Rico Open vs Arnold Palmer) — one may be skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.espn_pga_results import fetch_season_results

ARCHIVE = Path(project_root) / "src" / "data" / "archive"
DEFAULT_OUT = ARCHIVE / "pga_results_espn_supplement.tsv"
DEFAULT_TOURS = "pga,liv,eur"
DEFAULT_SEASONS = "2026"
DEFAULT_COVERAGE_NAMES = [
    "Jon Rahm",
    "Bryson DeChambeau",
    "Patrick Reed",
    "Joaquin Niemann",
    "Tyrrell Hatton",
]

KEEP_COLS = (
    "season", "start", "end", "tournament", "location", "position",
    "name", "score", "round1", "round2", "round3", "round4", "total",
    "earnings", "fedex_points",
)


def _print_coverage_report(df: pd.DataFrame, names: List[str]) -> None:
    if df.empty:
        print("\nCoverage report: empty dataframe")
        return

    work = df.copy()
    work["start"] = pd.to_datetime(work["start"], errors="coerce")
    tname = work["tournament"].astype(str)

    is_liv = tname.str.contains(r"\bliv\b", case=False, na=False)
    is_eur = tname.str.contains(
        r"dubai|bahrain|qatar|joburg|sa open|kenya|hero indian|european|dp world",
        case=False,
        na=False,
    )
    is_major = tname.str.contains(
        r"masters|pga championship|u\.s\. open|the open|open championship",
        case=False,
        na=False,
        regex=True,
    )

    print("\n=== Coverage report ===")
    print(f"Rows: {len(work):,} | events: {work['tournament'].nunique():,}")
    print(
        f"Tour/event mix: LIV={int(is_liv.sum()):,}, EUR/DP-like={int(is_eur.sum()):,}, "
        f"Majors={int(is_major.sum()):,}, Other={int((~(is_liv | is_eur | is_major)).sum()):,}"
    )

    print("\nKey player coverage:")
    for nm in names:
        sub = work[work["name"] == nm].sort_values("start", ascending=False)
        n = len(sub)
        if n == 0:
            print(f"  {nm:<22} rows=0  latest=—  [MISSING]")
            continue
        latest = sub["start"].iloc[0]
        latest_s = latest.strftime("%Y-%m-%d") if pd.notna(latest) else "—"
        liv_n = int(sub["tournament"].astype(str).str.contains(r"\bliv\b", case=False, na=False).sum())
        eur_n = int(
            sub["tournament"]
            .astype(str)
            .str.contains(r"dubai|bahrain|qatar|joburg|sa open|kenya|hero indian", case=False, na=False)
            .sum()
        )
        print(f"  {nm:<22} rows={n:<3} latest={latest_s}  liv={liv_n:<2} eur_like={eur_n:<2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--season",
        type=str,
        default=DEFAULT_SEASONS,
        help="Season year(s), comma-separated (e.g. 2022,2023,2024,2025,2026)",
    )
    ap.add_argument(
        "--tour",
        type=str,
        default=DEFAULT_TOURS,
        help="Tour abbreviation (e.g. pga, liv, eur). Comma-separated for multiple.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-players", type=int, default=25, help="Min leaderboard size to accept")
    ap.add_argument(
        "--as-of",
        type=str,
        default="",
        help="ISO date; only fetch events ending on or before this (UTC). Default: now",
    )
    ap.add_argument(
        "--coverage-names",
        type=str,
        default=",".join(DEFAULT_COVERAGE_NAMES),
        help="Comma-separated player names for post-fetch coverage check",
    )
    ap.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite instead of merging with existing supplement",
    )
    args = ap.parse_args()

    cutoff = (
        datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
        if args.as_of
        else datetime.now(timezone.utc)
    )

    seasons = [int(s.strip()) for s in args.season.split(",") if s.strip()]
    tours = [t.strip() for t in args.tour.split(",") if t.strip()]

    all_dfs: list[pd.DataFrame] = []
    for season in seasons:
        for tour in tours:
            print(f"\nFetching {tour.upper()} {season}...")
            try:
                df, log = fetch_season_results(
                    season,
                    tour=tour,
                    only_completed_before=cutoff,
                    min_players=args.min_players,
                )
            except Exception as e:
                print(f"  ERROR fetching {tour} {season}: {e}")
                continue
            for line in log:
                print(line)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        print("No rows fetched; supplement not written.")
        sys.exit(1)

    df_new = pd.concat(all_dfs, ignore_index=True)
    drop_cols = [c for c in df_new.columns if c not in KEEP_COLS]
    df_new = df_new.drop(columns=drop_cols, errors="ignore")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_merge and args.out.exists():
        df_existing = pd.read_csv(args.out, sep="\t")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["season", "start", "tournament", "name"], keep="last"
        )
        print(f"\nMerged with existing supplement ({len(df_existing)} old + {len(df_new)} new)")
    else:
        df_combined = df_new

    df_combined.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(df_combined)} rows -> {args.out}")

    coverage_names = [n.strip() for n in args.coverage_names.split(",") if n.strip()]
    _print_coverage_report(df_combined, coverage_names)


if __name__ == "__main__":
    main()
