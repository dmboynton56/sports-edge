#!/usr/bin/env python3
"""
Pull completed PGA Tour results from ESPN and write pga_results_espn_supplement.tsv.

Merge into the feature pipeline via build_pga_feature_store (auto-loads supplement).

Limitation: if two tournaments share the same week, the scoreboard ?dates= probe may
return the "wrong" event (e.g. Puerto Rico Open vs Arnold Palmer) — one may be skipped.

  cd data-core && .venv/bin/python scripts/fetch_espn_pga_results.py --season 2026
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.espn_pga_results import fetch_season_results

ARCHIVE = Path(project_root) / "src" / "data" / "archive"
DEFAULT_OUT = ARCHIVE / "pga_results_espn_supplement.tsv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-players", type=int, default=25, help="Min leaderboard size to accept")
    ap.add_argument(
        "--as-of",
        type=str,
        default="",
        help="ISO date; only fetch events ending on or before this (UTC). Default: now",
    )
    args = ap.parse_args()

    cutoff = (
        datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
        if args.as_of
        else datetime.now(timezone.utc)
    )

    df, log = fetch_season_results(
        args.season,
        only_completed_before=cutoff,
        min_players=args.min_players,
    )
    for line in log:
        print(line)

    if df.empty:
        print("No rows fetched; supplement not written.")
        sys.exit(1)

    drop_cols = [c for c in df.columns if c not in (
        "season", "start", "end", "tournament", "location", "position",
        "name", "score", "round1", "round2", "round3", "round4", "total",
        "earnings", "fedex_points",
    )]
    df = df.drop(columns=drop_cols, errors="ignore")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
