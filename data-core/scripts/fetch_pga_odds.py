#!/usr/bin/env python3
"""
Fetch PGA outright winner odds from The Odds API and cache to disk.

  cd data-core && .venv/bin/python scripts/fetch_pga_odds.py
  .venv/bin/python scripts/fetch_pga_odds.py --tournament masters
  .venv/bin/python scripts/fetch_pga_odds.py --tournament masters --cache-dir notebooks/cache
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.pga_odds_fetcher import (
    GOLF_SPORT_KEYS,
    TOURNAMENT_DISPLAY,
    detect_active_tournament,
    fetch_outrights,
    parse_outrights,
    fetch_and_summarize,
)

LOG = logging.getLogger("fetch_pga_odds")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fetch PGA outright odds from The Odds API.")
    parser.add_argument(
        "--tournament",
        choices=list(GOLF_SPORT_KEYS.keys()),
        help="Tournament to fetch (default: auto-detect active major)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(project_root) / "notebooks" / "cache",
        help="Directory to write cached files",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Path to predictions CSV for player name matching",
    )
    args = parser.parse_args()

    tournament = args.tournament
    if not tournament:
        LOG.info("Auto-detecting active major tournament...")
        tournament = detect_active_tournament()
        if not tournament:
            LOG.error("No active golf major found. Specify --tournament explicitly.")
            sys.exit(1)
        LOG.info("Detected active tournament: %s", TOURNAMENT_DISPLAY.get(tournament, tournament))

    prediction_names = None
    if args.predictions_csv and args.predictions_csv.exists():
        pdf = pd.read_csv(args.predictions_csv)
        if "player" in pdf.columns:
            prediction_names = pdf["player"].tolist()
            LOG.info("Loaded %d player names from predictions for matching", len(prediction_names))

    summary = fetch_and_summarize(tournament, prediction_names=prediction_names)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    json_path = args.cache_dir / f"pga_odds_{tournament}_{date_str}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s (%d players, %d books)", json_path, len(summary["playerOdds"]), len(summary["books"]))

    raw, _ = fetch_outrights(tournament)
    df = parse_outrights(raw)
    if not df.empty:
        csv_path = args.cache_dir / f"pga_odds_{tournament}_{date_str}.csv"
        df.to_csv(csv_path, index=False)
        LOG.info("Wrote %s (%d rows)", csv_path, len(df))

    print(f"\nAPI credits remaining: {summary.get('apiCreditsRemaining', '?')}")
    print(f"Tournament: {summary['tournament']}")
    print(f"Books: {', '.join(summary['books'])}")

    if summary.get("overrounds"):
        print("\nOverrounds (vig) per book:")
        for book, ov in summary["overrounds"].items():
            print(f"  {book:<20} {ov:.1%}")

    print(f"\nTop 15 by consensus implied probability:")
    for i, po in enumerate(summary["playerOdds"][:15]):
        print(
            f"  {i+1:>2}. {po['player']:<30} "
            f"consensus={po['consensusImplied']:.1%}  "
            f"best={po['bestPrice']:>+6} ({po['bestBook']})"
        )

    if summary.get("unmatched"):
        print(f"\nUnmatched API names ({len(summary['unmatched'])}):")
        for name in list(summary["unmatched"])[:10]:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
