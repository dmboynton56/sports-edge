#!/usr/bin/env python3
"""
Fetch NBA odds from The Odds API and load into BigQuery raw_nba_odds.

Run as part of daily refresh, or standalone to update odds for a specific date.

Requires: ODDS_API_KEY in .env (from the-odds-api.com)

Example:
    python scripts/fetch_nba_odds.py --project learned-pier-478122-p7
    python scripts/fetch_nba_odds.py --project learned-pier-478122-p7 --date 2026-02-16
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NBA odds from The Odds API into BigQuery.")
    parser.add_argument("--project", required=True, help="GCP project ID.")
    parser.add_argument("--date", type=str, default=None, help="Start date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--days", type=int, default=1, help="Number of days to fetch (default: 1).")
    args = parser.parse_args()

    today = datetime.now(tz=timezone.utc).date()
    start_str = args.date or today.strftime("%Y-%m-%d")
    end_dt = datetime.strptime(start_str, "%Y-%m-%d").date() + timedelta(days=args.days - 1)
    end_str = end_dt.strftime("%Y-%m-%d")

    from src.data.nba_odds_api import fetch_and_load_odds_for_range

    n = fetch_and_load_odds_for_range(args.project, start_str, end_str, replace_existing=True)
    print(f"Loaded {n} odds rows for {start_str} to {end_str}")


if __name__ == "__main__":
    main()
