#!/usr/bin/env python3
"""
Score MLB research markets for a given date: moneyline v3, run-line v1, and totals v1.

Usage:
    python scripts/predict_mlb_research_markets.py --date 2026-08-26 --output-dir notebooks/cache/research
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pipeline.mlb_research_markets import (
    load_mlb_moneyline_v3,
    load_mlb_runline_v1,
    load_mlb_totals_v1,
    score_research_markets_for_date,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONEYLINE_MODEL = ROOT / "models" / "mlb_winner_model_v3.pkl"
DEFAULT_TOTALS_MODEL = ROOT / "models" / "mlb_totals_model_v1.pkl"
DEFAULT_RUNLINE_MODEL = ROOT / "models" / "mlb_runline_model_v1.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MLB research markets for a date.")
    parser.add_argument("--date", required=True, help="Game date in YYYY-MM-DD.")
    parser.add_argument("--season", type=int, help="MLB season year (defaults to --date year).")
    parser.add_argument("--moneyline-model", default=str(DEFAULT_MONEYLINE_MODEL))
    parser.add_argument("--totals-model", default=str(DEFAULT_TOTALS_MODEL))
    parser.add_argument("--runline-model", default=str(DEFAULT_RUNLINE_MODEL))
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="notebooks/cache/research",
        help="Directory to write CSVs for moneyline, run-line, and totals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_date = pd.to_datetime(args.date).date()
    season = args.season or game_date.year

    print(f"Loading model artifacts...")
    print(f"  Moneyline v3: {args.moneyline_model}")
    print(f"  Totals v1: {args.totals_model}")
    print(f"  Run-line v1: {args.runline_model}")

    moneyline_artifact = load_mlb_moneyline_v3(args.moneyline_model)
    totals_artifact = load_mlb_totals_v1(args.totals_model)
    runline_artifact = load_mlb_runline_v1(args.runline_model)

    print(f"Scoring research markets for {game_date} (season={season})...")
    result = score_research_markets_for_date(
        game_date=game_date,
        season=season,
        moneyline_artifact=moneyline_artifact,
        totals_artifact=totals_artifact,
        runline_artifact=runline_artifact,
        min_prior_games=args.min_prior_games,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    moneyline_path = output_dir / f"mlb_moneyline_v3_{game_date}.csv"
    totals_path = output_dir / f"mlb_totals_v1_{game_date}.csv"
    runline_path = output_dir / f"mlb_runline_v1_{game_date}.csv"

    result.moneyline.to_csv(moneyline_path, index=False)
    result.totals.to_csv(totals_path, index=False)
    result.run_line.to_csv(runline_path, index=False)

    print(f"\nScored {len(result.moneyline)} moneyline games -> {moneyline_path}")
    print(f"Scored {len(result.totals)} totals games -> {totals_path}")
    print(f"Scored {len(result.run_line)} run-line games -> {runline_path}")


if __name__ == "__main__":
    main()
