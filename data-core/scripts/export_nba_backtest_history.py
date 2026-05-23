"""
Export NBA BigQuery spread-backtest history to cache artifacts.

This wraps `backtest_nba_spread.py` functions so the same backtest can feed
notebooks and the cross-sport performance hub without parsing console output.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from backtest_nba_spread import (
    join_predictions_odds_actuals,
    load_schedule_and_logs,
    run_predictions,
    simulate_betting,
    threshold_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NBA backtest history artifacts.")
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID"))
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--start-date", default="2025-10-22")
    parser.add_argument("--end-date")
    parser.add_argument("--model-version", default="v3")
    parser.add_argument("--edge-threshold", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--output-csv", default="data-core/notebooks/cache/nba_backtest_2025_v3.csv")
    parser.add_argument("--metrics-output", default="data-core/notebooks/cache/nba_backtest_2025_v3_metrics.json")
    return parser.parse_args()


def main() -> None:
    load_dotenv("data-core/.env")
    load_dotenv(".env")
    args = parse_args()
    project = args.project or os.getenv("GCP_PROJECT_ID")
    if not project:
        raise ValueError("GCP project is required via --project or GCP_PROJECT_ID.")

    end_date = args.end_date or (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Loading NBA BigQuery backtest data for {args.start_date} to {end_date}...")
    schedule, odds, game_logs = load_schedule_and_logs(
        project,
        args.season,
        args.start_date,
        end_date,
    )

    start_date_obj = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
    completed = schedule[
        schedule["home_score"].notna()
        & schedule["away_score"].notna()
        & (schedule["game_date"] >= start_date_obj)
        & (schedule["game_date"] <= end_date_obj)
    ].copy()
    if completed.empty:
        raise ValueError("No completed NBA games in requested date range.")

    predictions = run_predictions(completed, schedule, game_logs, args.model_version)
    merged = join_predictions_odds_actuals(predictions, schedule, odds)
    _, default_metrics = simulate_betting(
        merged,
        edge_threshold=args.edge_threshold,
        min_confidence=args.min_confidence,
    )
    sweep = threshold_sweep(merged)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    best_sweep = {}
    if not sweep.empty:
        best = sweep.sort_values(["roi", "n_bets"], ascending=[False, False]).iloc[0]
        best_sweep = {
            "edge_threshold": float(best["edge_threshold"]),
            "min_confidence": float(best["min_confidence"]),
            "n_bets": int(best["n_bets"]),
            "accuracy": float(best["accuracy"]),
            "roi": float(best["roi"]),
        }

    metrics = {
        "league": "NBA",
        "season": args.season,
        "model_version": args.model_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "date_min": args.start_date,
        "date_max": end_date,
        "schedule_rows": int(len(schedule)),
        "completed_games": int(len(completed)),
        "predictions_generated": int(len(predictions)),
        "odds_rows": int(len(odds)),
        "games_with_book_odds": int(merged["book_spread"].notna().sum()),
        "default_strategy": {
            "edge_threshold": args.edge_threshold,
            "min_confidence": args.min_confidence,
            **default_metrics,
        },
        "best_sweep": best_sweep,
        "threshold_sweep": sweep.to_dict(orient="records"),
        "results_csv": args.output_csv,
    }
    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.metrics_output}")
    print(json.dumps(metrics["default_strategy"], indent=2, sort_keys=True))
    print(json.dumps(metrics["best_sweep"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
