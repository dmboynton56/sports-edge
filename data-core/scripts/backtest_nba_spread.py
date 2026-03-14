#!/usr/bin/env python3
"""
Backtest NBA spread predictions against book odds and simulate betting.

Runs day-by-day predictions (leakage-free), joins with raw_nba_odds,
and evaluates betting strategies for EV and accuracy.

Example:
    python scripts/backtest_nba_spread.py --project learned-pier-478122-p7 --season 2025 --start-date 2025-11-01 --end-date 2026-01-15
    python scripts/backtest_nba_spread.py --project learned-pier-478122-p7 --season 2025 --output-csv backtest_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from tqdm import tqdm

from src.data.nba_game_logs_loader import load_nba_game_logs_from_bq
from src.models.predictor import GamePredictor


def load_schedule_and_logs(
    project: str,
    season: int,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load schedule (with actuals), game logs, and odds from BigQuery."""
    client = bigquery.Client(project=project)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    season_start = datetime(season, 10, 1).date()

    # Schedule: full season for predictor context
    sched_query = f"""
        SELECT game_id, season, game_date, home_team, away_team, home_score, away_score
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league = 'NBA'
          AND game_date BETWEEN @season_start AND @end_date
        ORDER BY game_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("season_start", "DATE", season_start),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_dt),
        ]
    )
    schedule = client.query(sched_query, job_config=job_config).to_dataframe()
    schedule["game_date"] = pd.to_datetime(schedule["game_date"]).dt.date

    if schedule.empty:
        return schedule, pd.DataFrame(), pd.DataFrame()

    # Odds: spread market, prefer pinnacle
    odds_query = f"""
        SELECT game_id, game_date, home_team, away_team, book, line AS book_spread, price AS book_price
        FROM `{project}.sports_edge_raw.raw_nba_odds`
        WHERE league = 'NBA' AND market = 'spread' AND line IS NOT NULL
          AND game_date BETWEEN @odds_start AND @odds_end
    """
    odds_job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("odds_start", "DATE", start_dt),
            bigquery.ScalarQueryParameter("odds_end", "DATE", end_dt),
        ]
    )
    odds_raw = client.query(odds_query, job_config=odds_job_config).to_dataframe()
    odds_raw["game_date"] = pd.to_datetime(odds_raw["game_date"]).dt.date

    # Dedupe: prefer pinnacle, else first book per game
    if not odds_raw.empty:
        odds_raw["prefer"] = (odds_raw["book"] == "pinnacle").astype(int)
        odds = (
            odds_raw.sort_values("prefer", ascending=False)
            .drop_duplicates(subset=["game_id"], keep="first")
            .drop(columns=["prefer"])
        )
    else:
        odds = pd.DataFrame()

    # Game logs
    game_logs = load_nba_game_logs_from_bq([season], project_id=project)
    if game_logs is None:
        game_logs = pd.DataFrame()

    return schedule, odds, game_logs


def run_predictions(
    test_games: pd.DataFrame,
    full_schedule: pd.DataFrame,
    game_logs: pd.DataFrame,
    model_version: str,
) -> pd.DataFrame:
    """Run leakage-free predictions for each game."""
    predictor = GamePredictor("NBA", model_version)
    rows = []

    for _, game in tqdm(test_games.iterrows(), total=len(test_games), desc="Predicting"):
        game_date = game["game_date"]
        if hasattr(game_date, "date"):
            game_date = game_date.date()
        game_date_ts = pd.Timestamp(game_date) if isinstance(game_date, (str, datetime)) else pd.Timestamp(game_date)

        logs_cutoff = game_logs[
            pd.to_datetime(game_logs["game_date"]).dt.date < game_date_ts.date()
        ].copy() if not game_logs.empty else pd.DataFrame()

        hist_cutoff = full_schedule[
            pd.to_datetime(full_schedule["game_date"]).dt.date < game_date_ts.date()
        ].copy()

        try:
            pred = predictor.predict(
                pd.DataFrame([game]),
                hist_cutoff,
                game_logs=logs_cutoff if not logs_cutoff.empty else None,
            )
            rows.append({
                "game_id": game["game_id"],
                "game_date": game_date,
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "predicted_spread": pred["predicted_spread"],
                "home_win_prob": pred["home_win_probability"],
                "confidence": pred.get("confidence", 0.5),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def join_predictions_odds_actuals(
    predictions: pd.DataFrame,
    schedule: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    """Join predictions with odds and actuals."""
    pred = predictions.copy()
    pred["game_date"] = pd.to_datetime(pred["game_date"]).dt.date

    # Merge actuals from schedule
    sched = schedule[["game_id", "game_date", "home_team", "away_team", "home_score", "away_score"]].copy()
    sched["game_date"] = pd.to_datetime(sched["game_date"]).dt.date
    merged = pred.merge(
        sched,
        on=["game_id", "game_date", "home_team", "away_team"],
        how="left",
    )

    # Merge odds (join on game_id; if multiple books, odds already deduped)
    if not odds.empty:
        odds_join = odds[["game_id", "book_spread", "book_price"]].copy()
        merged = merged.merge(odds_join, on="game_id", how="left")
    else:
        merged["book_spread"] = np.nan
        merged["book_price"] = np.nan

    # Actual margin and cover
    merged["actual_margin"] = merged["home_score"].astype(float) - merged["away_score"].astype(float)
    # Home covers if actual_margin + book_spread > 0 (book spread is home perspective, e.g. -5.5)
    merged["home_cover"] = (merged["actual_margin"] + merged["book_spread"]) > 0

    return merged


def simulate_betting(
    df: pd.DataFrame,
    edge_threshold: float = 1.0,
    min_confidence: float = 0.0,
    unit_stake: float = 1.0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Simulate spread betting based on model edge vs book.

    Edge = predicted_spread - book_spread. Negative edge -> bet home, positive -> bet away.
    """
    has_odds = df["book_spread"].notna()
    df = df[has_odds].copy()

    df["edge"] = df["predicted_spread"] - df["book_spread"]

    # Bet home when edge < -threshold, away when edge > threshold
    df["bet_home"] = (df["edge"] < -edge_threshold) & (df["confidence"] >= min_confidence)
    df["bet_away"] = (df["edge"] > edge_threshold) & (df["confidence"] >= min_confidence)

    # Payout: profit per unit when we win. Pinnacle uses decimal (1.917); CSV may use American.
    def profit_per_unit(price: float) -> float:
        if pd.isna(price) or price == 0:
            return 100 / 110  # -110 American default
        p = float(price)
        if 1 < p < 10:  # Decimal odds (e.g. 1.917)
            return p - 1
        if p > 100:  # American +150
            return p / 100
        if p < -100:  # American -110
            return 100 / abs(p)
        return 100 / 110

    df["payout_mult"] = df["book_price"].apply(profit_per_unit)

    # Outcome: did our bet win?
    df["bet_won"] = np.where(
        df["bet_home"],
        df["home_cover"],
        np.where(df["bet_away"], ~df["home_cover"], np.nan),
    )
    df["bet_side"] = np.where(df["bet_home"], "home", np.where(df["bet_away"], "away", None))

    # Only rows where we actually bet
    bets = df[df["bet_home"] | df["bet_away"]].copy()

    if bets.empty:
        metrics = {
            "n_bets": 0,
            "n_wins": 0,
            "accuracy": 0.0,
            "units_won": 0.0,
            "units_lost": 0.0,
            "roi": 0.0,
        }
        return df, metrics

    won = bets["bet_won"].fillna(False).astype(bool)
    n_wins = int(won.sum())
    units_won = (won * bets["payout_mult"] * unit_stake).sum()
    units_lost = (len(bets) - n_wins) * unit_stake
    total_staked = len(bets) * unit_stake
    roi = (units_won - units_lost) / total_staked if total_staked > 0 else 0

    metrics = {
        "n_bets": len(bets),
        "n_wins": n_wins,
        "accuracy": n_wins / len(bets),
        "units_won": float(units_won),
        "units_lost": float(units_lost),
        "roi": roi,
    }

    return df, metrics


def threshold_sweep(
    df: pd.DataFrame,
    edge_thresholds: List[float] = (0.5, 1.0, 1.5, 2.0),
    min_confidences: List[float] = (0.0, 0.2, 0.4),
) -> pd.DataFrame:
    """Sweep edge and confidence thresholds; return ROI and sample size per cell."""
    rows = []
    for et in edge_thresholds:
        for mc in min_confidences:
            _, m = simulate_betting(df, edge_threshold=et, min_confidence=mc)
            rows.append({
                "edge_threshold": et,
                "min_confidence": mc,
                "n_bets": m["n_bets"],
                "accuracy": m["accuracy"],
                "roi": m["roi"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Backtest NBA spread predictions vs book odds.")
    parser.add_argument("--project", required=True, help="GCP project ID.")
    parser.add_argument("--season", type=int, default=2025, help="NBA season year.")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-11-01",
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Backtest end date (default: yesterday).",
    )
    parser.add_argument(
        "--full-season",
        action="store_true",
        help="Backtest entire 2025-26 season so far (start 2025-10-22, end yesterday).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v2",
        help="Model version (default: v2 to match predict_nba_date).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Write full results to CSV for notebook analysis.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=1.0,
        help="Min |edge| in points to place bet (default: 1.0).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Min confidence to place bet (default: 0).",
    )
    args = parser.parse_args()

    if args.full_season:
        args.start_date = "2025-10-22"  # First 2025-26 regular season games
        args.end_date = None

    end_date = args.end_date or (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Loading schedule, odds, logs for {args.season} ({args.start_date} to {end_date})...")
    schedule, odds, game_logs = load_schedule_and_logs(
        args.project, args.season, args.start_date, end_date
    )

    completed = schedule[
        schedule["home_score"].notna()
        & schedule["away_score"].notna()
        & (schedule["game_date"] >= datetime.strptime(args.start_date, "%Y-%m-%d").date())
        & (schedule["game_date"] <= datetime.strptime(end_date, "%Y-%m-%d").date())
    ].copy()

    if completed.empty:
        print("No completed games in date range. Exiting.")
        return

    print(f"  Schedule: {len(schedule)} games (full season)")
    print(f"  Completed in range: {len(completed)} games")
    print(f"  Odds: {len(odds)} spread lines")
    print(f"  Game logs: {len(game_logs)} entries")

    print(f"\nRunning predictions (model {args.model_version})...")
    predictions = run_predictions(completed, schedule, game_logs, args.model_version)
    print(f"  Generated {len(predictions)} predictions")

    merged = join_predictions_odds_actuals(predictions, schedule, odds)
    n_with_odds = merged["book_spread"].notna().sum()
    print(f"  Joined: {n_with_odds} games with book odds")

    # Betting simulation
    _, metrics = simulate_betting(
        merged,
        edge_threshold=args.edge_threshold,
        min_confidence=args.min_confidence,
    )

    print("\n" + "=" * 60)
    print("BETTING SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Edge threshold: {args.edge_threshold} pts | Min confidence: {args.min_confidence}")
    print(f"  Bets placed: {metrics['n_bets']}")
    print(f"  Wins: {metrics['n_wins']} | Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Units: +{metrics['units_won']:.2f} / -{metrics['units_lost']:.2f}")
    print(f"  ROI: {metrics['roi']:.1%}")
    print("=" * 60)

    # Threshold sweep
    print("\nThreshold sweep (edge_threshold x min_confidence):")
    sweep = threshold_sweep(merged)
    pivot = sweep.pivot_table(
        index="edge_threshold",
        columns="min_confidence",
        values=["roi", "n_bets"],
        aggfunc="first",
    )
    print(pivot.to_string())

    if args.output_csv:
        merged.to_csv(args.output_csv, index=False)
        print(f"\nWrote full results to {args.output_csv}")


if __name__ == "__main__":
    main()
