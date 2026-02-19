#!/usr/bin/env python3
"""
Generate NBA spread betting picks for today or the next N days.

Predicts upcoming games, joins with book odds from BigQuery (if available),
and outputs picks based on model edge vs book spread.

Example:
    python scripts/predict_nba_picks.py --project learned-pier-478122-p7 --days 3
    python scripts/predict_nba_picks.py --project learned-pier-478122-p7 --date 2026-02-16 --edge-threshold 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.data.nba_game_logs_loader import load_nba_game_logs_from_bq
from src.data.nba_fetcher import fetch_nba_schedule, fetch_nba_games_for_date
from src.models.predictor import GamePredictor


def load_upcoming_games(
    project: str,
    start_date: datetime.date,
    end_date: datetime.date,
    season: int,
) -> pd.DataFrame:
    """Load upcoming games (no scores) from BigQuery or API."""
    client = bigquery.Client(project=project)

    query = f"""
        SELECT game_id, season, game_date, home_team, away_team, home_score, away_score
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league = 'NBA'
          AND game_date BETWEEN @start_date AND @end_date
          AND (home_score IS NULL OR away_score IS NULL)
        ORDER BY game_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()

    if df.empty:
        # Fallback: fetch from API for each date
        rows = []
        for d in pd.date_range(start_date, end_date):
            day_str = d.strftime("%Y-%m-%d")
            try:
                games = fetch_nba_games_for_date(day_str)
                if not games.empty and ("home_score" not in games.columns or games["home_score"].isna().all()):
                    games["season"] = season
                    rows.append(games)
            except Exception:
                pass
        if rows:
            df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["game_id"], keep="first")

    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def load_odds_for_dates(project: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Load spread odds for the date range."""
    client = bigquery.Client(project=project)
    query = f"""
        SELECT game_id, game_date, home_team, away_team, book, line AS book_spread, price AS book_price
        FROM `{project}.sports_edge_raw.raw_nba_odds`
        WHERE league = 'NBA' AND market = 'spread' AND line IS NOT NULL
          AND game_date BETWEEN @start_date AND @end_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    odds = client.query(query, job_config=job_config).to_dataframe()
    if not odds.empty:
        odds["game_date"] = pd.to_datetime(odds["game_date"]).dt.date
        odds["prefer"] = (odds["book"] == "pinnacle").astype(int)
        odds = odds.sort_values("prefer", ascending=False).drop_duplicates(subset=["game_id"], keep="first").drop(columns=["prefer"])
    return odds


def run_predictions_for_upcoming(
    games: pd.DataFrame,
    season: int,
    project: str,
    model_version: str,
) -> pd.DataFrame:
    """Run predictions for upcoming games using full season data up to yesterday."""
    predictor = GamePredictor("NBA", model_version)

    # Full schedule and logs up to yesterday (no leakage)
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1))
    schedule = fetch_nba_schedule(season, use_cache=True)
    if schedule.empty:
        schedule = pd.DataFrame(columns=["game_id", "game_date", "home_team", "away_team", "home_score", "away_score"])
    schedule["game_date"] = pd.to_datetime(schedule["game_date"]).dt.date
    schedule = schedule[schedule["game_date"] <= yesterday]

    game_logs = load_nba_game_logs_from_bq([season], project_id=project)
    if game_logs is None:
        game_logs = pd.DataFrame()

    rows = []
    for _, game in games.iterrows():
        game_date = game["game_date"]
        if hasattr(game_date, "date"):
            game_date = game_date.date()
        game_date_ts = pd.Timestamp(game_date)

        logs_cutoff = game_logs[
            pd.to_datetime(game_logs["game_date"]).dt.date < game_date_ts.date()
        ].copy() if not game_logs.empty else pd.DataFrame()

        hist_cutoff = schedule[schedule["game_date"] < game_date_ts.date()].copy()

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
        except Exception as e:
            print(f"  Skip {game['away_team']} @ {game['home_team']}: {e}")
            continue

    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="NBA spread picks for upcoming games.")
    parser.add_argument("--project", required=True, help="GCP project ID.")
    parser.add_argument("--date", type=str, default=None, help="Start date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--days", type=int, default=1, help="Number of days ahead (default: 1 = today only).")
    parser.add_argument("--model-version", type=str, default="v2", help="Model version.")
    parser.add_argument("--edge-threshold", type=float, default=1.0, help="Min |edge| to recommend bet (pts).")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Min confidence to recommend bet.")
    parser.add_argument(
        "--fetch-odds",
        action="store_true",
        default=True,
        help="Fetch current odds from The Odds API before loading picks (default: True).",
    )
    parser.add_argument(
        "--no-fetch-odds",
        action="store_false",
        dest="fetch_odds",
        help="Skip odds fetch; use only odds already in BigQuery.",
    )
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    start_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else today
    end_date = start_date + timedelta(days=args.days - 1)

    season = start_date.year if start_date.month >= 10 else start_date.year - 1

    # Fetch current odds from The Odds API so we have book spread for strategy
    if args.fetch_odds:
        try:
            from src.data.nba_odds_api import fetch_and_load_odds_for_range
            n_odds = fetch_and_load_odds_for_range(
                args.project,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                replace_existing=True,
            )
            if n_odds > 0:
                print(f"Fetched {n_odds} odds rows from The Odds API")
        except Exception as e:
            print(f"Odds fetch skipped ({e}). Using odds from BigQuery if available.")

    print(f"Loading upcoming games ({start_date} to {end_date})...")
    games = load_upcoming_games(args.project, start_date, end_date, season)

    if games.empty:
        print("No upcoming games found.")
        return

    print(f"  Found {len(games)} games")
    print("\nRunning predictions...")
    predictions = run_predictions_for_upcoming(games, season, args.project, args.model_version)

    if predictions.empty:
        print("No predictions generated.")
        return

    # Join odds
    odds = load_odds_for_dates(args.project, start_date, end_date)
    pred = predictions.copy()
    pred["game_date"] = pd.to_datetime(pred["game_date"]).dt.date
    if not odds.empty:
        odds_join = odds[["game_id", "book_spread", "book_price"]].copy()
        pred = pred.merge(odds_join, on="game_id", how="left")
    else:
        pred["book_spread"] = pd.NA
        pred["book_price"] = pd.NA

    pred["edge"] = pred["predicted_spread"] - pred["book_spread"]
    pred["bet_home"] = (pred["edge"] < -args.edge_threshold) & (pred["confidence"] >= args.min_confidence)
    pred["bet_away"] = (pred["edge"] > args.edge_threshold) & (pred["confidence"] >= args.min_confidence)

    print("\n" + "=" * 70)
    print("NBA SPREAD PICKS")
    print("=" * 70)
    print(f"  Edge threshold: {args.edge_threshold} pts | Min confidence: {args.min_confidence}")
    print("=" * 70)

    picks = pred[(pred["bet_home"]) | (pred["bet_away"])].copy()
    if not picks.empty:
        picks["side"] = picks["bet_home"].map({True: "HOME", False: "AWAY"})
        picks["team"] = picks.apply(lambda r: r["home_team"] if r["bet_home"] else r["away_team"], axis=1)
        for _, row in picks.iterrows():
            # Book spread is home perspective: -5.5 = home favored. Bet home = take -5.5, bet away = take +5.5
            spread = row["book_spread"]
            if pd.notna(spread):
                spread_str = f" ({spread:+.1f})" if row["bet_home"] else f" ({-spread:+.1f})"
            else:
                spread_str = ""
            print(f"\n  BET {row['side']}: {row['away_team']} @ {row['home_team']}")
            print(f"    Pick: {row['team']}{spread_str}")
            print(f"    Model spread: {row['predicted_spread']:.1f} | Book: {row['book_spread']:.1f} | Edge: {row['edge']:.1f} pts")
            print(f"    Home win prob: {row['home_win_prob']:.1%} | Confidence: {row['confidence']:.1%}")
        print(f"\n  Total picks: {len(picks)}")
    else:
        print("\n  No picks meet threshold.")

    print("\n--- All games (no bet) ---")
    no_bet = pred[~(pred["bet_home"] | pred["bet_away"])]
    for _, row in no_bet.iterrows():
        edge_str = f" | Edge: {row['edge']:.1f}" if pd.notna(row.get("edge")) and pd.notna(row.get("book_spread")) else ""
        book_str = f"{row['book_spread']:.1f}" if pd.notna(row.get("book_spread")) else "N/A"
        print(f"  {row['away_team']} @ {row['home_team']}: Model {row['predicted_spread']:.1f} | Book {book_str}{edge_str}")

    if pred["book_spread"].isna().any():
        print("\n  Note: Some games have no odds in DB. Add odds manually to compute edge.")


if __name__ == "__main__":
    main()
