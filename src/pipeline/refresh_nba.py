#!/usr/bin/env python3
"""
Daily NBA refresh script.

Pulls games for the current date, builds features using historical data from BigQuery,
and writes predictions to BigQuery curated tables.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.pipeline.refresh import build_features
from src.models.predictor import GamePredictor
from src.data.nba_fetcher import fetch_nba_games_for_date, fetch_nba_schedule
from src.data.nba_game_logs_loader import load_nba_game_logs_from_bq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NBA predictions for a specific date.")
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project ID (e.g., learned-pier-478122-p7).",
    )
    parser.add_argument(
        "--model-version",
        default="v3",
        help="Model version tag for GamePredictor (default: v3).",
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Date to predict (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year override (defaults to date year).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute predictions but do not write to BigQuery.",
    )
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip fetching odds from The Odds API (requires ODDS_API_KEY).",
    )
    return parser.parse_args()


def _query_games(client: bigquery.Client, project: str, target_date: datetime.date) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE game_date = @target_date
          AND game_date IS NOT NULL
          AND league = 'NBA'
        ORDER BY game_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_date", "DATE", target_date),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    # BigQuery DATE columns are already local gameday
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.tz_localize(None)
    return df


def _delete_existing_predictions(client: bigquery.Client, project: str, game_ids: List[str], model_version: str) -> None:
    if not game_ids:
        return
    table_id = f"{project}.sports_edge_curated.model_predictions"
    query = f"DELETE FROM `{table_id}` WHERE game_id IN UNNEST(@game_ids) AND model_version = @model_version"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("game_ids", "STRING", game_ids),
            bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    )
    client.query(query, job_config=job_config).result()
    print(f"Removed existing predictions for {len(game_ids)} games (version {model_version}).")


def _log_model_run(
    client: bigquery.Client,
    project: str,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    rows_written: int,
    status: str,
    error_text: Optional[str] = None,
) -> None:
    table_id = f"{project}.sports_edge_curated.model_runs"
    df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "league": "NBA",
                "rows_written": rows_written,
                "status": status,
                "error_text": error_text,
            }
        ]
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"))
    job.result()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)

    target_date = args.date or datetime.now(tz=timezone.utc).date()
    season = args.season or (target_date.year if target_date.month >= 10 else target_date.year - 1)
    print(f"Building NBA predictions for {target_date}. Target season={season}.")

    # Fetch current odds from The Odds API and load into raw_nba_odds
    if not args.skip_odds:
        try:
            from src.data.nba_odds_api import fetch_and_load_odds
            date_str = target_date.strftime("%Y-%m-%d")
            n_odds = fetch_and_load_odds(args.project, date_str, replace_existing=True)
            print(f"Loaded {n_odds} odds rows for {date_str}")
        except Exception as e:
            print(f"Odds fetch skipped: {e}")

    games_df = _query_games(client, args.project, target_date)
    if games_df.empty:
        print(f"No NBA games found in BigQuery for {target_date}. Trying NBA API...")
        games_df = fetch_nba_games_for_date(target_date.strftime("%Y-%m-%d"), raise_on_error=True)

    if games_df.empty:
        print(f"No NBA games scheduled for {target_date}. Exiting.")
        return

    hist_seasons = [season]
    # Use helper functions for consistency with local pipeline
    # historical_games: ET naive from BQ or API
    historical_games = fetch_nba_schedule(season, use_cache=True)
    # game_logs: ET naive from BQ
    game_logs = load_nba_game_logs_from_bq(hist_seasons, project_id=args.project)
    historical_data = {"historical_games": historical_games, "game_logs": game_logs}

    # Predictor handles feature building automatically
    predictor = GamePredictor("NBA", model_version=args.model_version)
    # Ensure games_df dates are normalized before passing to batch
    games_df['game_date'] = predictor._normalize_datetime(games_df['game_date'])
    predictions = predictor.predict_batch(games_df, historical_games, game_logs=game_logs)
    
    if predictions.empty:
        print("No predictions were generated.")
        return

    # Merge with games_df to get game_id
    # Normalize dates to midnight to ensure merge matches even if times differ
    predictions["game_date"] = pd.to_datetime(predictions["game_date"]).dt.normalize().dt.tz_localize(None)
    games_df_normalized = games_df[["game_id", "home_team", "away_team", "game_date"]].copy()
    games_df_normalized["game_date"] = pd.to_datetime(games_df_normalized["game_date"]).dt.normalize().dt.tz_localize(None)

    predictions = predictions.merge(
        games_df_normalized,
        on=["home_team", "away_team", "game_date"],
        how="left",
    )

    # Drop any predictions where game_id could not be matched
    if predictions["game_id"].isna().any():
        missing_games = predictions[predictions["game_id"].isna()]
        print(f"Warning: Could not match game_id for {len(missing_games)} games. Dropping them.")
        for _, row in missing_games.iterrows():
            print(f"  Missing: {row['away_team']} @ {row['home_team']} on {row['game_date']}")
        predictions = predictions.dropna(subset=["game_id"])

    if predictions.empty:
        print("No predictions remaining after game_id matching. Exiting.")
        return

    predictions["league"] = "NBA"
    predictions["model_version"] = args.model_version
    predictions["prediction_ts"] = datetime.now(tz=timezone.utc)
    predictions["prediction_id"] = predictions.apply(
        lambda row: f"{row['game_id']}_{args.model_version}_{row['prediction_ts'].strftime('%Y%m%dT%H%M%S')}",
        axis=1,
    )
    predictions = predictions.rename(columns={"home_win_probability": "home_win_prob"})
    
    target_columns = [
        "prediction_id",
        "game_id",
        "league",
        "season",
        "season_week",
        "model_version",
        "predicted_spread",
        "home_win_prob",
        "prediction_ts",
        "input_hash",
    ]
    for column in target_columns:
        if column not in predictions.columns:
            predictions[column] = None
    predictions = predictions[target_columns]

    if args.dry_run:
        print(predictions[['game_id', 'predicted_spread', 'home_win_prob']])
        return

    run_id = f"nba_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    started_at = datetime.now(tz=timezone.utc)
    try:
        _delete_existing_predictions(client, args.project, predictions["game_id"].dropna().tolist(), args.model_version)
        table_id = f"{args.project}.sports_edge_curated.model_predictions"
        load_job = client.load_table_from_dataframe(
            predictions,
            table_id,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
        )
        load_job.result()
        finished_at = datetime.now(tz=timezone.utc)
        _log_model_run(client, args.project, run_id, started_at, finished_at, len(predictions), "SUCCESS")
        print(f"Wrote {len(predictions)} predictions to {table_id}")
    except Exception as exc:
        finished_at = datetime.now(tz=timezone.utc)
        _log_model_run(client, args.project, run_id, started_at, finished_at, 0, "FAILED", str(exc))
        raise


if __name__ == "__main__":
    main()
