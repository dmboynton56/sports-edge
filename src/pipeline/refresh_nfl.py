#!/usr/bin/env python3
"""
Weekly NFL refresh script.

Pulls upcoming games for the next NFL week, builds features using historical data,
and writes predictions to BigQuery model tables. Designed to run every Tuesday.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.models.predictor import GamePredictor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NFL predictions for upcoming week.")
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project ID (e.g., learned-pier-478122-p7).",
    )
    parser.add_argument(
        "--model-version",
        default="v1",
        help="Model version tag for GamePredictor (default: v1).",
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Override the week start date (YYYY-MM-DD). Default: upcoming Thursday.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=6,
        help="Number of days to include after start date (default: 6 for Thu-Mon).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year override (defaults to start-date year).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute predictions but do not write to BigQuery.",
    )
    return parser.parse_args()


def _nfl_week_start(today: datetime.date) -> datetime.date:
    """Return the Thursday that starts the current NFL week."""
    weekday = today.weekday()  # Monday=0
    # Tue (1), Wed (2) -> Look forward to upcoming Thursday (3)
    if weekday in [1, 2]:
        return today + timedelta(days=(3 - weekday))
    # Thu (3), Fri (4), Sat (5), Sun (6), Mon (0) -> Look back to most recent Thursday
    else:
        if weekday == 0:  # Monday
            return today - timedelta(days=4)
        else:  # Thursday-Sunday
            return today - timedelta(days=(weekday - 3))


def _query_games(client: bigquery.Client, project: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE game_date BETWEEN @start_date AND @end_date
          AND game_date IS NOT NULL
        ORDER BY game_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _query_historical_games(client: bigquery.Client, project: str, seasons: List[int]) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE season IN UNNEST(@seasons)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("seasons", "INT64", seasons)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _query_pbp(client: bigquery.Client, project: str, seasons: List[int]) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM `{project}.sports_edge_raw.raw_pbp`
        WHERE season IN UNNEST(@seasons)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("seasons", "INT64", seasons)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _delete_existing_predictions(client: bigquery.Client, project: str, game_ids: List[str]) -> None:
    if not game_ids:
        return
    table_id = f"{project}.sports_edge_curated.model_predictions"
    query = f"DELETE FROM `{table_id}` WHERE game_id IN UNNEST(@game_ids)"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("game_ids", "STRING", game_ids)]
    )
    client.query(query, job_config=job_config).result()
    print(f"Removed existing predictions for {len(game_ids)} games.")


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
                "league": "NFL",
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

    today = datetime.now(tz=timezone.utc).date()
    start_date = args.start_date or _nfl_week_start(today)
    end_date = start_date + timedelta(days=args.window_days)
    season = args.season or (start_date.year if start_date.month >= 8 else start_date.year - 1)
    print(f"Building predictions for games between {start_date} and {end_date}. Target season={season}.")

    games_df = _query_games(client, args.project, start_date, end_date)
    if games_df.empty:
        print("No NFL games scheduled in the requested window. Exiting.")
        return

    hist_seasons = list(range(season - 3, season + 1))
    historical_games = _query_historical_games(client, args.project, hist_seasons)
    pbp = _query_pbp(client, args.project, hist_seasons)
    historical_data = {"historical_games": historical_games, "play_by_play": pbp}

    predictor = GamePredictor("NFL", model_version=args.model_version)
    predictions = predictor.predict_batch(games_df, historical_games, pbp)
    if predictions.empty:
        print("No predictions were generated.")
        return

    # Merge with games_df to get game_id
    # Normalize dates to midnight to ensure merge matches even if times differ
    predictions["game_date"] = pd.to_datetime(predictions["game_date"]).dt.normalize()
    games_df_normalized = games_df[["game_id", "home_team", "away_team", "game_date"]].copy()
    games_df_normalized["game_date"] = pd.to_datetime(games_df_normalized["game_date"]).dt.normalize()

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

    predictions["league"] = "NFL"
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
        print(predictions)
        return

    run_id = f"nfl_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    started_at = datetime.now(tz=timezone.utc)
    try:
        _delete_existing_predictions(client, args.project, predictions["game_id"].dropna().tolist())
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
    except Exception as exc:  # noqa: BLE001
        finished_at = datetime.now(tz=timezone.utc)
        _log_model_run(client, args.project, run_id, started_at, finished_at, 0, "FAILED", str(exc))
        raise


if __name__ == "__main__":
    main()
