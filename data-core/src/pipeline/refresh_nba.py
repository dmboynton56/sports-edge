#!/usr/bin/env python3
"""
Daily NBA refresh script.

Pulls games for the current date, builds features using historical data from BigQuery,
and writes predictions to BigQuery curated tables.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.pipeline.refresh import build_features
from src.models.predictor import GamePredictor
from src.data.nba_fetcher import (
    ensure_nba_schedule_covers_dates,
    fetch_nba_schedule,
    nba_season_start_year,
)
from src.data.nba_game_logs_loader import load_nba_game_logs_from_bq
from src.utils.injury_loader import load_injury_impacts_from_supabase
from src.utils.explanation_export import build_explanation_rows, write_explanation_cache


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
        "--lookback-days",
        type=int,
        default=0,
        help="Days before --date to include. Default: 0.",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=0,
        help="Days after --date to include. Dashboard NBA slate is today+1.",
    )
    parser.add_argument(
        "--require-games",
        action="store_true",
        help="Exit non-zero when the prediction window has no NBA games after BQ+ESPN lookup.",
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
    parser.add_argument(
        "--injury-aware",
        action="store_true",
        help="Load player_impact_estimates from Supabase and adjust features.",
    )
    parser.add_argument(
        "--include-explanations",
        action="store_true",
        help="Emit top feature drivers and write explanation cache for dashboard sync.",
    )
    return parser.parse_args()


def _prediction_window(anchor: date, lookback_days: int, lookahead_days: int) -> tuple[date, date]:
    if lookback_days < 0 or lookahead_days < 0:
        raise ValueError("lookback-days and lookahead-days must be non-negative.")
    return anchor - timedelta(days=lookback_days), anchor + timedelta(days=lookahead_days)


def _query_games(
    client: bigquery.Client,
    project: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE game_date IS NOT NULL
          AND league = 'NBA'
          AND CAST(game_date AS DATE) BETWEEN @start_date AND @end_date
        ORDER BY game_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    # BigQuery DATE columns are already local gameday
    if not df.empty:
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
    start_date, end_date = _prediction_window(
        target_date,
        args.lookback_days,
        args.lookahead_days,
    )
    season = args.season or nba_season_start_year(target_date)
    print(
        f"Building NBA predictions for {start_date} to {end_date}. Target season={season}."
    )

    # Fetch current odds from The Odds API and load into raw_nba_odds
    if not args.skip_odds:
        try:
            from src.data.nba_odds_api import fetch_and_load_odds
            for day in pd.date_range(start=start_date, end=end_date, freq="D"):
                date_str = day.strftime("%Y-%m-%d")
                n_odds = fetch_and_load_odds(args.project, date_str, replace_existing=True)
                print(f"Loaded {n_odds} odds rows for {date_str}")
        except Exception as e:
            print(f"Odds fetch skipped: {e}")

    games_df = _query_games(client, args.project, start_date, end_date)
    if games_df.empty:
        print(
            f"No NBA games found in BigQuery for {start_date} to {end_date}. Trying NBA API..."
        )
    games_df = ensure_nba_schedule_covers_dates(
        games_df,
        start_date,
        end_date,
        raise_on_error=True,
    )

    if games_df.empty:
        message = f"No NBA games scheduled for {start_date} to {end_date}."
        print(f"{message} Exiting.")
        if args.require_games:
            raise SystemExit(message)
        return
    games_df["season"] = [
        nba_season_start_year(game_date) for game_date in games_df["game_date"]
    ]

    hist_seasons = [season - 1, season]
    # Prior season covers leftover rows tagged with ESPN's fallback/championship year.
    # Current-season BQ may only have the incremental backfill window; pass dates so
    # fetch_nba_schedule trusts that cache instead of scanning ESPN day-by-day.
    schedule_frames = []
    for hist_season in hist_seasons:
        kwargs = {"use_cache": True}
        if hist_season == season:
            kwargs["date_from"] = date(season, 10, 1)
            kwargs["date_to"] = end_date
        frame = fetch_nba_schedule(hist_season, **kwargs)
        if frame is not None and not frame.empty:
            schedule_frames.append(frame)
    historical_games = (
        pd.concat(schedule_frames, ignore_index=True).drop_duplicates(subset=["game_id"])
        if schedule_frames
        else pd.DataFrame()
    )
    game_logs = load_nba_game_logs_from_bq(hist_seasons, project_id=args.project)

    # Predictor handles feature building automatically
    predictor = GamePredictor("NBA", model_version=args.model_version)
    # Ensure games_df dates are normalized before passing to batch
    games_df['game_date'] = predictor._normalize_datetime(games_df['game_date'])

    injury_impacts = None
    if args.injury_aware:
        game_ids = [str(gid) for gid in games_df["game_id"].dropna().tolist()]
        injury_impacts = load_injury_impacts_from_supabase("NBA", game_ids=game_ids)
        print(f"Loaded {len(injury_impacts)} injury impact rows for NBA refresh.")

    predictions = predictor.predict_batch(
        games_df,
        historical_games,
        game_logs=game_logs,
        injury_impacts=injury_impacts,
        include_explanations=args.include_explanations,
    )
    
    if predictions.empty:
        raise SystemExit(
            f"Found {len(games_df)} NBA games for {start_date} to {end_date} "
            "but generated 0 predictions."
        )

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
        raise SystemExit("No predictions remaining after game_id matching.")

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

    if args.include_explanations:
        explanation_rows = build_explanation_rows(
            predictions,
            league="NBA",
            model_version=args.model_version,
        )
        cache_dir = Path(__file__).resolve().parent.parent.parent / "notebooks" / "cache"
        out_path = write_explanation_cache(explanation_rows, league="NBA", cache_dir=cache_dir)
        print(f"Wrote {len(explanation_rows)} NBA explanations to {out_path}")

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
