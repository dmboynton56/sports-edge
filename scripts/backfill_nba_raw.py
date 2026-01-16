#!/usr/bin/env python3
"""
Backfill BigQuery raw tables with NBA data from nba_api.

Example:
    python scripts/backfill_nba_raw.py --project learned-pier-478122-p7 --seasons 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.data import nba_fetcher
from src.data.nba_game_logs_loader import load_nba_game_logs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load NBA data into BigQuery raw tables.")
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project ID (e.g., learned-pier-478122-p7).",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(2020, 2026)),
        help="Seasons to import (default: 2020-2025).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows for each season before inserting.",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out


def _ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    elif "GAME_DATE" in df.columns:
        df["game_date"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _build_raw_records(df: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    clean = df.where(pd.notna(df), None)
    records = clean.to_dict(orient="records")
    processed = []
    for rec in records:
        processed.append(
            json.dumps({k: _to_jsonable(v) for k, v in rec.items() if k not in exclude})
        )
    return processed


def _load_dataframe(
    client: bigquery.Client,
    df: pd.DataFrame,
    table_id: str,
    write_disposition: str = "WRITE_APPEND",
) -> None:
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Loaded {len(df):,} rows into {table_id}")


def _delete_season(client: bigquery.Client, table_id: str, season: int) -> None:
    query = f"DELETE FROM `{table_id}` WHERE season = @season"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("season", "INT64", season)]
    )
    client.query(query, job_config=job_config).result()
    print(f"Cleared {table_id} for season {season}")


GAME_LOGS_COLUMNS = [
    "game_id",
    "game_date",
    "team",
    "team_id",
    "season",
    "points_scored",
    "points_allowed",
    "net_rating",
    "point_diff",
    "ingested_at",
]

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "ingested_at",
]


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)
    utc_now = datetime.now(tz=timezone.utc)

    for season in args.seasons:
        print(f"Processing NBA season {season}")
        
        # Fetch schedule
        print(f"  Fetching schedule for {season}...")
        schedules_df = nba_fetcher.fetch_nba_schedule(season, raise_on_error=True)
        
        if schedules_df.empty:
            print(f"  Warning: No schedule data for {season}")
            continue
        
        schedules_df = _ensure_game_date(schedules_df)
        schedules_df["ingested_at"] = utc_now
        
        # Handle nulls (rare cases in some seasons or exhibition games)
        schedules_df = schedules_df[
            schedules_df['game_id'].notna() & 
            schedules_df['home_team'].notna() & 
            schedules_df['away_team'].notna()
        ].copy()
        
        # Fetch game logs
        print(f"  Fetching game logs for {season}...")
        game_logs_df = load_nba_game_logs([season], strict=False)
        
        if game_logs_df is None or game_logs_df.empty:
            print(f"  Warning: No game logs for {season}")
            game_logs_df = pd.DataFrame()
        else:
            game_logs_df = _ensure_game_date(game_logs_df)
            game_logs_df["ingested_at"] = utc_now
        
        # Prepare schedule data
        schedules_df = _ensure_columns(schedules_df, SCHEDULE_COLUMNS[:-1])
        schedules_selected = schedules_df[SCHEDULE_COLUMNS[:-1]].copy()
        schedules_selected["ingested_at"] = schedules_df["ingested_at"]
        schedules_selected["raw_record"] = _build_raw_records(schedules_df, SCHEDULE_COLUMNS[:-1])
        
        # Prepare game logs data
        if not game_logs_df.empty:
            game_logs_df = _ensure_columns(game_logs_df, GAME_LOGS_COLUMNS[:-1])
            game_logs_selected = game_logs_df[GAME_LOGS_COLUMNS[:-1]].copy()
            game_logs_selected["ingested_at"] = game_logs_df["ingested_at"]
            game_logs_selected["raw_record"] = _build_raw_records(game_logs_df, GAME_LOGS_COLUMNS[:-1])
        else:
            game_logs_selected = pd.DataFrame(columns=GAME_LOGS_COLUMNS)
        
        if args.replace:
            _delete_season(client, f"{args.project}.sports_edge_raw.raw_schedules", season)
            if not game_logs_selected.empty:
                _delete_season(client, f"{args.project}.sports_edge_raw.raw_nba_game_logs", season)
        
        _load_dataframe(client, schedules_selected, f"{args.project}.sports_edge_raw.raw_schedules")
        if not game_logs_selected.empty:
            _load_dataframe(client, game_logs_selected, f"{args.project}.sports_edge_raw.raw_nba_game_logs")
    
    print("NBA raw backfill complete.")


if __name__ == "__main__":
    main()

