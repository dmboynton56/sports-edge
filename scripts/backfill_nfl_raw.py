#!/usr/bin/env python3
"""
Backfill BigQuery raw tables with NFL data from nflreadpy.

Example:
    python scripts/backfill_nfl_raw.py --project learned-pier-478122-p7 --seasons 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Any

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load nflreadpy data into BigQuery raw tables.")
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
    elif "gameday" in df.columns:
        df["game_date"] = pd.to_datetime(df["gameday"], errors="coerce")
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


def _load_module():
    try:
        import nflreadpy as nfl  # noqa: WPS433
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("Install nflreadpy to run this script (`pip install nflreadpy`).") from exc
    return nfl


PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "game_date",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "play_type",
    "yards_gained",
    "epa",
    "ingested_at",
]

SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "venue",
    "result",
    "ingested_at",
]

TEAM_STATS_COLUMNS = [
    "team",
    "season",
    "week",
    "points_for",
    "points_against",
    "epa_off",
    "epa_def",
    "win_pct",
    "ingested_at",
]


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)
    nfl = _load_module()
    utc_now = datetime.now(tz=timezone.utc)

    for season in args.seasons:
        print(f"Processing season {season}")
        pbp_rel = nfl.load_pbp(seasons=season)
        schedules_rel = nfl.load_schedules(seasons=season)
        team_stats_rel = nfl.load_team_stats(seasons=season, summary_level="week")

        pbp_df = pbp_rel.to_pandas()
        schedules_df = schedules_rel.to_pandas()
        team_stats_df = team_stats_rel.to_pandas()

        pbp_df = _ensure_game_date(pbp_df)
        schedules_df = _ensure_game_date(schedules_df)

        pbp_df["ingested_at"] = utc_now
        schedules_df["ingested_at"] = utc_now
        team_stats_df["ingested_at"] = utc_now

        pbp_df = _ensure_columns(pbp_df, PBP_COLUMNS[:-1])
        schedules_df = _ensure_columns(schedules_df, SCHEDULE_COLUMNS[:-1])
        team_stats_df = _ensure_columns(team_stats_df, TEAM_STATS_COLUMNS[:-1])

        pbp_df_selected = pbp_df[PBP_COLUMNS[:-1]].copy()
        pbp_df_selected["ingested_at"] = pbp_df["ingested_at"]
        pbp_df_selected["raw_record"] = _build_raw_records(pbp_df, PBP_COLUMNS[:-1])

        schedules_selected = schedules_df[SCHEDULE_COLUMNS[:-1]].copy()
        schedules_selected["ingested_at"] = schedules_df["ingested_at"]
        if "result" in schedules_selected.columns:
            schedules_selected["result"] = schedules_selected["result"].astype(str)
        schedules_selected["raw_record"] = _build_raw_records(schedules_df, SCHEDULE_COLUMNS[:-1])

        team_stats_selected = team_stats_df[TEAM_STATS_COLUMNS[:-1]].copy()
        team_stats_selected["ingested_at"] = team_stats_df["ingested_at"]
        team_stats_selected["raw_record"] = _build_raw_records(team_stats_df, TEAM_STATS_COLUMNS[:-1])

        if args.replace:
            _delete_season(client, f"{args.project}.sports_edge_raw.raw_pbp", season)
            _delete_season(client, f"{args.project}.sports_edge_raw.raw_schedules", season)
            _delete_season(client, f"{args.project}.sports_edge_raw.raw_team_stats", season)

        _load_dataframe(client, pbp_df_selected, f"{args.project}.sports_edge_raw.raw_pbp")
        _load_dataframe(client, schedules_selected, f"{args.project}.sports_edge_raw.raw_schedules")
        _load_dataframe(client, team_stats_selected, f"{args.project}.sports_edge_raw.raw_team_stats")

    print("Raw backfill complete.")


if __name__ == "__main__":
    main()
