#!/usr/bin/env python3
"""
Build feature snapshots from BigQuery raw tables and store them in sports_edge_curated.

Example:
    python scripts/build_feature_snapshots.py --project learned-pier-478122-p7 --seasons 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.refresh import build_features


FEATURE_COLUMNS = [
    "game_id",
    "league",
    "season",
    "game_date",
    "as_of_ts",
    "home_team",
    "away_team",
    "home_win",
    "home_margin",
    "rest_home",
    "rest_away",
    "b2b_home",
    "b2b_away",
    "opp_strength_home_season",
    "opp_strength_away_season",
    "home_team_win_pct",
    "away_team_win_pct",
    "home_team_point_diff",
    "away_team_point_diff",
    "rest_differential",
    "win_pct_differential",
    "point_diff_differential",
    "opp_strength_differential",
    "week_number",
    "month",
    "is_playoff",
    "form_home_epa_off_3",
    "form_home_epa_off_5",
    "form_home_epa_off_10",
    "form_home_epa_def_3",
    "form_home_epa_def_5",
    "form_home_epa_def_10",
    "form_away_epa_off_3",
    "form_away_epa_off_5",
    "form_away_epa_off_10",
    "form_away_epa_def_3",
    "form_away_epa_def_5",
    "form_away_epa_def_10",
    "form_epa_off_diff_3",
    "form_epa_off_diff_5",
    "form_epa_off_diff_10",
    "form_epa_def_diff_3",
    "form_epa_def_diff_5",
    "form_epa_def_diff_10",
    "feature_version",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature snapshots for specified seasons.")
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
        help="Seasons to process (default: 2020-2025).",
    )
    parser.add_argument(
        "--feature-version",
        default="v1",
        help="Feature version tag stored in feature_snapshots.feature_version.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows for the provided seasons before inserting.",
    )
    return parser.parse_args()


def _fetch_table(client: bigquery.Client, project: str, table: str, seasons: List[int]) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM `{project}.sports_edge_raw.{table}`
        WHERE season IN UNNEST(@seasons)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("seasons", "INT64", seasons)]
    )
    return client.query(query, job_config=job_config).to_dataframe()


def _delete_existing_features(client: bigquery.Client, table_id: str, seasons: List[int]) -> None:
    query = f"DELETE FROM `{table_id}` WHERE season IN UNNEST(@seasons)"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("seasons", "INT64", seasons)]
    )
    client.query(query, job_config=job_config).result()
    print(f"Cleared {table_id} for seasons: {', '.join(map(str, seasons))}")


def _load_features(client: bigquery.Client, df: pd.DataFrame, table_id: str) -> None:
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Wrote {len(df):,} feature rows to {table_id}")


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)

    print(f"Loading raw schedules for seasons: {args.seasons}")
    schedules = _fetch_table(client, args.project, "raw_schedules", args.seasons)
    print(f"Loading raw play-by-play for seasons: {args.seasons}")
    pbp = _fetch_table(client, args.project, "raw_pbp", args.seasons)

    schedules["game_date"] = pd.to_datetime(schedules["game_date"], errors="coerce")
    pbp["game_date"] = pd.to_datetime(pbp["game_date"], errors="coerce")
    schedules = schedules.drop(columns=["raw_record"], errors="ignore")
    pbp = pbp.drop(columns=["raw_record"], errors="ignore")

    historical_data: Dict[str, pd.DataFrame] = {
        "historical_games": schedules,
        "play_by_play": pbp,
    }

    feature_rows = build_features(schedules, "NFL", historical_data)
    home_scores = schedules["home_score"]
    away_scores = schedules["away_score"]
    feature_rows["league"] = "NFL"
    feature_rows["home_win"] = (home_scores > away_scores).where(
        ~(home_scores.isna() | away_scores.isna()), None
    )
    feature_rows["home_margin"] = schedules["home_score"] - schedules["away_score"]
    feature_rows["as_of_ts"] = datetime.now(tz=timezone.utc)
    feature_rows["feature_version"] = args.feature_version

    # Ensure deterministic ordering and remove duplicates.
    feature_rows = feature_rows.sort_values(["season", "game_date", "game_id"]).drop_duplicates("game_id", keep="last")

    for column in FEATURE_COLUMNS:
        if column not in feature_rows.columns:
            feature_rows[column] = None
    feature_rows["game_date"] = pd.to_datetime(feature_rows["game_date"]).dt.date
    int_columns = ["season", "week_number", "month"]
    for col in int_columns:
        if col in feature_rows.columns:
            feature_rows[col] = pd.to_numeric(feature_rows[col], errors="coerce").astype("Int64")
    bool_columns = ["home_win", "b2b_home", "b2b_away", "is_playoff"]
    for col in bool_columns:
        if col in feature_rows.columns:
            feature_rows[col] = feature_rows[col].astype("boolean")
    feature_rows = feature_rows[FEATURE_COLUMNS]

    table_id = f"{args.project}.sports_edge_curated.feature_snapshots"
    if args.replace:
        _delete_existing_features(client, table_id, args.seasons)

    _load_features(client, feature_rows, table_id)
    print("Feature snapshot build complete.")


if __name__ == "__main__":
    main()
