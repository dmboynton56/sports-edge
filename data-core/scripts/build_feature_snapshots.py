#!/usr/bin/env python3
"""
Build feature snapshots from BigQuery raw tables and store them in sports_edge_curated.

Example:
    python scripts/build_feature_snapshots.py --project learned-pier-478122-p7 --league NBA --seasons 2025
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from google.api_core import exceptions

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.refresh import build_features
from src.data.nba_fetcher import fetch_nba_games_for_date


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
    "is_3in4_home",
    "is_3in4_away",
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
    "is_3in4_differential",
    "week_number",
    "month",
    "is_playoff",
    # NFL Form Metrics
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
    # NBA Form Metrics
    "form_home_net_rating_3",
    "form_home_net_rating_5",
    "form_home_net_rating_10",
    "form_away_net_rating_3",
    "form_away_net_rating_5",
    "form_away_net_rating_10",
    "form_net_rating_diff_3",
    "form_net_rating_diff_5",
    "form_net_rating_diff_10",
    # Injury-aware adjustments
    "home_injury_epa_delta",
    "away_injury_epa_delta",
    "home_injury_net_rating_delta",
    "away_injury_net_rating_delta",
    "home_injured_players",
    "away_injured_players",
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
        "--league",
        choices=["NFL", "NBA"],
        default="NFL",
        help="League to process (default: NFL).",
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
    parser.add_argument(
        "--destination-table",
        default=None,
        help="Fully qualified output table. Defaults to sports_edge_curated.feature_snapshots.",
    )
    parser.add_argument(
        "--date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=None,
        help="Anchor date for incremental builds. Default: today UTC when lookback/lookahead is provided.",
    )
    parser.add_argument(
        "--start-date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=None,
        help="First game_date to build, inclusive.",
    )
    parser.add_argument(
        "--end-date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=None,
        help="Last game_date to build, inclusive.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Days before --date to include in an incremental build.",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=None,
        help="Days after --date to include in an incremental build.",
    )
    return parser.parse_args()


def _resolve_date_window(args: argparse.Namespace) -> Optional[tuple[date, date]]:
    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise ValueError("--start-date and --end-date must be provided together.")
        if args.start_date > args.end_date:
            raise ValueError("--start-date must be on or before --end-date.")
        return args.start_date, args.end_date

    if args.lookback_days is None and args.lookahead_days is None:
        return None

    lookback_days = args.lookback_days or 0
    lookahead_days = args.lookahead_days or 0
    if lookback_days < 0 or lookahead_days < 0:
        raise ValueError("--lookback-days and --lookahead-days must be non-negative.")

    anchor = args.date or datetime.now(tz=timezone.utc).date()
    return anchor - timedelta(days=lookback_days), anchor + timedelta(days=lookahead_days)


def _fetch_table(
    client: bigquery.Client,
    project: str,
    dataset: str,
    table: str,
    seasons: List[int],
    *,
    league: Optional[str] = None,
) -> pd.DataFrame:
    league_clause = ""
    query_parameters: list[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter] = [
        bigquery.ArrayQueryParameter("seasons", "INT64", seasons)
    ]
    if league is not None:
        league_clause = "AND league = @league"
        query_parameters.append(bigquery.ScalarQueryParameter("league", "STRING", league))

    query = f"""
        SELECT *
        FROM `{project}.{dataset}.{table}`
        WHERE season IN UNNEST(@seasons)
          {league_clause}
    """
    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return client.query(query, job_config=job_config).to_dataframe()
        except exceptions.ServiceUnavailable as e:
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                print(f"ServiceUnavailable error fetching {table}, retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise


def _assert_schedule_league(schedules: pd.DataFrame, expected_league: str) -> None:
    """Fail closed when a feature build contains another league's games."""
    if schedules.empty:
        return
    if "league" not in schedules.columns:
        raise ValueError("Schedule rows are missing the required league column.")

    normalized = schedules["league"].astype("string").str.upper()
    unexpected = sorted(normalized.dropna().loc[normalized != expected_league.upper()].unique().tolist())
    missing_count = int(normalized.isna().sum())
    if unexpected or missing_count:
        raise ValueError(
            "Schedule league isolation failed: "
            f"expected={expected_league.upper()}, unexpected={unexpected}, missing={missing_count}."
        )


def _assert_feature_grain(feature_rows: pd.DataFrame, target_schedules: pd.DataFrame) -> None:
    """Require exactly one feature row for every requested game."""
    if "game_id" not in feature_rows.columns or "game_id" not in target_schedules.columns:
        raise ValueError("Feature grain validation requires game_id on schedules and features.")

    expected_ids = set(target_schedules["game_id"].dropna().astype(str))
    actual_ids = feature_rows["game_id"].dropna().astype(str)
    duplicate_ids = sorted(actual_ids.loc[actual_ids.duplicated(keep=False)].unique().tolist())
    actual_id_set = set(actual_ids)
    missing_ids = sorted(expected_ids - actual_id_set)
    unexpected_ids = sorted(actual_id_set - expected_ids)
    null_ids = int(feature_rows["game_id"].isna().sum())

    if duplicate_ids or missing_ids or unexpected_ids or null_ids:
        raise ValueError(
            "Feature grain validation failed: expected one row per requested game; "
            f"duplicates={duplicate_ids[:10]}, missing={missing_ids[:10]}, "
            f"unexpected={unexpected_ids[:10]}, null_ids={null_ids}."
        )


def _delete_existing_features(client: bigquery.Client, table_id: str, seasons: List[int], league: str) -> None:
    query = f"DELETE FROM `{table_id}` WHERE season IN UNNEST(@seasons) AND league = @league"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("seasons", "INT64", seasons),
            bigquery.ScalarQueryParameter("league", "STRING", league)
        ]
    )
    client.query(query, job_config=job_config).result()
    print(f"Cleared {table_id} for {league} seasons: {', '.join(map(str, seasons))}")


def _delete_existing_feature_window(
    client: bigquery.Client,
    table_id: str,
    *,
    league: str,
    start_date: date,
    end_date: date,
) -> None:
    query = f"""
        DELETE FROM `{table_id}`
        WHERE league = @league
          AND game_date BETWEEN @start_date AND @end_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("league", "STRING", league),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    client.query(query, job_config=job_config).result()
    print(f"Cleared {table_id} for {league} game_date {start_date} to {end_date}")


def _load_features(client: bigquery.Client, df: pd.DataFrame, table_id: str) -> None:
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Wrote {len(df):,} feature rows to {table_id}")


def _ensure_staging_schema(
    client: bigquery.Client,
    *,
    destination_table: Optional[str],
    canonical_table: str,
) -> None:
    """Create custom destinations with the canonical schema, never inferred types."""
    if not destination_table or destination_table == canonical_table:
        return
    try:
        client.get_table(destination_table)
    except exceptions.NotFound:
        client.query(f"CREATE TABLE `{destination_table}` LIKE `{canonical_table}`").result()
        print(f"Created staging table {destination_table} with canonical schema.")


def _filter_schedules_for_window(
    schedules: pd.DataFrame,
    date_window: Optional[tuple[date, date]],
) -> pd.DataFrame:
    if date_window is None or schedules.empty:
        return schedules.copy()

    start_date, end_date = date_window
    schedule_dates = pd.to_datetime(schedules["game_date"], errors="coerce").dt.date
    return schedules[
        (schedule_dates >= start_date)
        & (schedule_dates <= end_date)
    ].copy()


def _ensure_nba_window_games(
    schedules: pd.DataFrame,
    date_window: Optional[tuple[date, date]],
) -> pd.DataFrame:
    if date_window is None:
        target_days = [datetime.now(tz=timezone.utc).date()]
    else:
        start_date, end_date = date_window
        target_days = [day.date() for day in pd.date_range(start=start_date, end=end_date, freq="D")]

    if schedules.empty:
        existing_dates = set()
    else:
        existing_dates = set(pd.to_datetime(schedules["game_date"], errors="coerce").dt.date.dropna())

    fetched_frames = []
    for target_day in target_days:
        if target_day in existing_dates:
            continue

        target_str = target_day.strftime("%Y-%m-%d")
        print(f"No NBA games for {target_str} found in BQ. Fetching from API...")
        api_games = fetch_nba_games_for_date(target_str, raise_on_error=True)
        if api_games.empty:
            continue

        print(f"Found {len(api_games)} NBA games on API for {target_str}. Adding to processing queue.")
        api_games["game_date"] = pd.to_datetime(api_games["game_date"], utc=True).dt.tz_localize(None)
        api_games["league"] = "NBA"
        fetched_frames.append(api_games)

    if fetched_frames:
        schedules = pd.concat([schedules, *fetched_frames], ignore_index=True)
        schedules = schedules.drop_duplicates(subset=["game_id"])
        schedules = schedules.drop_duplicates(subset=["home_team", "away_team", "game_date"])
        schedules = schedules.reset_index(drop=True)

    return schedules


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)
    date_window = _resolve_date_window(args)

    print(f"Processing {args.league} for seasons: {args.seasons}")
    if date_window is not None:
        print(f"Incremental feature window: {date_window[0]} to {date_window[1]}")
    schedules = _fetch_table(
        client,
        args.project,
        "sports_edge_raw",
        "raw_schedules",
        args.seasons,
        league=args.league,
    )
    _assert_schedule_league(schedules, args.league)
    
    # 1. Aggressive deduplication of schedules to avoid row explosion
    if not schedules.empty:
        # Standardize dates for comparison - using utc=True to avoid mixed timezone errors
        schedules["game_date"] = pd.to_datetime(schedules["game_date"], errors="coerce", utc=True).dt.tz_localize(None)
        schedules = schedules.sort_values("ingested_at", ascending=False) if "ingested_at" in schedules.columns else schedules
        
        # Deduplicate by game_id first, then by the core game identifiers
        schedules = schedules.drop_duplicates(subset=["game_id"])
        schedules = schedules.drop_duplicates(subset=["home_team", "away_team", "game_date"])
        schedules = schedules.reset_index(drop=True)
    
    if args.league == "NBA":
        schedules = _ensure_nba_window_games(schedules, date_window)
        _assert_schedule_league(schedules, args.league)
    
    historical_data: Dict[str, pd.DataFrame] = {
        "historical_games": schedules,
    }

    if args.league == "NFL":
        print(f"Loading raw play-by-play for NFL...")
        pbp = _fetch_table(
            client,
            args.project,
            "sports_edge_raw",
            "raw_pbp",
            args.seasons,
            league="NFL",
        )
        if not pbp.empty:
            pbp["game_date"] = pd.to_datetime(pbp["game_date"], errors="coerce", utc=True).dt.tz_localize(None)
        historical_data["play_by_play"] = pbp
    else:
        print(f"Loading raw game logs for NBA...")
        logs = _fetch_table(client, args.project, "sports_edge_raw", "raw_nba_game_logs", args.seasons)
        if not logs.empty:
            logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce", utc=True).dt.tz_localize(None)
            # Deduplicate logs to avoid cartesian explosion in feature building
            logs = logs.sort_values("ingested_at", ascending=False) if "ingested_at" in logs.columns else logs
            logs = logs.drop_duplicates(subset=["team", "game_date"])
            logs = logs.reset_index(drop=True)
        historical_data["game_logs"] = logs

    schedules = schedules.drop(columns=["raw_record"], errors="ignore")

    # Final check: schedules must have unique game_id and unique (home, away, date)
    schedules = schedules.drop_duplicates(subset=["game_id"])
    schedules = schedules.drop_duplicates(subset=["home_team", "away_team", "game_date"])
    schedules = schedules.reset_index(drop=True)

    target_schedules = _filter_schedules_for_window(schedules, date_window)
    if target_schedules.empty:
        window_text = "requested window" if date_window is not None else "requested seasons"
        print(f"No {args.league} games found for {window_text}. Exiting.")
        return

    print(f"Building {args.league} features for {len(target_schedules):,} target games.")
    feature_rows = build_features(target_schedules, args.league, historical_data)
    _assert_feature_grain(feature_rows, target_schedules)
    
    # Ensure no duplicates in feature_rows index labels
    feature_rows = feature_rows.reset_index(drop=True)
    
    # Convert scores to numeric directly in feature_rows
    feature_rows["home_score"] = pd.to_numeric(feature_rows["home_score"], errors="coerce")
    feature_rows["away_score"] = pd.to_numeric(feature_rows["away_score"], errors="coerce")
    
    feature_rows["league"] = args.league
    feature_rows["home_win"] = (feature_rows["home_score"] > feature_rows["away_score"]).where(
        ~(feature_rows["home_score"].isna() | feature_rows["away_score"].isna()), None
    )
    feature_rows["home_margin"] = feature_rows["home_score"] - feature_rows["away_score"]
    feature_rows["as_of_ts"] = datetime.now(tz=timezone.utc)
    feature_rows["feature_version"] = args.feature_version

    # Ensure deterministic ordering. Grain validation above rejects duplicates.
    feature_rows = feature_rows.sort_values(["season", "game_date", "game_id"])

    for column in FEATURE_COLUMNS:
        if column not in feature_rows.columns:
            feature_rows[column] = None

    injury_delta_columns = [
        "home_injury_epa_delta",
        "away_injury_epa_delta",
        "home_injury_net_rating_delta",
        "away_injury_net_rating_delta",
    ]
    for col in injury_delta_columns:
        feature_rows[col] = pd.to_numeric(feature_rows[col], errors="coerce").fillna(0.0)

    injury_count_columns = ["home_injured_players", "away_injured_players"]
    for col in injury_count_columns:
        feature_rows[col] = pd.to_numeric(feature_rows[col], errors="coerce").fillna(0).astype("Int64")
            
    feature_rows["game_date"] = pd.to_datetime(feature_rows["game_date"], utc=True).dt.date
    
    int_columns = ["season", "week_number", "month"]
    for col in int_columns:
        if col in feature_rows.columns:
            feature_rows[col] = pd.to_numeric(feature_rows[col], errors="coerce").astype("Int64")
            
    bool_columns = ["home_win", "b2b_home", "b2b_away", "is_playoff", "is_3in4_home", "is_3in4_away"]
    for col in bool_columns:
        if col in feature_rows.columns:
            feature_rows[col] = feature_rows[col].astype("boolean")
            
    # Select and order columns
    feature_rows = feature_rows[FEATURE_COLUMNS]

    canonical_table = f"{args.project}.sports_edge_curated.feature_snapshots"
    table_id = args.destination_table or canonical_table
    _ensure_staging_schema(
        client,
        destination_table=args.destination_table,
        canonical_table=canonical_table,
    )
    if date_window is not None:
        _delete_existing_feature_window(
            client,
            table_id,
            league=args.league,
            start_date=date_window[0],
            end_date=date_window[1],
        )
    elif args.replace:
        _delete_existing_features(client, table_id, args.seasons, args.league)

    _load_features(client, feature_rows, table_id)
    print(f"{args.league} feature snapshot build complete.")


if __name__ == "__main__":
    main()
