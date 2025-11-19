#!/usr/bin/env python3
"""
Fetch actual results from nfl-data-py and update BigQuery tables:
- raw_schedules table (home_score, away_score)
- game_results table (actual_home_win, actual_home_points, actual_away_points, actual_home_margin)

Example:
    python update_week_results.py --project learned-pier-478122-p7 --season 2025 --weeks 11 12
"""

import argparse
from datetime import datetime, timezone
from typing import List

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.data import nfl_fetcher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NFL game results from nfl-data-py and update BigQuery tables (raw_schedules and game_results)."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project ID (e.g., learned-pier-478122-p7).",
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season year (e.g., 2025).",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        nargs="+",
        required=True,
        help="Week numbers to update (e.g., 11 12).",
    )
    return parser.parse_args()


def fetch_week_results(season: int, week: int) -> pd.DataFrame:
    """Fetch completed games for a specific week using nfl-data-py."""
    print(f"Fetching week {week} results for season {season}...")
    schedule = nfl_fetcher.fetch_nfl_schedule(season, week=week)
    
    if schedule.empty:
        print(f"  No games found for season {season}, week {week}")
        return pd.DataFrame()
    
    # Filter to completed games (where scores exist)
    completed = schedule[
        schedule['home_score'].notna() & schedule['away_score'].notna()
    ].copy()
    
    if completed.empty:
        print(f"  No completed games found for season {season}, week {week}")
        return pd.DataFrame()
    
    print(f"  Found {len(completed)} completed games")
    
    # Ensure we have the required columns
    required_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']
    missing_cols = [col for col in required_cols if col not in completed.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Include game_date if available (for game_results table)
    result = completed[required_cols].copy()
    if 'gameday' in completed.columns:
        result['game_date'] = pd.to_datetime(completed['gameday']).dt.date
    elif 'game_date' in completed.columns:
        result['game_date'] = pd.to_datetime(completed['game_date']).dt.date
    
    # Ensure game_id is string type to match BigQuery format
    result['game_id'] = result['game_id'].astype(str)
    result['season'] = result['season'].astype(int)
    result['week'] = result['week'].astype(int)
    result['home_score'] = result['home_score'].astype(int)
    result['away_score'] = result['away_score'].astype(int)
    
    return result


def update_bigquery_scores(
    client: bigquery.Client,
    project: str,
    results_df: pd.DataFrame,
) -> None:
    """Update BigQuery raw_schedules table with actual scores."""
    if results_df.empty:
        print("No results to update.")
        return
    
    table_id = f"{project}.sports_edge_raw.raw_schedules"
    
    # Build UPDATE statements for each game
    updated = 0
    for _, row in results_df.iterrows():
        update_query = f"""
            UPDATE `{table_id}`
            SET home_score = @home_score,
                away_score = @away_score
            WHERE game_id = @game_id
              AND season = @season
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("game_id", "STRING", str(row['game_id'])),
                bigquery.ScalarQueryParameter("season", "INT64", int(row['season'])),
                bigquery.ScalarQueryParameter("home_score", "INT64", int(row['home_score'])),
                bigquery.ScalarQueryParameter("away_score", "INT64", int(row['away_score'])),
            ]
        )
        try:
            client.query(update_query, job_config=job_config).result()
            updated += 1
            print(f"  Updated {row['away_team']} @ {row['home_team']}: {row['away_score']}-{row['home_score']}")
        except Exception as e:
            print(f"  ERROR updating {row['game_id']}: {e}")
    
    print(f"\nUpdated {updated} games in raw_schedules table.")


def update_game_results(
    client: bigquery.Client,
    project: str,
    results_df: pd.DataFrame,
) -> None:
    """Upsert actual outcomes into BigQuery game_results table (INSERT if not exists, UPDATE if exists)."""
    if results_df.empty:
        print("No results to update in game_results.")
        return
    
    table_id = f"{project}.sports_edge_results.game_results"
    
    # Calculate derived fields
    results_df = results_df.copy()
    results_df['actual_home_margin'] = results_df['home_score'] - results_df['away_score']
    results_df['actual_home_win'] = (results_df['actual_home_margin'] > 0).astype(bool)
    
    updated = 0
    inserted = 0
    for _, row in results_df.iterrows():
        # Use MERGE to upsert (UPDATE if exists, INSERT if not)
        # Match on game_id, league, season, week (unique combination)
        # BigQuery MERGE requires VALUES clause for single-row source
        merge_query = f"""
            MERGE `{table_id}` AS target
            USING (
                SELECT 
                    @game_id AS game_id,
                    @league AS league,
                    @season AS season,
                    @week AS week,
                    @game_date AS game_date,
                    @actual_home_win AS actual_home_win,
                    @actual_home_points AS actual_home_points,
                    @actual_away_points AS actual_away_points,
                    @actual_home_margin AS actual_home_margin
            ) AS source
            ON target.game_id = source.game_id
               AND target.league = source.league
               AND target.season = source.season
               AND COALESCE(target.week, -1) = COALESCE(source.week, -1)
            WHEN MATCHED THEN
                UPDATE SET
                    actual_home_win = source.actual_home_win,
                    actual_home_points = source.actual_home_points,
                    actual_away_points = source.actual_away_points,
                    actual_home_margin = source.actual_home_margin
            WHEN NOT MATCHED THEN
                INSERT (
                    game_id, league, season, week, game_date,
                    actual_home_win, actual_home_points, actual_away_points, actual_home_margin
                )
                VALUES (
                    source.game_id, source.league, source.season, source.week, source.game_date,
                    source.actual_home_win, source.actual_home_points, source.actual_away_points, source.actual_home_margin
                )
        """
        
        # Prepare parameters
        params = [
            bigquery.ScalarQueryParameter("game_id", "STRING", str(row['game_id'])),
            bigquery.ScalarQueryParameter("league", "STRING", "NFL"),
            bigquery.ScalarQueryParameter("season", "INT64", int(row['season'])),
            bigquery.ScalarQueryParameter("week", "INT64", int(row['week'])),
            bigquery.ScalarQueryParameter("actual_home_win", "BOOL", bool(row['actual_home_win'])),
            bigquery.ScalarQueryParameter("actual_home_points", "INT64", int(row['home_score'])),
            bigquery.ScalarQueryParameter("actual_away_points", "INT64", int(row['away_score'])),
            bigquery.ScalarQueryParameter("actual_home_margin", "INT64", int(row['actual_home_margin'])),
        ]
        
        # Add game_date if available
        if 'game_date' in row and pd.notna(row['game_date']):
            params.append(bigquery.ScalarQueryParameter("game_date", "DATE", row['game_date']))
        else:
            params.append(bigquery.ScalarQueryParameter("game_date", "DATE", None))
        
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        
        try:
            # Check if row exists before MERGE to distinguish INSERT vs UPDATE
            check_query = f"""
                SELECT COUNT(*) as cnt
                FROM `{table_id}`
                WHERE game_id = @game_id
                  AND league = @league
                  AND season = @season
                  AND week = @week
            """
            check_params = [
                bigquery.ScalarQueryParameter("game_id", "STRING", str(row['game_id'])),
                bigquery.ScalarQueryParameter("league", "STRING", "NFL"),
                bigquery.ScalarQueryParameter("season", "INT64", int(row['season'])),
                bigquery.ScalarQueryParameter("week", "INT64", int(row['week'])),
            ]
            check_job = client.query(check_query, job_config=bigquery.QueryJobConfig(query_parameters=check_params))
            existing_count = list(check_job.result())[0].cnt
            was_insert = existing_count == 0
            
            # Execute MERGE
            query_job = client.query(merge_query, job_config=job_config)
            try:
                query_job.result()  # Wait for completion
            except Exception as merge_error:
                print(f"  MERGE failed for {row['away_team']} @ {row['home_team']}: {merge_error}")
                # Print debug info
                print(f"    game_id: {row['game_id']}, season: {row['season']}, week: {row['week']}")
                print(f"    actual_home_win: {row['actual_home_win']}, margin: {row['actual_home_margin']}")
                raise
            
            # Verify MERGE worked by checking if row exists with actual results
            verify_query = f"""
                SELECT COUNT(*) as cnt
                FROM `{table_id}`
                WHERE game_id = @game_id
                  AND league = @league
                  AND season = @season
                  AND week = @week
                  AND actual_home_win IS NOT NULL
            """
            verify_job = client.query(verify_query, job_config=bigquery.QueryJobConfig(query_parameters=check_params))
            verify_count = list(verify_job.result())[0].cnt
            
            if verify_count > existing_count:
                # Row was inserted
                rows_affected = verify_count - existing_count
                inserted += rows_affected
                print(f"  Inserted game_results for {row['away_team']} @ {row['home_team']}: {rows_affected} row(s)")
            elif verify_count == existing_count and existing_count > 0:
                # Row was updated (already existed)
                rows_affected = existing_count
                updated += rows_affected
                print(f"  Updated game_results for {row['away_team']} @ {row['home_team']}: {rows_affected} row(s)")
            elif verify_count > 0:
                # Row exists but wasn't there before check (race condition or timing issue)
                rows_affected = verify_count
                inserted += rows_affected
                print(f"  Inserted game_results for {row['away_team']} @ {row['home_team']}: {rows_affected} row(s)")
            else:
                print(f"  WARNING: MERGE completed but no row found with actual results for {row['away_team']} @ {row['home_team']} (game_id: {row['game_id']})")
        except Exception as e:
            import traceback
            print(f"  ERROR upserting game_results for {row['game_id']}: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
    
    print(f"\nUpserted {inserted} new rows, updated {updated} existing rows in game_results table.")


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)
    
    print(f"Updating results for season {args.season}, weeks {args.weeks}")
    
    all_results = []
    for week in args.weeks:
        results = fetch_week_results(args.season, week)
        if not results.empty:
            all_results.append(results)
    
    if not all_results:
        print("No completed games found to update.")
        return
    
    combined_results = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal completed games to update: {len(combined_results)}")
    
    # Update raw_schedules table
    print("\n=== Updating raw_schedules table ===")
    update_bigquery_scores(client, args.project, combined_results)
    
    # Update game_results table
    print("\n=== Updating game_results table ===")
    update_game_results(client, args.project, combined_results)
    
    print("\nUpdate complete.")


if __name__ == "__main__":
    main()

