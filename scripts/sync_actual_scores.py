#!/usr/bin/env python3
"""
Sync actual scores from BigQuery raw_schedules to Supabase games table.

Example:
    python sync_actual_scores.py --project learned-pier-478122-p7 --season 2025 --weeks 11 12
"""

import argparse
from typing import List

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from predict_WEEK_11 import (
    create_pg_connection,
    load_supabase_credentials,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync actual scores from BigQuery to Supabase games table."
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
        help="Week numbers to sync (optional; if not provided, syncs all completed games for season).",
    )
    return parser.parse_args()


def fetch_completed_games(
    client: bigquery.Client,
    project: str,
    season: int,
    weeks: List[int] = None,
) -> pd.DataFrame:
    """Fetch completed games from BigQuery raw_schedules."""
    query = f"""
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            home_score,
            away_score
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE season = @season
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """
    
    params = [bigquery.ScalarQueryParameter("season", "INT64", season)]
    
    if weeks:
        query += " AND week IN UNNEST(@weeks)"
        params.append(bigquery.ArrayQueryParameter("weeks", "INT64", weeks))
    
    query += " ORDER BY week, home_team, away_team"
    
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    df = client.query(query, job_config=job_config).to_dataframe()
    
    print(f"Fetched {len(df)} completed games from BigQuery")
    if weeks:
        print(f"  Weeks: {weeks}")
    return df


def update_supabase_scores(
    conn,
    scores_df: pd.DataFrame,
) -> int:
    """Update Supabase games table with actual scores."""
    if scores_df.empty:
        print("No scores to update.")
        return 0
    
    update_sql = """
        UPDATE games
        SET home_score = %s,
            away_score = %s
        WHERE league = 'NFL'
          AND season = %s
          AND week = %s
          AND home_team = %s
          AND away_team = %s
    """
    
    updated = 0
    with conn.cursor() as cur:
        for _, row in scores_df.iterrows():
            cur.execute(
                update_sql,
                (
                    int(row['home_score']),
                    int(row['away_score']),
                    int(row['season']),
                    int(row['week']) if pd.notna(row['week']) else None,
                    row['home_team'],
                    row['away_team'],
                )
            )
            if cur.rowcount > 0:
                updated += 1
                print(f"  Updated {row['away_team']} @ {row['home_team']}: {row['away_score']}-{row['home_score']}")
            else:
                print(f"  WARNING: No matching game found for {row['away_team']} @ {row['home_team']} (week {row['week']})")
    
    conn.commit()
    print(f"\nUpdated {updated} games in Supabase.")
    return updated


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)
    
    print(f"Syncing scores for season {args.season}")
    if args.weeks:
        print(f"  Weeks: {args.weeks}")
    
    # Fetch completed games from BigQuery
    scores_df = fetch_completed_games(client, args.project, args.season, args.weeks)
    
    if scores_df.empty:
        print("No completed games found to sync.")
        return
    
    # Connect to Supabase and update scores
    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds['url'],
        password=creds['db_password'],
        host_override=creds.get('db_host'),
        port=creds['db_port'],
        database=creds['db_name'],
        user=creds['db_user']
    )
    
    try:
        updated = update_supabase_scores(conn, scores_df)
        print(f"\nSync complete. Updated {updated} games.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

