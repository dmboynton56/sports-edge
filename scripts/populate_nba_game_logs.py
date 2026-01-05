#!/usr/bin/env python3
"""
Populate BigQuery raw_nba_game_logs table with 2025 season data.
Fetches data from nba_api and loads it into GCP BigQuery.

Required Environment Variables (in .env):
- GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON key
- GCP_PROJECT_ID: Your BigQuery project ID
"""

import os
import json
import argparse
from datetime import datetime, timezone, date
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.nba_game_logs_loader import load_nba_game_logs
from src.data.nba_fetcher import fetch_nba_schedule

def _to_jsonable(value):
    """Convert non-serializable objects to strings for JSON."""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value

def build_raw_record(row):
    """Convert a dataframe row to a JSON string."""
    # Convert row to dict, handling NaNs and timestamps
    d = row.to_dict()
    clean_dict = {k: _to_jsonable(v) for k, v in d.items() if pd.notna(v)}
    return json.dumps(clean_dict)

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Populate NBA game logs in BigQuery.")
    parser.add_argument("--season", type=int, default=2025, help="NBA season year (default 2025)")
    parser.add_argument("--project", type=str, default=os.getenv("GCP_PROJECT_ID"), help="GCP Project ID")
    parser.add_argument("--replace", action="store_true", help="Replace existing data for the season")
    args = parser.parse_args()

    if not args.project:
        print("ERROR: GCP_PROJECT_ID not set in environment or provided via --project.")
        sys.exit(1)

    client = bigquery.Client(project=args.project)
    table_id = f"{args.project}.sports_edge_raw.raw_nba_game_logs"
    utc_now = datetime.now(tz=timezone.utc)

    print(f"Starting ingestion for NBA {args.season} season...")

    # 1. Fetch Schedule (needed to compute points_allowed in logs)
    print("Fetching schedule to resolve opponent scores...")
    schedule_df = fetch_nba_schedule(args.season)
    
    # 2. Fetch Game Logs
    print(f"Fetching team game logs for {args.season}...")
    logs_df = load_nba_game_logs([args.season], schedule_df=schedule_df)

    if logs_df is None or logs_df.empty:
        print("No logs found. Exiting.")
        return

    # 3. Prepare Data for BigQuery
    print("Preparing data for BigQuery...")
    
    # Standardize columns to match SQL schema
    bq_df = pd.DataFrame()
    bq_df['game_id'] = logs_df['game_id'].astype(str)
    bq_df['game_date'] = pd.to_datetime(logs_df['game_date']).dt.date
    bq_df['team'] = logs_df['team']
    bq_df['team_id'] = logs_df['team_id'].astype(int)
    bq_df['season'] = logs_df['season'].astype(int)
    bq_df['points_scored'] = pd.to_numeric(logs_df['points_scored'], errors='coerce').fillna(0).astype(int)
    bq_df['points_allowed'] = pd.to_numeric(logs_df['points_allowed'], errors='coerce').fillna(0).astype(int)
    bq_df['net_rating'] = pd.to_numeric(logs_df['net_rating'], errors='coerce')
    bq_df['point_diff'] = pd.to_numeric(logs_df['point_diff'], errors='coerce')
    bq_df['ingested_at'] = utc_now
    
    # Generate the raw_record JSON string
    bq_df['raw_record'] = logs_df.apply(build_raw_record, axis=1)

    # 4. Handle Deletion or Incremental Filter
    if args.replace:
        print(f"Deleting existing records for season {args.season}...")
        delete_query = f"DELETE FROM `{table_id}` WHERE season = {args.season}"
        client.query(delete_query).result()
    else:
        # Incremental check: only upload games not already in BQ
        print("Checking for existing games in BigQuery to avoid duplicates...")
        try:
            query = f"SELECT DISTINCT game_id, team_id FROM `{table_id}` WHERE season = {args.season}"
            existing_df = client.query(query).to_dataframe()
            if not existing_df.empty:
                # Create a composite key for easy filtering
                existing_keys = set(zip(existing_df['game_id'].astype(str), existing_df['team_id'].astype(int)))
                
                # Filter bq_df
                initial_count = len(bq_df)
                bq_df['temp_key'] = list(zip(bq_df['game_id'].astype(str), bq_df['team_id'].astype(int)))
                bq_df = bq_df[~bq_df['temp_key'].isin(existing_keys)].drop(columns=['temp_key'])
                
                new_count = len(bq_df)
                print(f"  Found {len(existing_keys)} existing team-game records.")
                print(f"  Filtering out duplicates: {initial_count} -> {new_count} new records to load.")
            else:
                print("  No existing records found for this season.")
        except Exception as e:
            # Table might not exist yet or other error, proceed with full load
            print(f"  Note: Could not check for duplicates (table may be empty): {e}")

    if bq_df.empty:
        print("No new records to load. Done.")
        return

    # 5. Load to BigQuery
    print(f"Loading {len(bq_df)} rows to {table_id}...")
    
    # Use standard load job
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
    )
    
    try:
        job = client.load_table_from_dataframe(bq_df, table_id, job_config=job_config)
        job.result()  # Wait for completion
        print(f"SUCCESS: Loaded {len(bq_df)} rows.")
    except Exception as e:
        print(f"FAILED to load data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

