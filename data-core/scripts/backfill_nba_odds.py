#!/usr/bin/env python3
"""
Backfill BigQuery raw_nba_odds table with historical odds from CSV.

Example:
    python scripts/backfill_nba_odds.py --project learned-pier-478122-p7 --csv-path nba_2008-2025.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, date
from typing import Any, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery


# Mapping from CSV lowercase abbreviations to standard uppercase NBA abbreviations
NBA_TEAM_MAP = {
    "atl": "ATL",
    "bos": "BOS",
    "bkn": "BKN",  # Brooklyn Nets (was NJN)
    "cha": "CHA",  # Charlotte Hornets (was CHA/CHO)
    "chi": "CHI",
    "cle": "CLE",
    "dal": "DAL",
    "den": "DEN",
    "det": "DET",
    "gs": "GSW",  # Golden State Warriors
    "hou": "HOU",
    "ind": "IND",
    "lac": "LAC",
    "lal": "LAL",
    "mem": "MEM",
    "mia": "MIA",
    "mil": "MIL",
    "min": "MIN",
    "no": "NOP",  # New Orleans Pelicans (was NOH)
    "ny": "NYK",
    "okc": "OKC",
    "orl": "ORL",
    "phi": "PHI",
    "phx": "PHX",  # Phoenix Suns
    "por": "POR",
    "sac": "SAC",
    "sa": "SAS",  # San Antonio Spurs
    "tor": "TOR",
    "utah": "UTA",
    "wsh": "WAS",  # Washington Wizards
}

# Map raw_schedules team codes (ESPN/NBA API) to our odds format for matching.
# raw_schedules can have GS, NO, NY, SA, UTAH, WSH; odds use GSW, NOP, NYK, SAS, UTA, WAS.
SCHEDULE_TO_ODDS_MAP = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
    "PHO": "PHX",
    "BRK": "BKN",
    "BRX": "BKN",
}


def normalize_team(team_code: str) -> str:
    """Convert CSV team code to standard uppercase abbreviation."""
    team_lower = team_code.lower().strip()
    return NBA_TEAM_MAP.get(team_lower, team_code.upper())


def schedule_team_to_odds_format(team: str) -> str:
    """Convert raw_schedules team code to odds format for matching."""
    t = str(team).upper().strip()
    return SCHEDULE_TO_ODDS_MAP.get(t, t)


def convert_moneyline_to_price(moneyline: float) -> Optional[float]:
    """
    Convert American moneyline to decimal price.
    
    Note: For historical data, we store the moneyline value directly as 'price'
    since the CSV format uses American odds. This can be converted later if needed.
    """
    if pd.isna(moneyline) or moneyline == 0:
        return None
    # Store as-is (American format: +150, -150)
    return float(moneyline)


def transform_odds_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform CSV odds data into normalized format for BigQuery.
    
    Creates rows for each market (spread, total, moneyline) with proper line/price.
    """
    rows = []
    utc_now = datetime.now(tz=timezone.utc)
    
    for _, row in df.iterrows():
        game_date = pd.to_datetime(row["date"]).date()
        season = int(row["season"])
        home_team = normalize_team(row["home"])
        away_team = normalize_team(row["away"])
        
        # Generate game_id by matching with existing games in raw_schedules
        # Format: will be matched later, for now use placeholder
        game_id = f"NBA_{season}_{game_date}_{away_team}_{home_team}"
        
        # Store raw record (convert to JSON-serializable format)
        raw_record = {}
        for k, v in row.to_dict().items():
            if pd.isna(v):
                raw_record[k] = None
            elif isinstance(v, (pd.Timestamp, datetime)):
                raw_record[k] = v.isoformat() if hasattr(v, 'isoformat') else str(v)
            else:
                raw_record[k] = v
        
        # Spread market (convert to home team perspective)
        if pd.notna(row.get("spread")):
            spread = float(row["spread"])
            # CSV spread is from favored team's perspective
            # Convert to home team perspective for consistency
            favored = str(row.get("whos_favored", "")).lower()
            if favored == "away":
                # Away favored by X means home spread is -X
                home_spread = -spread
            elif favored == "home":
                # Home favored by X means home spread is +X
                home_spread = spread
            else:
                # Default to positive if unclear (shouldn't happen)
                home_spread = spread
            
            rows.append({
                "game_id": game_id,
                "league": "NBA",
                "season": season,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "book": "historical",
                "market": "spread",
                "line": home_spread,
                "price": None,  # CSV doesn't have spread prices
                "whos_favored": row.get("whos_favored"),
                "ingested_at": utc_now,
                "raw_record": json.dumps(raw_record),
            })
        
        # Total market
        if pd.notna(row.get("total")):
            rows.append({
                "game_id": game_id,
                "league": "NBA",
                "season": season,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "book": "historical",
                "market": "total",
                "line": float(row["total"]),
                "price": None,
                "whos_favored": None,
                "ingested_at": utc_now,
                "raw_record": json.dumps(raw_record),
            })
        
        # Moneyline - away team
        if pd.notna(row.get("moneyline_away")):
            ml_away = float(row["moneyline_away"])
            rows.append({
                "game_id": game_id,
                "league": "NBA",
                "season": season,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "book": "historical",
                "market": "moneyline",
                "line": None,
                "price": convert_moneyline_to_price(ml_away),
                "whos_favored": "away" if ml_away < 0 else None,
                "ingested_at": utc_now,
                "raw_record": json.dumps(raw_record),
            })
        
        # Moneyline - home team
        if pd.notna(row.get("moneyline_home")):
            ml_home = float(row["moneyline_home"])
            rows.append({
                "game_id": game_id,
                "league": "NBA",
                "season": season,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "book": "historical",
                "market": "moneyline",
                "line": None,
                "price": convert_moneyline_to_price(ml_home),
                "whos_favored": "home" if ml_home < 0 else None,
                "ingested_at": utc_now,
                "raw_record": json.dumps(raw_record),
            })
    
    return pd.DataFrame(rows)


def match_game_ids(
    client: bigquery.Client,
    project: str,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match odds rows to actual game_ids from raw_schedules table.
    
    Returns DataFrame with matched game_ids.
    """
    print("Matching odds to games in raw_schedules...")
    
    # Get unique game identifiers from odds
    game_keys = odds_df[["season", "game_date", "home_team", "away_team"]].drop_duplicates()
    
    if game_keys.empty:
        print("No games to match.")
        return odds_df
    
    # Get date range for efficient querying
    date_min = game_keys["game_date"].min()
    date_max = game_keys["game_date"].max()
    seasons = sorted(game_keys["season"].unique().tolist())
    
    # Query all NBA games in the date/season range
    query = f"""
        SELECT game_id, season, game_date, home_team, away_team
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league = 'NBA'
          AND game_date BETWEEN @date_min AND @date_max
          AND season IN UNNEST(@seasons)
    """
    
    try:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date_min", "DATE", date_min),
                bigquery.ScalarQueryParameter("date_max", "DATE", date_max),
                bigquery.ArrayQueryParameter("seasons", "INT64", seasons),
            ]
        )
        schedule_df = client.query(query, job_config=job_config).to_dataframe()
        print(f"Found {len(schedule_df)} matching games in raw_schedules.")
        
        if schedule_df.empty:
            print("Warning: No games found in raw_schedules. Using placeholder game_ids.")
            return odds_df
        
        # Normalize dates in schedule_df
        schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"]).dt.date

        # Build lookup with schedule teams normalized to odds format (GS->GSW, NO->NOP, etc.)
        # and both season encodings. raw_schedules uses start year (2024), CSV uses end year (2025).
        game_lookup = {}
        for _, sched_row in schedule_df.iterrows():
            season = int(sched_row["season"])
            home_norm = schedule_team_to_odds_format(sched_row["home_team"])
            away_norm = schedule_team_to_odds_format(sched_row["away_team"])
            sched_date = sched_row["game_date"]
            game_id = sched_row["game_id"]

            # Key with schedule's season (start year)
            key = (season, sched_date, home_norm, away_norm)
            game_lookup[key] = game_id
            # Key with CSV season (end year = start year + 1)
            key_csv = (season + 1, sched_date, home_norm, away_norm)
            game_lookup[key_csv] = game_id

        print(f"Created lookup with {len(game_lookup)} game keys.")

        # Match game_ids - odds use our format, lookup has both normalized and raw keys
        matched_ids = []
        for _, row in odds_df.iterrows():
            row_date = (
                pd.to_datetime(row["game_date"]).date()
                if not isinstance(row["game_date"], date)
                else row["game_date"]
            )
            key = (
                int(row["season"]),
                row_date,
                str(row["home_team"]).upper().strip(),
                str(row["away_team"]).upper().strip(),
            )
            matched_id = game_lookup.get(key, row["game_id"])
            matched_ids.append(matched_id)

        odds_df["game_id"] = matched_ids

        matched_count = (~odds_df["game_id"].str.startswith("NBA_")).sum()
        matched_games = odds_df.loc[~odds_df["game_id"].str.startswith("NBA_")]["game_id"].nunique()
        print(f"Matched {matched_count} odds rows ({matched_games} unique games) to game_ids.")
        
    except Exception as e:
        print(f"Warning: Could not match game_ids: {e}")
        print("Using placeholder game_ids.")
        import traceback
        traceback.print_exc()
    
    return odds_df


def _load_dataframe(
    client: bigquery.Client,
    df: pd.DataFrame,
    table_id: str,
    write_disposition: str = "WRITE_APPEND",
) -> None:
    """Load DataFrame into BigQuery table."""
    if df.empty:
        print(f"No data to load into {table_id}")
        return
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        schema=[
            bigquery.SchemaField("game_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("league", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("season", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("game_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("home_team", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("away_team", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("book", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("market", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("line", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("price", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("whos_favored", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("raw_record", "STRING", mode="NULLABLE"),
        ],
    )
    
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Loaded {len(df):,} rows into {table_id}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load historical NBA odds from CSV into BigQuery.")
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project ID (e.g., learned-pier-478122-p7).",
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to CSV file with historical NBA odds.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows before inserting (by date range).",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Specific seasons to process (default: all seasons in CSV).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)
    
    print(f"Loading CSV from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df):,} rows from CSV.")
    
    # Filter by seasons if specified
    if args.seasons:
        df = df[df["season"].isin(args.seasons)]
        print(f"Filtered to {len(df):,} rows for seasons {args.seasons}.")
    
    # Transform to normalized format
    print("Transforming odds data...")
    odds_df = transform_odds_dataframe(df)
    print(f"Created {len(odds_df):,} odds rows.")
    
    # Match game_ids
    odds_df = match_game_ids(client, args.project, odds_df)
    
    # Delete existing data if replace flag is set
    if args.replace:
        date_min = odds_df["game_date"].min()
        date_max = odds_df["game_date"].max()
        print(f"Deleting existing odds for date range {date_min} to {date_max}...")
        query = f"""
            DELETE FROM `{args.project}.sports_edge_raw.raw_nba_odds`
            WHERE league = 'NBA' AND game_date BETWEEN @date_min AND @date_max
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date_min", "DATE", date_min),
                bigquery.ScalarQueryParameter("date_max", "DATE", date_max),
            ]
        )
        client.query(query, job_config=job_config).result()
        print("Deleted existing rows.")
    
    # Load into BigQuery
    table_id = f"{args.project}.sports_edge_raw.raw_nba_odds"
    _load_dataframe(client, odds_df, table_id)
    
    print("NBA odds backfill complete.")


if __name__ == "__main__":
    main()
