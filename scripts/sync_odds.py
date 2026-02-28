#!/usr/bin/env python3
"""
Sync market odds from The Odds API to Supabase (games table) and BigQuery (model_predictions).
Matches teams for NFL and NBA using standard abbreviations and aliases.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from google.cloud import bigquery
import psycopg

from pathlib import Path
import sys

# Ensure project root on sys.path for importing helper functions
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import (
    create_pg_connection,
    game_map_key,
    load_supabase_credentials,
)

LOGGER = logging.getLogger("sync_odds")

# Team Mappings
NFL_MAPPING = {
    'ARI': ['Arizona Cardinals', 'Cardinals', 'ARZ'],
    'ATL': ['Atlanta Falcons'],
    'BAL': ['Baltimore Ravens'],
    'BUF': ['Buffalo Bills'],
    'CAR': ['Carolina Panthers'],
    'CHI': ['Chicago Bears'],
    'CIN': ['Cincinnati Bengals'],
    'CLE': ['Cleveland Browns'],
    'DAL': ['Dallas Cowboys'],
    'DEN': ['Denver Broncos'],
    'DET': ['Detroit Lions'],
    'GB': ['Green Bay Packers'],
    'HOU': ['Houston Texans'],
    'IND': ['Indianapolis Colts'],
    'JAX': ['Jacksonville Jaguars', 'Jags'],
    'KC': ['Kansas City Chiefs', 'KC Chiefs'],
    'LAC': ['Los Angeles Chargers', 'LA Chargers'],
    'LAR': ['Los Angeles Rams', 'LA Rams'],
    'LV': ['Las Vegas Raiders', 'Oakland Raiders', 'LA Raiders'],
    'MIA': ['Miami Dolphins'],
    'MIN': ['Minnesota Vikings'],
    'NE': ['New England Patriots', 'NE Patriots'],
    'NO': ['New Orleans Saints'],
    'NYG': ['New York Giants', 'NY Giants'],
    'NYJ': ['New York Jets', 'NY Jets'],
    'PHI': ['Philadelphia Eagles'],
    'PIT': ['Pittsburgh Steelers'],
    'SEA': ['Seattle Seahawks'],
    'SF': ['San Francisco 49ers', 'SF 49ers', 'Niners'],
    'TB': ['Tampa Bay Buccaneers', 'Buccaneers', 'Bucs'],
    'TEN': ['Tennessee Titans'],
    'WAS': ['Washington Commanders', 'Washington Football Team', 'Redskins']
}

NBA_MAPPING = {
    'ATL': ['Atlanta Hawks'],
    'BOS': ['Boston Celtics'],
    'BKN': ['Brooklyn Nets'],
    'CHA': ['Charlotte Hornets'],
    'CHI': ['Chicago Bulls'],
    'CLE': ['Cleveland Cavaliers'],
    'DAL': ['Dallas Mavericks'],
    'DEN': ['Denver Nuggets'],
    'DET': ['Detroit Pistons'],
    'GSW': ['Golden State Warriors'],
    'HOU': ['Houston Rockets'],
    'IND': ['Indiana Pacers'],
    'LAC': ['LA Clippers', 'Los Angeles Clippers'],
    'LAL': ['Los Angeles Lakers', 'LA Lakers'],
    'MEM': ['Memphis Grizzlies'],
    'MIA': ['Miami Heat'],
    'MIL': ['Milwaukee Bucks'],
    'MIN': ['Minnesota Timberwolves'],
    'NOP': ['New Orleans Pelicans'],
    'NYK': ['New York Knicks'],
    'OKC': ['Oklahoma City Thunder'],
    'ORL': ['Orlando Magic'],
    'PHI': ['Philadelphia 76ers'],
    'PHX': ['Phoenix Suns'],
    'POR': ['Portland Trail Blazers'],
    'SAC': ['Sacramento Kings'],
    'SAS': ['San Antonio Spurs'],
    'TOR': ['Toronto Raptors'],
    'UTA': ['Utah Jazz'],
    'WAS': ['Washington Wizards']
}

def normalize_text(text: str) -> str:
    """Normalize text for matching (lowercase, alphanumeric only)."""
    import re
    if not text:
        return ""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def get_team_code(team_name: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    """Get team code from mapping based on name or aliases."""
    normalized_name = normalize_text(team_name)
    for code, aliases in mapping.items():
        if normalized_name == normalize_text(code):
            return code
        for alias in aliases:
            if normalized_name == normalize_text(alias):
                return code
    return None

def fetch_odds_data(api_key: str, league: str) -> List[Dict[str, Any]]:
    """Fetch odds from The Odds API for a given league."""
    sport_key = 'americanfootball_nfl' if league == 'NFL' else 'basketball_nba'
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads',
        'oddsFormat': 'american',
        'dateFormat': 'iso',
        'bookmakers': 'draftkings,betmgm,fanduel'
    }
    
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        LOGGER.error(f"Error fetching odds from API: {resp.text}")
        return []
    return resp.json()

def ensure_bq_column(client: bigquery.Client, project: str, table_id: str, column_name: str, data_type: str = "FLOAT64"):
    """Add a column to a BigQuery table if it doesn't already exist."""
    table = client.get_table(table_id)
    schema = table.schema
    
    # Check if column exists
    if any(field.name == column_name for field in schema):
        LOGGER.debug(f"Column {column_name} already exists in {table_id}")
        return
    
    LOGGER.info(f"Adding column {column_name} to {table_id}")
    new_schema = schema[:]
    new_schema.append(bigquery.SchemaField(column_name, data_type, mode="NULLABLE"))
    table.schema = new_schema
    client.update_table(table, ["schema"])

def sync_odds_to_supabase(conn, league: str, odds_data: List[Dict[str, Any]]):
    """Update Supabase games table with book spreads."""
    mapping = NFL_MAPPING if league == 'NFL' else NBA_MAPPING
    updates = []
    
    # Define date window: today +/- 1 day for matching
    now = datetime.now(timezone.utc)
    start_window = (now - timedelta(days=2)).isoformat()
    end_window = (now + timedelta(days=10)).isoformat()
    
    # Fetch recent games from Supabase to match
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, home_team, away_team, game_time_utc FROM games WHERE league = %s AND game_time_utc >= %s AND game_time_utc <= %s",
            (league, start_window, end_window)
        )
        games = cur.fetchall()
        
    game_map = {} # key -> supabase_id
    for gid, home, away, gtime in games:
        key = game_map_key(home, away, gtime)
        game_map[key] = gid
        
    matched_count = 0
    for event in odds_data:
        home_code = get_team_code(event['home_team'], mapping)
        away_code = get_team_code(event['away_team'], mapping)
        if not home_code or not away_code:
            continue
            
        # Match using the same logic as sync_bq_to_supabase
        game_time = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        key = game_map_key(home_code, away_code, game_time)
        
        gid = game_map.get(key)
        if not gid:
            # Try a fuzzy date match (same day)
            day_key_prefix = key.split('_')[0]
            fuzzy_key = next((k for k in game_map.keys() if k.startswith(day_key_prefix) and home_code in k and away_code in k), None)
            if fuzzy_key:
                gid = game_map[fuzzy_key]
        
        if gid:
            # Find the best bookmaker spread
            bookmaker = next((b for b in event.get('bookmakers', []) if b['key'] in ['draftkings', 'betmgm', 'fanduel']), None)
            if not bookmaker and event.get('bookmakers'):
                bookmaker = event['bookmakers'][0]
                
            if bookmaker:
                market = next((m for m in bookmaker.get('markets', []) if m['key'] == 'spreads'), None)
                if market:
                    # Spread is for the home team if 'name' matches home_team
                    home_outcome = next((o for o in market.get('outcomes', []) if get_team_code(o['name'], mapping) == home_code), None)
                    if home_outcome and 'point' in home_outcome:
                        spread = home_outcome['point']
                        updates.append((spread, gid))
                        matched_count += 1
                        
    if updates:
        with conn.cursor() as cur:
            for spread, gid in updates:
                cur.execute("UPDATE games SET book_spread = %s WHERE id = %s", (spread, gid))
        conn.commit()
        LOGGER.info(f"Updated {len(updates)} games in Supabase with book spreads for {league}")
    else:
        LOGGER.warning(f"No odds matches found for {league} in Supabase window")

def sync_odds_to_bq(client: bigquery.Client, project: str, league: str, odds_data: List[Dict[str, Any]]):
    """Update BigQuery model_predictions and feature_snapshots tables with book spreads."""
    mapping = NFL_MAPPING if league == 'NFL' else NBA_MAPPING
    
    # Update model_predictions
    pred_table_id = f"{project}.sports_edge_curated.model_predictions"
    ensure_bq_column(client, project, pred_table_id, "book_spread")
    
    # Update feature_snapshots
    feat_table_id = f"{project}.sports_edge_curated.feature_snapshots"
    ensure_bq_column(client, project, feat_table_id, "book_spread")
    
    # Fetch recent predictions/features from BigQuery to update
    now = datetime.now(timezone.utc)
    target_date = now.date().isoformat()
    
    query = f"""
        SELECT prediction_id, p.game_id, home_team, away_team, game_date
        FROM `{pred_table_id}` p
        JOIN `{feat_table_id}` f ON p.game_id = f.game_id
        WHERE p.league = @league AND f.game_date >= @target_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("league", "STRING", league),
            bigquery.ScalarQueryParameter("target_date", "DATE", target_date),
        ]
    )
    preds = client.query(query, job_config=job_config).to_dataframe()
    # Deduplicate in case the join produced multiple rows per prediction
    if not preds.empty:
        preds = preds.drop_duplicates('prediction_id')
    
    if preds.empty:
        LOGGER.warning(f"No BigQuery predictions found for {league} on/after {target_date}")
        return

    pred_updates = []
    feat_updates = []
    for event in odds_data:
        home_code = get_team_code(event['home_team'], mapping)
        away_code = get_team_code(event['away_team'], mapping)
        if not home_code or not away_code:
            continue
            
        event_date = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00')).date()
        matches = preds[
            (preds['home_team'] == home_code) & 
            (preds['away_team'] == away_code) & 
            (pd.to_datetime(preds['game_date']).dt.date == event_date)
        ]
        
        if not matches.empty:
            bookmaker = next((b for b in event.get('bookmakers', []) if b['key'] in ['draftkings', 'betmgm', 'fanduel']), None)
            if not bookmaker and event.get('bookmakers'):
                bookmaker = event['bookmakers'][0]
                
            if bookmaker:
                market = next((m for m in bookmaker.get('markets', []) if m['key'] == 'spreads'), None)
                if market:
                    home_outcome = next((o for o in market.get('outcomes', []) if get_team_code(o['name'], mapping) == home_code), None)
                    if home_outcome and 'point' in home_outcome:
                        spread = float(home_outcome['point'])
                        for pid in matches['prediction_id']:
                            pred_updates.append({'prediction_id': pid, 'book_spread': spread})
                        for gid in matches['game_id'].unique():
                            feat_updates.append({'game_id': gid, 'book_spread': spread})

    if pred_updates:
        # Update model_predictions
        temp_pred_id = f"{project}.sports_edge_curated.temp_preds_update_{league.lower()}"
        df_pred = pd.DataFrame(pred_updates).drop_duplicates('prediction_id')
        client.load_table_from_dataframe(df_pred, temp_pred_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
        
        client.query(f"""
            MERGE `{pred_table_id}` t
            USING `{temp_pred_id}` s
            ON t.prediction_id = s.prediction_id
            WHEN MATCHED THEN UPDATE SET book_spread = s.book_spread
        """).result()
        client.delete_table(temp_pred_id)
        LOGGER.info(f"Updated {len(df_pred)} BigQuery model_predictions rows with book spreads for {league}")

    if feat_updates:
        # Update feature_snapshots
        temp_feat_id = f"{project}.sports_edge_curated.temp_feat_update_{league.lower()}"
        df_feat = pd.DataFrame(feat_updates).drop_duplicates('game_id')
        client.load_table_from_dataframe(df_feat, temp_feat_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
        
        client.query(f"""
            MERGE `{feat_table_id}` t
            USING `{temp_feat_id}` s
            ON t.game_id = s.game_id
            WHEN MATCHED THEN UPDATE SET book_spread = s.book_spread
        """).result()
        client.delete_table(temp_feat_id)
        LOGGER.info(f"Updated {len(df_feat)} BigQuery feature_snapshots rows with book spreads for {league}")

def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Sync market odds to Supabase and BigQuery.")
    parser.add_argument("--project", help="GCP project ID.")
    parser.add_argument("--league", choices=["NFL", "NBA"], help="League to sync (default: both).")
    args = parser.parse_args()
    
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        LOGGER.error("ODDS_API_KEY not found in environment")
        return
        
    project = args.project or os.getenv("GCP_PROJECT_ID")
    if not project:
        LOGGER.error("Project ID not found (use --project or GCP_PROJECT_ID env var)")
        return
        
    leagues = [args.league] if args.league else ["NFL", "NBA"]
    
    bq_client = bigquery.Client(project=project)
    creds = load_supabase_credentials()
    pg_conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    
    try:
        for league in leagues:
            LOGGER.info(f"Starting sync for {league}...")
            odds_data = fetch_odds_data(api_key, league)
            if not odds_data:
                continue
                
            sync_odds_to_supabase(pg_conn, league, odds_data)
            sync_odds_to_bq(bq_client, project, league, odds_data)
            
    finally:
        pg_conn.close()

if __name__ == "__main__":
    main()
