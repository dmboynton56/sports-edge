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
from src.utils.team_codes import canonical_nba_abbr

LOGGER = logging.getLogger("sync_odds")
PREFERRED_BOOKMAKERS = ["draftkings", "betmgm", "fanduel"]

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

def canonical_team_code(team_code: str, league: str) -> Optional[str]:
    if league == "NBA":
        return canonical_nba_abbr(team_code)
    return team_code

def ordered_bookmakers(bookmakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep preferred books first, then try the rest instead of giving up early."""
    preferred = [
        bookmaker
        for key in PREFERRED_BOOKMAKERS
        for bookmaker in bookmakers
        if bookmaker.get("key") == key
    ]
    preferred_keys = {bookmaker.get("key") for bookmaker in preferred}
    fallback = [bookmaker for bookmaker in bookmakers if bookmaker.get("key") not in preferred_keys]
    return preferred + fallback

def pick_home_spread_from_event(
    event: Dict[str, Any],
    home_code: str,
    mapping: Dict[str, List[str]],
) -> Optional[tuple[float, Optional[int], str]]:
    """Return the first usable home-team spread, trying preferred books before fallbacks."""
    for bookmaker in ordered_bookmakers(event.get("bookmakers", [])):
        market = next((m for m in bookmaker.get("markets", []) if m.get("key") == "spreads"), None)
        if not market:
            continue
        home_outcome = next(
            (o for o in market.get("outcomes", []) if get_team_code(o.get("name", ""), mapping) == home_code),
            None,
        )
        if home_outcome and "point" in home_outcome and home_outcome["point"] is not None:
            return (
                float(home_outcome["point"]),
                home_outcome.get("price"),
                bookmaker.get("key") or "unknown",
            )
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
    
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Error fetching {league} odds from API: {resp.status_code} {resp.text}")
    return resp.json()

def count_expected_games(conn, league: str) -> int:
    """Count games in the serving window where odds should be attempted."""
    now = datetime.now(timezone.utc)
    start_window = (now - timedelta(days=2)).isoformat()
    end_window = (now + timedelta(days=10)).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM games
            WHERE league = %s
              AND game_time_utc >= %s
              AND game_time_utc <= %s
            """,
            (league, start_window, end_window),
        )
        return int(cur.fetchone()[0])

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

def sync_odds_to_supabase(conn, league: str, odds_data: List[Dict[str, Any]]) -> tuple[int, int]:
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
        canonical_home = canonical_team_code(home, league)
        canonical_away = canonical_team_code(away, league)
        if canonical_home and canonical_away:
            game_map[game_map_key(canonical_home, canonical_away, gtime)] = gid
        
    if games:
        sample_keys = list(game_map.keys())[:12]
        LOGGER.info(
            "%s: %d Supabase games in matching window; sample keys: %s",
            league,
            len(games),
            sample_keys,
        )

    matched_count = 0
    for event in odds_data:
        home_raw = event.get("home_team") or ""
        away_raw = event.get("away_team") or ""
        home_code = get_team_code(home_raw, mapping)
        away_code = get_team_code(away_raw, mapping)
        if not home_code or not away_code:
            LOGGER.warning(
                "%s: unresolved Odds API teams %r @ %r → codes %s vs %s",
                league,
                away_raw,
                home_raw,
                away_code,
                home_code,
            )
            continue

        game_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        key = game_map_key(home_code, away_code, game_time)

        gid = game_map.get(key)
        if not gid:
            day_key_prefix = key.split("_")[0]
            fuzzy_key = next(
                (
                    k
                    for k in game_map.keys()
                    if k.startswith(day_key_prefix) and home_code in k and away_code in k
                ),
                None,
            )
            if fuzzy_key:
                gid = game_map[fuzzy_key]

        if not gid:
            fallback_key = next(
                (
                    k
                    for k in game_map.keys()
                    if home_code in k and away_code in k
                ),
                None,
            )
            if fallback_key:
                gid = game_map[fallback_key]

        if gid:
            spread = pick_home_spread_from_event(event, home_code, mapping)
            if spread:
                line, price, book = spread
                updates.append((line, price, book, gid))
                matched_count += 1
            else:
                book_keys = [b.get("key") for b in event.get("bookmakers", [])]
                LOGGER.warning(
                    "%s: Supabase id=%s matched but no spreads line for home %s; "
                    "commence=%s; bookmakers=%s",
                    league,
                    gid,
                    home_code,
                    event.get("commence_time"),
                    book_keys,
                )
        else:
            LOGGER.warning(
                "%s: no Supabase row for Odds event %s@%s commence=%s (lookup_key=%s)",
                league,
                away_code,
                home_code,
                event.get("commence_time"),
                key,
            )
                        
    if updates:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE games SET book_spread = %s WHERE id = %s",
                [(spread, gid) for spread, _price, _book, gid in updates],
            )
            cur.executemany(
                """
                INSERT INTO odds_snapshots (game_id, book, market, line, price, snapshot_ts)
                VALUES (%s, %s, 'spread', %s, %s, NOW())
                """,
                [(gid, book, spread, price) for spread, price, book, gid in updates],
            )
        conn.commit()
        LOGGER.info(f"Updated {len(updates)} games in Supabase with book spreads for {league}")
    else:
        LOGGER.warning(f"No odds matches found for {league} in Supabase window")
    return matched_count, len(games)

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
            
        event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        event_date_us = (event_time_utc - timedelta(hours=5)).date()
        canonical_home = preds["home_team"].apply(lambda team: canonical_team_code(team, league))
        canonical_away = preds["away_team"].apply(lambda team: canonical_team_code(team, league))
        matches = preds[
            (canonical_home == home_code) &
            (canonical_away == away_code) &
            (pd.to_datetime(preds['game_date']).dt.date == event_date_us)
        ]
        
        if not matches.empty:
            spread = pick_home_spread_from_event(event, home_code, mapping)
            if spread:
                line, _price, _book = spread
                for pid in matches['prediction_id']:
                    pred_updates.append({'prediction_id': pid, 'book_spread': line})
                for gid in matches['game_id'].unique():
                    feat_updates.append({'game_id': gid, 'book_spread': line})

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
        raise SystemExit("ODDS_API_KEY not found in environment")
        
    project = args.project or os.getenv("GCP_PROJECT_ID")
    if not project:
        raise SystemExit("Project ID not found (use --project or GCP_PROJECT_ID env var)")
        
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
    
    failures: list[str] = []
    try:
        for league in leagues:
            LOGGER.info(f"Starting sync for {league}...")
            expected_games = count_expected_games(pg_conn, league)
            odds_data = fetch_odds_data(api_key, league)
            if not odds_data:
                message = f"No odds events returned for {league}"
                if expected_games > 0:
                    failures.append(f"{message}; {expected_games} Supabase games are in the serving window.")
                    LOGGER.error(failures[-1])
                else:
                    LOGGER.warning(f"{message}; no Supabase games are in the serving window.")
                continue
                
            LOGGER.info(f"Retrieved {len(odds_data)} games from The Odds API for {league}")
            matched_count, supabase_games = sync_odds_to_supabase(pg_conn, league, odds_data)
            if supabase_games > 0 and matched_count == 0:
                failures.append(
                    f"{league}: matched zero odds to {supabase_games} Supabase games in the serving window."
                )
            sync_odds_to_bq(bq_client, project, league, odds_data)
        if failures:
            for failure in failures:
                LOGGER.error(failure)
            raise SystemExit(1)
            
    finally:
        pg_conn.close()

if __name__ == "__main__":
    main()
