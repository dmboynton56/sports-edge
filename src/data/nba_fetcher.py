"""
NBA data fetcher using nba_api.
Fetches schedule, team game logs, and advanced statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Callable, TypeVar, Any
import inspect
import os
import time
import random
import requests
from urllib3.exceptions import ReadTimeoutError
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.endpoints import (
    teamgamelog, 
    teamdashboardbygeneralsplits,
    leaguegamefinder,
    scoreboardv2,
    scheduleleaguev2
)
from nba_api.stats.static import teams


NBA_API_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}


def get_team_id(team_name: str) -> Optional[int]:
    """
    Convert team name/abbreviation to NBA team ID.
    
    Args:
        team_name: Team name or abbreviation (e.g., 'Lakers', 'LAL')
    
    Returns:
        Team ID or None if not found
    """
    nba_teams = teams.get_teams()
    
    # Try exact match first
    for team in nba_teams:
        if team_name.upper() in [team['abbreviation'], team['nickname'].upper(), team['full_name'].upper()]:
            return team['id']
    
    return None


T = TypeVar("T")


def _retry_nba_request(
    fn: Callable[[], T],
    label: str,
    retries: int,
    base_delay: int,
) -> T:
    """Retry NBA API calls with exponential backoff."""
    attempts = max(1, retries)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            # If it's a timeout, we still want to retry if we have attempts left
            last_exc = exc
            if attempt >= attempts:
                break
            
            # Increase delay for timeouts specifically
            multiplier = 2 if isinstance(exc, (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, ReadTimeoutError)) else 1
            sleep_for = (base_delay * multiplier) * (2 ** (attempt - 1))
            
            # Add small jitter to avoid bot detection
            sleep_for += random.uniform(0, 1)
            
            print(f"Warning: {label} failed (attempt {attempt}/{attempts}): {exc}. Retrying in {sleep_for:.1f}s...")
            time.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{label} failed without an exception")


def _supports_timeout(endpoint: Callable[..., object]) -> bool:
    try:
        return "timeout" in inspect.signature(endpoint).parameters
    except (ValueError, TypeError):
        return False


def _filter_kwargs(endpoint: Callable[..., object], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = inspect.signature(endpoint).parameters
    except (ValueError, TypeError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _format_nba_date(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%m/%d/%Y")
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").strftime("%m/%d/%Y")
        except ValueError:
            return value
    return str(value)


def fetch_nba_schedule(
    season: int,
    *,
    max_retries: int = 3,
    base_delay: int = 2,
    timeout: int = 30,
    date_from: Optional[object] = None,
    date_to: Optional[object] = None,
    raise_on_error: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch NBA schedule for a given season.
    Tries BigQuery first, then falls back to ESPN/NBA APIs.
    """
    season_str = f"{season}-{str(season + 1)[-2:]}"
    
    # 1. Try BigQuery first (Instant)
    if use_cache:
        try:
            from google.cloud import bigquery
            client = bigquery.Client()
            project = os.getenv("GCP_PROJECT_ID")
            if project:
                print(f"  Checking BigQuery for {season} schedule...")
                query = f"""
                    SELECT game_id, season, game_date, home_team, away_team, home_score, away_score
                    FROM `{project}.sports_edge_raw.raw_schedules`
                    WHERE season = {season}
                """
                if date_from:
                    query += f" AND game_date >= '{pd.to_datetime(date_from).strftime('%Y-%m-%d')}'"
                if date_to:
                    query += f" AND game_date <= '{pd.to_datetime(date_to).strftime('%Y-%m-%d')}'"
                
                df_bq = client.query(query).to_dataframe()
                if not df_bq.empty:
                    # If we have a decent number of games (or specifically requested a small range), trust cache
                    if (date_from and date_to) or len(df_bq) > 100:
                        print(f"  Successfully loaded {len(df_bq)} games from BigQuery.")
                        df_bq['game_date'] = pd.to_datetime(df_bq['game_date'], utc=True).dt.tz_localize(None)
                        return df_bq.sort_values('game_date').reset_index(drop=True)
        except Exception as bq_err:
            print(f"  BigQuery check skipped/failed: {bq_err}. Falling back to APIs...")

    # 2. Try ESPN first (Fast, no bot detection, very reliable)
    try:
        print(f"  Fetching {season} schedule from ESPN API...")
        espn_dfs = []
        
        # Determine date range to fetch
        if date_from and date_to:
            start_date = pd.to_datetime(date_from)
            end_date = pd.to_datetime(date_to)
        else:
            # Full season range: Oct to June
            start_date = datetime(season, 10, 1)
            # Fetch up to today + some buffer to ensure we get current results
            end_date = datetime.now() + timedelta(days=14)
            
        delta = (end_date - start_date).days
        if delta > 400: delta = 400 # Safety cap
        
        # Fetch in daily chunks
        for i in range(delta + 1):
            curr_date = start_date + timedelta(days=i)
            try:
                daily = fetch_nba_games_for_date(curr_date.strftime("%Y-%m-%d"))
                if daily is not None and not daily.empty:
                    espn_dfs.append(daily)
            except:
                continue
                
        if espn_dfs:
            schedule_df = pd.concat(espn_dfs, ignore_index=True).drop_duplicates(subset=["game_id"])
            print(f"  Successfully fetched {len(schedule_df)} games via ESPN.")
            # Standardize columns for the rest of the pipeline
            schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date'], utc=True).dt.tz_localize(None)
            return schedule_df.sort_values('game_date').reset_index(drop=True)
            
    except Exception as e:
        print(f"  ESPN Schedule fetch failed: {e}. Falling back to NBA API...")

    # 3. NBA API Fallback (Simplified)
    try:
        # Since ESPN is primary and very reliable for schedule/scores, 
        # we skip the problematic stats.nba.com schedule calls entirely.
        # This prevents the hangs and timeouts that plague stats.nba.com.
        print(f"  Skipping NBA API schedule fallback (relying on ESPN).")
        return pd.DataFrame()
    except Exception as e:
        if raise_on_error: raise
        return pd.DataFrame()
        
        games_df['home_team'] = np.where(
            games_df['is_home'],
            games_df['TEAM_ABBREVIATION'],
            games_df['MATCHUP'].apply(lambda x: parse_matchup(x, None) if pd.notna(x) else None)
        )
        
        games_df['away_team'] = np.where(
            ~games_df['is_home'],
            games_df['TEAM_ABBREVIATION'],
            games_df['MATCHUP'].apply(lambda x: parse_matchup(x, None) if pd.notna(x) else None)
        )
        
        # Get scores if available
        games_df['home_score'] = None
        games_df['away_score'] = None
        
        # For each unique game_id, get scores from both team perspectives
        for game_id in games_df['GAME_ID'].unique():
            game_rows = games_df[games_df['GAME_ID'] == game_id]
            if len(game_rows) == 2:
                # Both team perspectives exist
                row1, row2 = game_rows.iloc[0], game_rows.iloc[1]
                
                # Determine which is home/away based on matchup
                if row1['is_home']:
                    games_df.loc[games_df['GAME_ID'] == game_id, 'home_team'] = row1['TEAM_ABBREVIATION']
                    games_df.loc[games_df['GAME_ID'] == game_id, 'away_team'] = row2['TEAM_ABBREVIATION']
                    games_df.loc[games_df['GAME_ID'] == game_id, 'home_score'] = row1.get('PTS', None)
                    games_df.loc[games_df['GAME_ID'] == game_id, 'away_score'] = row2.get('PTS', None)
                else:
                    games_df.loc[games_df['GAME_ID'] == game_id, 'home_team'] = row2['TEAM_ABBREVIATION']
                    games_df.loc[games_df['GAME_ID'] == game_id, 'away_team'] = row1['TEAM_ABBREVIATION']
                    games_df.loc[games_df['GAME_ID'] == game_id, 'home_score'] = row2.get('PTS', None)
                    games_df.loc[games_df['GAME_ID'] == game_id, 'away_score'] = row1.get('PTS', None)
        
        # Create standard columns
        schedule_df = pd.DataFrame({
            'game_id': games_df['GAME_ID'],
            'season': games_df['season'],
            'game_date': games_df['game_date'],
            'home_team': games_df['home_team'],
            'away_team': games_df['away_team'],
            'home_score': games_df['home_score'],
            'away_score': games_df['away_score'],
        })
        
        # Remove duplicates (each game appears twice - once per team)
        schedule_df = schedule_df.drop_duplicates(subset=['game_id']).reset_index(drop=True)
        
        # Sort by date
        schedule_df = schedule_df.sort_values('game_date').reset_index(drop=True)
        
        return schedule_df
        
    except Exception as e:
        print(f"Error fetching NBA schedule for {season}: {e}")
        import traceback
        traceback.print_exc()
        if raise_on_error:
            raise
        return pd.DataFrame()


def fetch_nba_games_for_date(
    date: str,
    *,
    max_retries: int = 3,
    base_delay: int = 2,
    timeout: int = 30,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    Fetch NBA games scheduled for a specific date.
    Tries ESPN API first as it's more reliable than stats.nba.com.
    """
    # 1. Try ESPN first (much faster, no bot protection)
    try:
        date_clean = date.replace("-", "") # YYYYMMDD
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_clean}"
        print(f"  Fetching {date} from ESPN API...")
        resp = requests.get(url, timeout=10) # Short timeout
        resp.raise_for_status()
        data = resp.json()
        
        games = []
        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            status = event.get("status", {}).get("type", {}).get("name")
            is_completed = status in ["STATUS_FINAL", "STATUS_FULL_TIME"]
            
            # Home/Away can be in any order in competitors list
            home_team_node = next((t for t in competition.get("competitors", []) if t.get("homeAway") == "home"), None)
            away_team_node = next((t for t in competition.get("competitors", []) if t.get("homeAway") == "away"), None)
            
            if home_team_node and away_team_node:
                h_score = pd.to_numeric(home_team_node.get("score"), errors='coerce')
                a_score = pd.to_numeric(away_team_node.get("score"), errors='coerce')
                
                games.append({
                    'game_id': str(event.get("id")),
                    'season': data.get("season", {}).get("year", 2025),
                    'game_date': pd.to_datetime(event.get("date"), utc=True).tz_localize(None),
                    'home_team': home_team_node.get("team", {}).get("abbreviation"),
                    'away_team': away_team_node.get("team", {}).get("abbreviation"),
                    'home_score': h_score if is_completed else None,
                    'away_score': a_score if is_completed else None,
                    'status': status
                })
        
        # KEY FIX: If we successfully got data from ESPN (even 0 games), return it.
        # Do NOT filter strictly by UTC date here, as evening games shift to the next day.
        # ESPN's ?dates= query is already authoritative for the local day.
        print(f"  Successfully fetched {len(games)} games from ESPN.")
        return pd.DataFrame(games)

    except Exception as e:
        print(f"  ESPN API failed for {date}: {e}.")
        if raise_on_error:
            raise
        return pd.DataFrame()


def fetch_nba_team_stats(team_id: int, season: str) -> pd.DataFrame:
    """
    Fetch NBA team statistics for a given team and season.
    
    Args:
        team_id: NBA team ID
        season: Season string (e.g., '2024-25')
    
    Returns:
        DataFrame with team statistics
    """
    try:
        team_log = teamgamelog.TeamGameLog(
            team_id=team_id, 
            season=season,
            headers=NBA_API_HEADERS,
            timeout=60
        )
        df = team_log.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching team stats for team {team_id}: {e}")
        return pd.DataFrame()


def fetch_nba_advanced_stats(team_id: int, season: str) -> pd.DataFrame:
    """
    Fetch advanced NBA team statistics (net rating, pace, etc.).
    
    Args:
        team_id: NBA team ID
        season: Season string (e.g., '2024-25')
    
    Returns:
        DataFrame with advanced statistics
    """
    try:
        dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id, 
            season=season,
            headers=NBA_API_HEADERS,
            timeout=60
        )
        df = dashboard.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching advanced stats for team {team_id}: {e}")
        return pd.DataFrame()


def cache_nba_data(data: pd.DataFrame, league: str, date: str, data_type: str):
    """
    Cache NBA data to disk for reproducibility.
    
    Args:
        data: DataFrame to cache
        league: 'nba'
        date: Date string YYYY-MM-DD
        data_type: 'schedule', 'stats', etc.
    """
    cache_dir = f"data/raw/{league}/{date}"
    os.makedirs(cache_dir, exist_ok=True)
    
    filepath = f"{cache_dir}/{data_type}.parquet"
    data.to_parquet(filepath, index=False)
    print(f"Cached {data_type} to {filepath}")

