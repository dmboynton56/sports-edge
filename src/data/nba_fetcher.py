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
import requests
from urllib3.exceptions import ReadTimeoutError
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.endpoints import (
    teamgamelog, 
    teamdashboardbygeneralsplits,
    leaguegamefinder,
    scoreboardv2
)
from nba_api.stats.static import teams


NBA_API_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
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
            # Fail fast on timeouts (don't retry hangs)
            if isinstance(exc, (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout, ReadTimeoutError)):
                raise exc

            last_exc = exc
            if attempt >= attempts:
                break
            sleep_for = base_delay * (2 ** (attempt - 1))
            print(f"Warning: {label} failed (attempt {attempt}/{attempts}): {exc}. Retrying in {sleep_for}s...")
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
) -> pd.DataFrame:
    """
    Fetch NBA schedule for a given season using LeagueGameFinder.
    
    Args:
        season: NBA season year (e.g., 2024 for 2024-25 season)
    
    Returns:
        DataFrame with schedule information including game_id, game_date, home_team, away_team, etc.
    """
    season_str = f"{season}-{str(season + 1)[-2:]}"
    
    try:
        # Optimization: For small date ranges (e.g., daily updates), use ScoreboardV2 per day
        # effectively avoiding the heavy LeagueGameFinder which can timeout on full season scans.
        use_optimized_fetch = False
        parsed_date_from = None
        parsed_date_to = None
        
        if date_from and date_to:
            try:
                # Parse dates to check duration
                if isinstance(date_from, str):
                    parsed_date_from = datetime.strptime(date_from, "%Y-%m-%d").date()
                elif isinstance(date_from, (datetime, date)):
                    parsed_date_from = date_from
                    
                if isinstance(date_to, str):
                    parsed_date_to = datetime.strptime(date_to, "%Y-%m-%d").date()
                elif isinstance(date_to, (datetime, date)):
                    parsed_date_to = date_to
                
                if parsed_date_from and parsed_date_to:
                    delta = (parsed_date_to - parsed_date_from).days
                    # If window is small enough (arbitrary cutoff, 45 days is generous for "daily" but safe for loop)
                    if 0 <= delta <= 45:
                        use_optimized_fetch = True
            except Exception:
                # If date parsing fails, fall back to standard method
                pass

        if use_optimized_fetch:
            print(f"  Using optimized iterative fetch for {parsed_date_from} to {parsed_date_to}...")
            all_games = []
            current_date = parsed_date_from
            while current_date <= parsed_date_to:
                try:
                    date_str = current_date.strftime("%Y-%m-%d")
                    daily_games = fetch_nba_games_for_date(
                        date_str, 
                        max_retries=max_retries,
                        base_delay=base_delay,
                        timeout=timeout
                    )
                    if not daily_games.empty:
                        all_games.append(daily_games)
                except Exception as e:
                    print(f"    Error fetching specific date {current_date}: {e}")
                
                current_date += timedelta(days=1)
            
            if not all_games:
                return pd.DataFrame()
            
            games_df = pd.concat(all_games, ignore_index=True)
            # Ensure season column matches requested (though ScoreboardV2 returns correct season usually)
            # Filter to only keep games for the requested season if overlap exists
            if 'season' in games_df.columns:
                games_df = games_df[games_df['season'] == season]
            
            return games_df

        # Use LeagueGameFinder to get all games for the season
        # Explicitly filter for NBA (00) to avoid G-League/WNBA games
        def _fetch():
            kwargs: Dict[str, Any] = {
                "season_nullable": season_str,
                "league_id_nullable": "00",
                "headers": NBA_API_HEADERS,
            }
            date_from_value = _format_nba_date(date_from)
            date_to_value = _format_nba_date(date_to)
            if date_from_value:
                kwargs["date_from_nullable"] = date_from_value
            if date_to_value:
                kwargs["date_to_nullable"] = date_to_value
            if _supports_timeout(leaguegamefinder.LeagueGameFinder):
                kwargs["timeout"] = timeout
            kwargs = _filter_kwargs(leaguegamefinder.LeagueGameFinder, kwargs)
            return leaguegamefinder.LeagueGameFinder(**kwargs)

        game_finder = _retry_nba_request(
            _fetch,
            label=f"NBA schedule {season_str}",
            retries=max_retries,
            base_delay=base_delay,
        )
        games_df = game_finder.get_data_frames()[0]
        
        if games_df.empty:
            return pd.DataFrame()
        
        # Normalize columns and create standard schedule format
        games_df['game_date'] = pd.to_datetime(games_df['GAME_DATE'], errors='coerce')
        games_df['season'] = season
        
        # Extract home/away teams from MATCHUP column (format: "TEAM @ OPPONENT" or "TEAM vs. OPPONENT")
        def parse_matchup(matchup: str, team_abbr: str):
            """Extract opponent from matchup string."""
            if pd.isna(matchup):
                return None
            
            matchup = matchup.strip()
            if '@' in matchup:
                # Away game: "TEAM @ OPPONENT"
                parts = matchup.split('@')
                if len(parts) == 2:
                    return parts[1].strip()
            elif 'vs.' in matchup or 'vs' in matchup:
                # Home game: "TEAM vs. OPPONENT"
                parts = matchup.split('vs.')
                if len(parts) == 2:
                    return parts[1].strip()
                else:
                    parts = matchup.split('vs')
                    if len(parts) == 2:
                        return parts[1].strip()
            return None
        
        # Create home_team and away_team columns
        games_df['is_home'] = games_df['MATCHUP'].str.contains('vs.', case=False, na=False) | \
                             games_df['MATCHUP'].str.contains('vs', case=False, na=False)
        
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
    
    Args:
        date: Date string in YYYY-MM-DD format
    
    Returns:
        DataFrame with games for that date (standardized format)
    """
    try:
        # Convert date to MM/DD/YYYY format for NBA API
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        date_str = date_obj.strftime('%m/%d/%Y')
        
        # Use ScoreboardV2 for date-specific games
        # Explicitly filter for NBA (00)
        def _fetch():
            kwargs: Dict[str, Any] = {
                "game_date": date_str,
                "league_id": "00",
                "headers": NBA_API_HEADERS,
            }
            if _supports_timeout(scoreboardv2.ScoreboardV2):
                kwargs["timeout"] = timeout
            kwargs = _filter_kwargs(scoreboardv2.ScoreboardV2, kwargs)
            return scoreboardv2.ScoreboardV2(**kwargs)

        scoreboard_data = _retry_nba_request(
            _fetch,
            label=f"NBA games {date_str}",
            retries=max_retries,
            base_delay=base_delay,
        )
        game_header = scoreboard_data.get_data_frames()[0]
        line_score = scoreboard_data.get_data_frames()[1]
        
        if game_header.empty:
            return pd.DataFrame()
        
        # Standardize to match our schedule format
        # Line score has 2 rows per game (one for home, one for visitor)
        games = []
        for _, game in game_header.iterrows():
            game_id = game['GAME_ID']
            home_team_id = game['HOME_TEAM_ID']
            visitor_team_id = game['VISITOR_TEAM_ID']
            
            home_row = line_score[(line_score['GAME_ID'] == game_id) & (line_score['TEAM_ID'] == home_team_id)]
            visitor_row = line_score[(line_score['GAME_ID'] == game_id) & (line_score['TEAM_ID'] == visitor_team_id)]
            
            if not home_row.empty and not visitor_row.empty:
                games.append({
                    'game_id': game_id,
                    'season': int(game['SEASON']),
                    'game_date': pd.to_datetime(game['GAME_DATE_EST']),
                    'home_team': home_row.iloc[0]['TEAM_ABBREVIATION'],
                    'away_team': visitor_row.iloc[0]['TEAM_ABBREVIATION'],
                    'home_score': home_row.iloc[0]['PTS'] if pd.notna(home_row.iloc[0]['PTS']) else None,
                    'away_score': visitor_row.iloc[0]['PTS'] if pd.notna(visitor_row.iloc[0]['PTS']) else None,
                })
        
        result_df = pd.DataFrame(games)
        
        if result_df.empty:
            return pd.DataFrame()
            
        # Filter to requested date (sometimes Scoreboard returns adjacent days depending on timezone)
        result_df = result_df[result_df['game_date'].dt.date == date_obj.date()]
        
        return result_df.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error fetching NBA games for {date}: {e}")
        import traceback
        traceback.print_exc()
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
            timeout=30
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
            timeout=30
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

