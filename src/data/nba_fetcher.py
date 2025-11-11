"""
NBA data fetcher using nba_api.
Fetches schedule, team game logs, and advanced statistics.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import os
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import teamgamelog, teamdashboardbygeneralsplits
from nba_api.stats.static import teams


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


def fetch_nba_schedule(season: int) -> pd.DataFrame:
    """
    Fetch NBA schedule for a given season.
    
    Args:
        season: NBA season year (e.g., 2024 for 2024-25 season)
    
    Returns:
        DataFrame with schedule information
    """
    # NBA API uses season format like '2024-25'
    season_str = f"{season}-{str(season + 1)[-2:]}"
    
    # Fetch scoreboard for current date (we'll need to iterate through dates)
    # For now, return empty - will be implemented with date-specific fetching
    return pd.DataFrame()


def fetch_nba_games_for_date(date: str) -> pd.DataFrame:
    """
    Fetch NBA games scheduled for a specific date.
    
    Args:
        date: Date string in YYYY-MM-DD format
    
    Returns:
        DataFrame with games for that date
    """
    try:
        scoreboard_data = scoreboard.ScoreBoard()
        games = scoreboard_data.get_data_frames()[0]
        
        # Filter by date
        games['game_date'] = pd.to_datetime(games['GAME_DATE_EST'])
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        games = games[games['game_date'].dt.date == date_obj.date()]
        
        return games
    except Exception as e:
        print(f"Error fetching NBA games: {e}")
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
        team_log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
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
            team_id=team_id, season=season
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

