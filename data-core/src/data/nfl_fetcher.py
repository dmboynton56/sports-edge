"""
NFL data fetcher built on nfl_data_py.
Fetches schedule, play-by-play data, and team statistics.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import os

import nfl_data_py as nfl


def fetch_nfl_schedule(season: int, week: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch NFL schedule for a given season and optionally a specific week.
    
    Args:
        season: NFL season year (e.g., 2024)
        week: Optional week number (1-18 for regular season)
    
    Returns:
        DataFrame with columns: game_id, season, week, game_date, home_team, away_team, etc.
    """
    schedule = nfl.import_schedules([season])
    
    if week is not None:
        schedule = schedule[schedule['week'] == week]
    
    return schedule


def fetch_nfl_team_stats(season: int, week: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch NFL team-level statistics (EPA, success rate, etc.).
    
    Args:
        season: NFL season year
        week: Optional week number
    
    Returns:
        DataFrame with team statistics
    """
    pbp = nfl.import_pbp_data([season])
    if week is not None:
        pbp = pbp[pbp['week'] == week]
    
    team_stats = pbp.groupby(['posteam', 'week']).agg({
        'epa': 'mean',
        'success': 'mean',
        'yards_gained': 'sum'
    }).reset_index()
    
    return team_stats


def fetch_nfl_games_for_date(date: str) -> pd.DataFrame:
    """
    Fetch NFL games scheduled for a specific date.
    
    Args:
        date: Date string in YYYY-MM-DD format
    
    Returns:
        DataFrame with games for that date
    """
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    season = date_obj.year if date_obj.month >= 9 else date_obj.year - 1
    
    schedule = fetch_nfl_schedule(season)
    schedule['game_date'] = pd.to_datetime(schedule['gameday'])
    
    games = schedule[schedule['game_date'].dt.date == date_obj.date()]
    return games


def fetch_nfl_weekly_data(seasons: List[int]) -> pd.DataFrame:
    """
    Fetch NFL weekly player statistics for a list of seasons.
    
    Attempts nflreadpy first and falls back to nfl_data_py.
    
    Args:
        seasons: List of season years
    
    Returns:
        DataFrame with weekly player stats
    """
    try:
        import nflreadpy as nflr
        stats = nflr.load_player_stats(seasons).to_pandas()
        if not stats.empty:
            # Rename some columns to match nfl_data_py expectations if needed
            # nfl_data_py: passing_tds, rushing_tds, receiving_tds
            # nflreadpy: same.
            
            # nflreadpy 'team' corresponds to nfl_data_py 'recent_team'
            if 'team' in stats.columns and 'recent_team' not in stats.columns:
                stats['recent_team'] = stats['team']
            return stats
    except ImportError:
        pass
        
    return nfl.import_weekly_data(seasons)


def cache_nfl_data(data: pd.DataFrame, league: str, date: str, data_type: str):
    """
    Cache NFL data to disk for reproducibility.
    
    Args:
        data: DataFrame to cache
        league: 'nfl'
        date: Date string YYYY-MM-DD
        data_type: 'schedule', 'stats', etc.
    """
    cache_dir = f"data/raw/{league}/{date}"
    os.makedirs(cache_dir, exist_ok=True)
    
    filepath = f"{cache_dir}/{data_type}.parquet"
    data.to_parquet(filepath, index=False)
    print(f"Cached {data_type} to {filepath}")
