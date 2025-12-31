"""
NBA data fetcher using nba_api.
Fetches schedule, team game logs, and advanced statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict
import os
import time
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.endpoints import (
    teamgamelog, 
    teamdashboardbygeneralsplits,
    leaguegamefinder,
    scoreboardv2
)
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
    Fetch NBA schedule for a given season using LeagueGameFinder.
    
    Args:
        season: NBA season year (e.g., 2024 for 2024-25 season)
    
    Returns:
        DataFrame with schedule information including game_id, game_date, home_team, away_team, etc.
    """
    season_str = f"{season}-{str(season + 1)[-2:]}"
    
    try:
        # Use LeagueGameFinder to get all games for the season
        game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season_str)
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
        return pd.DataFrame()


def fetch_nba_games_for_date(date: str) -> pd.DataFrame:
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
        scoreboard_data = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_df = scoreboard_data.get_data_frames()[0]
        
        if games_df.empty:
            return pd.DataFrame()
        
        # Standardize to match our schedule format
        games_df['game_date'] = pd.to_datetime(games_df['GAME_DATE_EST'], errors='coerce')
        games_df['season'] = games_df['game_date'].dt.year
        
        # Create standardized format
        result_df = pd.DataFrame({
            'game_id': games_df.get('GAME_ID', None),
            'season': games_df['season'],
            'game_date': games_df['game_date'],
            'home_team': games_df.get('HOME_TEAM_ABBREVIATION', games_df.get('HOME_TEAM_NAME', None)),
            'away_team': games_df.get('VISITOR_TEAM_ABBREVIATION', games_df.get('VISITOR_TEAM_NAME', None)),
            'home_score': games_df.get('HOME_TEAM_SCORE', None),
            'away_score': games_df.get('VISITOR_TEAM_SCORE', None),
        })
        
        # Filter to requested date
        result_df = result_df[result_df['game_date'].dt.date == date_obj.date()]
        
        return result_df.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error fetching NBA games for {date}: {e}")
        import traceback
        traceback.print_exc()
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

