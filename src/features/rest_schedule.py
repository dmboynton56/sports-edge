"""
Rest and schedule feature engineering.
Computes rest days, back-to-back flags, and travel distance.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in km).
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance


# Team city coordinates (approximate)
TEAM_COORDINATES = {
    # NFL teams
    'ARI': (33.5275, -112.2625), 'ATL': (33.7550, -84.4010), 'BAL': (39.2780, -76.6227),
    'BUF': (42.7738, -78.7869), 'CAR': (35.2258, -80.8528), 'CHI': (41.8625, -87.6167),
    'CIN': (39.0950, -84.5160), 'CLE': (41.5061, -81.6996), 'DAL': (32.7473, -97.0945),
    'DEN': (39.7439, -105.0200), 'DET': (42.3400, -83.0456), 'GB': (44.5013, -88.0622),
    'HOU': (29.6847, -95.4107), 'IND': (39.7601, -86.1639), 'JAX': (30.3239, -81.6372),
    'KC': (39.0489, -94.4839), 'LV': (37.7519, -122.2009), 'LAC': (33.8643, -118.2611),
    'LAR': (34.0141, -118.2877), 'MIA': (25.9581, -80.2389), 'MIN': (44.9740, -93.2581),
    'NE': (42.0909, -71.2643), 'NO': (29.9511, -90.0815), 'NYG': (40.8135, -74.0745),
    'NYJ': (40.8135, -74.0745), 'PHI': (39.9008, -75.1673), 'PIT': (40.4468, -80.0158),
    'SF': (37.4033, -121.9694), 'SEA': (47.5952, -122.3316), 'TB': (27.9753, -82.5033),
    'TEN': (36.1665, -86.7713), 'WAS': (38.9077, -76.8644),
    # NBA teams
    'ATL': (33.7550, -84.4010), 'BOS': (42.3662, -71.0621), 'BKN': (40.6826, -73.9744),
    'CHA': (35.2258, -80.8528), 'CHI': (41.8625, -87.6167), 'CLE': (41.5061, -81.6996),
    'DAL': (32.7903, -96.8102), 'DEN': (39.7439, -105.0200), 'DET': (42.3400, -83.0456),
    'GSW': (37.7680, -122.3879), 'HOU': (29.7508, -95.3621), 'IND': (39.7639, -86.1554),
    'LAC': (34.0430, -118.2673), 'LAL': (34.0430, -118.2673), 'MEM': (35.1380, -90.0506),
    'MIA': (25.7814, -80.1866), 'MIL': (43.0436, -87.9169), 'MIN': (44.9794, -93.2771),
    'NOP': (29.9490, -90.0821), 'NYK': (40.7505, -73.9934), 'OKC': (35.4634, -97.5151),
    'ORL': (28.5392, -81.3839), 'PHI': (39.9012, -75.1720), 'PHX': (33.4453, -112.0712),
    'POR': (45.5316, -122.6668), 'SAC': (38.5802, -121.4998), 'SAS': (29.4269, -98.4375),
    'TOR': (43.6435, -79.3791), 'UTA': (40.7683, -111.9011), 'WAS': (38.8981, -77.0209),
}


def _filter_games_by_season(games_df: pd.DataFrame, season: Optional[int]) -> pd.DataFrame:
    """Return only games from the requested season."""
    if season is None or games_df.empty:
        return games_df
    if 'season' in games_df.columns:
        season_values = pd.to_numeric(games_df['season'], errors='coerce')
        return games_df[season_values == season]
    game_years = pd.to_datetime(games_df['game_date']).dt.year
    return games_df[game_years == season]


def compute_rest_days(team: str, game_date: datetime, previous_games: pd.DataFrame,
                      season: Optional[int] = None) -> Optional[int]:
    """
    Compute days of rest for a team before a game.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        previous_games: DataFrame with previous games (must have 'game_date' and team columns)
    
    Returns:
        Number of days rest (or None if no previous game found)
    """
    # Find team's last game before this date
    season_games = _filter_games_by_season(previous_games, season)
    team_games = season_games[
        ((season_games['home_team'] == team) | (season_games['away_team'] == team)) &
        (pd.to_datetime(season_games['game_date']) < game_date)
    ]
    
    if team_games.empty:
        return None
    
    last_game_date = pd.to_datetime(team_games['game_date']).max()
    rest_days = (game_date - last_game_date).days
    
    return rest_days


def is_back_to_back(team: str, game_date: datetime, previous_games: pd.DataFrame,
                    season: Optional[int] = None) -> bool:
    """
    Check if team is playing back-to-back games.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        previous_games: DataFrame with previous games
    
    Returns:
        True if playing back-to-back (1 day rest or less)
    """
    rest_days = compute_rest_days(team, game_date, previous_games, season=season)
    return rest_days is not None and rest_days <= 1


def compute_travel_distance(team: str, game_date: datetime, previous_games: pd.DataFrame,
                           is_home: bool, season: Optional[int] = None) -> Optional[float]:
    """
    Compute travel distance (km) from team's last game location to current game.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        previous_games: DataFrame with previous games (must have 'home_team', 'away_team', 'game_date')
        is_home: Whether current game is at home
    
    Returns:
        Travel distance in km (or None if can't compute)
    """
    # Find team's last game
    season_games = _filter_games_by_season(previous_games, season)
    team_games = season_games[
        ((season_games['home_team'] == team) | (season_games['away_team'] == team)) &
        (pd.to_datetime(season_games['game_date']) < game_date)
    ]
    
    if team_games.empty:
        return None
    
    last_game = team_games.iloc[-1]
    was_home_last = last_game['home_team'] == team
    
    # Get coordinates
    if was_home_last:
        last_location = TEAM_COORDINATES.get(team)
    else:
        opponent = last_game['home_team'] if was_home_last else last_game['away_team']
        last_location = TEAM_COORDINATES.get(opponent)
    
    if is_home:
        current_location = TEAM_COORDINATES.get(team)
    else:
        # Need opponent - this would come from the game data
        # For now, return None if away (would need game context)
        return None
    
    if last_location is None or current_location is None:
        return None
    
    distance = haversine_distance(
        last_location[0], last_location[1],
        current_location[0], current_location[1]
    )
    
    return distance


def add_rest_features(games_df: pd.DataFrame, historical_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add rest and schedule features to games DataFrame efficiently.
    """
    df = games_df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    if df['game_date'].dt.tz is not None:
        df['game_date'] = df['game_date'].dt.tz_localize(None)
        
    hist = historical_games.copy()
    hist['game_date'] = pd.to_datetime(hist['game_date'])
    if hist['game_date'].dt.tz is not None:
        hist['game_date'] = hist['game_date'].dt.tz_localize(None)
    
    # 1. Melt history to get all games per team
    h_games = hist[['game_date', 'season', 'home_team']].rename(columns={'home_team': 'team'})
    a_games = hist[['game_date', 'season', 'away_team']].rename(columns={'away_team': 'team'})
    team_games = pd.concat([h_games, a_games]).sort_values(['team', 'game_date'])
    
    # 2. Compute rest days
    team_games['prev_game_date'] = team_games.groupby(['team', 'season'])['game_date'].shift(1)
    team_games['rest_days'] = (team_games['game_date'] - team_games['prev_game_date']).dt.days
    
    # 3. Add fatigue indicators
    # B2B
    team_games['is_b2b'] = (team_games['rest_days'] <= 1).astype(int)
    
    # 3 games in 4 nights (3in4)
    # A team has played 3 in 4 if their game 2 days ago was also within 4 days of today
    team_games['game_3_ago_date'] = team_games.groupby(['team', 'season'])['game_date'].shift(2)
    team_games['is_3in4'] = ((team_games['game_date'] - team_games['game_3_ago_date']).dt.days <= 4).astype(int)
    
    # 4. Merge back to games_df
    # We want the rest/fatigue status AT THE TIME OF THE GAME
    # The team_games table already has this for each game
    lookup = team_games[['team', 'game_date', 'rest_days', 'is_b2b', 'is_3in4']]
    
    # Home team
    df = pd.merge(df, lookup.rename(columns={
        'team': 'home_team', 
        'rest_days': 'rest_home', 
        'is_b2b': 'b2b_home', 
        'is_3in4': 'is_3in4_home'
    }), on=['home_team', 'game_date'], how='left')
    
    # Away team
    df = pd.merge(df, lookup.rename(columns={
        'team': 'away_team', 
        'rest_days': 'rest_away', 
        'is_b2b': 'b2b_away', 
        'is_3in4': 'is_3in4_away'
    }), on=['away_team', 'game_date'], how='left')
    
    # Fill NAs (first game of season)
    df['rest_home'] = df['rest_home'].fillna(7) # Assume full rest for first game
    df['rest_away'] = df['rest_away'].fillna(7)
    df['b2b_home'] = df['b2b_home'].fillna(0)
    df['b2b_away'] = df['b2b_away'].fillna(0)
    df['is_3in4_home'] = df['is_3in4_home'].fillna(0)
    df['is_3in4_away'] = df['is_3in4_away'].fillna(0)
    
    return df
