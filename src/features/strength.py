"""
Opponent strength feature engineering.
Computes season-long opponent strength metrics (SRS, Elo, etc.).
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_season_opponent_strength(team: str, game_date: pd.Timestamp,
                                    historical_games: pd.DataFrame,
                                    league: str = 'nfl',
                                    season: Optional[int] = None) -> Optional[float]:
    """
    Compute opponent strength for a team based on season performance.
    
    For NFL: Uses point differential and strength of schedule
    For NBA: Uses net rating and strength of schedule
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        historical_games: DataFrame with historical games
        league: 'nfl' or 'nba'
    
    Returns:
        Opponent strength metric (higher = stronger opponents faced)
    """
    # Determine season to use
    season_val = season or game_date.year
    
    if 'season' in historical_games.columns:
        season_mask = pd.to_numeric(historical_games['season'], errors='coerce') == season_val
    else:
        season_mask = pd.to_datetime(historical_games['game_date']).dt.year == season_val
    
    game_dates = pd.to_datetime(historical_games['game_date'])
    
    # Get all games before this date for the same season
    season_games = historical_games[
        season_mask & (game_dates < game_date)
    ]
    
    # Find games where team played
    team_games = season_games[
        (season_games['home_team'] == team) | (season_games['away_team'] == team)
    ]
    
    if team_games.empty:
        return None
    
    # Get opponents
    opponents = []
    for _, game in team_games.iterrows():
        if game['home_team'] == team:
            opponents.append(game['away_team'])
        else:
            opponents.append(game['home_team'])
    
    # Compute average opponent strength
    # Simple approach: average point differential of opponents
    opponent_strengths = []
    
    for opp in set(opponents):
        opp_games = season_games[
            ((season_games['home_team'] == opp) | (season_games['away_team'] == opp))
        ]
        
        if opp_games.empty:
            continue
        
        # Compute average point differential
        if 'home_score' in opp_games.columns and 'away_score' in opp_games.columns:
            opp_games = opp_games.copy()
            opp_games['point_diff'] = np.where(
                opp_games['home_team'] == opp,
                opp_games['home_score'] - opp_games['away_score'],
                opp_games['away_score'] - opp_games['home_score']
            )
            avg_diff = opp_games['point_diff'].mean()
            opponent_strengths.append(avg_diff)
    
    if not opponent_strengths:
        return None
    
    return np.mean(opponent_strengths)


def add_opponent_strength_features(games_df: pd.DataFrame, 
                                   historical_games: pd.DataFrame,
                                   league: str = 'nfl') -> pd.DataFrame:
    """
    Add opponent strength features to games DataFrame.
    
    Args:
        games_df: DataFrame with games
        historical_games: Historical games for computing strength
        league: 'nfl' or 'nba'
    
    Returns:
        DataFrame with added columns: opp_strength_home_season, opp_strength_away_season
    """
    df = games_df.copy()
    
    def _season_for_row(row):
        season_val = row.get('season')
        if pd.notna(season_val):
            try:
                return int(season_val)
            except (TypeError, ValueError):
                return pd.to_datetime(row['game_date']).year
        return pd.to_datetime(row['game_date']).year
    
    df['opp_strength_home_season'] = df.apply(
        lambda row: compute_season_opponent_strength(
            row['home_team'],
            pd.to_datetime(row['game_date']),
            historical_games,
            league,
            season=_season_for_row(row)
        ),
        axis=1
    )
    
    df['opp_strength_away_season'] = df.apply(
        lambda row: compute_season_opponent_strength(
            row['away_team'],
            pd.to_datetime(row['game_date']),
            historical_games,
            league,
            season=_season_for_row(row)
        ),
        axis=1
    )
    
    return df
