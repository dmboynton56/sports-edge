"""
Form metrics feature engineering.
Computes rolling averages for team performance (net rating for NBA, EPA for NFL).
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_rolling_net_rating(team: str, game_date: pd.Timestamp, 
                               game_logs: pd.DataFrame, window: int = 3) -> Optional[float]:
    """
    Compute rolling net rating (points per 100 possessions) for NBA team.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        game_logs: DataFrame with game logs (must have 'game_date', 'team', 'off_rating', 'def_rating')
        window: Number of games to look back
    
    Returns:
        Rolling net rating (off_rating - def_rating) or None
    """
    team_games = game_logs[
        (game_logs['team'] == team) &
        (pd.to_datetime(game_logs['game_date']) < game_date)
    ].sort_values('game_date')
    
    if len(team_games) < window:
        return None
    
    recent_games = team_games.tail(window)
    
    if 'net_rating' in recent_games.columns:
        return recent_games['net_rating'].mean()
    elif 'off_rating' in recent_games.columns and 'def_rating' in recent_games.columns:
        net_rating = recent_games['off_rating'] - recent_games['def_rating']
        return net_rating.mean()
    
    return None


def compute_rolling_epa(team: str, game_date: pd.Timestamp, 
                        play_by_play: pd.DataFrame, window: int = 3,
                        side: str = 'offense') -> Optional[float]:
    """
    Compute rolling EPA (Expected Points Added) for NFL team.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        play_by_play: DataFrame with play-by-play data (must have 'posteam', 'game_date', 'epa')
        window: Number of games to look back
        side: 'offense' or 'defense'
    
    Returns:
        Rolling EPA per play or None
    """
    if side == 'offense':
        team_games = play_by_play[
            (play_by_play['posteam'] == team) &
            (pd.to_datetime(play_by_play['game_date']) < game_date)
        ]
    else:  # defense
        team_games = play_by_play[
            (play_by_play['defteam'] == team) &
            (pd.to_datetime(play_by_play['game_date']) < game_date)
        ]
    
    if team_games.empty:
        return None
    
    # Group by game and compute average EPA, keeping game_date for sorting
    game_epa = team_games.groupby('game_id').agg({
        'epa': 'mean',
        'game_date': 'first'  # Get first game_date for each game_id
    }).reset_index()
    game_epa['game_date'] = pd.to_datetime(game_epa['game_date'])
    game_epa = game_epa.sort_values('game_date')
    
    if len(game_epa) < window:
        return None
    
    recent_epa = game_epa.tail(window)['epa'].mean()
    return recent_epa


def compute_rolling_success_rate(team: str, game_date: pd.Timestamp,
                                play_by_play: pd.DataFrame, window: int = 3) -> Optional[float]:
    """
    Compute rolling success rate for NFL team.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        play_by_play: DataFrame with play-by-play data (must have 'posteam', 'success')
        window: Number of games to look back
    
    Returns:
        Rolling success rate (0-1) or None
    """
    team_games = play_by_play[
        (play_by_play['posteam'] == team) &
        (pd.to_datetime(play_by_play['game_date']) < game_date)
    ]
    
    if team_games.empty:
        return None
    
    # Group by game, keeping game_date for sorting
    game_success = team_games.groupby('game_id').agg({
        'success': 'mean',
        'game_date': 'first'  # Get first game_date for each game_id
    }).reset_index()
    game_success['game_date'] = pd.to_datetime(game_success['game_date'])
    game_success = game_success.sort_values('game_date')
    
    if len(game_success) < window:
        return None
    
    recent_success = game_success.tail(window)['success'].mean()
    return recent_success


def add_form_features_nba(games_df: pd.DataFrame, game_logs: pd.DataFrame, 
                         window: int = 3) -> pd.DataFrame:
    """
    Add form features for NBA games.
    
    Args:
        games_df: DataFrame with games
        game_logs: DataFrame with game logs
        window: Rolling window size
    
    Returns:
        DataFrame with added columns: form_home_net_rating_3, form_away_net_rating_3, etc.
    """
    df = games_df.copy()
    
    df[f'form_home_net_rating_{window}'] = df.apply(
        lambda row: compute_rolling_net_rating(
            row['home_team'], pd.to_datetime(row['game_date']), game_logs, window
        ),
        axis=1
    )
    
    df[f'form_away_net_rating_{window}'] = df.apply(
        lambda row: compute_rolling_net_rating(
            row['away_team'], pd.to_datetime(row['game_date']), game_logs, window
        ),
        axis=1
    )
    
    return df


def add_form_features_nfl(games_df: pd.DataFrame, play_by_play: pd.DataFrame,
                          window: int = 3) -> pd.DataFrame:
    """
    Add form features for NFL games.
    
    Args:
        games_df: DataFrame with games
        play_by_play: DataFrame with play-by-play data
        window: Rolling window size
    
    Returns:
        DataFrame with added columns: form_home_epa_off_3, form_away_epa_off_3, etc.
    """
    df = games_df.copy()
    
    df[f'form_home_epa_off_{window}'] = df.apply(
        lambda row: compute_rolling_epa(
            row['home_team'], pd.to_datetime(row['game_date']), play_by_play, window, 'offense'
        ),
        axis=1
    )
    
    df[f'form_away_epa_off_{window}'] = df.apply(
        lambda row: compute_rolling_epa(
            row['away_team'], pd.to_datetime(row['game_date']), play_by_play, window, 'offense'
        ),
        axis=1
    )
    
    df[f'form_home_epa_def_{window}'] = df.apply(
        lambda row: compute_rolling_epa(
            row['home_team'], pd.to_datetime(row['game_date']), play_by_play, window, 'defense'
        ),
        axis=1
    )
    
    df[f'form_away_epa_def_{window}'] = df.apply(
        lambda row: compute_rolling_epa(
            row['away_team'], pd.to_datetime(row['game_date']), play_by_play, window, 'defense'
        ),
        axis=1
    )
    
    return df

