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
    Compute rolling net rating for NBA team.
    
    Net rating = Average point differential per game over the window.
    This is a proxy for true net rating (points per 100 possessions).
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        game_logs: DataFrame with game logs (must have 'game_date', 'team', 'points_scored', 'points_allowed')
        window: Number of games to look back
    
    Returns:
        Rolling net rating (average point differential) or None
    """
    team_games = game_logs[
        (game_logs['team'] == team) &
        (pd.to_datetime(game_logs['game_date']) < game_date)
    ].sort_values('game_date')
    
    if len(team_games) < window:
        return None
    
    recent_games = team_games.tail(window)
    
    # Try to use pre-computed net_rating
    if 'net_rating' in recent_games.columns:
        net_rating = pd.to_numeric(recent_games['net_rating'], errors='coerce')
        valid_rating = net_rating[net_rating.notna()]
        if len(valid_rating) >= window:
            return valid_rating.mean()
    
    # Compute from point differential
    if 'point_diff' in recent_games.columns:
        point_diff = pd.to_numeric(recent_games['point_diff'], errors='coerce')
        valid_diff = point_diff[point_diff.notna()]
        if len(valid_diff) >= window:
            return valid_diff.mean()
    
    # Compute from points scored/allowed
    if 'points_scored' in recent_games.columns and 'points_allowed' in recent_games.columns:
        points_scored = pd.to_numeric(recent_games['points_scored'], errors='coerce')
        points_allowed = pd.to_numeric(recent_games['points_allowed'], errors='coerce')
        
        # Need both to be valid
        valid_mask = points_scored.notna() & points_allowed.notna()
        if valid_mask.sum() >= window:
            point_diff = points_scored[valid_mask] - points_allowed[valid_mask]
            return point_diff.mean()
    
    return None


def compute_rolling_offensive_rating(team: str, game_date: pd.Timestamp,
                                     game_logs: pd.DataFrame, window: int = 3) -> Optional[float]:
    """
    Compute rolling offensive rating (points scored per game) for NBA team.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        game_logs: DataFrame with game logs
        window: Number of games to look back
    
    Returns:
        Average points scored per game over window
    """
    team_games = game_logs[
        (game_logs['team'] == team) &
        (pd.to_datetime(game_logs['game_date']) < game_date)
    ].sort_values('game_date')
    
    if len(team_games) < window:
        return None
    
    recent_games = team_games.tail(window)
    
    # Try points_scored column
    if 'points_scored' in recent_games.columns:
        pts = pd.to_numeric(recent_games['points_scored'], errors='coerce')
        valid_pts = pts[pts.notna()]
        if len(valid_pts) >= window:  # Need at least window games with valid data
            return valid_pts.mean()
    
    # Fallback to PTS column
    if 'PTS' in recent_games.columns:
        pts = pd.to_numeric(recent_games['PTS'], errors='coerce')
        valid_pts = pts[pts.notna()]
        if len(valid_pts) >= window:
            return valid_pts.mean()
    
    return None


def compute_rolling_defensive_rating(team: str, game_date: pd.Timestamp,
                                     game_logs: pd.DataFrame, window: int = 3) -> Optional[float]:
    """
    Compute rolling defensive rating (points allowed per game) for NBA team.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
        game_logs: DataFrame with game logs (must have 'points_allowed' or opponent points)
        window: Number of games to look back
    
    Returns:
        Average points allowed per game over window
    """
    team_games = game_logs[
        (game_logs['team'] == team) &
        (pd.to_datetime(game_logs['game_date']) < game_date)
    ].sort_values('game_date')
    
    if len(team_games) < window:
        return None
    
    recent_games = team_games.tail(window)
    
    # Try points_allowed column (preferred - computed from schedule)
    if 'points_allowed' in recent_games.columns:
        pts = pd.to_numeric(recent_games['points_allowed'], errors='coerce')
        valid_pts = pts[pts.notna()]
        if len(valid_pts) >= window:  # Need at least window games with valid data
            return valid_pts.mean()
    
    # Fallback to OPP_PTS column (if available in raw game logs)
    if 'OPP_PTS' in recent_games.columns:
        pts = pd.to_numeric(recent_games['OPP_PTS'], errors='coerce')
        valid_pts = pts[pts.notna()]
        if len(valid_pts) >= window:
            return valid_pts.mean()
    
    # If we have points_scored and can compute from schedule, try that
    # (This would require schedule data, which we don't have here)
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
    
    Computes rolling averages for:
    - Net rating (point differential)
    - Offensive rating (points scored)
    - Defensive rating (points allowed)
    
    Args:
        games_df: DataFrame with games
        game_logs: DataFrame with game logs (must have 'team', 'game_date', 'points_scored', 'points_allowed')
        window: Rolling window size (3, 5, or 10 games)
    
    Returns:
        DataFrame with added columns:
        - form_home_net_rating_{window}
        - form_away_net_rating_{window}
        - form_home_off_rating_{window}
        - form_away_off_rating_{window}
        - form_home_def_rating_{window}
        - form_away_def_rating_{window}
    """
    df = games_df.copy()
    
    # Net rating (point differential)
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
    
    # Offensive rating (points scored)
    df[f'form_home_off_rating_{window}'] = df.apply(
        lambda row: compute_rolling_offensive_rating(
            row['home_team'], pd.to_datetime(row['game_date']), game_logs, window
        ),
        axis=1
    )
    
    df[f'form_away_off_rating_{window}'] = df.apply(
        lambda row: compute_rolling_offensive_rating(
            row['away_team'], pd.to_datetime(row['game_date']), game_logs, window
        ),
        axis=1
    )
    
    # Defensive rating (points allowed)
    df[f'form_home_def_rating_{window}'] = df.apply(
        lambda row: compute_rolling_defensive_rating(
            row['home_team'], pd.to_datetime(row['game_date']), game_logs, window
        ),
        axis=1
    )
    
    df[f'form_away_def_rating_{window}'] = df.apply(
        lambda row: compute_rolling_defensive_rating(
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

