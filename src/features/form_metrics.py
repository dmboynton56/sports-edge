"""
Form metrics feature engineering.
Computes rolling averages for team performance (net rating for NBA, EPA for NFL).
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_possessions(df: pd.DataFrame) -> pd.Series:
    """
    Estimate possessions for a set of games.
    Formula: FGA + 0.44 * FTA - OREB + TOV
    """
    fga = pd.to_numeric(df.get('FGA', 0), errors='coerce').fillna(0)
    fta = pd.to_numeric(df.get('FTA', 0), errors='coerce').fillna(0)
    oreb = pd.to_numeric(df.get('OREB', 0), errors='coerce').fillna(0)
    tov = pd.to_numeric(df.get('TOV', 0), errors='coerce').fillna(0)
    
    return fga + (0.44 * fta) - oreb + tov


def compute_rolling_net_rating(team: str, game_date: pd.Timestamp, 
                               game_logs: pd.DataFrame, window: int = 3) -> Optional[float]:
    """
    Compute rolling net rating for NBA team.
    
    If box score stats are available, computes True Net Rating (Points per 100 possessions).
    Otherwise falls back to average point differential.
    
    Args:
        team: Team abbreviation
        game_date: Date of current game
    """
    team_games = game_logs[
        (game_logs['team'] == team) &
        (pd.to_datetime(game_logs['game_date']) < game_date)
    ].sort_values('game_date')
    
    if len(team_games) < window:
        return None
    
    recent_games = team_games.tail(window).copy()
    
    # Check for box score stats to compute pace-adjusted net rating
    box_cols = ['FGA', 'FTA', 'TOV', 'OREB']
    has_box = all(col in recent_games.columns for col in box_cols)
    
    if has_box:
        poss = compute_possessions(recent_games)
        pts_scored = pd.to_numeric(recent_games.get('pts', recent_games.get('PTS', 0)), errors='coerce').fillna(0)
        pts_allowed = pd.to_numeric(recent_games.get('points_allowed', 0), errors='coerce').fillna(0)
        
        # Calculate totals over window to avoid division by zero in single games
        total_poss = poss.sum()
        if total_poss > 0:
            off_rating = (pts_scored.sum() / total_poss) * 100
            def_rating = (pts_allowed.sum() / total_poss) * 100
            return off_rating - def_rating

    # Fallback to pre-computed or simple point differential
    for col in ['net_rating', 'point_diff', 'PLUS_MINUS']:
        if col in recent_games.columns:
            vals = pd.to_numeric(recent_games[col], errors='coerce').dropna()
            if len(vals) >= window:
                return vals.mean()
    
    return None


def compute_rolling_offensive_rating(team: str, game_date: pd.Timestamp,
                                     game_logs: pd.DataFrame, window: int = 3) -> Optional[float]:
    """
    Compute rolling offensive rating (Points per 100 possessions) for NBA team.
    """
    team_games = game_logs[
        (game_logs['team'] == team) &
        (pd.to_datetime(game_logs['game_date']) < game_date)
    ].sort_values('game_date')
    
    if len(team_games) < window:
        return None
    
    recent_games = team_games.tail(window).copy()
    
    # Try pace-adjusted
    if all(col in recent_games.columns for col in ['FGA', 'FTA', 'TOV', 'OREB']):
        poss = compute_possessions(recent_games)
        pts = pd.to_numeric(recent_games.get('pts', recent_games.get('PTS', 0)), errors='coerce').fillna(0)
        if poss.sum() > 0:
            return (pts.sum() / poss.sum()) * 100
            
    # Fallback to points per game
    for col in ['points_scored', 'PTS']:
        if col in recent_games.columns:
            vals = pd.to_numeric(recent_games[col], errors='coerce').dropna()
            if len(vals) >= window:
                return vals.mean()
    
    return None


def compute_rolling_defensive_rating(team: str, game_date: pd.Timestamp,
                                     game_logs: pd.DataFrame, window: int = 3) -> Optional[float]:
    """
    Compute rolling defensive rating (Points allowed per 100 possessions) for NBA team.
    """
    team_games = game_logs[
        (game_logs['team'] == team) &
        (pd.to_datetime(game_logs['game_date']) < game_date)
    ].sort_values('game_date')
    
    if len(team_games) < window:
        return None
    
    recent_games = team_games.tail(window).copy()
    
    # Try pace-adjusted
    if all(col in recent_games.columns for col in ['FGA', 'FTA', 'TOV', 'OREB']):
        poss = compute_possessions(recent_games)
        pts_allowed = pd.to_numeric(recent_games.get('points_allowed', 0), errors='coerce').fillna(0)
        if poss.sum() > 0:
            return (pts_allowed.sum() / poss.sum()) * 100

    # Fallback to points allowed per game
    for col in ['points_allowed', 'OPP_PTS']:
        if col in recent_games.columns:
            vals = pd.to_numeric(recent_games[col], errors='coerce').dropna()
            if len(vals) >= window:
                return vals.mean()
    
    return None


def compute_rolling_pace(team: str, game_date: pd.Timestamp,
                         game_logs: pd.DataFrame, window: int = 3) -> Optional[float]:
    """
    Compute rolling pace (Possessions per 48 minutes).
    """
    team_games = game_logs[
        (game_logs['team'] == team) &
        (pd.to_datetime(game_logs['game_date']) < game_date)
    ].sort_values('game_date')
    
    if len(team_games) < window:
        return None
    
    recent_games = team_games.tail(window).copy()
    
    if all(col in recent_games.columns for col in ['FGA', 'FTA', 'TOV', 'OREB']):
        poss = compute_possessions(recent_games)
        # Assuming 48 mins unless 'MIN' is available and different
        return poss.mean()  # Simplified pace
        
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
    Add form features for NBA games efficiently.
    """
    df = games_df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    logs = game_logs.copy()
    logs['game_date'] = pd.to_datetime(logs['game_date'])
    
    # Ensure necessary columns are numeric and exist
    box_cols = ['FGA', 'FTA', 'TOV', 'OREB']
    for col in box_cols:
        logs[col] = pd.to_numeric(logs.get(col, 0), errors='coerce').fillna(0)
    
    logs['pts_scored'] = pd.to_numeric(logs.get('pts', logs.get('PTS', 0)), errors='coerce').fillna(0)
    logs['pts_allowed'] = pd.to_numeric(logs.get('points_allowed', 0), errors='coerce').fillna(0)
    
    # Calculate possessions per game
    logs['possessions'] = logs['FGA'] + (0.44 * logs['FTA']) - logs['OREB'] + logs['TOV']
    
    # Sort logs for rolling calculations
    logs = logs.sort_values(['team', 'game_date'])
    
    # Group by team to calculate rolling sums
    team_gb = logs.groupby('team')
    
    # Calculate rolling totals over the window
    rolling_totals = team_gb[['possessions', 'pts_scored', 'pts_allowed']].rolling(window=window).sum().reset_index(level=0, drop=True)
    
    # Calculate ratings (shifted by 1 to get stats BEFORE the current game)
    logs['roll_poss'] = rolling_totals['possessions']
    logs['roll_pts_scored'] = rolling_totals['pts_scored']
    logs['roll_pts_allowed'] = rolling_totals['pts_allowed']
    
    # Shift per team
    logs['prev_roll_poss'] = team_gb['roll_poss'].shift(1)
    logs['prev_roll_pts_scored'] = team_gb['roll_pts_scored'].shift(1)
    logs['prev_roll_pts_allowed'] = team_gb['roll_pts_allowed'].shift(1)
    
    # Compute the final metrics
    logs[f'form_off_rating_{window}'] = (logs['prev_roll_pts_scored'] / logs['prev_roll_poss'].replace(0, np.nan)) * 100
    logs[f'form_def_rating_{window}'] = (logs['prev_roll_pts_allowed'] / logs['prev_roll_poss'].replace(0, np.nan)) * 100
    logs[f'form_net_rating_{window}'] = logs[f'form_off_rating_{window}'] - logs[f'form_def_rating_{window}']
    
    # NEW: Form Volatility (Std Dev of point differential)
    # This helps identify "swingy" teams like WAS or SAC
    logs['actual_pts_diff'] = logs['pts_scored'] - logs['pts_allowed']
    logs[f'form_volatility_{window}'] = team_gb['actual_pts_diff'].transform(lambda x: x.rolling(window=window).std().shift(1))
    
    # Handle cases where box score stats are missing (fallback to point diff)
    # If off_rating is still NaN after the shift, use simple point differential if available
    fallback_col = 'net_rating' if 'net_rating' in logs.columns else 'PLUS_MINUS'
    if fallback_col in logs.columns:
        logs[fallback_col] = pd.to_numeric(logs[fallback_col], errors='coerce')
        logs[f'form_net_rating_fallback_{window}'] = team_gb[fallback_col].transform(lambda x: x.rolling(window=window).mean().shift(1))
        logs[f'form_net_rating_{window}'] = logs[f'form_net_rating_{window}'].fillna(logs[f'form_net_rating_fallback_{window}'])

    # Create lookup table
    lookup_cols = ['team', 'game_date', f'form_net_rating_{window}', f'form_off_rating_{window}', f'form_def_rating_{window}', f'form_volatility_{window}']
    lookup = logs[lookup_cols].dropna(subset=[f'form_net_rating_{window}'])
    
    # Merge back for home team
    df = pd.merge(df, lookup.rename(columns={
        'team': 'home_team', 
        f'form_net_rating_{window}': f'form_home_net_rating_{window}',
        f'form_off_rating_{window}': f'form_home_off_rating_{window}',
        f'form_def_rating_{window}': f'form_home_def_rating_{window}',
        f'form_volatility_{window}': f'form_home_volatility_{window}'
    }), on=['home_team', 'game_date'], how='left')
    
    # Merge back for away team
    df = pd.merge(df, lookup.rename(columns={
        'team': 'away_team', 
        f'form_net_rating_{window}': f'form_away_net_rating_{window}',
        f'form_off_rating_{window}': f'form_away_off_rating_{window}',
        f'form_def_rating_{window}': f'form_away_def_rating_{window}',
        f'form_volatility_{window}': f'form_away_volatility_{window}'
    }), on=['away_team', 'game_date'], how='left')
    
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

