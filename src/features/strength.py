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
                pd.to_numeric(opp_games['home_score'], errors='coerce') - pd.to_numeric(opp_games['away_score'], errors='coerce'),
                pd.to_numeric(opp_games['away_score'], errors='coerce') - pd.to_numeric(opp_games['home_score'], errors='coerce')
            )
            avg_diff = opp_games['point_diff'].dropna().mean()
            if pd.notna(avg_diff):
                opponent_strengths.append(avg_diff)
    
    if not opponent_strengths:
        return 0.0
    
    return np.mean(opponent_strengths)


def add_opponent_strength_features(games_df: pd.DataFrame, 
                                   historical_games: pd.DataFrame,
                                   league: str = 'nfl') -> pd.DataFrame:
    """
    Add opponent strength features to games DataFrame efficiently.
    """
    import numpy as np
    
    df = games_df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    hist = historical_games.copy()
    hist['game_date'] = pd.to_datetime(hist['game_date'])
    
    # Ensure scores are numeric
    hist['home_score'] = pd.to_numeric(hist['home_score'], errors='coerce')
    hist['away_score'] = pd.to_numeric(hist['away_score'], errors='coerce')
    hist['home_diff'] = hist['home_score'] - hist['away_score']
    hist['away_diff'] = hist['away_score'] - hist['home_score']
    
    # 1. Pre-calculate seasonal averages for each team at each game date
    # Melt history to get per-team game outcomes
    h_stats = hist[['game_date', 'season', 'home_team', 'home_diff', 'away_team']].rename(
        columns={'home_team': 'team', 'home_diff': 'diff', 'away_team': 'opponent'}
    )
    a_stats = hist[['game_date', 'season', 'away_team', 'away_diff', 'home_team']].rename(
        columns={'away_team': 'team', 'away_diff': 'diff', 'home_team': 'opponent'}
    )
    t_stats = pd.concat([h_stats, a_stats]).sort_values(['team', 'game_date'])
    
    # Compute cumulative average point diff per team per season
    t_stats['cum_diff_sum'] = t_stats.groupby(['team', 'season'])['diff'].cumsum()
    t_stats['cum_games'] = t_stats.groupby(['team', 'season']).cumcount() + 1
    t_stats['prev_avg_diff'] = t_stats.groupby(['team', 'season'])['cum_diff_sum'].shift(1) / t_stats.groupby(['team', 'season'])['cum_games'].shift(1)
    
    # 2. For each game, find the average 'prev_avg_diff' of the opponents played so far
    # Join t_stats with itself to get the opponent's strength at that time
    # We want to know: for team T's game on date D, what was the average 'prev_avg_diff' of all opponents T played before D?
    
    # Get opponent strengths at the time they played
    t_stats = pd.merge(
        t_stats,
        t_stats[['team', 'game_date', 'prev_avg_diff']].rename(columns={'team': 'opponent', 'prev_avg_diff': 'opp_prev_avg_diff'}),
        on=['opponent', 'game_date'],
        how='left'
    )
    
    # Compute rolling average of opponent's strength
    t_stats['cum_opp_strength_sum'] = t_stats.groupby(['team', 'season'])['opp_prev_avg_diff'].cumsum()
    t_stats['opp_strength_season'] = t_stats.groupby(['team', 'season'])['cum_opp_strength_sum'].shift(1) / t_stats.groupby(['team', 'season'])['cum_games'].shift(1)
    
    # 3. Merge back to the games_df
    lookup = t_stats[['team', 'game_date', 'opp_strength_season']].dropna(subset=['opp_strength_season'])
    
    # Merge for home team
    df = pd.merge(df, lookup.rename(columns={'team': 'home_team', 'opp_strength_season': 'opp_strength_home_season'}), 
                  on=['home_team', 'game_date'], how='left')
    
    # Merge for away team
    df = pd.merge(df, lookup.rename(columns={'team': 'away_team', 'opp_strength_season': 'opp_strength_away_season'}), 
                  on=['away_team', 'game_date'], how='left')
    
    return df
