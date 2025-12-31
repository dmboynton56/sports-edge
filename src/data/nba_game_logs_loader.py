"""
NBA game logs loader.
Loads team game logs for computing form metrics (net rating, etc.).
Similar to pbp_loader.py for NFL.
"""

import pandas as pd
import numpy as np
from typing import Optional, Sequence, List
import time
import warnings
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams

# Suppress pandas date format inference warnings for NBA game logs
warnings.filterwarnings('ignore', category=UserWarning, 
                       message='.*Could not infer format.*')


def _normalize_game_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a game_date column exists and is datetime."""
    if 'game_date' in df.columns:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    elif 'GAME_DATE' in df.columns:
        # NBA API TeamGameLog returns dates as "APR 13, 2025" format (e.g., "%b %d, %Y")
        # Suppress the format inference warning since we're explicitly providing the format
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # Use exact format - NBA API consistently uses "MON DD, YYYY" format
            # Convert to string first to ensure consistent handling, replace 'nan' strings
            date_series = df['GAME_DATE'].astype(str).replace('nan', pd.NA)
            df['game_date'] = pd.to_datetime(
                date_series, 
                format='%b %d, %Y', 
                errors='coerce'
            )
    return df


def _compute_net_rating_from_logs(game_logs: pd.DataFrame, schedule_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute net rating, offensive rating, and defensive rating from game logs.
    
    Net Rating = Offensive Rating - Defensive Rating
    Offensive Rating = Points scored per 100 possessions
    Defensive Rating = Points allowed per 100 possessions
    
    Args:
        game_logs: DataFrame with game logs (must have PTS, OPP_PTS or similar)
    
    Returns:
        DataFrame with added rating columns
    """
    df = game_logs.copy()
    
    # Try to find points columns
    pts_col = None
    opp_pts_col = None
    
    for col in df.columns:
        col_upper = col.upper()
        if 'PTS' in col_upper and 'OPP' not in col_upper and 'AGAINST' not in col_upper:
            pts_col = col
        elif 'OPP_PTS' in col_upper or 'PTS_AGAINST' in col_upper:
            opp_pts_col = col
    
    # Get points scored
    if pts_col:
        df['points_scored'] = pd.to_numeric(df[pts_col], errors='coerce')
    else:
        df['points_scored'] = None
    
    # Try to get opponent points from game logs
    if opp_pts_col:
        df['points_allowed'] = pd.to_numeric(df[opp_pts_col], errors='coerce')
    elif schedule_df is not None:
        # Merge with schedule to get opponent scores
        # Use game_date + team to match (more reliable than game_id which may have format differences)
        df['points_allowed'] = None
        
        # Normalize schedule dates
        schedule_df_copy = schedule_df.copy()
        if 'game_date' in schedule_df_copy.columns:
            schedule_df_copy['game_date'] = pd.to_datetime(schedule_df_copy['game_date']).dt.date
        else:
            schedule_df_copy['game_date'] = None
        
        # Normalize log dates
        if 'game_date' in df.columns:
            df['game_date_normalized'] = pd.to_datetime(df['game_date']).dt.date
        else:
            df['game_date_normalized'] = None
        
        # Create lookup by (game_date, team) -> opponent score
        matched_count = 0
        for idx, log_row in df.iterrows():
            log_date = log_row.get('game_date_normalized')
            team = log_row.get('team')
            
            if pd.isna(log_date) or not team:
                continue
            
            # Find matching game in schedule
            matching_games = schedule_df_copy[
                (schedule_df_copy['game_date'] == log_date) &
                ((schedule_df_copy['home_team'] == team) | (schedule_df_copy['away_team'] == team))
            ]
            
            if len(matching_games) > 0:
                game = matching_games.iloc[0]  # Take first match
                home_score = pd.to_numeric(game.get('home_score'), errors='coerce')
                away_score = pd.to_numeric(game.get('away_score'), errors='coerce')
                
                if pd.notna(home_score) and pd.notna(away_score):
                    if game['home_team'] == team:
                        df.loc[idx, 'points_allowed'] = away_score
                    else:
                        df.loc[idx, 'points_allowed'] = home_score
                    matched_count += 1
        
        if len(df) > 0:
            pct_matched = (matched_count / len(df)) * 100 if len(df) > 0 else 0
            if matched_count > 0:
                print(f"    Matched {matched_count}/{len(df)} ({pct_matched:.1f}%) game log entries with schedule")
            else:
                print(f"    Warning: Could not match game logs with schedule using date+team matching")
    else:
        df['points_allowed'] = None
    
    # Compute point differential (simplified net rating)
    if df['points_scored'].notna().any() and df['points_allowed'].notna().any():
        df['point_diff'] = df['points_scored'] - df['points_allowed']
        df['net_rating'] = df['point_diff']  # Net rating = point differential per game
    else:
        df['point_diff'] = None
        df['net_rating'] = None
    
    # If we have pace data, compute true ratings
    if 'PACE' in df.columns or 'PACE_PER_40' in df.columns:
        pace_col = 'PACE' if 'PACE' in df.columns else 'PACE_PER_40'
        # Net rating = (Points scored - Points allowed) per 100 possessions
        # Simplified: point diff per game * (100 / avg pace)
        # For now, use point diff as proxy
        pass
    
    return df


def load_nba_game_logs(seasons: Sequence[int], strict: bool = False, schedule_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Load NBA team game logs for the requested seasons.
    
    Args:
        seasons: Iterable of season years (e.g., [2024, 2025])
        strict: If True, raise if data cannot be loaded. Otherwise return None.
    
    Returns:
        DataFrame with game logs including: game_id, game_date, team, points_scored, 
        points_allowed, net_rating, etc.
    """
    seasons = list(seasons)
    all_logs = []
    
    try:
        # Get all NBA teams
        nba_teams = teams.get_teams()
        team_ids = [team['id'] for team in nba_teams]
        
        print(f"Loading game logs for {len(team_ids)} teams across {len(seasons)} seasons...")
        
        for season in seasons:
            season_str = f"{season}-{str(season + 1)[-2:]}"
            print(f"  Processing {season_str}...")
            
            for i, team_id in enumerate(team_ids):
                try:
                    # Fetch team game log
                    team_log = teamgamelog.TeamGameLog(team_id=team_id, season=season_str)
                    df = team_log.get_data_frames()[0]
                    
                    if df.empty:
                        continue
                    
                    # Add team identifier
                    team_info = next((t for t in nba_teams if t['id'] == team_id), None)
                    if team_info:
                        df['team'] = team_info['abbreviation']
                        df['team_id'] = team_id
                    else:
                        df['team'] = None
                        df['team_id'] = team_id
                    
                    # Add season
                    df['season'] = season
                    
                    # Normalize game date (suppress warnings during date parsing)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning)
                        df = _normalize_game_dates(df)
                    
                    # Don't compute ratings yet - we'll do it after combining all logs
                    # so we can match with schedule more efficiently
                    
                    all_logs.append(df)
                    
                    # Rate limiting - be nice to the API
                    if (i + 1) % 10 == 0:
                        time.sleep(0.5)  # Small delay every 10 teams
                    
                except Exception as e:
                    print(f"    Warning: Could not load logs for team {team_id} ({season_str}): {e}")
                    continue
            
            # Longer delay between seasons
            time.sleep(1)
        
        if not all_logs:
            message = "No game logs were loaded"
            if strict:
                raise RuntimeError(message)
            print(f"WARNING: {message}")
            return None
        
        # Combine all logs
        combined_df = pd.concat(all_logs, ignore_index=True)
        
        # Standardize columns
        standard_cols = ['game_id', 'game_date', 'team', 'team_id', 'season']
        
        # Map common column names
        col_mapping = {
            'Game_ID': 'game_id',
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'MATCHUP': 'matchup',
            'WL': 'win_loss',
            'PTS': 'points_scored',
            'OPP_PTS': 'points_allowed',
        }
        
        for old_col, new_col in col_mapping.items():
            if old_col in combined_df.columns and new_col not in combined_df.columns:
                combined_df[new_col] = combined_df[old_col]
        
        # Ensure game_id exists and is string format
        if 'game_id' not in combined_df.columns:
            if 'Game_ID' in combined_df.columns:
                combined_df['game_id'] = combined_df['Game_ID'].astype(str)
            elif 'GAME_ID' in combined_df.columns:
                combined_df['game_id'] = combined_df['GAME_ID'].astype(str)
        
        # Convert game_id to string for matching
        if 'game_id' in combined_df.columns:
            combined_df['game_id'] = combined_df['game_id'].astype(str)
        
        # Now compute ratings with schedule lookup (if available)
        # This matches game logs with schedule to get opponent points
        print("  Computing ratings from game logs (matching with schedule for opponent points)...")
        combined_df = _compute_net_rating_from_logs(combined_df, schedule_df=schedule_df)
        
        # Sort by date
        combined_df = combined_df.sort_values('game_date').reset_index(drop=True)
        
        print(f"Loaded {len(combined_df)} game log entries")
        return combined_df
        
    except Exception as e:
        message = f"Unable to load NBA game logs ({e})"
        if strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        import traceback
        traceback.print_exc()
        return None

