"""
NBA game logs loader.
Loads team game logs for computing form metrics (net rating, etc.).
Similar to pbp_loader.py for NFL.
"""

import pandas as pd
import numpy as np
from typing import Optional, Sequence, List, Dict, Any
import time
import warnings
from nba_api.stats.endpoints import teamgamelog, leaguegamefinder
from nba_api.stats.static import teams
from google.cloud import bigquery
import os
from dotenv import load_dotenv
import inspect
from datetime import datetime, date

# Load environment variables
load_dotenv()

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
        # NBA API has two common formats:
        # 1. "APR 13, 2025" (TeamGameLog)
        # 2. "2025-04-13" (LeagueGameFinder)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            # Try flexible parsing first, then fallback to specific format if needed
            df['game_date'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            
            # If everything is still null, try the specific legacy format
            if df['game_date'].isna().all():
                date_series = df['GAME_DATE'].astype(str).replace('nan', pd.NA)
                df['game_date'] = pd.to_datetime(
                    date_series, 
                    format='%b %d, %Y', 
                    errors='coerce'
                )
    return df


def _supports_timeout(endpoint) -> bool:
    try:
        return "timeout" in inspect.signature(endpoint).parameters
    except (ValueError, TypeError):
        return False


def _filter_kwargs(endpoint, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = inspect.signature(endpoint).parameters
    except (ValueError, TypeError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _format_nba_date(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%m/%d/%Y")
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").strftime("%m/%d/%Y")
        except ValueError:
            return value
    return str(value)


def _retry_nba_request(fn, label: str, retries: int, base_delay: int):
    attempts = max(1, retries)
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= attempts:
                break
            sleep_for = base_delay * (2 ** (attempt - 1))
            print(f"Warning: {label} failed (attempt {attempt}/{attempts}): {exc}. Retrying in {sleep_for}s...")
            time.sleep(sleep_for)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{label} failed without an exception")


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


def load_nba_game_logs_from_bq(seasons: Sequence[int], project_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load NBA game logs from BigQuery instead of the API.
    
    Args:
        seasons: List of seasons to load
        project_id: GCP Project ID
        
    Returns:
        DataFrame with game logs
    """
    project_id = project_id or os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("WARNING: GCP_PROJECT_ID not found. Cannot load from BigQuery.")
        return None
        
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.sports_edge_raw.raw_nba_game_logs"
    
    season_list = ", ".join(map(str, seasons))
    query = f"""
        SELECT * 
        FROM `{table_id}` 
        WHERE season IN ({season_list})
        ORDER BY game_date ASC
    """
    
    try:
        print(f"Loading NBA game logs from BigQuery ({table_id})...")
        df = client.query(query).to_dataframe()
        if df.empty:
            print(f"  No logs found in BigQuery for seasons {seasons}.")
            return None
            
        print(f"  Successfully loaded {len(df)} logs from BigQuery.")
        
        # Expand raw_record JSON if present
        if 'raw_record' in df.columns:
            import json
            print("  Expanding raw_record JSON data...")
            try:
                # Convert JSON strings to dicts and then to a DataFrame
                raw_df = pd.json_normalize(df['raw_record'].apply(json.loads))
                
                # Combine with original DF, prioritizing the expanded columns
                # We drop columns from df that exist in raw_df to avoid duplicates during concat
                df = df.drop(columns=['raw_record'])
                cols_to_keep = [c for c in df.columns if c not in raw_df.columns]
                df = pd.concat([df[cols_to_keep], raw_df], axis=1)
            except Exception as e:
                print(f"  Warning: Could not expand raw_record: {e}")
        
        # Ensure game_date is datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        return df
    except Exception as e:
        print(f"ERROR: Failed to load from BigQuery: {e}")
        return None


def load_nba_game_logs(
    seasons: Sequence[int],
    strict: bool = False,
    schedule_df: Optional[pd.DataFrame] = None,
    date_from: Optional[object] = None,
    date_to: Optional[object] = None,
    timeout: int = 120,
    max_retries: int = 3,
    base_delay: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Load NBA team game logs for the requested seasons using LeagueGameFinder (efficient).
    
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
        print(f"Loading game logs for {len(seasons)} seasons using LeagueGameFinder...")
        
        for season in seasons:
            season_str = f"{season}-{str(season + 1)[-2:]}"
            print(f"  Processing {season_str}...")
            
            try:
                # Use LeagueGameFinder to get games for the season (optionally limited by date).
                def _fetch():
                    kwargs: Dict[str, Any] = {
                        "season_nullable": season_str,
                        "league_id_nullable": "00",
                    }
                    date_from_value = _format_nba_date(date_from)
                    date_to_value = _format_nba_date(date_to)
                    if date_from_value:
                        kwargs["date_from_nullable"] = date_from_value
                    if date_to_value:
                        kwargs["date_to_nullable"] = date_to_value
                    if _supports_timeout(leaguegamefinder.LeagueGameFinder):
                        kwargs["timeout"] = timeout
                    kwargs = _filter_kwargs(leaguegamefinder.LeagueGameFinder, kwargs)
                    return leaguegamefinder.LeagueGameFinder(**kwargs)

                game_finder = _retry_nba_request(
                    _fetch,
                    label=f"NBA game logs {season_str}",
                    retries=max_retries,
                    base_delay=base_delay,
                )
                df = game_finder.get_data_frames()[0]
                
                if df.empty:
                    print(f"    Warning: No logs found for {season_str}")
                    continue
                
                # Add season
                df['season'] = season
                
                # Normalize game date
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    df = _normalize_game_dates(df)

                if date_from or date_to:
                    start_date = pd.to_datetime(date_from, errors="coerce")
                    end_date = pd.to_datetime(date_to, errors="coerce")
                    if pd.notna(start_date):
                        df = df[df["game_date"] >= start_date]
                    if pd.notna(end_date):
                        df = df[df["game_date"] <= end_date]
                
                # Standardize team column
                df['team'] = df['TEAM_ABBREVIATION']
                df['team_id'] = df['TEAM_ID']
                
                all_logs.append(df)
                
                # Small delay between seasons to be safe
                time.sleep(1.0)
                
            except Exception as e:
                print(f"    Warning: Could not load logs for season {season_str}: {e}")
                if strict:
                    raise
                continue
        
        if not all_logs:
            message = "No game logs were loaded"
            if strict:
                raise RuntimeError(message)
            print(f"WARNING: {message}")
            return None
        
        # Combine all logs
        combined_df = pd.concat(all_logs, ignore_index=True)
        
        # Map common column names
        col_mapping = {
            'GAME_ID': 'game_id',
            'MATCHUP': 'matchup',
            'WL': 'win_loss',
            'PTS': 'points_scored',
        }
        
        for old_col, new_col in col_mapping.items():
            if old_col in combined_df.columns:
                combined_df[new_col] = combined_df[old_col]
        
        # Convert game_id to string for matching
        if 'game_id' in combined_df.columns:
            combined_df['game_id'] = combined_df['game_id'].astype(str)
        
        # Now compute ratings with schedule lookup (if available)
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

def load_nba_game_logs_legacy(seasons: Sequence[int], strict: bool = False, schedule_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Load NBA team game logs for the requested seasons (Legacy team-by-team version).
    """

