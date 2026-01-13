"""
CLI module for refreshing predictions.
Usage: python -m src.pipeline.refresh --league NFL --date 2025-11-06
"""

import argparse
import sys
import os
import pickle
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv
# Optional supabase import - only needed for production pushes
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data import nfl_fetcher, nba_fetcher, odds_fetcher
from src.data.pbp_loader import load_pbp
from src.features import rest_schedule, form_metrics, strength
from src.models.spread_model import SpreadModel
from src.models.win_prob_model import WinProbModel
from src.models.link_function import spread_to_win_prob
from src.models.predictor import GamePredictor

load_dotenv()


def get_supabase_client() -> Client:
    """Create Supabase client."""
    if not SUPABASE_AVAILABLE:
        return None
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not url or not key:
        return None
    
    return create_client(url, key)


def load_historical_data(league: str, current_season: int, seasons_back: int = 3) -> dict:
    """
    Load historical data for feature building.
    
    Args:
        league: 'NFL' or 'NBA'
        current_season: Current season year
        seasons_back: Number of seasons to load
    
    Returns:
        Dictionary with historical_games, play_by_play (NFL), or game_logs (NBA)
    """
    historical_data = {
        'historical_games': [],
        'play_by_play': None,
        'game_logs': None
    }
    
    seasons = list(range(current_season - seasons_back, current_season + 1))
    
    print(f"Loading historical data for {league} (seasons {seasons[0]}-{seasons[-1]})...")
    
    # Load schedules
    all_games = []
    for season in seasons:
        try:
            if league == 'NFL':
                schedule = nfl_fetcher.fetch_nfl_schedule(season)
            else:
                schedule = nba_fetcher.fetch_nba_schedule(season)
            
            if 'gameday' in schedule.columns:
                schedule['game_date'] = pd.to_datetime(schedule['gameday'])
            elif 'game_date' not in schedule.columns:
                schedule['game_date'] = pd.to_datetime(schedule.get('date', schedule.index))
            
            schedule['season'] = season
            all_games.append(schedule)
        except Exception as e:
            print(f"  Warning: Could not load {season}: {e}")
    
    if all_games:
        historical_data['historical_games'] = pd.concat(all_games, ignore_index=True)
        print(f"  Loaded {len(historical_data['historical_games'])} historical games")
    
    # Load play-by-play for NFL
    if league == 'NFL':
        pbp_df = load_pbp(seasons)
        if pbp_df is not None and len(pbp_df) > 0:
            historical_data['play_by_play'] = pbp_df
            print(f"  Loaded {len(pbp_df)} play-by-play records")
        else:
            print("  Warning: Could not load NFL play-by-play; skipping form features")
    else:  # NBA
        # Load game logs for NBA form metrics
        from src.data.nba_game_logs_loader import load_nba_game_logs
        # Pass historical_games to enable computing opponent points for defensive ratings
        game_logs_df = load_nba_game_logs(seasons, strict=False, schedule_df=historical_data.get('historical_games'))
        if game_logs_df is not None and len(game_logs_df) > 0:
            historical_data['game_logs'] = game_logs_df
            print(f"  Loaded {len(game_logs_df)} game log records")
        else:
            print("  Warning: Could not load NBA game logs; skipping form features")
    
    return historical_data


def build_features(games_df: pd.DataFrame, league: str, historical_data: dict) -> pd.DataFrame:
    """
    Build features for games using enhanced feature contract.
    Matches the feature engineering from the EDA notebook.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building features for {len(games_df)} games...")
    df = games_df.copy()
    
    # Ensure game_date exists
    if 'game_date' not in df.columns:
        if 'gameday' in df.columns:
            df['game_date'] = pd.to_datetime(df['gameday'])
        else:
            raise ValueError("game_date or gameday column required")
    else:
        df['game_date'] = pd.to_datetime(df['game_date'])
    
    historical_games = historical_data.get('historical_games')
    if historical_games is None or len(historical_games) == 0:
        raise ValueError("historical_games required for feature building")
    
    # Add rest and schedule features
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding rest features...")
    df = rest_schedule.add_rest_features(df, historical_games)
    
    # Add opponent strength features
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding opponent strength features...")
    df = strength.add_opponent_strength_features(df, historical_games, league.lower())
    
    # Add team strength features (win %, point diff)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding team strength features...")
    df = _add_team_strength_features(df, historical_games)
    
    # Add interaction features
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding interaction features...")
    df = _add_interaction_features(df)
    
    # Add form features if available
    if league == 'NFL':
        play_by_play = historical_data.get('play_by_play')
        if play_by_play is not None and len(play_by_play) > 0:
            for window in [3, 5, 10]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding NFL form features (window={window})...")
                df = form_metrics.add_form_features_nfl(df, play_by_play, window=window)
    else:  # NBA
        game_logs = historical_data.get('game_logs')
        if game_logs is not None and len(game_logs) > 0:
            # Add form features for NBA (net rating, offensive/defensive ratings)
            for window in [3, 5, 10]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding NBA form features (window={window})...")
                df = form_metrics.add_form_features_nba(df, game_logs, window=window)
    
    # Add form interaction features
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Adding form interaction features...")
    df = _add_form_interactions(df, league)
    
    return df


def _add_team_strength_features(games_df: pd.DataFrame, historical_games: pd.DataFrame) -> pd.DataFrame:
    """Add team strength features (win %, point diff) efficiently."""
    import numpy as np
    
    df = games_df.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    hist = historical_games.copy()
    hist['game_date'] = pd.to_datetime(hist['game_date'])
    
    # Sort history to use for cumulative stats
    hist = hist.sort_values('game_date')
    
    # Ensure scores are numeric
    hist['home_score'] = pd.to_numeric(hist['home_score'], errors='coerce')
    hist['away_score'] = pd.to_numeric(hist['away_score'], errors='coerce')
    
    # Pre-calculate points and wins for each game record
    hist['home_win'] = (hist['home_score'] > hist['away_score']).astype(int)
    hist['away_win'] = (hist['away_score'] > hist['home_score']).astype(int)
    hist['home_diff'] = hist['home_score'] - hist['away_score']
    hist['away_diff'] = hist['away_score'] - hist['home_score']
    
    # Melt history to get a per-team perspective
    home_stats = hist[['game_date', 'season', 'home_team', 'home_win', 'home_diff']].rename(
        columns={'home_team': 'team', 'home_win': 'win', 'home_diff': 'diff'}
    )
    away_stats = hist[['game_date', 'season', 'away_team', 'away_win', 'away_diff']].rename(
        columns={'away_team': 'team', 'away_win': 'win', 'away_diff': 'diff'}
    )
    team_stats = pd.concat([home_stats, away_stats]).sort_values(['team', 'game_date'])
    
    # Compute cumulative stats per team per season
    team_stats['cum_wins'] = team_stats.groupby(['team', 'season'])['win'].cumsum()
    team_stats['cum_games'] = team_stats.groupby(['team', 'season']).cumcount() + 1
    team_stats['cum_diff'] = team_stats.groupby(['team', 'season'])['diff'].cumsum()
    
    # Shift by 1 because we want stats *before* the current game
    team_stats['prev_win_pct'] = team_stats.groupby(['team', 'season'])['cum_wins'].shift(1) / team_stats.groupby(['team', 'season'])['cum_games'].shift(1)
    team_stats['prev_avg_diff'] = team_stats.groupby(['team', 'season'])['cum_diff'].shift(1) / team_stats.groupby(['team', 'season'])['cum_games'].shift(1)
    
    # Merge back to the games_df
    # We need to merge twice: once for home team and once for away team
    # Use merge_asof for efficient date-based merging if the indices were aligned, 
    # but a simple merge on team and game_date is safer here assuming game_date + team is unique in hist
    
    # Prepare lookup table
    lookup = team_stats[['team', 'game_date', 'prev_win_pct', 'prev_avg_diff']].dropna(subset=['prev_win_pct', 'prev_avg_diff'])
    
    # Merge for home team
    df = pd.merge(df, lookup.rename(columns={'team': 'home_team', 'prev_win_pct': 'home_team_win_pct', 'prev_avg_diff': 'home_team_point_diff'}), 
                  on=['home_team', 'game_date'], how='left')
    
    # Merge for away team
    df = pd.merge(df, lookup.rename(columns={'team': 'away_team', 'prev_win_pct': 'away_team_win_pct', 'prev_avg_diff': 'away_team_point_diff'}), 
                  on=['away_team', 'game_date'], how='left')
    
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and derived features."""
    import numpy as np
    
    # Rest differentials
    if 'rest_home' in df.columns and 'rest_away' in df.columns:
        df['rest_differential'] = df['rest_home'] - df['rest_away']
        df['rest_advantage_home'] = (df['rest_home'] > df['rest_away']).astype(int)
    
    # Team strength differentials
    if 'home_team_win_pct' in df.columns and 'away_team_win_pct' in df.columns:
        df['win_pct_differential'] = df['home_team_win_pct'] - df['away_team_win_pct']
    if 'home_team_point_diff' in df.columns and 'away_team_point_diff' in df.columns:
        df['point_diff_differential'] = df['home_team_point_diff'] - df['away_team_point_diff']
        df['point_diff_gap'] = df['away_team_point_diff'] - df['home_team_point_diff']
        df['abs_point_diff_gap'] = df['point_diff_gap'].abs()
        df['point_diff_gap_flag'] = (df['point_diff_gap'] > 5).astype(int)
    
    # Opponent strength differential
    if 'opp_strength_home_season' in df.columns and 'opp_strength_away_season' in df.columns:
        df['opp_strength_differential'] = df['opp_strength_home_season'] - df['opp_strength_away_season']
    
    # Time features
    if 'game_date' in df.columns:
        if 'game_type' in df.columns:
            df['is_playoff'] = df['game_type'].str.contains('POST', case=False, na=False).astype(int)
        else:
            df['is_playoff'] = 0
    
    return df


def _add_form_interactions(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """Add form feature interactions."""
    for window in [3, 5, 10]:
        if league == 'NFL':
            home_off = f'form_home_epa_off_{window}'
            away_off = f'form_away_epa_off_{window}'
            if home_off in df.columns and away_off in df.columns:
                df[f'form_epa_off_diff_{window}'] = df[home_off] - df[away_off]
            
            home_def = f'form_home_epa_def_{window}'
            away_def = f'form_away_epa_def_{window}'
            if home_def in df.columns and away_def in df.columns:
                df[f'form_epa_def_diff_{window}'] = df[home_def] - df[away_def]
        else:  # NBA
            # Net rating differential
            home_net = f'form_home_net_rating_{window}'
            away_net = f'form_away_net_rating_{window}'
            if home_net in df.columns and away_net in df.columns:
                df[f'form_net_rating_diff_{window}'] = df[home_net] - df[away_net]
            
            # Offensive rating differential
            home_off = f'form_home_off_rating_{window}'
            away_off = f'form_away_off_rating_{window}'
            if home_off in df.columns and away_off in df.columns:
                df[f'form_off_rating_diff_{window}'] = df[home_off] - df[away_off]
            
            # Defensive rating differential
            home_def = f'form_home_def_rating_{window}'
            away_def = f'form_away_def_rating_{window}'
            if home_def in df.columns and away_def in df.columns:
                df[f'form_def_rating_diff_{window}'] = df[home_def] - df[away_def]
    
    return df


def upsert_games(supabase: Client, games_df: pd.DataFrame) -> dict:
    """
    Upsert games to Supabase.
    
    Returns:
        Dict mapping (league, season, home_team, away_team, game_time_utc) -> game_id
    """
    game_id_map = {}
    
    for _, game in games_df.iterrows():
        # Check if game exists
        result = supabase.table('games').select('id').eq('league', game['league'])\
            .eq('season', game['season'])\
            .eq('home_team', game['home_team'])\
            .eq('away_team', game['away_team'])\
            .eq('game_time_utc', game['game_time_utc'].isoformat() if hasattr(game['game_time_utc'], 'isoformat') else str(game['game_time_utc']))\
            .execute()
        
        game_data = {
            'league': game['league'],
            'season': game['season'],
            'game_time_utc': game['game_time_utc'].isoformat() if hasattr(game['game_time_utc'], 'isoformat') else str(game['game_time_utc']),
            'home_team': game['home_team'],
            'away_team': game['away_team']
        }
        
        if result.data:
            # Update existing
            game_id = result.data[0]['id']
            supabase.table('games').update(game_data).eq('id', game_id).execute()
        else:
            # Insert new
            result = supabase.table('games').insert(game_data).execute()
            game_id = result.data[0]['id']
        
        key = (game['league'], game['season'], game['home_team'], game['away_team'], str(game['game_time_utc']))
        game_id_map[key] = game_id
    
    return game_id_map


def upsert_odds(supabase: Client, odds_df: pd.DataFrame, game_id_map: dict):
    """Upsert odds snapshots to Supabase."""
    for _, row in odds_df.iterrows():
        # Find game_id
        game_id = None
        for key, gid in game_id_map.items():
            if row['home_team'] in key and row['away_team'] in key:
                game_id = gid
                break
        
        if not game_id:
            continue
        
        odds_data = {
            'game_id': game_id,
            'book': row['book'],
            'market': row['market'],
            'line': float(row['line']) if pd.notna(row['line']) else None,
            'price': float(row['price']) if pd.notna(row['price']) else None,
            'snapshot_ts': datetime.now().isoformat()
        }
        
        supabase.table('odds_snapshots').insert(odds_data).execute()


def upsert_predictions(supabase: Client, predictions_df: pd.DataFrame, game_id_map: dict, model_version: str):
    """Upsert model predictions to Supabase."""
    for _, row in predictions_df.iterrows():
        # Find game_id
        game_id = None
        for key, gid in game_id_map.items():
            if row['home_team'] in key and row['away_team'] in key:
                game_id = gid
                break
        
        if not game_id:
            continue
        
        pred_data = {
            'game_id': game_id,
            'model_name': 'sports_edge',
            'model_version': model_version,
            'my_spread': float(row['my_spread']) if pd.notna(row['my_spread']) else None,
            'my_home_win_prob': float(row['my_home_win_prob']) if pd.notna(row['my_home_win_prob']) else None,
            'asof_ts': datetime.now().isoformat()
        }
        
        supabase.table('model_predictions').insert(pred_data).execute()


def upsert_features(supabase: Client, features_df: pd.DataFrame, game_id_map: dict):
    """Upsert features to Supabase."""
    for _, row in features_df.iterrows():
        # Find game_id
        game_id = None
        for key, gid in game_id_map.items():
            if row['home_team'] in key and row['away_team'] in key:
                game_id = gid
                break
        
        if not game_id:
            continue
        
        # Convert features to JSON
        feature_dict = row.to_dict()
        # Remove non-feature columns
        for col in ['league', 'season', 'game_time_utc', 'home_team', 'away_team']:
            feature_dict.pop(col, None)
        
        feature_data = {
            'game_id': game_id,
            'feature_json': json.dumps(feature_dict),
            'asof_ts': datetime.now().isoformat()
        }
        
        supabase.table('features').insert(feature_data).execute()


def log_model_run(supabase: Client, league: str, success: bool, rows_written: int, error_text: Optional[str] = None):
    """Log model run to audit table."""
    run_data = {
        'league': league,
        'started_at': datetime.now().isoformat(),
        'finished_at': datetime.now().isoformat(),
        'rows_written': rows_written,
        'success': success,
        'error_text': error_text
    }
    
    supabase.table('model_runs').insert(run_data).execute()


def refresh(league: str, date: str, model_version: str = 'v0.1.0'):
    """
    Main refresh function.
    
    Args:
        league: 'NFL' or 'NBA'
        date: Date string YYYY-MM-DD
        model_version: Model version string
    """
    supabase = get_supabase_client()
    rows_written = 0
    error_text = None
    
    try:
        # 1. Fetch schedule
        print(f"Fetching {league} schedule for {date}...")
        if league == 'NFL':
            games_df = nfl_fetcher.fetch_nfl_games_for_date(date)
        else:
            games_df = nba_fetcher.fetch_nba_games_for_date(date)
        
        if games_df.empty:
            print(f"No games found for {date}")
            log_model_run(supabase, league, True, 0)
            return
        
        games_df['league'] = league
        games_df['season'] = datetime.strptime(date, '%Y-%m-%d').year
        
        # 2. Fetch odds
        print(f"Fetching odds for {league}...")
        odds_df = odds_fetcher.fetch_odds_for_date(league, date)
        
        # 3. Load historical data and build features
        print("Loading historical data...")
        current_season = datetime.strptime(date, '%Y-%m-%d').year
        historical_data = load_historical_data(league, current_season, seasons_back=3)
        
        print("Building features...")
        features_df = build_features(games_df, league, historical_data)
        
        # 4. Load models and predict using GamePredictor
        print("Loading models and predicting...")
        try:
            predictor = GamePredictor(league, model_version)
            
            # Predict using predictor (handles feature alignment automatically)
            predictions_list = []
            for idx, game in games_df.iterrows():
                game_row = pd.DataFrame([game])
                pred = predictor.predict(
                    game_row,
                    historical_data['historical_games'],
                    historical_data.get('play_by_play'),
                    historical_data.get('game_logs')
                )
                predictions_list.append(pred)
            
            predictions_df_temp = pd.DataFrame(predictions_list)
            features_df['my_spread'] = predictions_df_temp['predicted_spread'].values
            features_df['my_home_win_prob'] = predictions_df_temp['home_win_probability'].values
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Falling back to direct model loading...")
            
            # Fallback to direct model loading
            spread_model_path = f"models/spread_model_{league.lower()}_{model_version}.pkl"
            if os.path.exists(spread_model_path):
                spread_model = SpreadModel.load(spread_model_path)
                feature_cols = spread_model.feature_names
                
                # Prepare feature matrix
                X = pd.DataFrame()
                for col in feature_cols:
                    if col in features_df.columns:
                        X[col] = features_df[col]
                    else:
                        X[col] = 0
                X = X.fillna(0)
                
                predictions = spread_model.predict(X)
                features_df['my_spread'] = predictions
                
                # Convert spread to win prob
                link_path = f"models/link_function_{league.lower()}_{model_version}.pkl"
                if os.path.exists(link_path):
                    with open(link_path, 'rb') as f:
                        link_data = pickle.load(f)
                    link_a = link_data.get('a', 0.15)
                    link_b = link_data.get('b', 2.5)
                else:
                    link_a, link_b = 0.15, 2.5
                
                features_df['my_home_win_prob'] = features_df['my_spread'].apply(
                    lambda x: spread_to_win_prob(x, link_a, link_b) if pd.notna(x) else None
                )
            else:
                print(f"Warning: Model not found at {spread_model_path}, skipping predictions")
                features_df['my_spread'] = None
                features_df['my_home_win_prob'] = None
        
        # 5. Upsert to Supabase
        print("Upserting to Supabase...")
        game_id_map = upsert_games(supabase, games_df)
        rows_written += len(game_id_map)
        
        if not odds_df.empty:
            upsert_odds(supabase, odds_df, game_id_map)
            rows_written += len(odds_df)
        
        predictions_df = features_df[['home_team', 'away_team', 'my_spread', 'my_home_win_prob']]
        upsert_predictions(supabase, predictions_df, game_id_map, model_version)
        rows_written += len(predictions_df)
        
        upsert_features(supabase, features_df, game_id_map)
        rows_written += len(features_df)
        
        print(f"Successfully processed {len(games_df)} games")
        log_model_run(supabase, league, True, rows_written)
        
    except Exception as e:
        error_text = str(e)
        print(f"Error: {error_text}")
        log_model_run(supabase, league, False, rows_written, error_text)
        raise


def main():
    parser = argparse.ArgumentParser(description='Refresh sports-edge predictions')
    parser.add_argument('--league', type=str, required=True, choices=['NFL', 'NBA'], help='League (NFL or NBA)')
    parser.add_argument('--date', type=str, required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--model-version', type=str, default='v0.1.0', help='Model version')
    
    args = parser.parse_args()
    
    refresh(args.league, args.date, args.model_version)


if __name__ == '__main__':
    main()
