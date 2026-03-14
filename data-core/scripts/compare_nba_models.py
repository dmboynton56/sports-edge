
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data import nba_fetcher, nba_game_logs_loader
from src.models.predictor import GamePredictor

def run_backtest_for_version(version, test_games, schedule, game_logs):
    predictor = GamePredictor('NBA', version)
    predictions = []
    
    for _, game in test_games.iterrows():
        game_date = pd.to_datetime(game['game_date']).date()
        logs_cutoff = game_logs[pd.to_datetime(game_logs['game_date']).dt.date < game_date].copy()
        hist_cutoff = schedule[pd.to_datetime(schedule['game_date']).dt.date < game_date].copy()
        
        try:
            pred = predictor.predict(pd.DataFrame([game]), hist_cutoff, game_logs=logs_cutoff)
            pred['actual_margin'] = game['home_score'] - game['away_score']
            pred['actual_winner'] = game['home_team'] if pred['actual_margin'] > 0 else game['away_team']
            pred['predicted_margin'] = -pred['predicted_spread']
            pred['error'] = pred['predicted_margin'] - pred['actual_margin']
            pred['is_correct'] = pred['predicted_winner'] == pred['actual_winner']
            predictions.append(pred)
        except:
            continue
            
    results_df = pd.DataFrame(predictions)
    if results_df.empty:
        return None
        
    return {
        'version': version,
        'accuracy': results_df['is_correct'].mean(),
        'mae': results_df['error'].abs().mean(),
        'rmse': np.sqrt((results_df['error']**2).mean()),
        'conf_40_plus_acc': results_df[results_df['confidence'] >= 0.4]['is_correct'].mean() if len(results_df[results_df['confidence'] >= 0.4]) > 0 else 0
    }

def compare_models(days_back=30):
    end_date = datetime(2026, 1, 12).date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"--- Comparing NBA Models (v2, v3, v4) ---")
    print(f"Window: {start_date} to {end_date} ({days_back} days)")
    
    # Load Data once
    season = 2025
    schedule = nba_fetcher.fetch_nba_schedule(season)
    completed_games = schedule[schedule['home_score'].notna()].copy()
    game_logs = nba_game_logs_loader.load_nba_game_logs_from_bq([season])
    
    test_games = completed_games[
        (pd.to_datetime(completed_games['game_date']).dt.date >= start_date) &
        (pd.to_datetime(completed_games['game_date']).dt.date <= end_date)
    ]
    
    print(f"Found {len(test_games)} games in window.")
    
    all_metrics = []
    for v in ['v2', 'v3', 'v4']:
        print(f"Running backtest for {v}...")
        m = run_backtest_for_version(v, test_games, schedule, game_logs)
        if m:
            all_metrics.append(m)
            
    comparison_df = pd.DataFrame(all_metrics)
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY (30-DAY LOOKBACK)")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    compare_models(days_back=30)
