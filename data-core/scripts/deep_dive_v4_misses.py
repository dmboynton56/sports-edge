
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

def deep_dive_v4_misses(days_back=30):
    end_date = datetime(2026, 1, 12).date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"--- Deep Dive: NBA Model v4 High-Confidence Misses ---")
    print(f"Window: {start_date} to {end_date}")
    
    # Load Data
    season = 2025
    schedule = nba_fetcher.fetch_nba_schedule(season)
    completed_games = schedule[schedule['home_score'].notna()].copy()
    game_logs = nba_game_logs_loader.load_nba_game_logs_from_bq([season])
    
    test_games = completed_games[
        (pd.to_datetime(completed_games['game_date']).dt.date >= start_date) &
        (pd.to_datetime(completed_games['game_date']).dt.date <= end_date)
    ]
    
    predictor = GamePredictor('NBA', 'v4')
    results = []
    
    print(f"Running predictions for {len(test_games)} games...")
    for _, game in tqdm(test_games.iterrows(), total=len(test_games)):
        game_date = pd.to_datetime(game['game_date']).date()
        logs_cutoff = game_logs[pd.to_datetime(game_logs['game_date']).dt.date < game_date].copy()
        hist_cutoff = schedule[pd.to_datetime(schedule['game_date']).dt.date < game_date].copy()
        
        try:
            pred = predictor.predict(pd.DataFrame([game]), hist_cutoff, game_logs=logs_cutoff, include_explanations=True)
            pred['actual_margin'] = game['home_score'] - game['away_score']
            pred['actual_winner'] = game['home_team'] if pred['actual_margin'] > 0 else game['away_team']
            pred['predicted_margin'] = -pred['predicted_spread']
            pred['error'] = pred['predicted_margin'] - pred['actual_margin']
            pred['abs_error'] = abs(pred['error'])
            pred['is_correct'] = pred['predicted_winner'] == pred['actual_winner']
            results.append(pred)
        except:
            continue
            
    df = pd.DataFrame(results)
    
    # 1. Analyze "Big Misses" (Top 15% absolute error)
    big_misses = df.sort_values('abs_error', ascending=False).head(int(len(df)*0.15))
    print("\n" + "="*80)
    print(f"ANALYSIS: {len(big_misses)} games with largest spread error")
    print("="*80)
    
    miss_impacts = []
    for _, row in big_misses.iterrows():
        for feat in row['top_features']:
            # Did this feature push the model in the direction of the error?
            # error = predicted - actual. 
            # If error > 0 (model overestimated home), and impact > 0 (feature favored home), it pushed error.
            pushed = (feat['impact'] > 0 and row['error'] > 0) or (feat['impact'] < 0 and row['error'] < 0)
            miss_impacts.append({
                'feature': feat['feature'],
                'abs_impact': abs(feat['impact']),
                'pushed_error': pushed
            })
    
    big_df = pd.DataFrame(miss_impacts)
    summary = big_df.groupby('feature').agg({
        'abs_impact': 'sum',
        'pushed_error': 'mean',
        'feature': 'count'
    }).rename(columns={'feature': 'frequency'}).sort_values('abs_impact', ascending=False)
    
    print("Features that most frequently 'lied' to the model (pushed toward error):")
    print(summary.head(15))
    
    # Fatigue specific
    fatigue_feats = ['rest_differential', 'is_3in4_differential', 'rest_home', 'rest_away', 'b2b_home', 'b2b_away']
    print("\n--- Impact of Fatigue in Big Misses ---")
    print(summary[summary.index.isin(fatigue_feats)])

if __name__ == "__main__":
    deep_dive_v4_misses(days_back=30)
