
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

def run_analysis_backtest(league='NBA', version='v3', season=2025, start_date_str='2025-11-15', end_date_str='2026-01-12'):
    print(f"--- Analyzing {league} Model {version} ---")
    
    # 1. Load Data
    print(f"Loading {season} schedule...")
    schedule = nba_fetcher.fetch_nba_schedule(season)
    completed_games = schedule[schedule['home_score'].notna()].copy()
    
    print("Loading game logs from BigQuery...")
    game_logs = nba_game_logs_loader.load_nba_game_logs_from_bq([season])
    if game_logs is None or game_logs.empty:
        print("Falling back to API for logs...")
        game_logs = nba_game_logs_loader.load_nba_game_logs([season], strict=False, schedule_df=schedule)
    
    # 2. Setup Backtest
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    
    predictor = GamePredictor(league, version)
    predictions = []
    
    # Filter schedule for the backtest window
    test_games = completed_games[
        (pd.to_datetime(completed_games['game_date']).dt.date >= start_date) &
        (pd.to_datetime(completed_games['game_date']).dt.date <= end_date)
    ]
    
    print(f"Running backtest on {len(test_games)} games from {start_date} to {end_date}...")
    
    for _, game in tqdm(test_games.iterrows(), total=len(test_games)):
        game_date = pd.to_datetime(game['game_date']).date()
        
        # Leakage-free logs
        logs_cutoff = game_logs[pd.to_datetime(game_logs['game_date']).dt.date < game_date].copy()
        hist_cutoff = schedule[pd.to_datetime(schedule['game_date']).dt.date < game_date].copy()
        
        try:
            # We want explanations for the analysis
            pred = predictor.predict(pd.DataFrame([game]), hist_cutoff, game_logs=logs_cutoff, include_explanations=True)
            
            # Add actuals for error calculation
            pred['actual_home_score'] = game['home_score']
            pred['actual_away_score'] = game['away_score']
            pred['actual_margin'] = game['home_score'] - game['away_score']
            pred['actual_winner'] = game['home_team'] if pred['actual_margin'] > 0 else game['away_team']
            
            # Error metrics
            pred['predicted_margin'] = -pred['predicted_spread'] # Home - Away
            pred['error'] = pred['predicted_margin'] - pred['actual_margin']
            pred['abs_error'] = abs(pred['error'])
            pred['is_correct'] = pred['predicted_winner'] == pred['actual_winner']
            
            predictions.append(pred)
        except Exception as e:
            # print(f"Error predicting {game['away_team']} @ {game['home_team']}: {e}")
            continue
            
    results_df = pd.DataFrame(predictions)
    if results_df.empty:
        print("No predictions were generated.")
        return
        
    # 3. ANALYSIS OF "BIG MISSES"
    print("\n" + "="*80)
    print("BACKTEST ERROR ANALYSIS REPORT")
    print("="*80)
    
    accuracy = results_df['is_correct'].mean()
    mae = results_df['abs_error'].mean()
    rmse = np.sqrt((results_df['error']**2).mean())
    
    print(f"Overall Accuracy: {accuracy:.1%}")
    print(f"Overall MAE:      {mae:.2f}")
    print(f"Overall RMSE:     {rmse:.2f}")
    
    # Define "Big Misses" as top 15% of absolute errors
    error_threshold = results_df['abs_error'].quantile(0.85)
    big_misses = results_df[results_df['abs_error'] >= error_threshold].copy()
    
    print(f"\nAnalyzing 'Big Misses' (Error >= {error_threshold:.1f} pts, {len(big_misses)} games)")
    
    # A. Team-specific analysis
    print("\n--- Teams Involved in Most Big Misses ---")
    home_misses = big_misses['home_team'].value_counts()
    away_misses = big_misses['away_team'].value_counts()
    total_misses = (home_misses.add(away_misses, fill_value=0)).sort_values(ascending=False)
    print(total_misses.head(10))
    
    # B. Direction of misses
    print("\n--- Bias Analysis ---")
    avg_error = results_df['error'].mean()
    print(f"Mean Error (Pred - Actual): {avg_error:.2f}")
    if avg_error > 1.0:
        print("  Insight: Model is consistently overestimating the HOME team.")
    elif avg_error < -1.0:
        print("  Insight: Model is consistently overestimating the AWAY team.")
    else:
        print("  Insight: No significant home/away bias detected.")
        
    # C. "Upset" analysis (where we were very confident but wrong)
    upsets = big_misses[(big_misses['confidence'] > 0.3) & (big_misses['is_correct'] == False)]
    if not upsets.empty:
        print(f"\n--- Top 5 Most Confident Misses (False Favorites) ---")
        for _, row in upsets.sort_values('confidence', ascending=False).head(5).iterrows():
            print(f"  {row['away_team']} @ {row['home_team']}: Conf {row['confidence']:.1%}, "
                  f"Pred {row['predicted_margin']:+.1f}, Actual {row['actual_margin']:+.1f} (Error {row['error']:+.1f})")
            if 'top_features' in row and row['top_features']:
                feats = [f"{f['feature']}({f['impact']:+.2f})" for f in row['top_features'][:3]]
                print(f"    Key Features: {', '.join(feats)}")

    # D. Feature correlation with error
    print("\n--- Feature Correlation with Abs Error ---")
    # Extract feature values into a flat DF for correlation analysis
    feat_data = []
    for _, row in results_df.iterrows():
        d = {'abs_error': row['abs_error']}
        if 'top_features' in row and row['top_features']:
            for f in row['top_features']:
                d[f['feature']] = f['value']
        feat_data.append(d)
    
    feat_df = pd.DataFrame(feat_data)
    if not feat_df.empty:
        corrs = feat_df.corr()['abs_error'].abs().sort_values(ascending=False)
        print("Features most correlated with prediction error:")
        print(corrs.drop('abs_error').head(5))

    print("\n" + "="*80)

if __name__ == "__main__":
    # Run analysis for v3
    run_analysis_backtest(version='v3', start_date_str='2025-11-20')
