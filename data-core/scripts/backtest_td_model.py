import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.td_predictor import TDScorerPredictor
from src.data import nfl_fetcher
from src.data.pbp_loader import load_pbp

def backtest():
    predictor = TDScorerPredictor(model_version='backtest')
    
    # Train on 2021-2023
    train_seasons = [2021, 2022, 2023]
    print(f"Training on {train_seasons}...")
    predictor.train(train_seasons)
    
    # Test on 2024
    test_season = 2024
    print(f"Testing on {test_season}...")
    weekly_test = nfl_fetcher.fetch_nfl_weekly_data([test_season])
    
    # Prepare features for the test set
    # Note: we need to use the full history to get rolling features for the start of 2024
    full_data = nfl_fetcher.fetch_nfl_weekly_data([2023, 2024])
    full_pbp = load_pbp([2023, 2024])
    full_sched = pd.concat([nfl_fetcher.fetch_nfl_schedule(2023), nfl_fetcher.fetch_nfl_schedule(2024)])
    
    all_features = predictor._prepare_features(full_data, full_sched, full_pbp)
    
    # Filter to 2024 rows and relevant positions
    test_df = all_features[
        (all_features['season'] == test_season) & 
        (all_features['position'].isin(predictor.target_positions))
    ].dropna(subset=predictor.feature_names)
    
    if test_df.empty:
        print("No test data available after filtering.")
        return

    print(f"Running predictions on {len(test_df)} samples...")
    # Get probabilities
    X_test = test_df[predictor.feature_names]
    probs = predictor.model.predict_proba(X_test)[:, 1]
    y_test = test_df['has_td']
    
    # Metrics
    brier = brier_score_loss(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ll = log_loss(y_test, probs)
    
    print("\n" + "=" * 30)
    print("BACKTEST RESULTS (2024)")
    print("=" * 30)
    print(f"Brier Score: {brier:.4f}")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Log Loss:    {ll:.4f}")
    
    # Baseline: what if we just predicted the mean TD rate for every player?
    mean_rate = y_test.mean()
    baseline_probs = np.full_like(probs, mean_rate)
    baseline_brier = brier_score_loss(y_test, baseline_probs)
    print(f"Baseline Brier: {baseline_brier:.4f}")
    print(f"Improvement over baseline: {(1 - brier/baseline_brier):.2%}")
    print("=" * 30)

if __name__ == "__main__":
    backtest()
