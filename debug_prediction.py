# Deep Feature Analysis for Model Disagreement
# This script investigates why models disagree and why predictions differ from book spreads

import pandas as pd
import numpy as np
import pickle
import os
import sys
from src.models.predictor import GamePredictor
from src.data import nfl_fetcher

# Load historical data
schedule_df = nfl_fetcher.fetch_nfl_schedule(2024)
schedule_df['game_date'] = pd.to_datetime(schedule_df['gameday'])

# Initialize predictor
predictor = GamePredictor('NFL', 'v2')

# Game to analyze
game_row = pd.DataFrame({
    'home_team': ['DEN'],
    'away_team': ['LV'],
    'game_date': ['2025-11-06'],
    'season': [2025]
})

print("=" * 80)
print("DEEP FEATURE ANALYSIS")
print("=" * 80)

# 1. Build features for this game
print("\n1. BUILDING FEATURES FOR PREDICTION")
print("-" * 80)
features_df = predictor.build_features_for_game(game_row, schedule_df)
win_feature_names = predictor.win_feature_names or predictor.feature_names
spread_feature_names = predictor.spread_feature_names or predictor.feature_names

spread_matrix = pd.DataFrame()
for col in spread_feature_names:
    spread_matrix[col] = features_df[col] if col in features_df.columns else 0.0
spread_matrix = spread_matrix.fillna(0.0)
spread_pred_vector = predictor.spread_model.predict(spread_matrix)

win_matrix = pd.DataFrame()
for col in win_feature_names:
    if col == 'model_spread_feature':
        win_matrix[col] = spread_pred_vector
    elif col in features_df.columns:
        win_matrix[col] = features_df[col]
    else:
        win_matrix[col] = 0.0
win_matrix = win_matrix.fillna(0.0)

# Show key computed features
key_features_to_check = [
    'home_team_win_pct', 'away_team_win_pct',
    'home_team_point_diff', 'away_team_point_diff',
    'opp_strength_home_season', 'opp_strength_away_season',
    'rest_home', 'rest_away',
    'win_pct_differential', 'point_diff_differential'
]

print("\nKey features computed:")
for col in key_features_to_check:
    if col in features_df.columns:
        val = features_df[col].iloc[0]
        if pd.notna(val):
            print(f"  {col:35s}: {val:10.4f}")
        else:
            print(f"  {col:35s}: {'NaN':>10}")

# 2. Check which features the model expects vs what we have
print("\n2. MODEL FEATURE REQUIREMENTS")
print("-" * 80)

X = win_matrix
missing_features = []
for col in win_feature_names:
    if col == 'model_spread_feature':
        continue
    if col not in features_df.columns or pd.isna(features_df[col].iloc[0]):
        missing_features.append(col)

print(f"Model expects: {len(win_feature_names)} features")
print(f"Missing/defaulted: {len(missing_features)} ({len(missing_features)/len(win_feature_names)*100:.1f}%)")

# Check form features specifically
form_features = [f for f in win_feature_names if f.startswith('form_')]
missing_form = [f for f in form_features if f in missing_features]
print(f"\nForm features (EPA): {len(form_features)} total, {len(missing_form)} missing ({len(missing_form)/len(form_features)*100:.1f}%)")

# 3. Load training data statistics
print("\n3. COMPARING TO TRAINING DATA")
print("-" * 80)

models_dir = 'models'
medians_path = os.path.join(models_dir, 'feature_medians_nfl_v2.pkl')
if os.path.exists(medians_path):
    with open(medians_path, 'rb') as f:
        training_medians = pickle.load(f)
    
    print("\nTop features - Prediction vs Training Median:")
    print(f"{'Feature':<35} {'Predicted':>12} {'Training Median':>15} {'Difference':>12}")
    print("-" * 80)
    
    # Show features with largest differences
    diffs = []
    for col in win_feature_names:
        pred_val = X[col].iloc[0]
        train_median = training_medians.get(col, 0)
        diff = abs(pred_val - train_median)
        diffs.append((col, pred_val, train_median, diff))
    
    diffs.sort(key=lambda x: x[3], reverse=True)
    for col, pred_val, train_median, diff in diffs[:20]:
        print(f"{col:<35} {pred_val:>12.4f} {train_median:>15.4f} {diff:>12.4f}")

# 4. Feature importance analysis
print("\n4. FEATURE IMPORTANCE ANALYSIS")
print("-" * 80)

win_prob_path = os.path.join(models_dir, 'win_prob_model_nfl_v2.pkl')
spread_path = os.path.join(models_dir, 'spread_model_nfl_v2.pkl')

with open(win_prob_path, 'rb') as f:
    win_prob_data = pickle.load(f)
with open(spread_path, 'rb') as f:
    spread_data = pickle.load(f)

win_prob_model = win_prob_data['model']
spread_model = spread_data['model']

# Get feature importances
def get_importance(model):
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'calibrated_classifiers_'):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, 'feature_importances_'):
            return base.feature_importances_
    return None

win_imp = get_importance(win_prob_model)
spread_imp = get_importance(spread_model)

if win_imp is not None:
    win_importance = pd.DataFrame({
        'feature': win_feature_names,
        'importance': win_imp,
        'value': X.iloc[0].values
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Features - Win Probability Model:")
    print(f"{'Feature':<35} {'Importance':>12} {'Value':>12}")
    print("-" * 60)
    for _, row in win_importance.head(15).iterrows():
        print(f"{row['feature']:<35} {row['importance']:>12.4f} {row['value']:>12.4f}")

if spread_imp is not None:
    spread_importance = pd.DataFrame({
        'feature': spread_feature_names,
        'importance': spread_imp,
        'value': spread_matrix.iloc[0].values
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Features - Spread Model:")
    print(f"{'Feature':<35} {'Importance':>12} {'Value':>12}")
    print("-" * 60)
    for _, row in spread_importance.head(15).iterrows():
        print(f"{row['feature']:<35} {row['importance']:>12.4f} {row['value']:>12.4f}")

# 5. Data availability check
print("\n5. DATA AVAILABILITY CHECK")
print("-" * 80)

game_date = pd.to_datetime('2025-11-06')
print(f"Predicting game on: {game_date.date()}")
print(f"Using historical data through: {schedule_df['game_date'].max().date()}")

# Check team strength features - these should be based on 2025 season, but we only have 2024
print(f"\nCRITICAL ISSUE:")
print(f"   Predicting {game_date.year} game but only have data through {schedule_df['game_date'].max().year}")
print(f"   Team strength features (win_pct, point_diff) will be:")
print(f"     - NaN (if looking for 2025 season games)")
print(f"     - Based on 2024 season (if falling back)")

# 6. Make prediction
print("\n6. PREDICTION ANALYSIS")
print("-" * 80)

prediction = predictor.predict(game_row, schedule_df)

print(f"\nOur Prediction: DEN by {prediction['predicted_spread']:.2f}")
print(f"Official Spread: DEN by 9.5")
print(f"Difference: {abs(prediction['predicted_spread'] - 9.5):.2f} points")
print(f"\nModel Disagreement: {prediction.get('model_disagreement', 0):.1%}")

# 7. Root Cause Analysis
print("\n7. ROOT CAUSE ANALYSIS")
print("-" * 80)

issues = []

# Check if we're using old season data
if 'home_team_win_pct' in X.columns:
    win_pct = X['home_team_win_pct'].iloc[0]
    if pd.isna(win_pct) or win_pct == 0:
        issues.append("ERROR: home_team_win_pct is NaN/0 - no 2025 season data available")
    else:
        issues.append(f"WARNING: home_team_win_pct = {win_pct:.3f} - based on 2024 season, not 2025")

# Check form features
if len(missing_form) > len(form_features) * 0.5:
    issues.append(f"ERROR: {len(missing_form)}/{len(form_features)} form features missing - no PBP data for future games")

# Check if key features are defaulting
high_importance_missing = []
if win_imp is not None:
    for feat, imp in zip(win_feature_names, win_imp):
        if feat in missing_features and imp > np.percentile(win_imp, 75):
            high_importance_missing.append((feat, imp))
    
    if high_importance_missing:
        issues.append(f"ERROR: {len(high_importance_missing)} high-importance features are missing/defaulted")

# Prediction accuracy
if abs(prediction['predicted_spread'] - 9.5) > 3:
    issues.append(f"ERROR: Prediction off by {abs(prediction['predicted_spread'] - 9.5):.1f} points")
    issues.append("   Likely causes:")
    issues.append("     1. Missing 2025 season data (team strength features)")
    issues.append("     2. Missing form/EPA features (no PBP for future games)")
    issues.append("     3. Models trained on historical data, predicting future with incomplete features")

print("\nIssues identified:")
for issue in issues:
    print(f"  {issue}")

# 8. Recommendations
print("\n8. RECOMMENDATIONS")
print("-" * 80)

print("""
To fix model disagreement and improve prediction accuracy:

1. **Get Current Season Data**
   - Load 2025 NFL schedule and results up to Nov 6, 2025
   - This will populate team strength features (win_pct, point_diff) correctly
   - Use: nfl_fetcher.fetch_nfl_schedule(2025)

2. **Handle Missing Form Features**
   - Form features (EPA) require play-by-play data
   - For future games, these will always be missing
   - Options:
     a) Use recent form (last 3-5 games) as proxy
     b) Train models without form features for future predictions
     c) Use season-to-date averages instead of rolling windows

3. **Separate Models for Future vs Historical**
   - Train one model with all features (for historical analysis)
   - Train another model without form features (for future predictions)
   - Or: Use feature importance to identify which features matter most

4. **Feature Engineering for Future Games**
   - Use preseason projections or early season data
   - Incorporate betting market data (closing lines) as features
   - Use team ratings from external sources

5. **Model Calibration**
   - The ensemble approach helps, but fixing root cause is better
   - Consider retraining with more recent data
   - Use temporal validation (train on older data, test on recent)
""")

print("\n" + "=" * 80)
