import argparse
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data import nfl_fetcher
from src.data.pbp_loader import load_pbp
from src.models.predictor import GamePredictor

def run_what_if(team: str, qb_name: str, replacement_epa: float = 0.0, week: str = 'CON', season: int = 2025):
    predictor = GamePredictor('NFL', 'v1')
    
    print(f"Loading {season} NFL season data...")
    schedule = nfl_fetcher.fetch_nfl_schedule(season)
    if 'gameday' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['gameday'])
    else:
        schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    schedule['season'] = season
    
    completed_games = schedule[schedule['home_score'].notna() & schedule['away_score'].notna()].copy()
    pbp = load_pbp([season])
    
    # Filter for the target week
    if week.isdigit():
        week_games = schedule[schedule['week'] == int(week)].copy()
    else:
        week_games = schedule[schedule['game_type'] == week.upper()].copy()
    
    target_game = week_games[(week_games['home_team'] == team) | (week_games['away_team'] == team)].copy()
    
    if target_game.empty:
        print(f"No game found for {team} in week {week}")
        return

    # 1. Calculate QB Delta
    nix_plays = pbp[pbp['passer_player_name'] == qb_name]
    if nix_plays.empty:
        print(f"No PBP data found for QB {qb_name}")
        return
    
    current_qb_epa = nix_plays['epa'].mean()
    delta = replacement_epa - current_qb_epa
    
    print(f"\nQB Analysis:")
    print(f"  {qb_name} current avg EPA/play: {current_qb_epa:.3f}")
    print(f"  Simulated replacement EPA: {replacement_epa:.3f}")
    print(f"  Estimated impact per play: {delta:.3f}")

    # 2. Base Prediction
    print("\nRunning Base Prediction (Bo Nix Active)...")
    base_pred = predictor.predict(target_game, schedule, play_by_play=pbp)
    
    # 3. Modified Prediction
    print(f"Running Injury Prediction ({qb_name} OUT)...")
    
    # We need to manually intercept the feature building to apply the delta
    features_df = predictor.build_features_for_game(target_game, schedule, play_by_play=pbp)
    
    # Apply delta to EPA features for the injured team
    epa_cols = [c for c in features_df.columns if 'epa_off' in c]
    is_home = (target_game.iloc[0]['home_team'] == team)
    prefix = 'form_home_' if is_home else 'form_away_'
    
    for col in epa_cols:
        if col.startswith(prefix):
            print(f"  Adjusting {col}: {features_df[col].iloc[0]:.3f} -> {features_df[col].iloc[0] + delta:.3f}")
            features_df[col] += delta
            
    # Now use a modified predict that takes features_df directly (internal hack)
    # We'll replicate the core of predictor.predict but with our adjusted features
    
    def predict_with_features(predictor, features_df):
        medians = predictor._load_feature_medians()
        spread_cols = predictor.spread_feature_names
        win_cols = predictor.win_feature_names
        
        X_spread = predictor._prepare_feature_matrix(features_df, spread_cols)
        X_spread = predictor._fill_with_medians(X_spread, medians)
        
        if predictor.is_ensemble:
            spread_preds = [m.predict(X_spread.values.astype(float)) for m in predictor.spread_model.values()]
            spread_pred = np.mean(spread_preds, axis=0)
        else:
            spread_pred = predictor.spread_model.predict(X_spread)
            
        X_win = predictor._prepare_feature_matrix(features_df, win_cols)
        if 'model_spread_feature' in X_win.columns:
            X_win['model_spread_feature'] = spread_pred
        X_win = predictor._fill_with_medians(X_win, medians)
        
        if predictor.is_ensemble:
            win_probs = [m.predict_proba(X_win.values.astype(float))[:, 1] for m in predictor.win_prob_model.values()]
            win_prob_proba = np.mean(win_probs, axis=0)
        else:
            win_prob_proba = predictor.win_prob_model.predict_proba(X_win)[:, 1]
            
        from src.models.link_function import spread_to_win_prob
        link_a, link_b = predictor.link_params
        win_prob_from_spread = spread_to_win_prob(spread_pred, link_a, link_b)
        
        # Simple blend
        final_win_prob = (0.65 * win_prob_proba + 0.35 * win_prob_from_spread)
        
        home_margin = float(spread_pred[0])
        home_spread = -home_margin
        
        return {
            'home_win_probability': float(final_win_prob[0]),
            'predicted_spread': home_spread,
            'predicted_winner': target_game.iloc[0]['home_team'] if final_win_prob[0] > 0.5 else target_game.iloc[0]['away_team']
        }

    injury_pred = predict_with_features(predictor, features_df)
    
    # 4. Compare
    print("\n" + "="*50)
    print(f"IMPACT ANALYSIS: {qb_name} Injury")
    print("="*50)
    print(f"Matchup: {target_game.iloc[0]['away_team']} @ {target_game.iloc[0]['home_team']}")
    print(f"\nOriginal Prediction:")
    print(f"  Winner: {base_pred['predicted_winner']}")
    print(f"  Spread: {base_pred['predicted_spread']:.1f}")
    print(f"  Win Prob: {base_pred['home_win_probability']:.1%}")
    
    print(f"\nInjury Scenario (Replacement EPA {replacement_epa}):")
    print(f"  Winner: {injury_pred['predicted_winner']}")
    print(f"  Spread: {injury_pred['predicted_spread']:.1f}")
    print(f"  Win Prob: {injury_pred['home_win_probability']:.1%}")
    
    spread_diff = injury_pred['predicted_spread'] - base_pred['predicted_spread']
    prob_diff = injury_pred['home_win_probability'] - base_pred['home_win_probability']
    
    print(f"\nNET IMPACT:")
    print(f"  Spread Change: {spread_diff:+.1f} points")
    print(f"  Win Prob Change: {prob_diff:+.1%}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="DEN")
    parser.add_argument("--qb", default="B.Nix")
    parser.add_argument("--replacement-epa", type=float, default=0.015, help="EPA/play for backup (league avg ~0.015)")
    args = parser.parse_args()
    
    run_what_if(args.team, args.qb, args.replacement_epa)
