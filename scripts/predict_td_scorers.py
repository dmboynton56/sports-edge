import argparse
import os
import sys
import pandas as pd
from typing import List

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data import nfl_fetcher
from src.models.td_predictor import TDScorerPredictor
from src.data.pbp_loader import load_pbp

def main():
    parser = argparse.ArgumentParser(description="Predict touchdown scorers for a specific week.")
    parser.add_argument("--week", type=int, required=True, help="Week number to predict (e.g., 19 for Wild Card).")
    parser.add_argument("--season", type=int, default=2025, help="Season year (default: 2025).")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top scorers to show per team (default: 10).")
    parser.add_argument("--show-features", action="store_true", help="Show top 5 players per team with feature importance breakdown.")
    args = parser.parse_args()

    # If show-features is set, we limit to 5 per team as requested
    top_n = 5 if args.show_features else args.top_n

    predictor = TDScorerPredictor(model_version='v1')
    if not predictor.load_model():
        print("Error: Model not found. Run scripts/train_td_model.py first.")
        return

    print(f"Loading {args.season} data...")
    weekly_data = nfl_fetcher.fetch_nfl_weekly_data([args.season])
    schedule = nfl_fetcher.fetch_nfl_schedule(args.season)
    pbp = load_pbp([args.season])
    
    # Filter schedule to the target week
    week_games = schedule[schedule['week'] == args.week]
    if week_games.empty:
        print(f"No games found for week {args.week}.")
        return
    
    # Get list of teams playing this week
    playing_teams = set(week_games['home_team'].tolist() + week_games['away_team'].tolist())
    
    print(f"Preparing features for week {args.week}...")
    # Calculate features for all available data
    all_players_features = predictor._prepare_features(weekly_data, schedule, pbp)
    
    # We want the BEST available prediction for each player on a playing team for the target week.
    # 1. Get all players on playing teams
    all_team_players = all_players_features[all_players_features['recent_team'].isin(playing_teams)].copy()
    
    # 2. For each player, pick their latest week entry that is <= args.week
    # (Usually this will be args.week if available, or args.week-1 if not)
    # This ensures that even if only Saturday games have week 19 data, we still get Sunday/Monday players.
    latest_idx = all_team_players[all_team_players['week'] <= args.week].groupby('player_id')['week'].idxmax()
    current_week_preds = all_team_players.loc[latest_idx].copy()
    
    # Update week to the target week for consistency in display/logic
    current_week_preds['week'] = args.week

    print(f"Running predictions for {len(current_week_preds)} players...")
    predictions = predictor.predict(current_week_preds, include_explanations=args.show_features)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"TOUCHDOWN SCORER PREDICTIONS - WEEK {args.week}")
    print("=" * 60)
    
    # Group by game
    for _, game in week_games.iterrows():
        home = game['home_team']
        away = game['away_team']
        print(f"\n{away} @ {home}")
        
        for team in [away, home]:
            print(f"  {team} Top Scorers:")
            team_preds = predictions[predictions['recent_team'] == team].sort_values('td_probability', ascending=False)
            for _, p in team_preds.head(top_n).iterrows():
                print(f"    {p['td_probability']:.1%} - {p['player_display_name']} ({p['position']})")
                if args.show_features and 'top_features' in p:
                    for feat in p['top_features']:
                        impact_str = f"{feat['impact']:+.3f}"
                        print(f"      - {feat['feature']:<20}: {feat['value']:>8.2f} | Impact: {impact_str}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
