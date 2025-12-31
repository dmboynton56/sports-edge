#!/usr/bin/env python3
"""
Predict NBA games for a specific date.

Unlike NFL which has weekly schedules, NBA games change day-to-day.
This script predicts games for a specific date.

Example:
    python scripts/predict_nba_date.py --date 2025-12-31 --season 2025
"""

import argparse
import os
import sys
from datetime import date, datetime
from typing import List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from dotenv import load_dotenv

from src.data import nba_fetcher
from src.data.nba_game_logs_loader import load_nba_game_logs
from src.models.predictor import GamePredictor


MODEL_VERSION = 'v1'


def load_season_schedule(season: int) -> pd.DataFrame:
    """Fetch the NBA schedule for a season and normalize columns."""
    print(f"Loading {season} NBA season data...")
    schedule = nba_fetcher.fetch_nba_schedule(season)
    if schedule.empty:
        raise ValueError(f"No schedule data found for {season}")
    
    schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    schedule['season'] = season
    print(f"  Loaded {len(schedule)} games for {season}")
    return schedule


def filter_completed_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return only games that have final scores logged."""
    if 'home_score' not in schedule.columns or 'away_score' not in schedule.columns:
        raise ValueError("Schedule is missing score columns needed to identify completed games.")
    
    completed = schedule[
        schedule['home_score'].notna() & schedule['away_score'].notna()
    ].copy()
    
    print(f"  Completed games: {len(completed)}")
    if len(completed) > 0:
        print(f"  Completed through: {completed['game_date'].max().date()}")
    else:
        print("  WARNING: No completed games yet; team-strength features will be empty.")
    
    return completed


def collect_date_games(schedule: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Gather games for the specified date.
    
    Args:
        schedule: Full season schedule
        target_date: Date string in YYYY-MM-DD format
    
    Returns:
        DataFrame with games for that date
    """
    date_obj = pd.to_datetime(target_date)
    
    date_games = schedule[
        pd.to_datetime(schedule['game_date']).dt.date == date_obj.date()
    ].copy()
    
    if date_games.empty:
        # Try fetching directly from API
        print(f"  No games found in schedule for {target_date}, fetching from API...")
        date_games = nba_fetcher.fetch_nba_games_for_date(target_date)
        if date_games.empty:
            raise ValueError(f"No games found for date {target_date}.")
    
    date_games = date_games.sort_values('game_date').reset_index(drop=True)
    print(f"\nFound {len(date_games)} games for {target_date}:")
    for _, row in date_games.iterrows():
        print(f"  {row['away_team']} @ {row['home_team']}")
    
    base_cols = ['home_team', 'away_team', 'game_date', 'season']
    optional_cols = [
        col for col in ['game_id', 'home_score', 'away_score']
        if col in date_games.columns
    ]
    return date_games[base_cols + optional_cols].copy()


def team_has_data(team: str, game_date: pd.Timestamp, completed_games: pd.DataFrame) -> bool:
    """Check whether the team has at least one current-season completed game before the given date."""
    games_before = completed_games[
        (completed_games['game_date'] < game_date) &
        ((completed_games['home_team'] == team) | (completed_games['away_team'] == team))
    ]
    return len(games_before) > 0


def predict_games(games: pd.DataFrame,
                  schedule: pd.DataFrame,
                  completed_games: pd.DataFrame,
                  game_logs: Optional[pd.DataFrame]) -> List[dict]:
    """Predict a batch of games, skipping any without sufficient data."""
    predictor = GamePredictor('NBA', MODEL_VERSION)
    
    predictions = []
    for _, game in games.iterrows():
        game_date = pd.to_datetime(game['game_date'])
        home_team = game['home_team']
        away_team = game['away_team']
        
        if not team_has_data(home_team, game_date, completed_games):
            print(f"\nSkipping {away_team} @ {home_team} ({game_date.date()}): "
                  f"No completed games for {home_team} before this date.")
            continue
        if not team_has_data(away_team, game_date, completed_games):
            print(f"\nSkipping {away_team} @ {home_team} ({game_date.date()}): "
                  f"No completed games for {away_team} before this date.")
            continue
        
        game_df = pd.DataFrame([game])
        result = predictor.predict(game_df, schedule, game_logs=game_logs)
        predictions.append(result)
    
    return predictions


def display_predictions(predictions: List[dict], target_date: str) -> None:
    """Pretty-print prediction results."""
    if not predictions:
        print("\nNo predictions generated.")
        return
    
    print("\n" + "=" * 80)
    print(f"NBA PREDICTIONS FOR {target_date}")
    print("=" * 80)
    
    for pred in predictions:
        print(f"\n{pred['away_team']} @ {pred['home_team']}")
        print(f"  Spread: {pred['predicted_spread']:.2f} ({pred['spread_interpretation']})")
        print(f"  Win Probabilities: Home {pred['home_win_probability']:.1%} | Away {pred['away_win_probability']:.1%}")
        print(f"  Predicted Winner: {pred['predicted_winner']} (Confidence {pred['confidence']:.1%})")
        if pred.get('model_disagreement', 0) > 0.15:
            print(f"  ⚠️  Disagreement: {pred['model_disagreement']:.1%}")
    
    print("\n" + "=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NBA games for a specific date."
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to predict (YYYY-MM-DD format, e.g., 2025-12-31).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="NBA season year (default: inferred from date).",
    )
    parser.add_argument(
        "--push-to-db",
        action="store_true",
        help="Persist predictions to Supabase using env credentials.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_date = args.date
    season = args.season or pd.to_datetime(target_date).year
    
    # Validate date format
    try:
        date_obj = datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        print(f"ERROR: Invalid date format '{target_date}'. Use YYYY-MM-DD format.")
        sys.exit(1)
    
    try:
        schedule_df = load_season_schedule(season)
    except Exception as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    
    try:
        completed_games = filter_completed_games(schedule_df)
    except Exception as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    
    print(f"\nUsing {season} season data:")
    print(f"  Total games: {len(schedule_df)}")
    print(f"  Date range: {schedule_df['game_date'].min().date()} to {schedule_df['game_date'].max().date()}")
    
    # Load game logs for form metrics
    print(f"\nLoading game logs for form metrics...")
    game_logs = load_nba_game_logs([season], strict=False)
    if game_logs is None or game_logs.empty:
        print("  Warning: Could not load game logs; form features will be unavailable")
        game_logs = None
    else:
        print(f"  Loaded {len(game_logs)} game log entries")
    
    try:
        date_games = collect_date_games(schedule_df, target_date)
    except Exception as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    
    predictions = predict_games(date_games, schedule_df, completed_games, game_logs)
    display_predictions(predictions, target_date)
    
    if args.push_to_db:
        # TODO: Implement Supabase push for NBA (similar to NFL)
        print("\nSupabase push not yet implemented for NBA")
        # from scripts.predict_week import push_predictions_to_supabase
        # push_predictions_to_supabase(predictions, date_games, schedule_df)


if __name__ == "__main__":
    main()

