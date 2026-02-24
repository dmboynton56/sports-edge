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
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from dotenv import load_dotenv

from src.data import nba_fetcher
from src.data.nba_game_logs_loader import load_nba_game_logs, load_nba_game_logs_from_bq
from src.models.predictor import GamePredictor


MODEL_VERSION = 'v3'


def load_season_schedule(season: int) -> pd.DataFrame:
    """Fetch the NBA schedule for a season and normalize columns."""
    print(f"Loading {season} NBA season data...")
    schedule = nba_fetcher.fetch_nba_schedule(season)
    if schedule.empty:
        raise ValueError(f"No schedule data found for {season}")
    
    # Normalize to ET naive for consistent day-of-game filtering
    predictor = GamePredictor('NBA', MODEL_VERSION)
    schedule['game_date'] = predictor._normalize_datetime(schedule['game_date'])
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
        # Standardize dates to date objects for display
        print(f"  Completed through: {pd.to_datetime(completed['game_date']).max().date()}")
    else:
        print("  WARNING: No completed games yet; team-strength features will be empty.")
    
    return completed


def collect_date_games(schedule: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Gather games for the specified date.
    Uses a 24-hour window to catch games that might have shifted in UTC.
    """
    date_obj = pd.to_datetime(target_date).date()
    
    # Standardize schedule dates to naive date objects for comparison
    sched_dates = pd.to_datetime(schedule['game_date']).dt.date
    
    date_games = schedule[sched_dates == date_obj].copy()
    
    if date_games.empty:
        # Try fetching directly from API
        print(f"  No games found in schedule for {target_date}, fetching from API...")
        date_games = nba_fetcher.fetch_nba_games_for_date(target_date)
        if date_games.empty:
            raise ValueError(f"No games found for date {target_date}.")
        
        # Normalize API dates too
        predictor = GamePredictor('NBA', MODEL_VERSION)
        date_games['game_date'] = predictor._normalize_datetime(date_games['game_date'])
        
    # Final filter: Ensure we are only looking at games for THIS specific local day
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
    # Ensure game_date is naive
    if game_date.tzinfo is not None:
        game_date = game_date.replace(tzinfo=None)
        
    hist_dates = pd.to_datetime(completed_games['game_date'])
    if hist_dates.dt.tz is not None:
        hist_dates = hist_dates.dt.tz_localize(None)
        
    games_before = completed_games[
        (hist_dates < game_date) &
        ((completed_games['home_team'] == team) | (completed_games['away_team'] == team))
    ]
    return len(games_before) > 0


def predict_games(games: pd.DataFrame,
                  schedule: pd.DataFrame,
                  completed_games: pd.DataFrame,
                  game_logs: Optional[pd.DataFrame],
                  include_explanations: bool = False) -> List[dict]:
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
        result = predictor.predict(game_df, schedule, game_logs=game_logs, include_explanations=include_explanations)
        predictions.append(result)
    
    return predictions


def display_predictions(predictions: List[dict], target_date: str, show_features: bool = False) -> None:
    """Pretty-print prediction results."""
    if not predictions:
        print(f"\nNo predictions generated for {target_date}.")
        return
    
    print("\n" + "=" * 80)
    print(f"NBA PREDICTIONS FOR {target_date}")
    print("=" * 80)
    
    for pred in predictions:
        print(f"\n{pred['away_team']} @ {pred['home_team']}")
        print(f"  Spread: {pred['predicted_spread']:.2f} ({pred['spread_interpretation']})")
        print(f"  Win Probabilities: Home {pred['home_win_probability']:.1%} | Away {pred['away_win_probability']:.1%}")
        print(f"  Predicted Winner: {pred['predicted_winner']} (Confidence {pred['confidence']:.1%})")
        
        if show_features and 'top_features' in pred:
            print(f"  Top Contributing Features:")
            for feat in pred['top_features']:
                impact_str = f"{feat['impact']:+.3f}"
                heur_marker = " (h)" if feat.get('is_heuristic') else ""
                print(f"    - {feat['feature']:<30}: {feat['value']:>8.2f} | Impact: {impact_str}{heur_marker}")
        
        if pred.get('model_disagreement', 0) > 0.15:
            print(f"  (!) Disagreement: {pred['model_disagreement']:.1%}")
    
    print("\n" + "=" * 80)


def evaluate_predictions(predictions: List[dict], schedule: pd.DataFrame) -> List[dict]:
    """
    Evaluate predictions against actual results if available.
    
    Args:
        predictions: List of prediction dictionaries
        schedule: Full schedule with actual scores
        
    Returns:
        List of dictionaries with evaluation results
    """
    eval_results = []
    
    for pred in predictions:
        # Find matching game in schedule
        target_date = pd.to_datetime(pred['game_date']).date()
        match = schedule[
            (schedule['home_team'] == pred['home_team']) &
            (schedule['away_team'] == pred['away_team']) &
            (pd.to_datetime(schedule['game_date']).dt.date == target_date)
        ]
        
        if match.empty or pd.isna(match.iloc[0]['home_score']):
            continue
            
        actual = match.iloc[0]
        actual_home_margin = actual['home_score'] - actual['away_score']
        actual_winner = pred['home_team'] if actual_home_margin > 0 else pred['away_team']
        
        # In our predictions, home_spread = -predicted_home_margin
        # So predicted_home_margin = -pred['predicted_spread']
        predicted_home_margin = -pred['predicted_spread']
        
        spread_error = abs(actual_home_margin - predicted_home_margin)
        is_correct = actual_winner == pred['predicted_winner']
        
        eval_results.append({
            'game': f"{pred['away_team']} @ {pred['home_team']}",
            'date': pred['game_date'],
            'is_correct': is_correct,
            'spread_error': spread_error,
            'confidence': pred['confidence'],
            'actual_margin': actual_home_margin,
            'predicted_margin': predicted_home_margin
        })
        
    return eval_results


def display_backtest_summary(eval_results: List[dict]):
    """Print summary statistics for backtest results."""
    if not eval_results:
        print("\nNo completed games found to evaluate.")
        return
        
    df = pd.DataFrame(eval_results)
    
    accuracy = df['is_correct'].mean()
    mae = df['spread_error'].mean()
    rmse = (df['spread_error']**2).mean()**0.5
    
    print("\n" + "#" * 80)
    print("BACKTEST SUMMARY STATISTICS")
    print("#" * 80)
    print(f"Total Games Evaluated: {len(df)}")
    print(f"Win Prediction Accuracy: {accuracy:.1%}")
    print(f"Spread MAE: {mae:.2f}")
    print(f"Spread RMSE: {rmse:.2f}")
    
    # Accuracy by confidence buckets
    if len(df) >= 5:
        print("\nAccuracy by Confidence:")
        df['conf_bucket'] = pd.cut(df['confidence'], bins=[0, 0.1, 0.2, 0.4, 1.0], 
                                  labels=['0-10%', '10-20%', '20-40%', '40%+'])
        conf_summary = df.groupby('conf_bucket', observed=False)['is_correct'].agg(['mean', 'count'])
        for bucket, row in conf_summary.iterrows():
            if row['count'] > 0:
                print(f"  {bucket}: {row['mean']:.1%} ({int(row['count'])} games)")

    print("#" * 80 + "\n")


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
        "--lookback",
        type=int,
        default=0,
        help="Number of days to look back from the target date for backtesting.",
    )
    parser.add_argument(
        "--push-to-db",
        action="store_true",
        help="Persist predictions to Supabase using env credentials.",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Show top contributing features for each prediction.",
    )
    return parser.parse_args()


def get_nba_season_from_date(date_obj: datetime) -> int:
    """
    Infers NBA season starting year from a date.
    NBA seasons typically start in October and end in June.
    If date is January-September, it belongs to the season that started the previous year.
    """
    if date_obj.month >= 10:
        return date_obj.year
    else:
        return date_obj.year - 1


def main():
    args = parse_args()
    target_date = args.date
    
    # Validate date format
    try:
        date_obj = datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        print(f"ERROR: Invalid date format '{target_date}'. Use YYYY-MM-DD format.")
        sys.exit(1)
        
    season = args.season or get_nba_season_from_date(date_obj)
    
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
    
    # Try loading from BigQuery first
    game_logs = load_nba_game_logs_from_bq([season])
    
    # Fallback to API if BigQuery fails or is empty
    if game_logs is None or game_logs.empty:
        print("  BigQuery logs unavailable, falling back to NBA API...")
        game_logs = load_nba_game_logs([season], strict=False, schedule_df=schedule_df)
    
    if game_logs is None or game_logs.empty:
        print("  Warning: Could not load game logs from any source; form features will be unavailable")
        game_logs = None
    else:
        print(f"  Loaded {len(game_logs)} total game log entries")
    
    # Handle lookback if requested
    dates_to_predict = [target_date]
    if args.lookback > 0:
        base_date = datetime.strptime(target_date, '%Y-%m-%d')
        dates_to_predict = [
            (base_date - timedelta(days=d)).strftime('%Y-%m-%d')
            for d in range(args.lookback, -1, -1)
        ]
        print(f"\nRunning backtest for {len(dates_to_predict)} days: {dates_to_predict[0]} to {dates_to_predict[-1]}")

    all_predictions = []
    all_eval_results = []

    for d_str in dates_to_predict:
        try:
            current_date_games = collect_date_games(schedule_df, d_str)
            if current_date_games.empty:
                continue
                
            day_preds = predict_games(current_date_games, schedule_df, completed_games, game_logs, include_explanations=args.show_features)
            all_predictions.extend(day_preds)
            
            # Display daily results if it's a single date or the last date in range
            if len(dates_to_predict) == 1 or d_str == target_date:
                display_predictions(day_preds, d_str, show_features=args.show_features)
                
            # Collect evaluation results for all dates
            day_evals = evaluate_predictions(day_preds, schedule_df)
            all_eval_results.extend(day_evals)
            
        except Exception as err:
            print(f"Error processing {d_str}: {err}")
            continue

    if args.lookback > 0:
        display_backtest_summary(all_eval_results)
    
    if args.push_to_db:
        # TODO: Implement Supabase push for NBA (similar to NFL)
        print("\nSupabase push not yet implemented for NBA")
        # from scripts.predict_week import push_predictions_to_supabase
        # push_predictions_to_supabase(predictions, date_games, schedule_df)


if __name__ == "__main__":
    main()
