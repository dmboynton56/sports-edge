"""
Utility script to compare saved predictions against actual results.

Example:
    python3 evaluate_predictions.py \
        --predictions data/predictions/week10.csv \
        --season 2025 \
        --week 10
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

from src.data import nfl_fetcher


def load_actual_results(season: int, week: int) -> pd.DataFrame:
    """Fetch completed games for a season/week with scores."""
    schedule = nfl_fetcher.fetch_nfl_schedule(season)
    if 'gameday' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['gameday'])
    elif 'game_date' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    else:
        raise ValueError("Schedule missing game_date/gameday column.")
    
    week_games = schedule[
        (schedule['week'] == week) &
        schedule['home_score'].notna() &
        schedule['away_score'].notna()
    ].copy()
    
    if week_games.empty:
        raise ValueError(f"No completed games found for season {season}, week {week}.")
    
    week_games['actual_margin'] = week_games['home_score'] - week_games['away_score']
    week_games['actual_home_win'] = (week_games['actual_margin'] > 0).astype(int)
    week_games['game_date'] = pd.to_datetime(week_games['game_date']).dt.normalize()
    
    return week_games[[
        'home_team', 'away_team', 'game_date',
        'home_score', 'away_score',
        'actual_margin', 'actual_home_win'
    ]]


def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions saved via predict_WEEK_10/11 CSV export."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")
    
    df = pd.read_csv(path)
    required_cols = {
        'home_team', 'away_team', 'game_date',
        'predicted_spread', 'home_win_probability',
        'predicted_winner'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing columns: {missing}")
    
    df['game_date'] = pd.to_datetime(df['game_date']).dt.normalize()
    return df


def evaluate(predictions: pd.DataFrame, actuals: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Join predictions with actuals and compute error metrics."""
    merged = predictions.merge(
        actuals,
        on=['home_team', 'away_team', 'game_date'],
        how='inner',
        suffixes=('', '_actual')
    )
    
    if merged.empty:
        raise ValueError("No overlapping games found between predictions and actual results.")
    
    merged['spread_error'] = merged['predicted_spread'] - merged['actual_margin']
    merged['abs_spread_error'] = merged['spread_error'].abs()
    merged['home_win_predicted_label'] = (merged['predicted_spread'] > 0).astype(int)
    merged['home_win_correct'] = (merged['home_win_predicted_label'] == merged['actual_home_win']).astype(int)
    merged['brier_component'] = (merged['home_win_probability'] - merged['actual_home_win']) ** 2
    
    metrics = {
        'games_evaluated': len(merged),
        'direction_accuracy': merged['home_win_correct'].mean(),
        'mean_abs_spread_error': merged['abs_spread_error'].mean(),
        'spread_rmse': np.sqrt((merged['spread_error'] ** 2).mean()),
        'brier_score': merged['brier_component'].mean()
    }
    
    return merged, metrics


def print_summary(metrics: dict) -> None:
    """Pretty-print aggregate metrics."""
    print("\nEvaluation Summary")
    print("------------------")
    print(f"Games evaluated:        {metrics['games_evaluated']}")
    print(f"Direction accuracy:     {metrics['direction_accuracy']:.3f}")
    print(f"Mean abs spread error:  {metrics['mean_abs_spread_error']:.2f} pts")
    print(f"Spread RMSE:            {metrics['spread_rmse']:.2f} pts")
    print(f"Brier score:            {metrics['brier_score']:.3f}")


def print_miss_list(merged: pd.DataFrame, top_n: int = 5) -> None:
    """Show largest misses to guide feature diagnostics."""
    largest_misses = merged.sort_values('abs_spread_error', ascending=False).head(top_n)
    if largest_misses.empty:
        return
    
    print(f"\nTop {len(largest_misses)} Biggest Misses")
    print("-----------------------------------")
    for _, row in largest_misses.iterrows():
        print(
            f"{row['away_team']} @ {row['home_team']} ({row['game_date'].date()}): "
            f"predicted {row['predicted_spread']:.1f}, actual margin {row['actual_margin']:.1f}, "
            f"error {row['spread_error']:.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved predictions against actual results.")
    parser.add_argument('--predictions', required=True, help="Path to CSV saved from predict_WEEK_*.py")
    parser.add_argument('--season', type=int, required=True, help="Season year (e.g., 2025)")
    parser.add_argument('--week', type=int, required=True, help="Week number to evaluate")
    parser.add_argument('--export-path', type=str, help="Optional path to write the merged evaluation CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    preds = load_predictions(args.predictions)
    actuals = load_actual_results(args.season, args.week)
    merged, metrics = evaluate(preds, actuals)
    print_summary(metrics)
    print_miss_list(merged)
    
    if args.export_path:
        os.makedirs(os.path.dirname(args.export_path), exist_ok=True)
        merged.to_csv(args.export_path, index=False)
        print(f"\nDetailed evaluation saved to {args.export_path}")


if __name__ == "__main__":
    main()
