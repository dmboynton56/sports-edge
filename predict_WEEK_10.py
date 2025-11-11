import argparse
import os
import sys
from typing import List

import pandas as pd

from src.data import nfl_fetcher
from src.models.predictor import GamePredictor


MODEL_VERSION = 'v2'


def load_season_schedule(season: int) -> pd.DataFrame:
    """Fetch the NFL schedule for a season and normalize columns."""
    print(f"Loading {season} NFL season data...")
    schedule = nfl_fetcher.fetch_nfl_schedule(season)
    if 'gameday' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['gameday'])
    elif 'game_date' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    else:
        raise ValueError("Schedule missing game_date/gameday column.")
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


def collect_week_10_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Gather Week 10 games for the target window (Nov 6, Nov 9, Nov 10).
    """
    target_dates = pd.to_datetime(['2025-11-06', '2025-11-09', '2025-11-10']).date
    if 'week' not in schedule.columns:
        raise ValueError("Schedule missing 'week' column required to filter Week 10 games.")
    
    mask = (schedule['week'] == 10) & schedule['game_date'].dt.date.isin(target_dates)
    week_10_games = schedule[mask].copy()
    
    if week_10_games.empty:
        raise ValueError("No Week 10 games found for Nov 6/9/10.")
    
    week_10_games = week_10_games.sort_values('game_date').reset_index(drop=True)
    print(f"\nFound {len(week_10_games)} Week 10 games on Nov 6/9/10:")
    for _, row in week_10_games.iterrows():
        print(f"  {row['game_date'].date()} - {row['away_team']} @ {row['home_team']}")
    
    return week_10_games[['home_team', 'away_team', 'game_date', 'season']]


def team_has_data(team: str, game_date: pd.Timestamp, completed_games: pd.DataFrame) -> bool:
    """Check whether the team has at least one current-season completed game before the given date."""
    games_before = completed_games[
        (completed_games['game_date'] < game_date) &
        ((completed_games['home_team'] == team) | (completed_games['away_team'] == team))
    ]
    return len(games_before) > 0


def predict_games(games: pd.DataFrame, schedule: pd.DataFrame, completed_games: pd.DataFrame) -> List[dict]:
    """Predict a batch of games, skipping any without sufficient data."""
    predictor = GamePredictor('NFL', MODEL_VERSION)
    
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
        result = predictor.predict(game_df, schedule)
        predictions.append(result)
    
    return predictions


def display_predictions(predictions: List[dict]) -> None:
    """Pretty-print prediction results."""
    if not predictions:
        print("\nNo predictions generated.")
        return
    
    print("\n" + "=" * 80)
    print("WEEK 10 PREDICTIONS (NOV 6 / NOV 9 / NOV 10)")
    print("=" * 80)
    
    for pred in predictions:
        print(f"\n{pred['away_team']} @ {pred['home_team']}  |  {pred['game_date']}")
        print(f"  Spread: {pred['predicted_spread']:.2f} ({pred['spread_interpretation']})")
        print(f"  Win Probabilities: Home {pred['home_win_probability']:.1%} | Away {pred['away_win_probability']:.1%}")
        print(f"  Predicted Winner: {pred['predicted_winner']} (Confidence {pred['confidence']:.1%})")
        print(f"  Model win prob: {pred.get('home_win_prob_from_model', float('nan')):.1%}")
        print(f"  From spread: {pred.get('win_prob_from_spread', float('nan')):.1%}")
        if pred.get('model_disagreement', 0) > 0.15:
            print(f"  ⚠️  Disagreement: {pred['model_disagreement']:.1%}")
    
    print("\n" + "=" * 80)

def save_predictions(predictions: List[dict], path: str) -> None:
    """Persist predictions to CSV."""
    if not predictions:
        print("No predictions to save; skipping write.")
        return
    
    df = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} predictions to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Week 10 games and optionally save results.")
    parser.add_argument("--season", type=int, default=2025, help="NFL season to score (default: 2025)")
    parser.add_argument("--save-path", type=str, help="Optional CSV path to save predictions.")
    return parser.parse_args()


def main():
    args = parse_args()
    season = args.season
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
    
    try:
        week_10_games = collect_week_10_games(schedule_df)
    except Exception as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    
    predictions = predict_games(week_10_games, schedule_df, completed_games)
    display_predictions(predictions)
    
    if args.save_path:
        save_predictions(predictions, args.save_path)


if __name__ == "__main__":
    main()
