import sys
from typing import List, Optional

import pandas as pd

from src.data import nfl_fetcher
from src.data.pbp_loader import load_pbp
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


def collect_week_11_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Gather Week 11 games for the target window (Nov 13, Nov 14, Nov 16, Nov 17).
    """
    target_dates = pd.to_datetime([
        '2025-11-13',
        '2025-11-14',
        '2025-11-16',
        '2025-11-17'
    ]).date
    if 'week' not in schedule.columns:
        raise ValueError("Schedule missing 'week' column required to filter Week 11 games.")
    
    mask = (schedule['week'] == 11) & schedule['game_date'].dt.date.isin(target_dates)
    week_11_games = schedule[mask].copy()
    
    if week_11_games.empty:
        raise ValueError("No Week 11 games found for Nov 13/14/16/17.")
    
    week_11_games = week_11_games.sort_values('game_date').reset_index(drop=True)
    print(f"\nFound {len(week_11_games)} Week 11 games on Nov 13/14/16/17:")
    for _, row in week_11_games.iterrows():
        print(f"  {row['game_date'].date()} - {row['away_team']} @ {row['home_team']}")
    
    return week_11_games[['home_team', 'away_team', 'game_date', 'season']]


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
                  play_by_play: Optional[pd.DataFrame]) -> List[dict]:
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
        result = predictor.predict(game_df, schedule, play_by_play=play_by_play)
        predictions.append(result)
    
    return predictions


def display_predictions(predictions: List[dict]) -> None:
    """Pretty-print prediction results."""
    if not predictions:
        print("\nNo predictions generated.")
        return
    
    print("\n" + "=" * 80)
    print("WEEK 11 PREDICTIONS (NOV 13 / NOV 14 / NOV 16 / NOV 17)")
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


def main():
    season = 2025
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

    play_by_play = load_pbp([season])
    
    try:
        week_11_games = collect_week_11_games(schedule_df)
    except Exception as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    
    predictions = predict_games(week_11_games, schedule_df, completed_games, play_by_play)
    display_predictions(predictions)


if __name__ == "__main__":
    main()
