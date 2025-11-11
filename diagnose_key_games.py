import os
import sys
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from src.data import nfl_fetcher
from src.models.predictor import GamePredictor


TARGET_GAMES = [
    ('LAC', 'PIT'),  # PIT @ LAC
    ('GB', 'PHI'),   # PHI @ GB
]

SEASON = 2025


def load_schedule(season: int) -> pd.DataFrame:
    """Fetch schedule and ensure game_date column exists."""
    print(f"Loading {season} schedule via nfl_data_py...")
    schedule = nfl_fetcher.fetch_nfl_schedule(season)
    if 'gameday' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['gameday'])
    elif 'game_date' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    else:
        raise ValueError("Schedule missing 'game_date'/'gameday' column.")
    schedule['season'] = season
    return schedule


def find_games(schedule: pd.DataFrame, targets: List[Tuple[str, str]]) -> List[pd.Series]:
    """Locate the target matchups in the schedule."""
    games = []
    for home, away in targets:
        match = schedule[
            (schedule['home_team'] == home) &
            (schedule['away_team'] == away)
        ]
        if match.empty:
            raise ValueError(f"Game {away} @ {home} not found in schedule.")
        games.append(match.iloc[0])
    return games


def load_feature_medians(predictor: GamePredictor) -> Dict[str, float]:
    """Load feature medians used for imputation."""
    medians_path = os.path.join(
        predictor.models_dir,
        f"feature_medians_{predictor.league.lower()}_{predictor.model_version}.pkl"
    )
    try:
        import pickle
        with open(medians_path, 'rb') as f:
            medians = pickle.load(f)
        if isinstance(medians, dict):
            return medians
    except Exception:
        # Fall back to zeros if medians missing
        pass
    return {}


def get_feature_matrix(features_df: pd.DataFrame,
                       feature_names: List[str],
                       spread_override: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Project engineered features onto the model's expected columns."""
    X = pd.DataFrame()
    for col in feature_names:
        if col == 'model_spread_feature' and spread_override is not None:
            X[col] = spread_override
        elif col in features_df.columns:
            X[col] = features_df[col]
        else:
            X[col] = 0.0
    return X.fillna(0.0)


def average_feature_importances(win_prob_model, feature_names: List[str]) -> np.ndarray:
    """Average feature importances across calibrated estimators."""
    importances = np.zeros(len(feature_names))
    if hasattr(win_prob_model, 'calibrated_classifiers_'):
        used = 0
        for clf in win_prob_model.calibrated_classifiers_:
            est = getattr(clf, 'estimator', None)
            if est is not None and hasattr(est, 'feature_importances_'):
                importances += est.feature_importances_
                used += 1
        if used > 0:
            return importances / used
    if hasattr(win_prob_model, 'base_estimator') and hasattr(win_prob_model.base_estimator, 'feature_importances_'):
        return win_prob_model.base_estimator.feature_importances_
    return np.ones(len(feature_names))


def compute_feature_impacts(values: pd.Series, medians: Dict[str, float],
                            importances: np.ndarray, top_n: int = 8) -> List[Tuple[str, float, float, float]]:
    """Return top features ranked by |delta from median| * importance."""
    rows = []
    for idx, feat in enumerate(values.index):
        imp = importances[idx] if idx < len(importances) else 0.0
        if imp <= 0:
            continue
        median_val = medians.get(feat, 0.0)
        delta = values.iloc[idx] - median_val
        impact = abs(delta) * imp
        if impact == 0:
            continue
        rows.append((feat, values.iloc[idx], delta, impact))
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows[:top_n]


def print_prediction_details(predictor: GamePredictor, game_series: pd.Series,
                             schedule: pd.DataFrame,
                             feature_medians: Dict[str, float],
                             win_importances: np.ndarray,
                             spread_importances: np.ndarray):
    """Predict a game and display feature diagnostics."""
    game_row = pd.DataFrame([{
        'home_team': game_series['home_team'],
        'away_team': game_series['away_team'],
        'game_date': game_series['game_date'],
        'season': game_series['season']
    }])
    
    prediction = predictor.predict(game_row, schedule)
    
    engineered_features = predictor.build_features_for_game(game_row, schedule)
    spread_feature_names = predictor.spread_feature_names or predictor.feature_names
    win_feature_names = predictor.win_feature_names or predictor.feature_names
    
    spread_matrix = get_feature_matrix(engineered_features, spread_feature_names)
    spread_values = predictor.spread_model.predict(spread_matrix)
    win_matrix = get_feature_matrix(engineered_features, win_feature_names, spread_override=spread_values)
    
    feature_impacts_win = compute_feature_impacts(win_matrix.iloc[0], feature_medians, win_importances)
    feature_impacts_spread = compute_feature_impacts(spread_matrix.iloc[0], feature_medians, spread_importances)
    
    print("\n" + "=" * 90)
    print(f"{prediction['away_team']} @ {prediction['home_team']} | {prediction['game_date']}")
    print(f"  Spread: {prediction['predicted_spread']:.2f} ({prediction['spread_interpretation']})")
    print(f"  Win Probabilities: Home {prediction['home_win_probability']:.1%} | Away {prediction['away_win_probability']:.1%}")
    print(f"  Model vs Spread: {prediction['home_win_prob_from_model']:.1%} vs {prediction['win_prob_from_spread']:.1%}")
    print(f"  Ensemble Confidence: {prediction['confidence']:.1%}")
    if prediction.get('model_disagreement', 0) > 0.15:
        print(f"  ⚠️  Disagreement: {prediction['model_disagreement']:.1%}")
    
    def _print_table(title: str, impacts: List[Tuple[str, float, float, float]]):
        print(f"\n  {title}")
        if not impacts:
            print("    (No informative features)")
            return
        print("    Feature                      Value      Δ vs Median     Weighted Impact")
        for feat, val, delta, impact in impacts:
            print(f"    {feat:25s} {val:9.3f} {delta:14.3f} {impact:18.4f}")
    
    _print_table("Top Win-Probability Drivers", feature_impacts_win)
    _print_table("Top Spread Drivers", feature_impacts_spread)


def main():
    try:
        schedule = load_schedule(SEASON)
    except Exception as exc:
        print(f"ERROR: Failed to load schedule: {exc}")
        sys.exit(1)
    
    games = find_games(schedule, TARGET_GAMES)
    
    predictor = GamePredictor('NFL', 'v2')
    feature_medians = load_feature_medians(predictor)
    
    win_feature_names = predictor.win_feature_names or predictor.feature_names
    spread_feature_names = predictor.spread_feature_names or predictor.feature_names
    win_importances = average_feature_importances(predictor.win_prob_model, win_feature_names)
    spread_importances = getattr(predictor.spread_model, 'feature_importances_', np.ones(len(spread_feature_names)))
    
    for game in games:
        print_prediction_details(
            predictor,
            game,
            schedule,
            feature_medians,
            win_importances,
            spread_importances
        )
    
    print("\n" + "=" * 90)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
