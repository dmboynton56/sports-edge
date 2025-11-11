"""
Model training and export script.
Trains models from notebook data and saves them in production format.
Can be run standalone or imported from notebook.
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Optional, Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from pandas.api.types import is_numeric_dtype

from src.models.link_function import fit_link_function
from src.pipeline.refresh import build_features
from src.data import nfl_fetcher
from src.data.pbp_loader import load_pbp


def compute_season_sample_weights(df: pd.DataFrame, season_col: str = 'season',
                                  growth_per_year: float = 1.2) -> np.ndarray:
    """
    Create sample weights that emphasize more recent seasons.
    
    Args:
        df: DataFrame containing a season column
        season_col: Column with season/year values
        growth_per_year: Multiplicative boost applied per season step
    
    Returns:
        Numpy array of weights normalized to mean 1.
    """
    if season_col not in df.columns:
        return np.ones(len(df))
    
    seasons = pd.to_numeric(df[season_col], errors='coerce')
    if seasons.isna().all():
        return np.ones(len(df))
    
    min_season = seasons.min()
    weights = np.power(growth_per_year, seasons - min_season)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    return weights / np.mean(weights)


def _normalize_schedule(schedule: pd.DataFrame, season: int, league: str) -> pd.DataFrame:
    """Ensure schedule has game_date and standard columns."""
    df = schedule.copy()
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    elif 'gameday' in df.columns:
        df['game_date'] = pd.to_datetime(df['gameday'])
    else:
        raise ValueError("Schedule missing 'game_date'/'gameday' column.")
    
    df['season'] = season
    df['league'] = league.upper()
    return df


def load_completed_games(league: str, seasons: List[int]) -> pd.DataFrame:
    """
    Fetch and combine completed games for the requested seasons.
    """
    if not seasons:
        raise ValueError("At least one season must be provided.")
    
    frames = []
    league = league.upper()
    
    for season in seasons:
        if league == 'NFL':
            schedule = nfl_fetcher.fetch_nfl_schedule(season)
        else:
            raise NotImplementedError("Automated training currently supports NFL only.")
        
        schedule = _normalize_schedule(schedule, season, league)
        completed = schedule[
            schedule['home_score'].notna() & schedule['away_score'].notna()
        ].copy()
        
        if completed.empty:
            print(f"Warning: No completed games found for {league} {season}.")
            continue
        
        frames.append(completed)
    
    if not frames:
        raise ValueError("No completed games available for the provided seasons.")
    
    games = pd.concat(frames, ignore_index=True)
    games['game_date'] = pd.to_datetime(games['game_date'])
    games['season'] = pd.to_numeric(games['season'], errors='coerce')
    games = games.sort_values('game_date').reset_index(drop=True)
    return games


def load_play_by_play(seasons: List[int]) -> Optional[pd.DataFrame]:
    """
    Load play-by-play data for seasons (used for form features).
    """
    pbp = load_pbp(seasons)
    if pbp is None or pbp.empty:
        print("Warning: Failed to load play-by-play data; form features will be skipped.")
        return None
    
    keep_cols = [col for col in ['game_id', 'posteam', 'defteam', 'epa', 'success', 'game_date']
                 if col in pbp.columns]
    if not keep_cols:
        print("Warning: Play-by-play data missing required columns; skipping form features.")
        return None
    
    pbp = pbp[keep_cols].copy()
    pbp['game_date'] = pd.to_datetime(pbp['game_date'])
    return pbp


def build_training_dataset(league: str, seasons: List[int], include_form: bool = True) -> pd.DataFrame:
    """
    Build training features by fetching historical games and engineering features.
    """
    games = load_completed_games(league, seasons)
    historical_data = {
        'historical_games': games.copy(),
        'play_by_play': None,
        'game_logs': None
    }
    
    if league.upper() == 'NFL' and include_form:
        pbp = load_play_by_play(seasons)
        if pbp is not None:
            historical_data['play_by_play'] = pbp
    
    features = build_features(games, league.upper(), historical_data)
    features = features[features['home_score'].notna() & features['away_score'].notna()].copy()
    features['home_win'] = (features['home_score'] > features['away_score']).astype(int)
    features['margin'] = features['home_score'] - features['away_score']
    return features


def _load_features_from_path(path: str) -> pd.DataFrame:
    """Load cached features from disk."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        return pd.read_parquet(path)
    if ext == '.csv':
        return pd.read_csv(path)
    raise ValueError(f"Unsupported features file type: {ext}")


def _save_features(df: pd.DataFrame, path: str) -> None:
    """Persist engineered feature set to disk."""
    ext = os.path.splitext(path)[1].lower()
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if ext == '.parquet':
        df.to_parquet(path, index=False)
    elif ext == '.csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported save format for {path}")


def train_and_save_models(features_df: pd.DataFrame, 
                          target_col: str = 'home_win',
                          margin_col: str = 'margin',
                          feature_cols: Optional[list] = None,
                          league: str = 'NFL',
                          model_version: str = 'v1',
                          use_lgbm: bool = True,
                          test_size: float = 0.2,
                          random_state: int = 42):
    """
    Train and save win probability and spread models.
    
    Args:
        features_df: DataFrame with features and targets
        target_col: Column name for binary win target
        margin_col: Column name for margin/spread target
        feature_cols: List of feature column names (if None, auto-detect)
        league: 'NFL' or 'NBA'
        model_version: Version string for saved models
        use_lgbm: Whether to use LightGBM (True) or Random Forest (False)
        test_size: Test set size fraction
        random_state: Random seed
    
    Returns:
        Dictionary with model metrics and paths
    """
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Determine feature columns
    if feature_cols is None:
        # Auto-detect feature columns (exclude target, margin, and metadata columns)
        exclude_cols = [
            target_col, margin_col, 'game_id', 'game_date', 'gameday',
            'home_team', 'away_team', 'home_score', 'away_score',
            'league', 'season', 'game_type', 'week', 'weekday', 'gametime',
            'result', 'total', 'overtime', 'old_game_id', 'gsis',
            'nfl_detail_id', 'pfr', 'pff', 'espn', 'ftn',
            'away_qb_id', 'home_qb_id', 'stadium_id', 'referee',
            'temp', 'wind'
        ]
        feature_cols = [
            col for col in features_df.columns
            if col not in exclude_cols and is_numeric_dtype(features_df[col])
        ]
        
        # Explicitly drop sportsbook/odds inputs until we feed real-time prices
        odds_keywords = (
            'moneyline',
            'spread_line',
            'spread_odds',
            'total_line',
            'over_odds',
            'under_odds',
            'book'
        )
        feature_cols = [
            col for col in feature_cols
            if not any(keyword in col.lower() for keyword in odds_keywords)
        ]
    
    # Remove duplicates while preserving order
    seen = set()
    feature_cols = [col for col in feature_cols if not (col in seen or seen.add(col))]
    
    # Prepare data
    extra_cols = [target_col, margin_col]
    has_season_col = 'season' in features_df.columns
    if has_season_col:
        extra_cols.append('season')
    else:
        print("WARNING: 'season' column missing from features; sample weights will be uniform.")
    
    model_df = features_df[feature_cols + extra_cols].dropna(subset=[target_col, margin_col])
    
    if len(model_df) < 100:
        raise ValueError(f"Insufficient data: {len(model_df)} rows (need >= 100)")
    
    X = model_df[feature_cols]
    y_win = model_df[target_col]
    y_margin = model_df[margin_col]
    # Clip margin target to reduce blowout noise and emphasize realistic spreads
    y_margin_clipped = np.clip(y_margin, -21, 21)
    if has_season_col:
        sample_weights = compute_season_sample_weights(model_df, season_col='season')
    else:
        sample_weights = np.ones(len(model_df))
    
    # Boost weights for large mismatches so models learn blowout signals
    mismatch_boost = 1 + 0.5 * np.clip(np.abs(y_margin_clipped) / 14.0, 0, 1)
    sample_weights = sample_weights * mismatch_boost
    
    # Split data
    split_result = train_test_split(
        X, y_win, y_margin_clipped, sample_weights,
        test_size=test_size,
        random_state=random_state,
        stratify=y_win
    )
    (X_train, X_test,
     y_train_win, y_test_win,
     y_train_margin, y_test_margin,
     w_train, w_test) = split_result
    
    w_train = np.asarray(w_train)
    w_test = np.asarray(w_test)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")
    
    # Keep base feature matrices for each model
    spread_feature_cols = feature_cols.copy()
    X_train_base = X_train.copy()
    X_test_base = X_test.copy()
    X_full_base = X.copy()
    
    # Train spread model first
    print("\nTraining spread model...")
    if use_lgbm:
        spread_estimator = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=random_state,
            verbose=-1
        )
    else:
        spread_estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
    
    n_splits = min(5, max(2, len(X_train_base) // 75))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_spread = np.zeros(len(X_train_base))
    for train_idx, val_idx in kfold.split(X_train_base):
        reg = clone(spread_estimator)
        reg.fit(
            X_train_base.iloc[train_idx],
            y_train_margin.iloc[train_idx],
            sample_weight=w_train[train_idx]
        )
        oof_spread[val_idx] = reg.predict(X_train_base.iloc[val_idx])
    
    spread_model = clone(spread_estimator)
    spread_model.fit(X_train_base, y_train_margin, sample_weight=w_train)
    
    y_pred_spread = spread_model.predict(X_test_base)
    mae = mean_absolute_error(y_test_margin, y_pred_spread, sample_weight=w_test)
    rmse = np.sqrt(mean_squared_error(y_test_margin, y_pred_spread, sample_weight=w_test))
    
    print(f"Spread Model ({'LGBM' if use_lgbm else 'RF'}):")
    print(f"  MAE: {mae:.2f} points")
    print(f"  RMSE: {rmse:.2f} points")
    
    # Inject spread predictions as an anchor feature for the win model
    spread_feature_name = 'model_spread_feature'
    spread_pred_train = oof_spread
    spread_pred_full = spread_model.predict(X_full_base)
    
    X_train_win = X_train_base.copy()
    X_test_win = X_test_base.copy()
    X_full_win = X_full_base.copy()
    X_train_win[spread_feature_name] = spread_pred_train
    X_test_win[spread_feature_name] = y_pred_spread
    X_full_win[spread_feature_name] = spread_pred_full
    win_feature_cols = X_train_win.columns.tolist()
    
    # Train win probability model (now aligned with spread outputs)
    print("\nTraining win probability model...")
    if use_lgbm:
        try:
            base_model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                random_state=random_state,
                verbose=-1
            )
            win_prob_model = CalibratedClassifierCV(
                base_model,
                method='isotonic',
                cv=5,
                n_jobs=-1
            )
            win_prob_model.fit(X_train_win, y_train_win, sample_weight=w_train)
            model_type = 'lightgbm_calibrated'
        except Exception as e:
            print(f"LightGBM failed: {e}, falling back to Random Forest")
            use_lgbm = False
    
    if not use_lgbm:
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
        win_prob_model = CalibratedClassifierCV(
            base_model,
            method='isotonic',
            cv=5,
            n_jobs=-1
        )
        win_prob_model.fit(X_train_win, y_train_win, sample_weight=w_train)
        model_type = 'rf_calibrated'
    
    y_pred_win = win_prob_model.predict(X_test_win)
    y_pred_proba = win_prob_model.predict_proba(X_test_win)[:, 1]
    accuracy = accuracy_score(y_test_win, y_pred_win, sample_weight=w_test)
    brier = brier_score_loss(y_test_win, y_pred_proba, sample_weight=w_test)
    
    print(f"Win Probability Model ({model_type.upper()}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Brier Score: {brier:.4f} (lower is better)")
    
    # Calibrate link function using model-predicted spreads
    print("\nCalibrating link function...")
    try:
        calibration_spreads = pd.Series(spread_pred_full, index=X.index)
        link_params = fit_link_function(calibration_spreads, y_win, max_abs_a=0.3)
    except Exception as e:
        print(f"  Warning: Link calibration failed ({e}); falling back to defaults.")
        link_params = (0.15, 2.5)
    link_a, link_b = link_params
    print(f"Link function parameters: a={link_a:.4f}, b={link_b:.4f}")
    
    # Check model consistency
    from src.models.link_function import spread_to_win_prob
    win_prob_from_spread = spread_to_win_prob(y_pred_spread, link_a, link_b)
    disagreement = np.abs(y_pred_proba - win_prob_from_spread)
    mean_disagreement = np.mean(disagreement)
    sign_agreement = ((y_pred_proba > 0.5) == (y_pred_spread > 0)).mean()
    
    print(f"\nModel Consistency Check:")
    print(f"  Mean disagreement: {mean_disagreement:.4f}")
    print(f"  Sign agreement: {sign_agreement:.1%}")
    
    if mean_disagreement > 0.15:
        print(f"  ⚠️  Warning: High disagreement between models. Consider ensemble approach.")
    
    # Calculate feature medians for missing value imputation
    feature_medians = X_train_win.median().to_dict()
    
    # Save models
    league_lower = league.lower()
    
    # Win probability model
    win_prob_path = os.path.join(models_dir, f'win_prob_model_{league_lower}_{model_version}.pkl')
    win_prob_data = {
        'model': win_prob_model,
        'model_type': model_type,
        'feature_names': win_feature_cols,
        'win_feature_names': win_feature_cols,
        'spread_feature_names': spread_feature_cols,
        'accuracy': accuracy,
        'brier_score': brier,
        'trained_date': datetime.now().isoformat(),
        'n_features': len(win_feature_cols),
        'n_samples': len(X_train_win),
        'league': league,
        'calibrated': True
    }
    with open(win_prob_path, 'wb') as f:
        pickle.dump(win_prob_data, f)
    print(f"\nSaved win probability model to {win_prob_path}")
    
    # Spread model
    spread_path = os.path.join(models_dir, f'spread_model_{league_lower}_{model_version}.pkl')
    spread_data = {
        'model': spread_model,
        'model_type': 'lightgbm' if use_lgbm else 'rf',
        'feature_names': spread_feature_cols,
        'spread_feature_names': spread_feature_cols,
        'mae': mae,
        'rmse': rmse,
        'trained_date': datetime.now().isoformat(),
        'n_features': len(spread_feature_cols),
        'n_samples': len(X_train_base),
        'league': league
    }
    with open(spread_path, 'wb') as f:
        pickle.dump(spread_data, f)
    print(f"Saved spread model to {spread_path}")
    
    # Link function
    link_path = os.path.join(models_dir, f'link_function_{league_lower}_{model_version}.pkl')
    link_data = {
        'a': link_a,
        'b': link_b,
        'calibrated_date': datetime.now().isoformat(),
        'league': league
    }
    with open(link_path, 'wb') as f:
        pickle.dump(link_data, f)
    print(f"Saved link function to {link_path}")
    
    # Feature medians
    medians_path = os.path.join(models_dir, f'feature_medians_{league_lower}_{model_version}.pkl')
    with open(medians_path, 'wb') as f:
        pickle.dump(feature_medians, f)
    print(f"Saved feature medians to {medians_path}")
    
    return {
        'win_prob_model_path': win_prob_path,
        'spread_model_path': spread_path,
        'link_function_path': link_path,
        'feature_medians_path': medians_path,
        'accuracy': accuracy,
        'brier_score': brier,
        'mae': mae,
        'rmse': rmse,
        'link_params': link_params,
        'mean_disagreement': mean_disagreement,
        'sign_agreement': sign_agreement,
        'win_feature_names': win_feature_cols,
        'spread_feature_names': spread_feature_cols,
        'n_samples': len(X_train_win)
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train and export Sports-Edge models.")
    parser.add_argument('--league', default='NFL', choices=['NFL'], help="League to train.")
    parser.add_argument('--start-season', type=int, default=2021, help="First season to include.")
    parser.add_argument('--end-season', type=int, default=2024, help="Last season to include (inclusive).")
    parser.add_argument('--model-version', default='v1', help="Model version tag for saved artifacts.")
    parser.add_argument('--features-path', type=str, help="Optional path to cached features (csv/parquet).")
    parser.add_argument('--save-features-path', type=str, help="Optional path to save engineered training set.")
    parser.add_argument('--use-rf', action='store_true', help="Use RandomForest instead of LightGBM.")
    parser.add_argument('--no-form', action='store_true', help="Skip form metrics (no play-by-play fetch).")
    parser.add_argument('--test-size', type=float, default=0.2, help="Test size fraction.")
    parser.add_argument('--random-state', type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    league = args.league.upper()
    
    if args.features_path:
        features_df = _load_features_from_path(args.features_path)
        print(f"Loaded cached features from {args.features_path} ({len(features_df)} rows)")
    else:
        if args.start_season > args.end_season:
            raise ValueError("start-season must be <= end-season.")
        seasons = list(range(args.start_season, args.end_season + 1))
        print(f"Building training set for {league} seasons {seasons[0]}-{seasons[-1]}...")
        features_df = build_training_dataset(league, seasons, include_form=not args.no_form)
        print(f"Engineered features for {len(features_df)} completed games.")
        if args.save_features_path:
            _save_features(features_df, args.save_features_path)
            print(f"Saved training features to {args.save_features_path}")
    
    metrics = train_and_save_models(
        features_df,
        league=league,
        model_version=args.model_version,
        use_lgbm=not args.use_rf,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print("\nTraining complete:")
    for key in ['accuracy', 'brier_score', 'mae', 'rmse', 'mean_disagreement', 'sign_agreement']:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
