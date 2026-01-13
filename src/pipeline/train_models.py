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
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from pandas.api.types import is_numeric_dtype

from src.models.link_function import fit_link_function
# Lazy import for refresh to avoid supabase dependency issues
try:
    from src.pipeline.refresh import build_features
except ImportError:
    build_features = None
from src.data import nfl_fetcher, nba_fetcher
from src.data.pbp_loader import load_pbp
from src.data.nba_game_logs_loader import load_nba_game_logs, load_nba_game_logs_from_bq


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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading completed games for {league} seasons {seasons}...")
    if not seasons:
        raise ValueError("At least one season must be provided.")
    
    frames = []
    league = league.upper()
    
    for season in seasons:
        if league == 'NFL':
            schedule = nfl_fetcher.fetch_nfl_schedule(season)
        elif league == 'NBA':
            schedule = nba_fetcher.fetch_nba_schedule(season)
        else:
            raise NotImplementedError(f"Automated training currently supports NFL and NBA only. Got {league}")
        
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
    elif league.upper() == 'NBA' and include_form:
        print(f"Loading NBA game logs for {seasons}...")
        # Try BQ first, then API
        game_logs = load_nba_game_logs_from_bq(seasons)
        if game_logs is None or game_logs.empty:
            game_logs = load_nba_game_logs(seasons, strict=False, schedule_df=games)
        
        if game_logs is not None and not game_logs.empty:
            historical_data['game_logs'] = game_logs
    
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
                          use_ensemble: bool = False,
                          test_size: float = 0.2,
                          random_state: int = 42):
    """
    Train and save win probability and spread models.
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting training pipeline for {league} {model_version}...")
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Determine feature columns
    if feature_cols is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Auto-detecting feature columns...")
        # Auto-detect feature columns (exclude target, margin, and metadata columns)
        exclude_cols = [
            target_col, margin_col, 'game_id', 'game_date', 'gameday',
            'home_team', 'away_team', 'home_score', 'away_score',
            'league', 'season', 'game_type', 'week', 'weekday', 'gametime',
            'week_number', 'month', 'point_diff_differential',
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
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Selected {len(feature_cols)} features.")
    
    # Remove duplicates while preserving order
    seen = set()
    feature_cols = [col for col in feature_cols if not (col in seen or seen.add(col))]
    
    # Prepare data - ensure no overlap between features and targets
    # This prevents duplicate column issues which can break dtype cleanup
    clean_feature_cols = [c for c in feature_cols if c not in [target_col, margin_col]]
    
    extra_cols = []
    if target_col in features_df.columns: extra_cols.append(target_col)
    if margin_col in features_df.columns: extra_cols.append(margin_col)
    
    has_season_col = 'season' in features_df.columns
    has_date_col = 'game_date' in features_df.columns
    if has_season_col:
        extra_cols.append('season')
    if has_date_col:
        extra_cols.append('game_date')
    else:
        print("WARNING: 'game_date' column missing; will use random split instead of time-series split")
    
    # Remove duplicates while preserving order
    all_cols = []
    for col in clean_feature_cols + extra_cols:
        if col not in all_cols:
            all_cols.append(col)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Cleaning and preparing dataset...")
    model_df = features_df[all_cols].dropna(subset=[target_col, margin_col]).copy()
    
    if len(model_df) < 100:
        raise ValueError(f"Insufficient data: {len(model_df)} rows (need >= 100)")
    
    # Ensure targets are numeric before splitting
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors='coerce').fillna(0).astype(float)
    model_df[margin_col] = pd.to_numeric(model_df[margin_col], errors='coerce').fillna(0).astype(float)
    
    X = model_df[clean_feature_cols]
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
    
    # Split data - USE TIME-SERIES SPLIT for sports data
    # Train on past games, test on future games (no data leakage)
    if 'game_date' in model_df.columns:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Performing time-series split...")
        # Sort by date
        model_df_sorted = model_df.sort_values('game_date').reset_index(drop=True)
        X_sorted = model_df_sorted[clean_feature_cols]
        y_win_sorted = model_df_sorted[target_col]
        y_margin_sorted = model_df_sorted[margin_col]
        
        # Use date-based split (e.g., train on first 80%, test on last 20%)
        split_idx = int(len(model_df_sorted) * (1 - test_size))
        
        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_train_win = y_win_sorted.iloc[:split_idx]
        y_test_win = y_win_sorted.iloc[split_idx:]
        y_train_margin = y_margin_sorted.iloc[:split_idx]
        y_test_margin = y_margin_sorted.iloc[split_idx:]
        
        if has_season_col:
            # We need to recompute sample weights for the sorted dataframe to keep them aligned
            sample_weights_sorted = compute_season_sample_weights(model_df_sorted, season_col='season')
            mismatch_boost_sorted = 1 + 0.5 * np.clip(np.abs(np.clip(y_margin_sorted, -21, 21)) / 14.0, 0, 1)
            sample_weights_sorted = sample_weights_sorted * mismatch_boost_sorted
            w_train = sample_weights_sorted[:split_idx]
            w_test = sample_weights_sorted[split_idx:]
        else:
            w_train = np.ones(len(X_train))
            w_test = np.ones(len(X_test))
    else:
        # Fallback to random split if no date column
        print("WARNING: No game_date column found, using random split (not ideal for time-series data)")
        split_result = train_test_split(
            X, y_win, y_margin_clipped, sample_weights,
            test_size=test_size,
            random_state=random_state,
            stratify=y_win if len(np.unique(y_win)) > 1 else None
        )
        (X_train, X_test,
         y_train_win, y_test_win,
         y_train_margin, y_test_margin,
         w_train, w_test) = split_result
    
    w_train = np.asarray(w_train)
    w_test = np.asarray(w_test)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Final train size: {len(X_train)}, test size: {len(X_test)}")
    
    # --- AGGRESSIVE DTYPE CLEANUP ---
    # Force targets to pure numpy float arrays to bypass pandas object-dtype issues
    y_train_win = np.asarray(y_train_win).astype(float).astype(int)
    y_test_win = np.asarray(y_test_win).astype(float).astype(int)
    y_train_margin = np.asarray(y_train_margin).astype(float)
    y_test_margin = np.asarray(y_test_margin).astype(float)
    
    # Force feature matrices to be purely numeric and fill any missing values
    # This prevents LightGBM from seeing 'object' dtypes in the features
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    # Ensure no duplicate columns and only keep numeric feature columns
    final_feature_cols = clean_feature_cols
    
    X_train = X_train[final_feature_cols]
    X_test = X_test[final_feature_cols]
    
    # --- END DTYPE CLEANUP ---

    def train_model_pair(lgbm_flag):
        """Helper to train a spread and win model pair."""
        name = "LGBM" if lgbm_flag else "Random Forest"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training {name} spread model...")
        # Spread model
        if lgbm_flag:
            spread_est = LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=7,
                num_leaves=31, random_state=random_state, verbose=-1
            )
        else:
            spread_est = RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_split=10,
                min_samples_leaf=5, random_state=random_state, n_jobs=-1
            )
        
        # OOF spread for win model anchor
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating OOF spread predictions for {name} win model...")
        n_splits_kf = min(5, max(2, len(X_train_base) // 75))
        kfold_kf = KFold(n_splits=n_splits_kf, shuffle=True, random_state=random_state)
        oof_spread_kf = np.zeros(len(X_train_base))
        
        X_train_base_np_kf = X_train_base.values.astype(float)
        for fold, (t_idx, v_idx) in enumerate(kfold_kf.split(X_train_base_np_kf)):
            print(f"  Fold {fold+1}/{n_splits_kf}...")
            reg_kf = clone(spread_est)
            reg_kf.fit(X_train_base_np_kf[t_idx], y_train_margin[t_idx], sample_weight=w_train[t_idx])
            oof_spread_kf[v_idx] = reg_kf.predict(X_train_base_np_kf[v_idx])
        
        # Final spread model
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fitting final {name} spread model...")
        final_spread = clone(spread_est)
        final_spread.fit(X_train_base_np_kf, y_train_margin, sample_weight=w_train)
        
        # Win model with spread as anchor
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training {name} win probability model...")
        X_train_win_kf = X_train_base.copy()
        X_train_win_kf[spread_feature_name] = oof_spread_kf
        
        if lgbm_flag:
            base_win = LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=7,
                num_leaves=31, random_state=random_state, verbose=-1
            )
        else:
            base_win = RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_split=10,
                min_samples_leaf=5, random_state=random_state, n_jobs=-1
            )
            
        final_win = CalibratedClassifierCV(base_win, method='isotonic', cv=5, n_jobs=-1)
        final_win.fit(X_train_win_kf.values.astype(float), y_train_win, sample_weight=w_train)
        
        return final_spread, final_win

    # Keep base feature matrices for each model
    spread_feature_cols = final_feature_cols.copy()
    X_train_base = X_train.copy()
    X_test_base = X_test.copy()
    X_full_base = X.copy()
    spread_feature_name = 'model_spread_feature'

    if use_ensemble:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training ENSEMBLE (LGBM + Random Forest)...")
        spread_lgbm, win_lgbm = train_model_pair(True)
        spread_rf, win_rf = train_model_pair(False)
        
        # Package as ensemble
        spread_model = {'lgbm': spread_lgbm, 'rf': spread_rf}
        win_prob_model = {'lgbm': win_lgbm, 'rf': win_rf}
        model_type = 'ensemble'
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training {'LGBM' if use_lgbm else 'Random Forest'} model pair...")
        spread_model, win_prob_model = train_model_pair(use_lgbm)
        model_type = 'lightgbm_calibrated' if use_lgbm else 'rf_calibrated'

    # Evaluation (using the ensemble or single model)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating model performance...")
    def get_predictions(s_mod, w_mod, X_spread_eval, X_win_eval):
        if isinstance(s_mod, dict):
            # Average predictions for ensemble
            s_preds = [m.predict(X_spread_eval.values.astype(float)) for m in s_mod.values()]
            s_pred = np.mean(s_preds, axis=0)
            
            # For win model, we need to inject the averaged spread
            X_win_eval_ens = X_win_eval.copy()
            X_win_eval_ens[spread_feature_name] = s_pred
            
            w_probs = [m.predict_proba(X_win_eval_ens.values.astype(float))[:, 1] for m in w_mod.values()]
            w_prob = np.mean(w_probs, axis=0)
            w_class = (w_prob > 0.5).astype(int)
            return s_pred, w_prob, w_class
        else:
            s_pred = s_mod.predict(X_spread_eval.values.astype(float))
            X_win_eval_single = X_win_eval.copy()
            X_win_eval_single[spread_feature_name] = s_pred
            w_prob = w_mod.predict_proba(X_win_eval_single.values.astype(float))[:, 1]
            w_class = (w_prob > 0.5).astype(int)
            return s_pred, w_prob, w_class

    y_pred_spread, y_pred_proba, y_pred_win = get_predictions(spread_model, win_prob_model, X_test_base, X_test_base)
    
    mae = mean_absolute_error(y_test_margin, y_pred_spread, sample_weight=w_test)
    rmse = np.sqrt(mean_squared_error(y_test_margin, y_pred_spread, sample_weight=w_test))
    accuracy = accuracy_score(y_test_win, y_pred_win, sample_weight=w_test)
    brier = brier_score_loss(y_test_win, y_pred_proba, sample_weight=w_test)

    print(f"\nModel Evaluation ({model_type.upper()}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  MAE: {mae:.2f} points")
    print(f"  Brier Score: {brier:.4f}")
    
    # Inject spread predictions for full dataset (needed for link calibration)
    spread_pred_full, _, _ = get_predictions(spread_model, win_prob_model, X_full_base, X_full_base)

    # Calibrate link function using model-predicted spreads
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calibrating link function...")
    print("\nCalibrating link function...")
    try:
        calibration_spreads = pd.Series(spread_pred_full)
        link_params = fit_link_function(calibration_spreads, y_win, max_abs_a=0.3)
    except Exception as e:
        print(f"  Warning: Link calibration failed ({e}); falling back to defaults.")
        link_params = (0.15, 2.5)
    link_a, link_b = link_params
    print(f"Link function parameters: a={link_a:.4f}, b={link_b:.4f}")
    
    # Sign agreement
    from src.models.link_function import spread_to_win_prob
    win_prob_from_spread = spread_to_win_prob(y_pred_spread, link_a, link_b)
    disagreement = np.abs(y_pred_proba - win_prob_from_spread)
    mean_disagreement = np.mean(disagreement)
    sign_agreement = ((y_pred_proba > 0.5) == (y_pred_spread > 0)).mean()
    print(f"  Sign agreement: {sign_agreement:.1%}")
    print(f"  Mean disagreement: {mean_disagreement:.4f}")

    # --- TRAIN OUTPUT ENSEMBLE (STACKING) ---
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training stacking meta-ensemble...")
    print("\nTraining output-level ensemble (stacking)...")
    # Get OOF predictions for the ensemble to train on
    # We use the same KFold as before for consistency
    oof_spread = np.zeros(len(X_train_base))
    oof_win_prob = np.zeros(len(X_train_base))
    
    n_splits_ens = min(5, max(2, len(X_train_base) // 75))
    kfold_ens = KFold(n_splits=n_splits_ens, shuffle=True, random_state=random_state)
    
    for fold, (t_idx, v_idx) in enumerate(kfold_ens.split(X_train_base.values)):
        print(f"  Stacking Fold {fold+1}/{n_splits_ens}...")
        # 1. Train spread model on t_idx, predict on v_idx
        if use_ensemble:
            # For simplicity in stacking, we use the first model or a simplified version
            # But better to use a combined prediction
            s_preds_oof = []
            w_preds_oof = []
            
            # LGBM
            s_lgbm = LGBMRegressor(n_estimators=100, max_depth=5, verbose=-1, random_state=random_state)
            s_lgbm.fit(X_train_base.iloc[t_idx].values, y_train_margin[t_idx], sample_weight=w_train[t_idx])
            s_pred_v = s_lgbm.predict(X_train_base.iloc[v_idx].values)
            s_preds_oof.append(s_pred_v)
            
            # RF
            s_rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_state)
            s_rf.fit(X_train_base.iloc[t_idx].values, y_train_margin[t_idx], sample_weight=w_train[t_idx])
            s_pred_v_rf = s_rf.predict(X_train_base.iloc[v_idx].values)
            s_preds_oof.append(s_pred_v_rf)
            
            s_pred_v_avg = np.mean(s_preds_oof, axis=0)
            oof_spread[v_idx] = s_pred_v_avg
            
            # Win prob (with spread anchor)
            X_win_v = X_train_base.iloc[v_idx].copy()
            X_win_v[spread_feature_name] = s_pred_v_avg
            
            # Win LGBM
            w_lgbm = LGBMClassifier(n_estimators=100, max_depth=5, verbose=-1, random_state=random_state)
            # Use X_train_win_kf logic but for this fold
            X_train_win_fold = X_train_base.iloc[t_idx].copy()
            X_train_win_fold[spread_feature_name] = y_train_margin[t_idx]
            w_lgbm.fit(X_train_win_fold.values, y_train_win[t_idx], sample_weight=w_train[t_idx])
            w_pred_v = w_lgbm.predict_proba(X_win_v.values)[:, 1]
            w_preds_oof.append(w_pred_v)
            
            # Win RF
            w_rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_state)
            w_rf.fit(X_train_win_fold.values, y_train_win[t_idx], sample_weight=w_train[t_idx])
            w_pred_v_rf = w_rf.predict_proba(X_win_v.values)[:, 1]
            w_preds_oof.append(w_pred_v_rf)
            
            oof_win_prob[v_idx] = np.mean(w_preds_oof, axis=0)
        else:
            # Single model pair
            cur_s, cur_w = train_model_pair(use_lgbm)
            oof_spread[v_idx] = cur_s.predict(X_train_base.iloc[v_idx].values)
            
            X_win_v = X_train_base.iloc[v_idx].copy()
            X_win_v[spread_feature_name] = oof_spread[v_idx]
            oof_win_prob[v_idx] = cur_w.predict_proba(X_win_v.values)[:, 1]
            
    # Also add link-function probability as a feature to the meta-model
    oof_win_from_spread = spread_to_win_prob(oof_spread, link_a, link_b)
    
    # Meta-features: [win_prob, win_from_spread, spread]
    meta_X_train = np.column_stack([oof_win_prob, oof_win_from_spread, oof_spread])
    meta_model = LogisticRegression(fit_intercept=True)
    meta_model.fit(meta_X_train, y_train_win, sample_weight=w_train)
    
    # Evaluate Meta-model on test set
    win_from_spread_test = spread_to_win_prob(y_pred_spread, link_a, link_b)
    meta_X_test = np.column_stack([y_pred_proba, win_from_spread_test, y_pred_spread])
    y_pred_meta_proba = meta_model.predict_proba(meta_X_test)[:, 1]
    y_pred_meta_class = (y_pred_meta_proba > 0.5).astype(int)
    
    meta_accuracy = accuracy_score(y_test_win, y_pred_meta_class, sample_weight=w_test)
    meta_brier = brier_score_loss(y_test_win, y_pred_meta_proba, sample_weight=w_test)
    
    print(f"\nMeta-Ensemble Evaluation:")
    print(f"  Accuracy: {meta_accuracy:.4f} (vs {accuracy:.4f} base)")
    print(f"  Brier Score: {meta_brier:.4f} (vs {brier:.4f} base)")
    
    # --- END ENSEMBLE TRAINING ---

    # Calculate feature medians
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating feature medians and saving models...")
    win_feature_cols = clean_feature_cols.copy() + [spread_feature_name]
    X_train_win_final = X_train_base.copy()
    X_train_win_final[spread_feature_name] = 0 # Dummy for median calc
    feature_medians = X_train_win_final.apply(pd.to_numeric, errors='coerce').median().to_dict()
    
    # Save models
    league_lower = league.lower()
    
    # Win probability model
    win_prob_path = os.path.join(models_dir, f'win_prob_model_{league_lower}_{model_version}.pkl')
    win_prob_data = {
        'model': win_prob_model,
        'model_type': model_type,
        'ensemble': use_ensemble,
        'feature_names': win_feature_cols,
        'win_feature_names': win_feature_cols,
        'spread_feature_names': spread_feature_cols,
        'accuracy': accuracy,
        'brier_score': brier,
        'trained_date': datetime.now().isoformat(),
        'n_features': len(win_feature_cols),
        'n_samples': len(X_train_base),
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
        'model_type': model_type,
        'ensemble': use_ensemble,
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
    
    # Meta-ensemble model
    meta_path = os.path.join(models_dir, f'meta_ensemble_{league_lower}_{model_version}.pkl')
    meta_data = {
        'model': meta_model,
        'features': ['win_prob_model', 'win_prob_from_spread', 'predicted_spread'],
        'trained_date': datetime.now().isoformat(),
        'accuracy': meta_accuracy,
        'brier_score': meta_brier,
        'league': league
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_data, f)
    print(f"Saved meta-ensemble to {meta_path}")
    
    return {
        'win_prob_model_path': win_prob_path,
        'spread_model_path': spread_path,
        'link_function_path': link_path,
        'feature_medians_path': medians_path,
        'meta_ensemble_path': meta_path,
        'accuracy': accuracy,
        'meta_accuracy': meta_accuracy,
        'brier_score': brier,
        'meta_brier': meta_brier,
        'mae': mae,
        'rmse': rmse,
        'link_params': link_params,
        'mean_disagreement': mean_disagreement,
        'sign_agreement': sign_agreement,
        'win_feature_names': win_feature_cols,
        'spread_feature_names': spread_feature_cols,
        'n_samples': len(X_train_base)
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train and export Sports-Edge models.")
    parser.add_argument('--league', default='NFL', choices=['NFL', 'NBA'], help="League to train.")
    parser.add_argument('--start-season', type=int, default=2021, help="First season to include.")
    parser.add_argument('--end-season', type=int, default=2024, help="Last season to include (inclusive).")
    parser.add_argument('--model-version', default='v1', help="Model version tag for saved artifacts.")
    parser.add_argument('--features-path', type=str, help="Optional path to cached features (csv/parquet).")
    parser.add_argument('--save-features-path', type=str, help="Optional path to save engineered training set.")
    parser.add_argument('--use-rf', action='store_true', help="Use RandomForest instead of LightGBM.")
    parser.add_argument('--ensemble', action='store_true', help="Train both LGBM and RF and ensemble them.")
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
        use_ensemble=args.ensemble,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print("\nTraining complete:")
    for key in ['accuracy', 'brier_score', 'mae', 'rmse', 'mean_disagreement', 'sign_agreement']:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
