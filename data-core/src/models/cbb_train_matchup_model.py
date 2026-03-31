"""
CBB Matchup Model Training
===========================
Trains LightGBM + XGBoost ensemble for predicting head-to-head
NCAA tournament game outcomes (P(Team A wins)).

Uses expanding window time-series CV to prevent future data leakage:
  Fold 1: Train 2010-2015 -> Validate 2016
  Fold 2: Train 2010-2016 -> Validate 2017
  ...

Final meta-ensemble via Logistic Regression stacker.
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved')
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'cache')

FEATURE_COLS = [
    'off_eff_diff', 'def_eff_diff', 'net_eff_diff',
    'tempo_diff', 'tempo_mismatch',
    'off_efg_diff', 'off_to_rate_diff', 'off_or_rate_diff', 'off_ftr_diff',
    'def_efg_diff', 'def_to_rate_diff', 'def_or_rate_diff', 'def_ftr_diff',
    'win_pct_diff', 'sos_diff',
    'recent_win_pct_diff', 'recent_avg_margin_diff',
    'tourney_exp_diff',
    'a_off_eff', 'a_def_eff', 'a_net_eff', 'a_tempo',
    'a_off_efg', 'a_off_to_rate', 'a_off_or_rate', 'a_off_ftr',
    'a_def_efg', 'a_def_to_rate', 'a_def_or_rate', 'a_def_ftr',
    'a_win_pct', 'a_sos',
    'b_off_eff', 'b_def_eff', 'b_net_eff', 'b_tempo',
    'b_off_efg', 'b_off_to_rate', 'b_off_or_rate', 'b_off_ftr',
    'b_def_efg', 'b_def_to_rate', 'b_def_or_rate', 'b_def_ftr',
    'b_win_pct', 'b_sos',
]

TARGET_COL = 'team_a_wins'


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_pred - avg_true)
    return ece


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, label: str = '') -> Dict[str, float]:
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    metrics = {
        'log_loss': log_loss(y_true, y_prob),
        'brier': brier_score_loss(y_true, y_prob),
        'auc': roc_auc_score(y_true, y_prob),
        'ece': expected_calibration_error(y_true, y_prob),
        'accuracy': np.mean((y_prob > 0.5) == y_true),
    }
    if label:
        print(f"\n  {label}:")
        print(f"    Log Loss: {metrics['log_loss']:.4f}  |  Brier: {metrics['brier']:.4f}")
        print(f"    AUC:      {metrics['auc']:.4f}  |  ECE:   {metrics['ece']:.4f}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def check_upset_calibration(
    y_true: np.ndarray, y_prob: np.ndarray,
    seed_diffs: np.ndarray
) -> None:
    """
    Checks model calibration on major upsets (1v16, 2v15, 3v14).
    Underdog probability should be 1-5%, not 0%.
    """
    print("\n  Upset Calibration Check:")
    for matchup, sd_range in [
        ('1v16', (-15, -15)), ('2v15', (-13, -13)), ('3v14', (-11, -11))
    ]:
        mask = (seed_diffs >= sd_range[0]) & (seed_diffs <= sd_range[1])
        alt_mask = (seed_diffs >= -sd_range[1]) & (seed_diffs <= -sd_range[0])
        combined = mask | alt_mask
        if combined.sum() == 0:
            continue
        underdog_probs = np.where(
            seed_diffs[combined] < 0,
            y_prob[combined],
            1 - y_prob[combined]
        )
        avg_underdog = np.mean(underdog_probs)
        n_games = combined.sum() // 2
        actual_upsets = np.sum(
            (seed_diffs[combined] < 0) & (y_true[combined] == 1)
        ) + np.sum(
            (seed_diffs[combined] > 0) & (y_true[combined] == 0)
        )
        print(f"    {matchup}: Avg underdog prob = {avg_underdog:.3f} "
              f"({n_games} games, {actual_upsets} actual upsets)")


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
class CBBMatchupTrainer:
    """Trains and evaluates LightGBM + XGBoost matchup models."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.lgbm_model = None
        self.xgb_model = None
        self.meta_model = None
        self.scaler = StandardScaler()

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].fillna(0).values.astype(np.float32)
        if TARGET_COL in df.columns:
            y = df[TARGET_COL].values.astype(np.float32)
        else:
            y = np.zeros(len(df), dtype=np.float32)
        return X, y

    def train_lightgbm(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> np.ndarray:
        print("\n--- LightGBM Classifier ---")
        available = [c for c in FEATURE_COLS if True]
        feature_names = [f'f_{i}' for i in range(X_train.shape[1])]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
        }

        self.lgbm_model = lgb.train(
            params, train_data, num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        preds = self.lgbm_model.predict(X_val)
        evaluate_predictions(y_val, preds, 'LightGBM')
        joblib.dump(self.lgbm_model, os.path.join(self.models_dir, 'lgbm_cbb_matchup.joblib'))
        return preds

    def train_xgboost(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> np.ndarray:
        print("\n--- XGBoost Classifier ---")
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            device='cuda' if use_gpu else 'cpu',
            early_stopping_rounds=100,
            eval_metric='logloss',
            verbosity=0,
        )

        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = self.xgb_model.predict_proba(X_val)[:, 1]
        evaluate_predictions(y_val, preds, 'XGBoost')
        joblib.dump(self.xgb_model, os.path.join(self.models_dir, 'xgb_cbb_matchup.joblib'))
        return preds

    def train_meta_ensemble(
        self,
        lgbm_preds: np.ndarray, xgb_preds: np.ndarray,
        y_val: np.ndarray
    ) -> np.ndarray:
        print("\n--- Meta Ensemble (Logistic Regression Stacker) ---")
        X_meta = np.column_stack([lgbm_preds, xgb_preds])

        n = len(y_val)
        mid = n // 2
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
        self.meta_model.fit(X_meta[:mid], y_val[:mid])

        meta_preds = self.meta_model.predict_proba(X_meta)[:, 1]

        weights = self.meta_model.coef_[0]
        print(f"  Meta weights: LGBM={weights[0]:.3f}, XGB={weights[1]:.3f}")
        print(f"  Intercept: {self.meta_model.intercept_[0]:.4f}")

        evaluate_predictions(y_val, meta_preds, 'Meta Ensemble')
        joblib.dump(self.meta_model, os.path.join(self.models_dir, 'meta_cbb_matchup.joblib'))
        return meta_preds

    def expanding_window_cv(
        self, df: pd.DataFrame,
        train_start: int = 2010,
        val_start: int = 2016,
        val_end: int = 2025
    ) -> Dict[str, List[float]]:
        """
        Expanding Window Time-Series CV.
        Each fold trains on all seasons up to val_year - 1,
        then validates on val_year's tournament games.
        """
        print("=" * 70)
        print("EXPANDING WINDOW CV")
        print("=" * 70)

        fold_results = {'year': [], 'lgbm_ll': [], 'xgb_ll': [], 'meta_ll': []}

        all_lgbm_preds = []
        all_xgb_preds = []
        all_meta_preds = []
        all_y_val = []
        all_seed_diffs = []

        for val_year in range(val_start, val_end + 1):
            if val_year == 2020:
                continue

            train_df = df[
                (df['Season'] >= train_start) &
                (df['Season'] < val_year) &
                (df['Season'] != 2020)
            ]
            val_df = df[df['Season'] == val_year]

            if train_df.empty or val_df.empty:
                continue

            X_train, y_train = self.prepare_data(train_df)
            X_val, y_val = self.prepare_data(val_df)

            print(f"\n{'='*50}")
            print(f"Fold: Train {train_start}-{val_year-1} -> Val {val_year}")
            print(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")

            lgbm_preds = self.train_lightgbm(X_train, y_train, X_val, y_val)
            xgb_preds = self.train_xgboost(X_train, y_train, X_val, y_val)
            meta_preds = self.train_meta_ensemble(lgbm_preds, xgb_preds, y_val)

            lgbm_ll = log_loss(y_val, np.clip(lgbm_preds, 1e-7, 1 - 1e-7))
            xgb_ll = log_loss(y_val, np.clip(xgb_preds, 1e-7, 1 - 1e-7))
            meta_ll = log_loss(y_val, np.clip(meta_preds, 1e-7, 1 - 1e-7))

            fold_results['year'].append(val_year)
            fold_results['lgbm_ll'].append(lgbm_ll)
            fold_results['xgb_ll'].append(xgb_ll)
            fold_results['meta_ll'].append(meta_ll)

            all_lgbm_preds.append(lgbm_preds)
            all_xgb_preds.append(xgb_preds)
            all_meta_preds.append(meta_preds)
            all_y_val.append(y_val)

            if 'seed_diff' in val_df.columns:
                all_seed_diffs.append(val_df['seed_diff'].values)

        # Summary
        print("\n" + "=" * 70)
        print("CV SUMMARY")
        print("=" * 70)
        cv_df = pd.DataFrame(fold_results)
        print(cv_df.to_string(index=False))
        print(f"\nMean Log Loss:")
        print(f"  LGBM: {cv_df['lgbm_ll'].mean():.4f}")
        print(f"  XGB:  {cv_df['xgb_ll'].mean():.4f}")
        print(f"  Meta: {cv_df['meta_ll'].mean():.4f}")

        if all_y_val and all_seed_diffs:
            y_all = np.concatenate(all_y_val)
            meta_all = np.concatenate(all_meta_preds)
            sd_all = np.concatenate(all_seed_diffs)
            check_upset_calibration(y_all, meta_all, sd_all)

        return fold_results

    def train_final_model(self, df: pd.DataFrame) -> None:
        """Trains the final production model on ALL available data."""
        print("\n" + "=" * 70)
        print("TRAINING FINAL PRODUCTION MODEL")
        print("=" * 70)

        X, y = self.prepare_data(df)

        n = len(y)
        split = int(0.85 * n)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        print(f"Train: {len(y_train)}, Hold-out: {len(y_val)}")

        lgbm_preds = self.train_lightgbm(X_train, y_train, X_val, y_val)
        xgb_preds = self.train_xgboost(X_train, y_train, X_val, y_val)
        meta_preds = self.train_meta_ensemble(lgbm_preds, xgb_preds, y_val)

        self.scaler.fit(X)
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'cbb_scaler.joblib'))

        print(f"\nAll models saved to {self.models_dir}")

    def predict_matchup(
        self, X: np.ndarray
    ) -> np.ndarray:
        """Predict win probability for matchup features using the meta ensemble."""
        lgbm_preds = self.lgbm_model.predict(X)
        xgb_preds = self.xgb_model.predict_proba(X)[:, 1]
        X_meta = np.column_stack([lgbm_preds, xgb_preds])
        return self.meta_model.predict_proba(X_meta)[:, 1]

    def build_probability_matrix(
        self, matchup_df: pd.DataFrame, team_ids: List[int]
    ) -> np.ndarray:
        """
        Builds the NxN win probability matrix for N teams.
        matrix[i][j] = P(team_ids[i] beats team_ids[j])
        """
        n = len(team_ids)
        prob_matrix = np.full((n, n), 0.5)
        np.fill_diagonal(prob_matrix, 0.0)

        id_to_idx = {tid: i for i, tid in enumerate(team_ids)}
        X, _ = self.prepare_data(matchup_df)

        if X.shape[0] == 0:
            return prob_matrix

        preds = self.predict_matchup(X)

        for k, (_, row) in enumerate(matchup_df.iterrows()):
            a_id = int(row['TeamA_ID'])
            b_id = int(row['TeamB_ID'])
            if a_id in id_to_idx and b_id in id_to_idx:
                i = id_to_idx[a_id]
                j = id_to_idx[b_id]
                prob_matrix[i][j] = preds[k]

        return prob_matrix

    @staticmethod
    def load_trained(models_dir: str = MODELS_DIR) -> 'CBBMatchupTrainer':
        trainer = CBBMatchupTrainer(models_dir)
        lgbm_path = os.path.join(models_dir, 'lgbm_cbb_matchup.joblib')
        xgb_path = os.path.join(models_dir, 'xgb_cbb_matchup.joblib')
        meta_path = os.path.join(models_dir, 'meta_cbb_matchup.joblib')

        if os.path.exists(lgbm_path):
            trainer.lgbm_model = joblib.load(lgbm_path)
        if os.path.exists(xgb_path):
            trainer.xgb_model = joblib.load(xgb_path)
        if os.path.exists(meta_path):
            trainer.meta_model = joblib.load(meta_path)

        return trainer


def main():
    feature_store_path = os.path.join(CACHE_DIR, 'cbb_matchup_feature_store.csv')
    if not os.path.exists(feature_store_path):
        print(f"Feature store not found at {feature_store_path}")
        print("Run cbb_feature_engineering.py first to build it.")
        sys.exit(1)

    df = pd.read_csv(feature_store_path)
    print(f"Loaded feature store: {df.shape}")
    print(f"Seasons: {sorted(df['Season'].unique())}")

    trainer = CBBMatchupTrainer()

    cv_results = trainer.expanding_window_cv(df)

    trainer.train_final_model(df)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
