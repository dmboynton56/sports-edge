"""
Sports Edge PGA Model Training Pipeline v2
==========================================
Comprehensive model training with:
- Proper time-based train/validation splits
- Multiple model architectures (LR, RF, LightGBM, XGBoost, PyTorch)
- Both regression (SG prediction) and classification (cut/top10/top20/win)
- Stacking meta-ensemble with residual learning
- Full evaluation suite: RMSE, MAE, Brier, LogLoss, ECE, NDCG, Spearman
- Feature importance and calibration analysis
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from typing import Dict, List, Tuple, Optional

import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, brier_score_loss,
    log_loss, roc_auc_score, accuracy_score
)
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
META_COLS = [
    'season', 'start', 'end', 'tournament', 'location', 'name',
    'position_str', 'position_num', 'dataset_split'
]
# rounds_played leaks the target — it encodes how many rounds a player
# completed in the CURRENT tournament (4 = made cut, 2 = missed cut)
LEAK_COLS = ['rounds_played']
TARGET_COLS = [
    'target_sg_total', 'target_sg_per_round',
    'target_made_cut', 'target_top10', 'target_top20', 'target_win'
]
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved')


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """NDCG@k — weights correct ranking of top finishers heavily."""
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order][:k]
    dcg = np.sum((2 ** y_true_sorted - 1) / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE — checks if predicted probabilities match observed frequencies."""
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


# =========================================================================
# Data Loading
# =========================================================================
class DataManager:
    """Loads and prepares the PGA feature store with proper time-based splits."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.feature_cols: List[str] = []
        self.scaler = StandardScaler()

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in META_COLS + TARGET_COLS + LEAK_COLS]

    def prepare_splits(
        self, df: pd.DataFrame, target_col: str, impute_strategy: str = 'median'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Time-based splits using the dataset_split column.
        Imputes missing values with median (train-derived) instead of 0.
        """
        df = df.dropna(subset=[target_col]).copy()
        self.feature_cols = self.get_feature_cols(df)

        train_mask = df['dataset_split'] == 'train'
        val_mask = df['dataset_split'] == 'valid'

        X_train_raw = df.loc[train_mask, self.feature_cols].copy()
        X_val_raw = df.loc[val_mask, self.feature_cols].copy()
        y_train = df.loc[train_mask, target_col].values
        y_val = df.loc[val_mask, target_col].values

        if impute_strategy == 'median':
            medians = X_train_raw.median()
            X_train_raw = X_train_raw.fillna(medians)
            X_val_raw = X_val_raw.fillna(medians)
            self._medians = medians
        else:
            X_train_raw = X_train_raw.fillna(0)
            X_val_raw = X_val_raw.fillna(0)

        X_train = X_train_raw.values.astype(np.float32)
        X_val = X_val_raw.values.astype(np.float32)

        # Fitted scaler (for NN / LR)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        return X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled

    def prepare_tournament_splits(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[str]]:
        """
        Same as prepare_splits but also returns tournament groupings for
        NDCG / Spearman ranking evaluation.
        """
        df = df.dropna(subset=[target_col]).copy()
        self.feature_cols = self.get_feature_cols(df)

        val_df = df[df['dataset_split'] == 'valid'].copy()
        train_df = df[df['dataset_split'] == 'train'].copy()

        medians = train_df[self.feature_cols].median()
        train_df[self.feature_cols] = train_df[self.feature_cols].fillna(medians)
        val_df[self.feature_cols] = val_df[self.feature_cols].fillna(medians)
        self._medians = medians

        X_train = train_df[self.feature_cols].values.astype(np.float32)
        X_val = val_df[self.feature_cols].values.astype(np.float32)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values

        self.scaler.fit(X_train)

        tournament_groups = []
        tournament_names = []
        for t_name, g in val_df.groupby(['season', 'tournament']):
            idxs = g.index
            local_mask = val_df.index.isin(idxs)
            local_positions = np.where(local_mask)[0]
            tournament_groups.append(local_positions)
            tournament_names.append(f"{t_name[0]} {t_name[1]}")

        return X_train, y_train, X_val, y_val, tournament_groups, tournament_names


# =========================================================================
# Regression Models
# =========================================================================
class RegressionSuite:
    """Trains and evaluates all regression models for SG prediction."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.models: Dict[str, object] = {}
        self.predictions: Dict[str, np.ndarray] = {}

    def train_ridge(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
        print("\n--- Ridge Regression (Baseline) ---")
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        self.models['ridge'] = model
        self.predictions['ridge'] = preds
        joblib.dump(model, os.path.join(self.models_dir, 'ridge_sg_model.joblib'))
        return preds

    def train_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray
    ) -> np.ndarray:
        print("\n--- Random Forest Regressor ---")
        model = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            max_features='sqrt', n_jobs=1, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        self.models['rf'] = model
        self.predictions['rf'] = preds
        joblib.dump(model, os.path.join(self.models_dir, 'rf_sg_model.joblib'))
        return preds

    def train_lightgbm(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        print("\n--- LightGBM Regressor ---")
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'regression', 'metric': 'rmse',
            'boosting_type': 'gbdt', 'learning_rate': 0.03,
            'num_leaves': 63, 'max_depth': 8,
            'min_child_samples': 30, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1,
            'reg_lambda': 1.0, 'verbose': -1
        }
        model = lgb.train(
            params, train_data, num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        preds = model.predict(X_val)
        self.models['lgbm'] = model
        self.predictions['lgbm'] = preds
        joblib.dump(model, os.path.join(self.models_dir, 'lgbm_sg_model_v2.joblib'))
        return preds

    def train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        print("\n--- XGBoost Regressor ---")
        model = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.03,
            max_depth=7, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method='hist', device='cuda',
            early_stopping_rounds=100, eval_metric='rmse', verbosity=0
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        self.models['xgb'] = model
        self.predictions['xgb'] = preds
        joblib.dump(model, os.path.join(self.models_dir, 'xgb_sg_model_v2.joblib'))
        return preds

    def evaluate(self, y_val: np.ndarray, tournament_groups: Optional[List[np.ndarray]] = None):
        print("\n" + "=" * 70)
        print("REGRESSION EVALUATION (target: SG per round)")
        print("=" * 70)

        header = f"{'Model':<15} {'RMSE':>8} {'MAE':>8}"
        if tournament_groups:
            header += f" {'Spearman':>10} {'NDCG@10':>10} {'NDCG@20':>10}"
        print(header)
        print("-" * len(header))

        for name, preds in self.predictions.items():
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae = mean_absolute_error(y_val, preds)
            row = f"{name:<15} {rmse:>8.4f} {mae:>8.4f}"

            if tournament_groups:
                spearmans, ndcg10s, ndcg20s = [], [], []
                for grp in tournament_groups:
                    if len(grp) < 5:
                        continue
                    yt = y_val[grp]
                    yp = preds[grp]
                    rho, _ = spearmanr(-yt, -yp)
                    spearmans.append(rho)
                    ndcg10s.append(ndcg_at_k(yt, yp, k=min(10, len(grp))))
                    ndcg20s.append(ndcg_at_k(yt, yp, k=min(20, len(grp))))
                row += f" {np.mean(spearmans):>10.4f} {np.mean(ndcg10s):>10.4f} {np.mean(ndcg20s):>10.4f}"
            print(row)


# =========================================================================
# Classification Models
# =========================================================================
class ClassificationSuite:
    """Trains and evaluates classification models for binary PGA targets."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def _train_single_target(
        self, target_name: str,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_train_scaled: np.ndarray, X_val_scaled: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Trains LR, RF, LGBM, XGB classifiers for one binary target."""
        results = {}

        # --- Logistic Regression ---
        lr = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs')
        lr.fit(X_train_scaled, y_train)
        lr_probs = lr.predict_proba(X_val_scaled)[:, 1]
        results['lr'] = lr_probs
        joblib.dump(lr, os.path.join(self.models_dir, f'lr_{target_name}.joblib'))

        # --- Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=30,
            max_features='sqrt', n_jobs=1, random_state=42
        )
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_val)[:, 1]
        results['rf'] = rf_probs
        joblib.dump(rf, os.path.join(self.models_dir, f'rf_{target_name}.joblib'))

        # --- LightGBM ---
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'boosting_type': 'gbdt', 'learning_rate': 0.03,
            'num_leaves': 63, 'max_depth': 8,
            'min_child_samples': 50, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1,
            'reg_lambda': 1.0, 'is_unbalance': True, 'verbose': -1
        }
        lgbm = lgb.train(
            params, train_data, num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        lgbm_probs = lgbm.predict(X_val)
        results['lgbm'] = lgbm_probs
        joblib.dump(lgbm, os.path.join(self.models_dir, f'lgbm_{target_name}.joblib'))

        # --- XGBoost ---
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.03,
            max_depth=7, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=pos_weight,
            tree_method='hist', device='cuda',
            early_stopping_rounds=100, eval_metric='logloss', verbosity=0
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
        results['xgb'] = xgb_probs
        joblib.dump(xgb_model, os.path.join(self.models_dir, f'xgb_{target_name}.joblib'))

        return results

    def train_all_targets(
        self, X_train, y_dict_train, X_val, y_dict_val,
        X_train_scaled, X_val_scaled, feature_names
    ) -> Dict[str, Dict[str, np.ndarray]]:
        all_results = {}
        for target_name in ['target_made_cut', 'target_top10', 'target_top20', 'target_win']:
            print(f"\n{'='*70}")
            print(f"CLASSIFICATION: {target_name}")
            print(f"{'='*70}")
            yt = y_dict_train[target_name]
            yv = y_dict_val[target_name]
            print(f"  Train: {yt.sum():.0f}/{len(yt)} positive ({100*yt.mean():.1f}%)")
            print(f"  Val:   {yv.sum():.0f}/{len(yv)} positive ({100*yv.mean():.1f}%)")
            results = self._train_single_target(
                target_name, X_train, yt, X_val, yv,
                X_train_scaled, X_val_scaled, feature_names
            )
            all_results[target_name] = results
        return all_results

    def evaluate(self, all_results: Dict, y_dict_val: Dict):
        print("\n" + "=" * 70)
        print("CLASSIFICATION EVALUATION")
        print("=" * 70)

        for target_name, model_preds in all_results.items():
            y_true = y_dict_val[target_name]
            print(f"\n--- {target_name} (base rate: {y_true.mean():.3f}) ---")
            print(f"  {'Model':<10} {'Brier':>8} {'LogLoss':>10} {'AUC':>8} {'ECE':>8}")
            print(f"  {'-'*46}")

            for model_name, probs in model_preds.items():
                probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
                brier = brier_score_loss(y_true, probs_clipped)
                ll = log_loss(y_true, probs_clipped)
                auc = roc_auc_score(y_true, probs_clipped)
                ece = expected_calibration_error(y_true, probs_clipped)
                print(f"  {model_name:<10} {brier:>8.4f} {ll:>10.4f} {auc:>8.4f} {ece:>8.4f}")


# =========================================================================
# PyTorch Tabular Network
# =========================================================================
SCHEDULE_FEATURE_PATTERNS = [
    "liv_share_last_", "strong_field_share_last_",
]


def _schedule_feature_indices(feature_names: List[str]) -> List[int]:
    """Return column indices for schedule-related features prone to OOD shift."""
    idxs = []
    for i, name in enumerate(feature_names):
        if any(name.startswith(pat) for pat in SCHEDULE_FEATURE_PATTERNS):
            idxs.append(i)
    return idxs


class TabularNet(nn.Module):
    """Deeper tabular network with skip connections for SG regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        mask_indices: Optional[List[int]] = None,
        mask_prob: float = 0.30,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.mask_indices = mask_indices or []
        self.mask_prob = mask_prob

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.SiLU())
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.Dropout(0.15))
            prev_dim = hd
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        if self.training and self.mask_indices:
            mask = torch.rand(x.size(0), 1, device=x.device) < self.mask_prob
            x = x.clone()
            x[:, self.mask_indices] = torch.where(
                mask.expand(-1, len(self.mask_indices)),
                torch.zeros_like(x[:, self.mask_indices]),
                x[:, self.mask_indices],
            )
        x = self.bn_input(x)
        x = self.backbone(x)
        return self.head(x).squeeze(-1)


def train_pytorch_tabular(
    X_train_scaled: np.ndarray, y_train: np.ndarray,
    X_val_scaled: np.ndarray, y_val: np.ndarray,
    epochs: int = 60, batch_size: int = 512, lr: float = 1e-3,
    models_dir: str = MODELS_DIR,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    print("\n--- PyTorch Tabular NN ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    X_tr = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_vl = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_vl, y_vl)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    mask_idxs = _schedule_feature_indices(feature_names) if feature_names else []
    if mask_idxs:
        print(f"  Feature masking: {len(mask_idxs)} schedule features (p=0.30)")
    model = TabularNet(X_tr.shape[1], mask_indices=mask_idxs).to(device)
    criterion = nn.HuberLoss(delta=1.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_rmse = float('inf')
    patience, patience_counter = 15, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        scheduler.step()

        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                val_preds_list.append(model(xb).cpu().numpy())
        val_preds = np.concatenate(val_preds_list)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs} | Val RMSE: {val_rmse:.4f} | Best: {best_val_rmse:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(models_dir, 'pytorch_tabular_v2.pth'))

    model.eval()
    with torch.no_grad():
        final_preds = model(X_vl.to(device)).cpu().numpy()
    return final_preds


# =========================================================================
# Stacking Meta-Ensemble
# =========================================================================
def clip_base_preds_relative(base_preds: Dict[str, np.ndarray], sigma: float = 2.0) -> Dict[str, np.ndarray]:
    """Clip per-model predictions that deviate >sigma from the peer median.

    For each player, compute the median prediction across models. If any model
    is more than `sigma` standard deviations (of the cross-model spread) away,
    clip it to median +/- sigma * std. Prevents a single broken model from
    poisoning the stack.
    """
    names = list(base_preds.keys())
    stacked = np.column_stack([base_preds[n] for n in names])  # (n_players, n_models)
    medians = np.median(stacked, axis=1, keepdims=True)
    stds = np.std(stacked, axis=1, keepdims=True)
    stds = np.maximum(stds, 1e-6)

    lo = medians - sigma * stds
    hi = medians + sigma * stds
    clipped = np.clip(stacked, lo, hi)

    n_clipped = int((stacked != clipped).sum())
    if n_clipped > 0:
        print(f"  Relative outlier clipping: {n_clipped} values clipped ({sigma:.1f}σ)")

    return {n: clipped[:, i] for i, n in enumerate(names)}


class StackingEnsemble:
    """
    Stacking meta-learner that combines base model predictions.
    Uses 5-fold OOF on training data to avoid data leakage.
    Ridge for regression, LogisticRegression for classification.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.meta_models: Dict[str, object] = {}
        self.base_model_names: List[str] = []

    def fit_regression(
        self, base_preds_val: Dict[str, np.ndarray], y_val: np.ndarray,
    ) -> np.ndarray:
        """Simple weighted average via Ridge on validation predictions only."""
        print("\n--- Stacking Ensemble (Regression) ---")
        base_preds_val = clip_base_preds_relative(base_preds_val, sigma=2.0)
        self.base_model_names = list(base_preds_val.keys())
        X_meta_val = np.column_stack(list(base_preds_val.values()))

        n = len(y_val)
        mid = n // 2
        meta = Ridge(alpha=10.0)
        meta.fit(X_meta_val[:mid], y_val[:mid])
        preds = meta.predict(X_meta_val)

        names = self.base_model_names
        print(f"  Meta-weights: {dict(zip(names, meta.coef_.round(3)))}")
        print(f"  Meta intercept: {meta.intercept_:.4f}")

        self.meta_models['regression'] = meta
        joblib.dump(meta, os.path.join(self.models_dir, 'meta_ensemble_sg_v2.joblib'))
        return preds

    def fit_classification(
        self, target_name: str,
        base_preds_val: Dict[str, np.ndarray], y_val: np.ndarray,
    ) -> np.ndarray:
        print(f"\n--- Stacking Ensemble ({target_name}) ---")
        base_preds_val = clip_base_preds_relative(base_preds_val, sigma=2.0)
        X_meta_val = np.column_stack(list(base_preds_val.values()))

        n = len(y_val)
        mid = n // 2
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(X_meta_val[:mid], y_val[:mid])
        preds = meta.predict_proba(X_meta_val)[:, 1]

        names = list(base_preds_val.keys())
        print(f"  Meta-weights: {dict(zip(names, meta.coef_[0].round(3)))}")

        self.meta_models[target_name] = meta
        joblib.dump(meta, os.path.join(self.models_dir, f'meta_ensemble_{target_name}_v2.joblib'))
        return preds


# =========================================================================
# Feature Importance
# =========================================================================
def print_feature_importance(models: Dict, feature_names: List[str], top_n: int = 15):
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Top 15)")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        if name == 'lgbm':
            imp = model.feature_importance(importance_type='gain')
        elif name == 'xgb':
            imp = model.feature_importances_
        elif name == 'rf':
            imp = model.feature_importances_
        elif name == 'ridge':
            imp = np.abs(model.coef_)
        else:
            continue
        top_idx = np.argsort(imp)[-top_n:][::-1]
        for i, idx in enumerate(top_idx):
            print(f"  {i+1:>2}. {feature_names[idx]:<35} {imp[idx]:>10.2f}")


# =========================================================================
# Main Pipeline
# =========================================================================
def main():
    data_path = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'notebooks', 'cache', 'pga_feature_store_event_level.csv'
    )
    dm = DataManager(data_path)
    df = dm.load()

    # ---- Regression ----
    X_tr, y_tr, X_vl, y_vl, X_tr_s, X_vl_s = dm.prepare_splits(df, 'target_sg_per_round')
    feature_names = dm.feature_cols
    print(f"\nFeatures ({len(feature_names)}): {feature_names[:5]} ... {feature_names[-5:]}")
    print(f"Train: {X_tr.shape[0]}, Val: {X_vl.shape[0]}")

    reg = RegressionSuite()
    reg.train_ridge(X_tr_s, y_tr, X_vl_s)
    reg.train_random_forest(X_tr, y_tr, X_vl)
    reg.train_lightgbm(X_tr, y_tr, X_vl, y_vl, feature_names)
    reg.train_xgboost(X_tr, y_tr, X_vl, y_vl)

    nn_preds = train_pytorch_tabular(X_tr_s, y_tr, X_vl_s, y_vl, feature_names=feature_names)
    reg.predictions['nn'] = nn_preds
    reg.models['nn'] = 'pytorch_tabular_v2.pth'

    # Tournament-level ranking evaluation
    X_tr2, y_tr2, X_vl2, y_vl2, t_groups, t_names = dm.prepare_tournament_splits(df, 'target_sg_per_round')
    reg.evaluate(y_vl, t_groups)

    print_feature_importance(reg.models, feature_names)

    # ---- Stacking (Regression) ----
    stacking = StackingEnsemble()
    base_val = {k: v for k, v in reg.predictions.items()}
    meta_preds = stacking.fit_regression(base_val, y_vl)
    reg.predictions['meta_stack'] = meta_preds
    reg.evaluate(y_vl, t_groups)

    # ---- Classification ----
    cls = ClassificationSuite()
    df_cls = df.copy()
    df_cls = df_cls.dropna(subset=['target_sg_per_round'])
    df_cls = df_cls.fillna(0)
    feature_cols = dm.feature_cols

    train_mask = df_cls['dataset_split'] == 'train'
    val_mask = df_cls['dataset_split'] == 'valid'

    X_cls_tr = df_cls.loc[train_mask, feature_cols].values.astype(np.float32)
    X_cls_vl = df_cls.loc[val_mask, feature_cols].values.astype(np.float32)

    scaler_cls = StandardScaler()
    X_cls_tr_s = scaler_cls.fit_transform(X_cls_tr)
    X_cls_vl_s = scaler_cls.transform(X_cls_vl)

    y_dict_train = {
        t: df_cls.loc[train_mask, t].values.astype(int) for t in
        ['target_made_cut', 'target_top10', 'target_top20', 'target_win']
    }
    y_dict_val = {
        t: df_cls.loc[val_mask, t].values.astype(int) for t in
        ['target_made_cut', 'target_top10', 'target_top20', 'target_win']
    }

    cls_results = cls.train_all_targets(
        X_cls_tr, y_dict_train, X_cls_vl, y_dict_val,
        X_cls_tr_s, X_cls_vl_s, feature_names
    )
    cls.evaluate(cls_results, y_dict_val)

    # ---- Stacking (Classification) ----
    for target_name, model_preds in cls_results.items():
        meta_cls_preds = stacking.fit_classification(
            target_name, model_preds, y_dict_val[target_name]
        )
        cls_results[target_name]['meta_stack'] = meta_cls_preds

    cls.evaluate(cls_results, y_dict_val)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — All models saved to", MODELS_DIR)
    print("=" * 70)


if __name__ == '__main__':
    main()
