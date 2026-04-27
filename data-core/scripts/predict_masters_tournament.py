#!/usr/bin/env python3
"""
Pre-tournament Masters predictions using saved v2 PGA models + rebuilt feature store.

  python scripts/predict_masters_tournament.py --rebuild-store
  python scripts/predict_masters_tournament.py --field-file data/masters_field_2026.txt

TSV archive ends at 2025; refresh ESPN supplement for 2026 form. Live features use the
last feature-store row with start < --as-of. Use --course-fit-weight 0 to ablate the
Augusta course-fit head. Run scripts/audit_masters_field_data.py to verify coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.preprocessing import StandardScaler

from src.models.train_models_v2 import (  # noqa: E402
    MODELS_DIR,
    TabularNet,
    clip_base_preds_relative,
    print_feature_importance,
)

META_COLS = [
    "season",
    "start",
    "end",
    "tournament",
    "location",
    "name",
    "position_str",
    "position_num",
    "dataset_split",
]
LEAK_COLS = ["rounds_played"]
TARGET_COLS = [
    "target_sg_total",
    "target_sg_per_round",
    "target_made_cut",
    "target_top10",
    "target_top20",
    "target_win",
]

DEFAULT_FEATURE_STORE = Path(project_root) / "notebooks" / "cache" / "pga_feature_store_event_level.csv"
DEFAULT_RESULTS_TSV = Path(project_root) / "src" / "data" / "archive" / "pga_results_2001-2025.tsv"
DEFAULT_RESULTS_SUPPLEMENT = Path(project_root) / "src" / "data" / "archive" / "pga_results_espn_supplement.tsv"

# Saved lr_*.joblib may predate these store columns; refit a 32-dim scaler on train for LR only.
CLASSIFIER_LEGACY_EXCLUDE = frozenset(
    {
        "history_5_plus",
        "history_20_plus",
        "liv_share_last_10",
        "liv_share_last_20",
        "strong_field_share_last_10",
        "strong_field_share_last_20",
    }
)

CALIBRATION_MODEL_BY_TARGET = {
    "target_made_cut": "meta_ensemble",
    "target_top10": "rf",
    "target_top20": "meta_ensemble",
    "target_win": "lr",
}


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS + TARGET_COLS + LEAK_COLS]


def train_medians_and_scalers(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, StandardScaler, List[str]]:
    """Match train_models_v2.main(): regression scaler + classification scaler on train split."""
    df = df.dropna(subset=["target_sg_per_round"]).copy()
    feature_cols = get_feature_cols(df)
    train_mask = df["dataset_split"] == "train"
    val_mask = df["dataset_split"] == "valid"

    X_train_raw = df.loc[train_mask, feature_cols].copy()
    medians = X_train_raw.median()
    X_train_raw = X_train_raw.fillna(medians)
    X_train = X_train_raw.values.astype(np.float32)

    scaler_reg = StandardScaler()
    scaler_reg.fit(X_train)

    df_cls = df.copy().fillna(0)
    X_cls_tr = df_cls.loc[train_mask, feature_cols].values.astype(np.float32)
    scaler_cls = StandardScaler()
    scaler_cls.fit(X_cls_tr)

    return medians.values.astype(np.float64), scaler_reg, scaler_cls, feature_cols


def masters_field_from_tsv(tsv_path: Path, season: int = 2025) -> List[str]:
    raw = pd.read_csv(tsv_path, sep="\t", usecols=["season", "tournament", "name"])
    m = raw[(raw["season"] == season) & (raw["tournament"] == "Masters Tournament")]
    return sorted(m["name"].astype(str).unique().tolist())


def load_field(path: Optional[Path], tsv_path: Path) -> List[str]:
    if path and path.exists():
        lines = path.read_text().strip().splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    return masters_field_from_tsv(tsv_path)


EVENT_HISTORY_FALLBACK = {
    "prev_event_avg_sg_round": "prev_avg_sg_round",
    "prev_event_cut_rate": "prev_cut_rate",
    "prev_event_top20_rate": "prev_top20_rate",
}

CAREER_PEER_FALLBACK = {
    "prev_avg_r4_sg": "prev_avg_sg_round",
    "prev_avg_close_delta": 0.0,
    "prev_avg_finish_num": "prev_avg_finish_num",
}


def latest_player_rows(
    df: pd.DataFrame,
    names: Sequence[str],
    feature_cols: Sequence[str],
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    One row per player: last feature-store row strictly before as_of (if given).
    If as_of is None, uses globally latest start (legacy / debugging).

    Applies event-history NaN fallback: when prev_event_* is NaN (new venue),
    substitute with the player's own career averages from the same row.
    """
    df = df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    out = []
    for name in names:
        sub = df[df["name"] == name]
        if sub.empty:
            sub = df[(df["name"] == name) & (df["tournament"].str.contains("Masters", case=False, na=False))]
        if as_of is not None:
            sub = sub[sub["start"] < as_of]
        if sub.empty:
            continue
        row = sub.sort_values("start").iloc[-1].copy()
        for event_col, career_col in EVENT_HISTORY_FALLBACK.items():
            if event_col in row.index and career_col in row.index:
                if pd.isna(row[event_col]) and pd.notna(row[career_col]):
                    row[event_col] = row[career_col]
        for feat, fallback in CAREER_PEER_FALLBACK.items():
            if feat not in row.index:
                continue
            if pd.isna(row[feat]):
                if isinstance(fallback, str) and fallback in row.index:
                    row[feat] = row[fallback]
                elif isinstance(fallback, (int, float)):
                    row[feat] = fallback
        out.append(row)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).reset_index(drop=True)


def _concat_results_tsv(main: Path, supplement: Optional[Path]) -> pd.DataFrame:
    usecols = ["season", "start", "tournament", "position", "total", "name"]
    frames = []
    if main.exists():
        df = pd.read_csv(main, sep="\t")
        frames.append(df[[c for c in usecols if c in df.columns]])
    if supplement and supplement.exists():
        sdf = pd.read_csv(supplement, sep="\t")
        frames.append(sdf[[c for c in usecols if c in sdf.columns]])
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def recent_starts(
    tsv_path: Path,
    player: str,
    as_of: pd.Timestamp,
    days: int,
    max_rows: int,
    supplement_path: Optional[Path] = None,
) -> pd.DataFrame:
    raw = _concat_results_tsv(tsv_path, supplement_path)
    if raw.empty:
        return raw
    raw["start"] = pd.to_datetime(raw["start"], errors="coerce")
    sub = raw[(raw["name"] == player) & (raw["start"] < as_of)].sort_values("start", ascending=False)
    cutoff = as_of - pd.Timedelta(days=days)
    recent = sub[sub["start"] >= cutoff]
    if len(recent) == 0:
        recent = sub.head(max_rows)
    else:
        recent = recent.head(max_rows)
    return recent


def print_global_importance(models_dir: Path, feature_names: List[str]) -> None:
    models: Dict = {}
    for key, fname in [
        ("ridge", "ridge_sg_model.joblib"),
        ("rf", "rf_sg_model.joblib"),
        ("lgbm", "lgbm_sg_model_v2.joblib"),
        ("xgb", "xgb_sg_model_v2.joblib"),
    ]:
        p = models_dir / fname
        if p.exists():
            models[key] = joblib.load(p)
    print_feature_importance(models, feature_names, top_n=15)


def per_player_ridge_contribs(
    ridge, feature_names: List[str], X_scaled: np.ndarray, names: List[str], top_k: int = 8
) -> None:
    coef = np.asarray(ridge.coef_).ravel()
    intercept = float(ridge.intercept_)
    print("\n" + "=" * 80)
    print(f"PER-PLAYER LINEAR (Ridge SG) CONTRIBUTIONS — top {top_k} |coef * x| (scaled x)")
    print("=" * 80)
    for i, name in enumerate(names):
        x = X_scaled[i]
        prod = coef * x
        order = np.argsort(np.abs(prod))[::-1][:top_k]
        print(f"\n{name}  (ridge intercept {intercept:+.3f}, sum linear {np.dot(coef, x):+.3f})")
        for j in order:
            print(f"    {feature_names[j]:<38} x={x[j]:>8.3f}  contrib={prod[j]:>+9.4f}")


def _course_fit_continuous_dim(cf_path: Path) -> Optional[int]:
    """Infer number of continuous inputs from saved BatchNorm weights (if present)."""
    try:
        sd = torch.load(cf_path, weights_only=True, map_location="cpu")
        w = sd.get("continuous_bn.weight")
        if w is not None:
            return int(w.shape[0])
    except Exception:
        return None
    return None


def winsorize_scaled_features(X: np.ndarray, limit: float = 3.5) -> np.ndarray:
    """
    Clip each column to [-limit, limit] in standard-deviation space.
    Stops Ridge / LR from extrapolating on live rows far outside train distribution
    (e.g. LIV-only schedules, sparse history) which yields absurd SG and win probs.
    """
    return np.clip(X.astype(np.float64), -limit, limit)


def clip_sg_per_round(preds: np.ndarray, lo: float = -3.75, hi: float = 3.75) -> np.ndarray:
    """Per-round SG on Tour almost always lies in a few strokes of this band."""
    return np.clip(preds.astype(np.float64), lo, hi)


def model_feature_count(model) -> int:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "num_feature"):
        try:
            return int(model.num_feature())
        except Exception:
            pass
    if hasattr(model, "feature_importance"):
        try:
            return int(len(model.feature_importance()))
        except Exception:
            pass
    return -1


def classifier_probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    else:
        cls_name = model.__class__.__name__
        cls_module = model.__class__.__module__.lower()
        if cls_name == "Booster" and "xgboost" in cls_module:
            import xgboost as xgb

            p = model.predict(xgb.DMatrix(X))
        elif cls_name == "Booster" and "lightgbm" in cls_module:
            p = model.predict(np.asarray(X, dtype=np.float32))
        else:
            p = model.predict(X)
    p = np.asarray(p)
    if p.ndim == 2:
        if p.shape[1] == 2:
            p = p[:, 1]
        elif p.shape[1] == 1:
            p = p.ravel()
    return np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)


def run_mc(
    meta_sg_per_round: np.ndarray,
    n_players: int,
    n_sims: int,
    n_rounds: int,
    sg_std: float,
    player_stds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if player_stds is not None:
        per_player_std = np.clip(player_stds, 1.5, 3.5)
    else:
        per_player_std = np.full(n_players, sg_std)
    sim_totals = np.zeros((n_sims, n_players))
    for _ in range(n_rounds):
        sim_totals += np.random.normal(
            loc=-meta_sg_per_round, scale=per_player_std, size=(n_sims, n_players)
        )
    win = np.zeros(n_players)
    t5 = np.zeros(n_players)
    t10 = np.zeros(n_players)
    t20 = np.zeros(n_players)
    for s in range(n_sims):
        ranks = np.argsort(sim_totals[s])
        win[ranks[0]] += 1
        for j in range(min(5, n_players)):
            t5[ranks[j]] += 1
        for j in range(min(10, n_players)):
            t10[ranks[j]] += 1
        for j in range(min(20, n_players)):
            t20[ranks[j]] += 1
    return win / n_sims, t5 / n_sims, t10 / n_sims, t20 / n_sims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    ap.add_argument("--results-tsv", type=Path, default=DEFAULT_RESULTS_TSV)
    ap.add_argument(
        "--results-supplement",
        type=Path,
        default=DEFAULT_RESULTS_SUPPLEMENT,
        help="ESPN supplement TSV (set nonexistent path to disable)",
    )
    ap.add_argument("--field-file", type=Path, default=None)
    ap.add_argument("--as-of", type=str, default="2026-04-09", help="Simulate form prior to this date")
    ap.add_argument("--recent-days", type=int, default=21)
    ap.add_argument("--n-sims", type=int, default=50000)
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--sg-std", type=float, default=2.5)
    ap.add_argument("--rebuild-store", action="store_true")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--skip-importance", action="store_true")
    ap.add_argument(
        "--scaled-clip",
        type=float,
        default=3.5,
        help="Winsor limit (std devs) on scaled features for Ridge/NN/LR (default 3.5)",
    )
    ap.add_argument(
        "--base-sg-clip",
        type=float,
        default=3.75,
        help="Clip each base regressor's SG/R prediction to [-v, v] before meta stack",
    )
    ap.add_argument(
        "--final-sg-clip",
        type=float,
        default=3.25,
        help="Clip blended expected SG/R before Monte Carlo (default 3.25)",
    )
    ap.add_argument(
        "--field-median-blend",
        type=float,
        default=0.18,
        help="Shrink exp SG/R toward field median: (1-w)*sg + w*median(sg). Dampens single-player spikes.",
    )
    ap.add_argument(
        "--course-fit-weight",
        type=float,
        default=0.18,
        help="Blend course-fit DL head: (1-w)*meta_sg + w*dl_cf. Use 0 to ablate Augusta embedding tail.",
    )
    args = ap.parse_args()

    if args.rebuild_store:
        from src.data.build_pga_feature_store import build_and_save

        build_and_save(out_path=args.feature_store)

    as_of = pd.Timestamp(args.as_of)
    sup_path = args.results_supplement if args.results_supplement.exists() else None
    comb = _concat_results_tsv(args.results_tsv, sup_path)
    if not comb.empty:
        comb["start"] = pd.to_datetime(comb["start"], errors="coerce")
        latest_data = comb["start"].max()
    else:
        latest_data = pd.NaT
    if pd.notna(latest_data) and latest_data < as_of - pd.Timedelta(days=30):
        print(
            f"\n*** NOTE: Latest result start in merged TSVs is {latest_data.date()}; "
            f"as-of is {as_of.date()}. Refresh ESPN supplement if you need fresher form.\n"
        )

    field = load_field(args.field_file, args.results_tsv)
    print(f"Field size: {len(field)} players")

    df = pd.read_csv(args.feature_store)
    medians_arr, scaler_reg, scaler_cls, feature_cols = train_medians_and_scalers(df)
    medians = pd.Series(medians_arr, index=feature_cols)

    df_fit = df.dropna(subset=["target_sg_per_round"]).copy()
    train_mask_fit = df_fit["dataset_split"] == "train"
    models_dir = Path(MODELS_DIR)

    rows_df = latest_player_rows(df, field, feature_cols, as_of=as_of)
    found = set(rows_df["name"].tolist())
    missing = [n for n in field if n not in found]
    if missing:
        print(f"Missing from feature store (no history): {len(missing)} e.g. {missing[:8]}...")
    names = rows_df["name"].tolist()
    X_live = rows_df[feature_cols].fillna(medians).values.astype(np.float32)
    X_scaled_reg = scaler_reg.transform(X_live)
    legacy_cols = [c for c in feature_cols if c not in CLASSIFIER_LEGACY_EXCLUDE]
    medians_legacy = df_fit.loc[train_mask_fit, legacy_cols].median()
    scaler_cls_legacy = StandardScaler()
    scaler_cls_legacy.fit(
        df_fit.loc[train_mask_fit, legacy_cols].fillna(medians_legacy).values.astype(np.float32)
    )

    X_live_full = rows_df[feature_cols].fillna(medians).values.astype(np.float32)
    X_live_legacy = rows_df[legacy_cols].fillna(medians_legacy).values.astype(np.float32)

    X_scaled_cls_full = winsorize_scaled_features(
        scaler_cls.transform(X_live_full),
        limit=args.scaled_clip,
    )
    X_scaled_cls_legacy = winsorize_scaled_features(
        scaler_cls_legacy.transform(X_live_legacy),
        limit=args.scaled_clip,
    )

    X_scaled_reg = winsorize_scaled_features(X_scaled_reg, limit=args.scaled_clip)
    v = args.base_sg_clip

    reg_preds: Dict[str, np.ndarray] = {}
    ridge = joblib.load(models_dir / "ridge_sg_model.joblib")
    reg_preds["ridge"] = clip_sg_per_round(ridge.predict(X_scaled_reg), -v, v)
    reg_preds["rf"] = clip_sg_per_round(
        joblib.load(models_dir / "rf_sg_model.joblib").predict(X_live), -v, v
    )
    reg_preds["lgbm"] = clip_sg_per_round(
        joblib.load(models_dir / "lgbm_sg_model_v2.joblib").predict(X_live), -v, v
    )
    reg_preds["xgb"] = clip_sg_per_round(
        joblib.load(models_dir / "xgb_sg_model_v2.joblib").predict(X_live), -v, v
    )

    nn_path = models_dir / "pytorch_tabular_v2.pth"
    nn = TabularNet(X_scaled_reg.shape[1])
    nn.load_state_dict(torch.load(nn_path, weights_only=True, map_location="cpu"))
    nn.eval()
    with torch.no_grad():
        reg_preds["nn"] = clip_sg_per_round(
            nn(torch.tensor(X_scaled_reg, dtype=torch.float32)).numpy(), -v, v
        )

    cf_path = models_dir / "pytorch_course_fit.pth"
    w_cf = float(np.clip(args.course_fit_weight, 0.0, 1.0))
    dl_cf_preds = np.zeros(len(names))
    if cf_path.exists() and w_cf > 0:
        from src.models.dl_course_fit import PGACourseFitNN

        player_mapping = joblib.load(models_dir / "player_mapping.joblib")
        course_mapping = joblib.load(models_dir / "course_mapping.joblib")

        num_p = len(player_mapping)
        num_c = len(course_mapping)
        n_cont_cf = _course_fit_continuous_dim(cf_path)
        cols_cf = list(feature_cols)
        X_cf = X_live
        if n_cont_cf is not None and n_cont_cf != len(feature_cols):
            cand = [c for c in feature_cols if c not in CLASSIFIER_LEGACY_EXCLUDE]
            if len(cand) == n_cont_cf:
                cols_cf = cand
                X_cf = rows_df[cols_cf].fillna(medians[cols_cf]).values.astype(np.float32)
            else:
                print(
                    f"\n*** WARN: course-fit checkpoint expects {n_cont_cf} continuous features; "
                    f"could not align (subset len {len(cand)}). Skipping course-fit blend.\n"
                )
                w_cf = 0.0

        if w_cf > 0:
            cf_model = PGACourseFitNN(num_p, num_c, len(cols_cf))
            cf_model.load_state_dict(torch.load(cf_path, weights_only=True, map_location="cpu"))
            cf_model.eval()

            course_name = "Masters Tournament"
            c_idx = course_mapping.get(course_name, 0)

            p_indices = [player_mapping.get(n, 0) for n in names]

            p_tensor = torch.tensor(p_indices, dtype=torch.long)
            c_tensor = torch.tensor([c_idx] * len(names), dtype=torch.long)
            cont_tensor = torch.tensor(X_cf, dtype=torch.float32)

            with torch.no_grad():
                cf_preds = cf_model(p_tensor, c_tensor, cont_tensor).numpy()
                dl_cf_preds = clip_sg_per_round(cf_preds, -v, v)

    meta = joblib.load(models_dir / "meta_ensemble_sg_v2.joblib")
    base_order = ["ridge", "rf", "lgbm", "xgb", "nn"]

    reg_preds = clip_base_preds_relative(reg_preds, sigma=2.0)

    X_meta = np.column_stack([reg_preds[k] for k in base_order])
    meta_sg = meta.predict(X_meta)

    # Course-fit: player×course embedding tail (skip load when weight is 0).
    if cf_path.exists() and w_cf > 0:
        meta_sg = (1.0 - w_cf) * meta_sg + w_cf * dl_cf_preds

    meta_sg = clip_sg_per_round(meta_sg, -args.final_sg_clip, args.final_sg_clip)
    w_med = float(np.clip(args.field_median_blend, 0.0, 0.5))
    if w_med > 0 and len(meta_sg) > 5:
        med = float(np.median(meta_sg))
        meta_sg = (1.0 - w_med) * meta_sg + w_med * med

    cls_targets = ["target_made_cut", "target_top10", "target_top20", "target_win"]
    base_model_types = ["lr", "rf", "lgbm", "xgb"]
    cls_probs_by_target: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in cls_targets}

    for t in cls_targets:
        for mt in base_model_types:
            pth = models_dir / f"{mt}_{t}.joblib"
            if not pth.exists():
                continue
            model = joblib.load(pth)
            expected = model_feature_count(model)
            needs_scaled = mt == "lr"
            if expected == len(legacy_cols):
                X = X_scaled_cls_legacy if needs_scaled else X_live_legacy
            else:
                X = X_scaled_cls_full if needs_scaled else X_live_full
            try:
                cls_probs_by_target[t][mt] = classifier_probability(model, X)
            except Exception as e:
                print(f"*** WARN: skip {mt}_{t}: {e}\n")

        meta_path = models_dir / f"meta_ensemble_{t}_v2.joblib"
        if meta_path.exists() and all(m in cls_probs_by_target[t] for m in base_model_types):
            try:
                base_for_meta = {m: cls_probs_by_target[t][m] for m in base_model_types}
                base_for_meta = clip_base_preds_relative(base_for_meta, sigma=2.0)
                meta_model = joblib.load(meta_path)
                X_meta_cls = np.column_stack([base_for_meta[m] for m in base_model_types])
                cls_probs_by_target[t]["meta_ensemble"] = classifier_probability(meta_model, X_meta_cls)
            except Exception as e:
                print(f"*** WARN: skip meta_ensemble_{t}: {e}\n")

    calibrated_market_probs: Dict[str, np.ndarray] = {}
    calibrated_market_model: Dict[str, str] = {}
    for t in cls_targets:
        preferred = CALIBRATION_MODEL_BY_TARGET.get(t)
        fallback_order = [preferred, "meta_ensemble", "lr", "rf", "lgbm", "xgb"]
        seen = set()
        chosen = None
        for cand in fallback_order:
            if cand is None or cand in seen:
                continue
            seen.add(cand)
            if cand in cls_probs_by_target[t]:
                chosen = cand
                break
        if chosen is not None:
            calibrated_market_probs[t] = cls_probs_by_target[t][chosen]
            calibrated_market_model[t] = chosen

    player_stds = None
    if "prev_round_std_10" in feature_cols:
        std_col_idx = feature_cols.index("prev_round_std_10")
        raw_stds = X_live[:, std_col_idx]
        valid_stds = raw_stds[np.isfinite(raw_stds) & (raw_stds > 0)]
        fallback = float(np.median(valid_stds)) if len(valid_stds) > 0 else args.sg_std
        player_stds = np.where(np.isfinite(raw_stds) & (raw_stds > 0), raw_stds, fallback)

    win_p, t5_p, t10_p, t20_p = run_mc(
        meta_sg, len(names), args.n_sims, args.n_rounds, args.sg_std,
        player_stds=player_stds,
    )

    ridge_coef = np.asarray(ridge.coef_).ravel()
    key_features = []
    for i, name in enumerate(names):
        x = X_scaled_reg[i]
        prod = ridge_coef * x
        order = np.argsort(prod)
        top_neg = order[:3]
        top_pos = order[-3:][::-1]
        kf = []
        for j in top_pos:
            if prod[j] > 0:
                kf.append({"feature": feature_cols[j], "contrib": float(prod[j]), "type": "positive"})
        for j in top_neg:
            if prod[j] < 0:
                kf.append({"feature": feature_cols[j], "contrib": float(prod[j]), "type": "negative"})
        key_features.append(json.dumps(kf))

    out = pd.DataFrame(
        {
            "player": names,
            "exp_sg_per_round": meta_sg,
            "sim_win_pct": 100 * win_p,
            "sim_top5_pct": 100 * t5_p,
            "sim_top10_pct": 100 * t10_p,
            "sim_top20_pct": 100 * t20_p,
            "model_ridge": reg_preds.get("ridge", np.zeros(len(names))),
            "model_rf": reg_preds.get("rf", np.zeros(len(names))),
            "model_lgbm": reg_preds.get("lgbm", np.zeros(len(names))),
            "model_xgb": reg_preds.get("xgb", np.zeros(len(names))),
            "model_nn": reg_preds.get("nn", np.zeros(len(names))),
            "key_features": key_features,
        }
    )
    for t, model_map in cls_probs_by_target.items():
        for mt, probs in model_map.items():
            out[f"{mt}_{t}_prob"] = probs
    for t, probs in calibrated_market_probs.items():
        out[f"best_calibrated_{t}_prob"] = probs
        out[f"best_calibrated_{t}_model"] = calibrated_market_model[t]
    primary_sort = "best_calibrated_target_win_prob"
    if primary_sort in out.columns:
        out = out.sort_values(primary_sort, ascending=False).reset_index(drop=True)
    else:
        out = out.sort_values("sim_win_pct", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 120)
    print(f"2026 MASTERS (pre-event) — classifier + MC {args.n_sims} sims × {args.n_rounds} rounds")
    print("=" * 120)
    has_cls = "best_calibrated_target_made_cut_prob" in out.columns
    hdr = f"{'#':<3} {'Player':<28} {'ExpSG/R':>8}"
    if has_cls:
        hdr += f" {'P Win':>8} {'P T10':>8} {'P T20':>8} {'P Cut':>8}"
    hdr += f" {'MC Win%':>8} {'MC T5%':>8} {'MC T10%':>8} {'MC T20%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in out.iterrows():
        line = f"{i+1:<3} {r['player']:<28} {r['exp_sg_per_round']:>+8.3f}"
        if has_cls:
            line += (
                f" {100*r['best_calibrated_target_win_prob']:>7.1f}%"
                f" {100*r['best_calibrated_target_top10_prob']:>7.1f}%"
                f" {100*r['best_calibrated_target_top20_prob']:>7.1f}%"
                f" {100*r['best_calibrated_target_made_cut_prob']:>7.1f}%"
            )
        line += (
            f" {r['sim_win_pct']:>7.1f}%"
            f" {r['sim_top5_pct']:>7.1f}%"
            f" {r['sim_top10_pct']:>7.1f}%"
            f" {r['sim_top20_pct']:>7.1f}%"
        )
        print(line)

    print("\n" + "=" * 80)
    print(f"RECENT TOUR ROWS (TSV, last {args.recent_days}d before {as_of.date()} or last 3 starts)")
    print("=" * 80)
    for name in names[:15]:
        rdf = recent_starts(
            args.results_tsv, name, as_of, args.recent_days, 3, supplement_path=sup_path
        )
        print(f"\n{name}:")
        if rdf.empty:
            print("  (none)")
        else:
            for _, rr in rdf.iterrows():
                print(
                    f"  {rr['start'].date()}  {rr['tournament'][:40]:<40}  pos {rr['position']!s:>4}  total {rr['total']!s}"
                )
    if len(names) > 15:
        print(f"\n... ({len(names) - 15} more players; full table in CSV)")

    if not args.skip_importance:
        print_global_importance(models_dir, feature_cols)
        per_player_ridge_contribs(ridge, feature_cols, X_scaled_reg, names, top_k=8)

        print("\n" + "=" * 80)
        print("LOGISTIC REGRESSION (win) — GLOBAL |coef| top 12")
        print("=" * 80)
        lr_win = joblib.load(models_dir / "lr_target_win.joblib")
        cw = np.asarray(lr_win.coef_).ravel()
        order = np.argsort(np.abs(cw))[::-1][:12]
        for j in order:
            print(f"  {feature_cols[j]:<40} {cw[j]:>+10.4f}")

    out_path = args.out_csv or (Path(project_root) / "notebooks" / "cache" / "masters_2026_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "as_of": str(as_of.date()),
                "n_players": len(names),
                "n_sims": args.n_sims,
                "feature_store": str(args.feature_store),
                "latest_result_start": str(latest_data.date()) if pd.notna(latest_data) else None,
                "calibration": {
                    "note": "LR classifiers can disagree with MC when features are OOD; inference uses winsorized scaled inputs, clipped base SG, and optional field-median shrink.",
                    "best_model_by_market_target": {
                        t: calibrated_market_model.get(t) for t in cls_targets
                    },
                    "scaled_feature_clip": args.scaled_clip,
                    "base_sg_clip": args.base_sg_clip,
                    "final_sg_clip": args.final_sg_clip,
                    "field_median_blend": w_med,
                    "course_fit_weight": w_cf,
                    "live_feature_rows_use_start_strictly_before_as_of": True,
                    "metrics_to_monitor": [
                        "Spearman rank vs actual finish (event holdout)",
                        "Brier / log-loss on win & top20 (calibrated vs base rate)",
                        "ECE on LR win prob bins",
                        "Mean abs error on exp SG/R vs realized SG/R",
                    ],
                },
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
