#!/usr/bin/env python3
"""
Pre-tournament Masters predictions using saved v2 PGA models + rebuilt feature store.

  python scripts/predict_masters_tournament.py --rebuild-store
  python scripts/predict_masters_tournament.py --field-file data/masters_field_2026.txt

TSV archive ends at 2025; there is no 2026 tour data in-repo yet — run with updated
pga_results TSV when you ingest Jan–Apr 2026 events for freshest form.
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


def latest_player_rows(df: pd.DataFrame, names: Sequence[str], feature_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    out = []
    for name in names:
        sub = df[df["name"] == name]
        if sub.empty:
            # Augusta history row as fallback (same tournament string as TSV)
            sub = df[(df["name"] == name) & (df["tournament"].str.contains("Masters", case=False, na=False))]
        if sub.empty:
            continue
        out.append(sub.sort_values("start").iloc[-1])
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


def run_mc(
    meta_sg_per_round: np.ndarray,
    n_players: int,
    n_sims: int,
    n_rounds: int,
    sg_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sim_totals = np.zeros((n_sims, n_players))
    for _ in range(n_rounds):
        sim_totals += np.random.normal(
            loc=-meta_sg_per_round, scale=sg_std, size=(n_sims, n_players)
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

    rows_df = latest_player_rows(df, field, feature_cols)
    found = set(rows_df["name"].tolist())
    missing = [n for n in field if n not in found]
    if missing:
        print(f"Missing from feature store (no history): {len(missing)} e.g. {missing[:8]}...")
    names = rows_df["name"].tolist()
    X_live = rows_df[feature_cols].fillna(medians).values.astype(np.float32)
    X_scaled_reg = scaler_reg.transform(X_live)
    X_scaled_cls = scaler_cls.transform(X_live)

    models_dir = Path(MODELS_DIR)

    reg_preds: Dict[str, np.ndarray] = {}
    ridge = joblib.load(models_dir / "ridge_sg_model.joblib")
    reg_preds["ridge"] = ridge.predict(X_scaled_reg)
    reg_preds["rf"] = joblib.load(models_dir / "rf_sg_model.joblib").predict(X_live)
    reg_preds["lgbm"] = joblib.load(models_dir / "lgbm_sg_model_v2.joblib").predict(X_live)
    reg_preds["xgb"] = joblib.load(models_dir / "xgb_sg_model_v2.joblib").predict(X_live)

    nn_path = models_dir / "pytorch_tabular_v2.pth"
    nn = TabularNet(X_scaled_reg.shape[1])
    nn.load_state_dict(torch.load(nn_path, weights_only=True, map_location="cpu"))
    nn.eval()
    with torch.no_grad():
        reg_preds["nn"] = nn(torch.tensor(X_scaled_reg, dtype=torch.float32)).numpy()

    meta = joblib.load(models_dir / "meta_ensemble_sg_v2.joblib")
    base_order = ["ridge", "rf", "lgbm", "xgb", "nn"]
    X_meta = np.column_stack([reg_preds[k] for k in base_order])
    meta_sg = meta.predict(X_meta)

    cls_targets = ["target_made_cut", "target_top10", "target_top20", "target_win"]
    cls_probs: Dict[str, np.ndarray] = {}
    for t in cls_targets:
        p = models_dir / f"lr_{t}.joblib"
        if p.exists():
            cls_probs[t] = joblib.load(p).predict_proba(X_scaled_cls)[:, 1]

    win_p, t5_p, t10_p, t20_p = run_mc(
        meta_sg, len(names), args.n_sims, args.n_rounds, args.sg_std
    )

    out = pd.DataFrame(
        {
            "player": names,
            "exp_sg_per_round": meta_sg,
            "sim_win_pct": 100 * win_p,
            "sim_top5_pct": 100 * t5_p,
            "sim_top10_pct": 100 * t10_p,
            "sim_top20_pct": 100 * t20_p,
        }
    )
    for t, probs in cls_probs.items():
        out[f"lr_{t}_prob"] = probs
    out = out.sort_values("sim_win_pct", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 120)
    print(f"2026 MASTERS (pre-event) — MC {args.n_sims} sims × {args.n_rounds} rounds, σ={args.sg_std}")
    print("=" * 120)
    hdr = (
        f"{'#':<3} {'Player':<28} {'ExpSG/R':>8} "
        f"{'Win%':>7} {'Top5%':>7} {'Top10%':>8} {'Top20%':>8}"
    )
    if cls_probs:
        hdr += f" {'P(cut)':>8} {'P T10':>8} {'P T20':>8} {'P Win':>8}"
    print(hdr)
    print("-" * len(hdr))
    for i, r in out.iterrows():
        line = (
            f"{i+1:<3} {r['player']:<28} {r['exp_sg_per_round']:>+8.3f} "
            f"{r['sim_win_pct']:>6.1f}% {r['sim_top5_pct']:>6.1f}% "
            f"{r['sim_top10_pct']:>7.1f}% {r['sim_top20_pct']:>7.1f}%"
        )
        if "lr_target_made_cut_prob" in out.columns:
            line += (
                f" {100*r['lr_target_made_cut_prob']:>7.1f}% "
                f"{100*r['lr_target_top10_prob']:>7.1f}% "
                f"{100*r['lr_target_top20_prob']:>7.1f}% "
                f"{100*r['lr_target_win_prob']:>7.1f}%"
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
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
