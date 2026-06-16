#!/usr/bin/env python3
"""Backtest MLB home run probability outputs from historical batter rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "notebooks" / "cache" / "mlb_home_run_predictions.csv"
DEFAULT_OUT = ROOT / "notebooks" / "cache" / "mlb_home_run_backtest_metrics.json"
DEFAULT_TRAINED_METRICS = ROOT / "models" / "mlb_hr_model_v1_metrics.json"


def _clip_prob(values: pd.Series) -> np.ndarray:
    return np.clip(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float), 1e-6, 1 - 1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate scored MLB HR predictions when outcomes are available.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--trained-metrics", type=Path, default=DEFAULT_TRAINED_METRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"Missing predictions file: {args.predictions}")
    df = pd.read_csv(args.predictions)
    model_version = (
        str(df["model_version"].dropna().iloc[0])
        if "model_version" in df.columns and not df["model_version"].dropna().empty
        else "mlb-hr-v1"
    )
    time_split_backtest = None
    if args.trained_metrics.exists():
        metrics = json.loads(args.trained_metrics.read_text(encoding="utf-8"))
        time_split_backtest = {
            "training_window": metrics.get("training_window"),
            "leakage_controls": metrics.get("leakage_controls", []),
            "feature_columns": metrics.get("feature_columns", []),
            "train": metrics.get("train"),
            "test": metrics.get("test"),
        }
    if "actual_home_run" not in df.columns:
        payload = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "status": "pending_outcomes",
            "rows": int(len(df)),
            "note": "Daily prediction file has no actual_home_run column yet; run after completed-game outcome join.",
            "modelVersion": model_version,
            "time_split_backtest": time_split_backtest,
        }
    else:
        y = pd.to_numeric(df["actual_home_run"], errors="coerce").fillna(0).astype(int)
        p = _clip_prob(df["hr_probability"])
        top_10 = df.sort_values("hr_probability", ascending=False).head(10)
        payload = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "status": "evaluated",
            "rows": int(len(df)),
            "positive_rate": float(y.mean()),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "top_10_hit_rate": float(pd.to_numeric(top_10["actual_home_run"], errors="coerce").fillna(0).mean()),
            "modelVersion": model_version,
            "leakage_note": "This evaluator expects prediction rows scored before first pitch; feature generation uses boxscores strictly before the scored date.",
            "time_split_backtest": time_split_backtest,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
