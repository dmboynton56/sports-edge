"""
Export CBB expanding-window CV metrics from the cached matchup feature store.

The script writes machine-readable fold metrics without touching the saved
production model directory. Raw Kaggle files are still required to rebuild the
cache from source, but the existing cache can be audited and backtested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.cbb_train_matchup_model import CBBMatchupTrainer


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CBB expanding-window CV metrics.")
    parser.add_argument("--feature-store", default="data-core/notebooks/cache/cbb_matchup_feature_store.csv")
    parser.add_argument("--models-dir", default="/tmp/sports-edge-cbb-cv-models")
    parser.add_argument("--train-start", type=int, default=2010)
    parser.add_argument("--val-start", type=int, default=2016)
    parser.add_argument("--val-end", type=int, default=2025)
    parser.add_argument("--folds-output", default="data-core/notebooks/cache/cbb_expanding_cv_2016_2025.csv")
    parser.add_argument("--summary-output", default="data-core/notebooks/cache/cbb_expanding_cv_2016_2025.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.feature_store)
    trainer = CBBMatchupTrainer(models_dir=args.models_dir)
    results = trainer.expanding_window_cv(
        df,
        train_start=args.train_start,
        val_start=args.val_start,
        val_end=args.val_end,
    )

    upset_calibration = results.pop("upset_calibration", [])
    folds = pd.DataFrame(results)
    folds_output = Path(args.folds_output)
    folds_output.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(folds_output, index=False)

    model_names = ["lgbm", "xgb", "meta"]
    mean_metrics = {}
    for model in model_names:
        mean_metrics[model] = {
            "log_loss": float(folds[f"{model}_ll"].mean()),
            "brier": float(folds[f"{model}_brier"].mean()),
            "auc": float(folds[f"{model}_auc"].mean()),
            "ece": float(folds[f"{model}_ece"].mean()),
            "accuracy": float(folds[f"{model}_accuracy"].mean()),
        }
    best_by_log_loss = min(mean_metrics, key=lambda name: mean_metrics[name]["log_loss"])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_store": args.feature_store,
        "feature_rows": int(len(df)),
        "seasons": sorted([int(season) for season in df["Season"].dropna().unique().tolist()]),
        "train_start": int(args.train_start),
        "validation_start": int(args.val_start),
        "validation_end": int(args.val_end),
        "folds": int(len(folds)),
        "folds_output": str(folds_output),
        "mean_metrics": mean_metrics,
        "best_by_mean_log_loss": best_by_log_loss,
        "upset_calibration": upset_calibration,
        "raw_data_status": "kaggle_mmlm_raw_files_absent_from_repo",
    }

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {folds_output}")
    print(f"Wrote {summary_output}")
    print(json.dumps(summary["mean_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
