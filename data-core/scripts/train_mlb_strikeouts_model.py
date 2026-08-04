#!/usr/bin/env python3
"""Train and evaluate the MLB probable-starter strikeouts model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.mlb_strikeouts_model import (  # noqa: E402
    coverage_metrics,
    reshape_starter_sides,
    train_and_evaluate_strikeouts,
)


DEFAULT_FEATURE_STORE = "data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet"
DEFAULT_METRICS = "data-core/notebooks/cache/mlb_strikeouts_metrics_2026_ytd.json"
DEFAULT_PREDICTIONS = "data-core/notebooks/cache/mlb_strikeouts_predictions_2026_ytd.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/validate/test an MLB starter strikeouts model from feature store v2."
    )
    parser.add_argument("--feature-store", default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--metrics-output", default=DEFAULT_METRICS)
    parser.add_argument("--predictions-output", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--train-start-season", type=int, default=2021)
    parser.add_argument("--train-end-season", type=int, default=2024)
    parser.add_argument("--validation-season", type=int, default=2025)
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--minimum-clean-coverage", type=float, default=0.70)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _write_json(payload: dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_clean_coverage <= 1.0:
        raise ValueError("--minimum-clean-coverage must be between 0 and 1")
    if not args.train_start_season <= args.train_end_season < args.validation_season < args.test_season:
        raise ValueError(
            "Seasons must satisfy train-start <= train-end < validation < test"
        )

    print(f"Loading MLB feature store: {args.feature_store}")
    store = pd.read_parquet(args.feature_store)
    sides = reshape_starter_sides(store)
    coverage = coverage_metrics(sides, season=args.test_season)
    print(
        f"{args.test_season} clean starter coverage: "
        f"{coverage['clean_sides']}/{coverage['total_sides']} "
        f"({coverage['clean_coverage']:.2%}); probable mismatch discarded: "
        f"{coverage['probable_mismatch_discarded_fraction']:.2%}"
    )
    if coverage["clean_coverage"] < args.minimum_clean_coverage:
        raise RuntimeError(
            f"Feasibility gate failed: {coverage['clean_coverage']:.2%} is below "
            f"{args.minimum_clean_coverage:.2%}"
        )

    result = train_and_evaluate_strikeouts(
        sides,
        train_start_season=args.train_start_season,
        train_end_season=args.train_end_season,
        validation_season=args.validation_season,
        test_season=args.test_season,
        random_state=args.random_state,
    )
    _write_json(result.metrics, args.metrics_output)
    predictions_output = Path(args.predictions_output)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(predictions_output, index=False, date_format="%Y-%m-%d")

    test = result.metrics["test"]
    print(
        f"Selected {result.metrics['selected_candidate']} ({result.metrics['loss']} loss). "
        f"Test MAE={test['model']['mae']:.4f}, RMSE={test['model']['rmse']:.4f}; "
        f"baseline MAE={test['k9_expected_outs_baseline']['mae']:.4f}, "
        f"RMSE={test['k9_expected_outs_baseline']['rmse']:.4f}."
    )
    print(f"Wrote metrics: {args.metrics_output}")
    print(f"Wrote {len(result.predictions)} predictions: {args.predictions_output}")


if __name__ == "__main__":
    main()
