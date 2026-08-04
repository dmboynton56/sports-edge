"""Train and evaluate the MLB total-runs model from the leakage-safe feature store."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.mlb_totals_model import save_mlb_totals_outputs, train_and_evaluate_mlb_totals


def _load_frame(path: str) -> pd.DataFrame:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".parquet":
        return pd.read_parquet(path)
    if extension == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}: {extension}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test MLB total-runs point and O/U models.")
    parser.add_argument(
        "--features-path",
        default="data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet",
    )
    parser.add_argument("--validation-season", type=int, default=2025)
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument(
        "--odds-path",
        help="Optional CSV/parquet with game_pk, total_line, over_price, and under_price.",
    )
    parser.add_argument(
        "--metrics-output",
        default="data-core/notebooks/cache/mlb_totals_metrics_2026_ytd.json",
    )
    parser.add_argument(
        "--predictions-output",
        default="data-core/notebooks/cache/mlb_totals_predictions_2026_ytd.csv",
    )
    parser.add_argument(
        "--output-model",
        default="data-core/models/mlb_totals_model_v1.pkl",
        help="Model pickle path; pass an empty string to skip the pickle.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = _load_frame(args.features_path)
    odds = _load_frame(args.odds_path) if args.odds_path else None
    print(f"Loaded {len(features):,} feature rows from {args.features_path}")
    result = train_and_evaluate_mlb_totals(
        features,
        validation_season=args.validation_season,
        test_season=args.test_season,
        random_state=args.random_state,
        odds=odds,
    )
    save_mlb_totals_outputs(
        result,
        metrics_path=args.metrics_output,
        predictions_path=args.predictions_output,
        model_path=args.output_model or None,
    )

    selected = result["selected_refit_test"]
    ablation = result["weather_ablation"]
    print(f"Selected model: {result['selected_model_name']}")
    print(f"Test rows: {result['splits']['test_rows']:,}")
    print(f"Test MAE: {selected['mae']:.4f}; RMSE: {selected['rmse']:.4f}")
    print(
        "Weather/park value: "
        f"delta MAE (ablation-full)={ablation['delta_mae_ablation_minus_full']:+.4f}; "
        f"delta AUC 8.5 (full-ablation)={ablation['delta_auc_full_minus_ablation_8_5']:+.4f}"
    )
    print(f"ROI: {result['roi_evaluation']}")
    print(f"Wrote metrics to {args.metrics_output}")
    print(f"Wrote predictions to {args.predictions_output}")
    if args.output_model:
        print(f"Wrote model to {args.output_model}")


if __name__ == "__main__":
    main()

