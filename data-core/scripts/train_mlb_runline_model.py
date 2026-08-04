"""Train and evaluate MLB home -1.5 cover and margin models."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.mlb_runline_model import (  # noqa: E402
    save_mlb_runline_artifact,
    train_and_evaluate_mlb_runline,
)


DEFAULT_FEATURES = "data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet"
DEFAULT_METRICS = "data-core/notebooks/cache/mlb_runline_metrics_2026_ytd.json"
DEFAULT_PREDICTIONS = "data-core/notebooks/cache/mlb_runline_predictions_2026_ytd.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test an MLB home -1.5 run-line model.")
    parser.add_argument("--features-path", default=DEFAULT_FEATURES)
    parser.add_argument("--validation-season", type=int, default=2025)
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--odds-path", help="Reserved for a future run-line price source.")
    parser.add_argument("--metrics-output", default=DEFAULT_METRICS)
    parser.add_argument("--predictions-output", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-model", default="data-core/models/mlb_runline_model_v1.pkl")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--moneyline-predictions",
        default="data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd_v4_free.csv",
        help="Optional same-game moneyline predictions used only for a correlation sanity check.",
    )
    return parser.parse_args()


def _moneyline_consistency(runline_predictions: pd.DataFrame, path: str) -> dict:
    if not path or not os.path.exists(path):
        return {"status": "skipped", "reason": "moneyline predictions CSV not found", "path": path}
    moneyline = pd.read_csv(path)
    probability_columns = (
        "model_probability", "p_home_win", "home_win_prob", "home_win_probability",
        "predicted_home_win_prob", "model_home_win_probability",
    )
    probability_column = next((column for column in probability_columns if column in moneyline), None)
    if "game_pk" not in moneyline or probability_column is None:
        return {
            "status": "skipped",
            "reason": "CSV lacks game_pk or a recognized home-win probability column",
            "path": path,
            "available_columns": moneyline.columns.tolist(),
        }
    joined = runline_predictions[["game_pk", "p_home_cover_15"]].merge(
        moneyline[["game_pk", probability_column]].drop_duplicates("game_pk"), on="game_pk"
    )
    if len(joined) < 2:
        return {"status": "skipped", "reason": "fewer than two matching games", "matched_rows": len(joined)}
    return {
        "status": "computed",
        "path": path,
        "moneyline_probability_column": probability_column,
        "matched_rows": int(len(joined)),
        "pearson_correlation": float(joined["p_home_cover_15"].corr(joined[probability_column])),
    }


def _json_metrics(result: dict, consistency: dict, odds_path: str | None) -> dict:
    omitted = {
        "selected_classifier",
        "selected_regressor",
        "predictions",
        "_selected_margin_residuals",
    }
    payload = {key: value for key, value in result.items() if key not in omitted}
    payload["moneyline_consistency"] = consistency
    payload["odds_input"] = {
        "path": odds_path,
        "status": "not supplied" if not odds_path else "reserved stub; ROI not evaluated",
    }
    return payload


def main() -> None:
    args = parse_args()
    print(f"Loading feature store: {args.features_path}")
    features = pd.read_parquet(args.features_path)
    print(f"Training on {len(features):,} rows...")
    result = train_and_evaluate_mlb_runline(
        features,
        validation_season=args.validation_season,
        test_season=args.test_season,
        random_state=args.random_state,
    )
    predictions = result["predictions"]
    consistency = _moneyline_consistency(predictions, args.moneyline_predictions)
    metrics = _json_metrics(result, consistency, args.odds_path)

    os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.predictions_output), exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    predictions.to_csv(args.predictions_output, index=False)
    if not args.no_save_model:
        save_mlb_runline_artifact(result, args.output_model)

    selected = result["selected_refit_test"]
    baseline = result["baseline"]["test"]
    margin = result["selected_margin_refit_test"]
    print(
        f"Selected classifier={result['selected_classifier_name']} "
        f"Brier={selected['brier']:.4f} AUC={selected.get('roc_auc', float('nan')):.4f} "
        f"ECE={selected['ece_10']:.4f}"
    )
    print(
        f"Base-rate Brier={baseline['brier']:.4f}; "
        f"delta={selected['brier'] - baseline['brier']:+.4f}"
    )
    print(
        f"Margin head={result['selected_regressor_name']} "
        f"MAE={margin['margin']['mae']:.4f} "
        f"cover Brier={margin['cover_probability']['brier']:.4f}"
    )
    print(f"Wrote metrics: {args.metrics_output}")
    print(f"Wrote predictions: {args.predictions_output}")
    if not args.no_save_model:
        print(f"Wrote model: {args.output_model}")


if __name__ == "__main__":
    main()
