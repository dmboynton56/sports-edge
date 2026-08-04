"""Run controlled MLB winner feature ablations on one feature-store window."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from backtest_mlb_winners import _attach_odds
from src.models.mlb_winner_model import default_feature_columns, train_and_evaluate_mlb_winner


STARTER_COLUMNS = [
    "home_starter_k9_last5",
    "home_starter_bb9_last5",
    "home_starter_era_proxy_last5",
    "home_starter_pitches_last3_avg",
    "home_starter_outs_per_start_last5",
    "home_starter_starts_last365",
    "home_starter_days_since_last_start",
    "home_starter_career_starts_prior",
    "home_starter_has_history",
    "home_starter_throws_r",
    "away_starter_k9_last5",
    "away_starter_bb9_last5",
    "away_starter_era_proxy_last5",
    "away_starter_pitches_last3_avg",
    "away_starter_outs_per_start_last5",
    "away_starter_starts_last365",
    "away_starter_days_since_last_start",
    "away_starter_career_starts_prior",
    "away_starter_has_history",
    "away_starter_throws_r",
    "starter_k9_last5_diff",
    "starter_bb9_last5_diff",
    "starter_era_proxy_last5_diff",
    "starter_pitches_last3_avg_diff",
    "starter_outs_per_start_last5_diff",
    "starter_starts_last365_diff",
    "starter_days_since_last_start_diff",
    "starter_career_starts_prior_diff",
    "starter_has_history_diff",
    "starter_throws_r_diff",
]

WEATHER_COLUMNS = [
    "temp_f",
    "wind_mph",
    "wind_out",
    "wind_in",
    "wind_cross",
    "is_dome_or_closed",
    "is_day_game",
    "elevation",
]

RUN_ENV_COLUMNS = [
    "home_runs_scored_pg_15",
    "home_runs_allowed_pg_15",
    "home_team_total_pg_15",
    "home_team_bat_k_pg_15",
    "away_runs_scored_pg_15",
    "away_runs_allowed_pg_15",
    "away_team_total_pg_15",
    "away_team_bat_k_pg_15",
    "runs_scored_pg_15_diff",
    "runs_allowed_pg_15_diff",
    "team_total_pg_15_diff",
    "team_bat_k_pg_15_diff",
    "combined_expected_total",
]

NEW_V2_COLUMNS = STARTER_COLUMNS + WEATHER_COLUMNS + RUN_ENV_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate MLB winner v2 feature groups.")
    parser.add_argument(
        "--features-path",
        default="data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet",
    )
    parser.add_argument(
        "--odds-path",
        default="data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv",
    )
    parser.add_argument("--validation-season", type=int, default=2025)
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data-core/notebooks/cache/mlb_ml_ablation_2026_ytd.json",
    )
    return parser.parse_args()


def _read_frame(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature-store extension: {os.path.splitext(path)[1]}")


def _arm_frame(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Retain metadata/labels plus exactly the requested numeric model columns."""
    all_model_columns = set(default_feature_columns(features))
    keep = [column for column in features.columns if column not in all_model_columns]
    return features[keep + feature_columns].copy()


def _prediction_frame(test_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    predictions = test_df[["game_pk", "home_win"]].copy()
    predictions["home_win_prob"] = probabilities
    predictions["pick_side"] = np.where(probabilities >= 0.5, "home", "away")
    return predictions


def _selected_metrics(result: dict) -> tuple[dict, dict]:
    selected_name = result["selected_model_name"]
    validation = result["model_metrics"][selected_name]["validation"]
    return validation, result["selected_refit_test"]


def main() -> None:
    args = parse_args()
    features = _read_frame(args.features_path)
    full_columns = default_feature_columns(features)
    missing = sorted(set(NEW_V2_COLUMNS) - set(full_columns))
    if missing:
        raise ValueError(f"Feature store is missing expected v2 model columns: {missing}")

    v1_columns = [column for column in full_columns if column not in NEW_V2_COLUMNS]
    arms = {
        "v1_baseline": v1_columns,
        "v1_plus_starter": v1_columns + STARTER_COLUMNS,
        "v1_plus_weather": v1_columns + WEATHER_COLUMNS,
        "full_v2": full_columns,
    }

    test_rows = features[features["season"].astype(int) == args.test_season].copy()
    test_rows = test_rows.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)
    output_arms: dict[str, dict] = {}

    for arm_name, columns in arms.items():
        print(f"Training {arm_name} with {len(columns)} columns...")
        arm_features = _arm_frame(features, columns)
        result = train_and_evaluate_mlb_winner(
            arm_features,
            validation_season=args.validation_season,
            test_season=args.test_season,
            random_state=args.random_state,
        )
        probabilities = result["selected_model"].predict_proba(test_rows[columns])[:, 1]
        predictions = _prediction_frame(test_rows, probabilities)
        _, odds_summary = _attach_odds(predictions, args.odds_path)
        odds_summary["coverage"] = float(odds_summary["odds_rows"] / len(predictions))
        validation_metrics, test_metrics = _selected_metrics(result)
        output_arms[arm_name] = {
            "feature_columns": columns,
            "feature_count": len(columns),
            "selected_model_name": result["selected_model_name"],
            "splits": result["splits"],
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "odds_summary": odds_summary,
            "prediction_distribution": {
                "p5": float(np.quantile(probabilities, 0.05)),
                "p50": float(np.quantile(probabilities, 0.50)),
                "p95": float(np.quantile(probabilities, 0.95)),
            },
        }

    test_counts = {arm["splits"]["test_rows"] for arm in output_arms.values()}
    if len(test_counts) != 1:
        raise RuntimeError(f"Ablation arms have different test row counts: {sorted(test_counts)}")

    baseline = output_arms["v1_baseline"]
    full = output_arms["full_v2"]
    payload = {
        "features_path": args.features_path,
        "odds_path": args.odds_path,
        "validation_season": args.validation_season,
        "test_season": args.test_season,
        "identical_test_rows": True,
        "test_rows": next(iter(test_counts)),
        "feature_groups": {
            "starter": STARTER_COLUMNS,
            "weather_park": WEATHER_COLUMNS,
            "cross_season_run_environment": RUN_ENV_COLUMNS,
        },
        "arms": output_arms,
        "full_v2_vs_v1_baseline": {
            "delta_brier": full["test_metrics"]["brier"] - baseline["test_metrics"]["brier"],
            "delta_roc_auc": full["test_metrics"]["roc_auc"] - baseline["test_metrics"]["roc_auc"],
            "delta_flat_roi": full["odds_summary"]["flat_roi"] - baseline["odds_summary"]["flat_roi"],
        },
        "published_v3_context_different_673_game_window": {
            "window_end": "2026-05-21",
            "test_rows": 673,
            "brier": 0.2478,
            "log_loss": 0.6888,
            "roc_auc": 0.5431,
            "accuracy": 0.5379,
            "flat_roi": -0.031,
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Saved MLB winner ablation to {args.output}")


if __name__ == "__main__":
    main()
