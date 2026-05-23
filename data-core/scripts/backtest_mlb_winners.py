"""
Backtest MLB winner predictions from an MLB feature store.

If an odds file is supplied, it should contain `game_pk` and either
`home_moneyline`/`away_moneyline` or `home_ml`/`away_ml`. The script will add
flat-stake ROI for model picks and simple edge buckets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.odds_math import american_to_decimal, american_to_implied, remove_vig
from src.models.mlb_winner_model import train_and_evaluate_mlb_winner


def _moneyline_profit(won: bool, price: float) -> float:
    if pd.isna(price):
        return np.nan
    if won:
        return american_to_decimal(int(price)) - 1.0
    return -1.0


def _attach_odds(predictions: pd.DataFrame, odds_path: str) -> tuple[pd.DataFrame, dict]:
    odds = pd.read_csv(odds_path) if odds_path.endswith(".csv") else pd.read_parquet(odds_path)
    odds = odds.rename(columns={"home_ml": "home_moneyline", "away_ml": "away_moneyline"})
    required = {"game_pk", "home_moneyline", "away_moneyline"}
    missing = required - set(odds.columns)
    if missing:
        raise ValueError(f"Odds file missing columns: {sorted(missing)}")

    out = predictions.merge(odds[list(required)], on="game_pk", how="left")
    implied_pairs = out.apply(
        lambda row: remove_vig(
            {
                "home": american_to_implied(int(row["home_moneyline"])),
                "away": american_to_implied(int(row["away_moneyline"])),
            }
        )
        if pd.notna(row["home_moneyline"]) and pd.notna(row["away_moneyline"])
        else {"home": np.nan, "away": np.nan},
        axis=1,
    )
    out["market_home_prob"] = [pair["home"] for pair in implied_pairs]
    out["model_edge_home"] = out["home_win_prob"] - out["market_home_prob"]
    out["pick_price"] = np.where(out["pick_side"] == "home", out["home_moneyline"], out["away_moneyline"])
    out["pick_won"] = np.where(out["pick_side"] == "home", out["home_win"] == 1, out["home_win"] == 0)
    out["profit"] = out.apply(lambda row: _moneyline_profit(bool(row["pick_won"]), row["pick_price"]), axis=1)
    odds_rows = out["profit"].notna().sum()
    summary = {
        "odds_rows": int(odds_rows),
        "flat_units": float(out["profit"].sum(skipna=True)) if odds_rows else None,
        "flat_roi": float(out["profit"].mean(skipna=True)) if odds_rows else None,
    }
    if odds_rows:
        out["abs_edge"] = out["model_edge_home"].abs()
        out["edge_bucket"] = pd.cut(
            out["abs_edge"],
            bins=[0, 0.02, 0.04, 0.06, 1],
            labels=["0-2%", "2-4%", "4-6%", "6%+"],
            include_lowest=True,
        )
        bucket = (
            out.dropna(subset=["profit"])
            .groupby("edge_bucket", observed=False)
            .agg(games=("game_pk", "count"), units=("profit", "sum"), roi=("profit", "mean"))
            .reset_index()
        )
        summary["edge_buckets"] = bucket.to_dict(orient="records")
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest MLB winner model.")
    parser.add_argument("--features-path", default="data-core/notebooks/cache/mlb_feature_store_2021_2025.parquet")
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--odds-path")
    parser.add_argument("--predictions-output", default="data-core/notebooks/cache/mlb_backtest_predictions_2025.csv")
    parser.add_argument("--metrics-output", default="data-core/notebooks/cache/mlb_backtest_metrics_2025.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_parquet(args.features_path)
    result = train_and_evaluate_mlb_winner(
        features,
        validation_season=args.validation_season,
        test_season=args.test_season,
    )

    test_df = features[features["season"].astype(int) == args.test_season].copy()
    test_df = test_df.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)
    probabilities = result["selected_model"].predict_proba(test_df[result["feature_columns"]])[:, 1]
    predictions = test_df[
        [
            "game_pk",
            "game_date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "home_win",
        ]
    ].copy()
    predictions["home_win_prob"] = probabilities
    predictions["away_win_prob"] = 1.0 - probabilities
    predictions["pick_side"] = np.where(predictions["home_win_prob"] >= 0.5, "home", "away")
    predictions["pick_won"] = np.where(
        predictions["pick_side"] == "home",
        predictions["home_win"] == 1,
        predictions["home_win"] == 0,
    )

    odds_summary = {"odds_rows": 0, "flat_units": None, "flat_roi": None}
    if args.odds_path:
        predictions, odds_summary = _attach_odds(predictions, args.odds_path)

    metrics = {
        "selected_model_name": result["selected_model_name"],
        "splits": result["splits"],
        "data_summary": result["data_summary"],
        "selected_refit_test": result["selected_refit_test"],
        "baseline": result["baseline"],
        "candidate_metrics": result["model_metrics"],
        "odds_summary": odds_summary,
    }

    os.makedirs(os.path.dirname(args.predictions_output), exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Saved MLB backtest predictions to {args.predictions_output}")
    print(f"Saved MLB backtest metrics to {args.metrics_output}")
    print(json.dumps(metrics["selected_refit_test"], indent=2, sort_keys=True))
    print(json.dumps(metrics["odds_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
