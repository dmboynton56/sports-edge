"""
Export NFL model-vs-results backtest metrics from BigQuery raw tables.

This is not an ATS/ROI backtest because the current NFL BigQuery path does not
include a documented spread-odds table. It measures the saved production model
against completed games and writes predictions plus a JSON metrics sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.mlb_winner_model import expected_calibration_error
from src.models.predictor import GamePredictor


def _load_completed_games(client: bigquery.Client, project: str, season: int) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league = 'NFL'
          AND season = @season
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY game_date, game_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("season", "INT64", season)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _load_pbp(client: bigquery.Client, project: str, season: int) -> pd.DataFrame:
    query = f"""
        SELECT game_id, posteam, defteam, epa, game_date
        FROM `{project}.sports_edge_raw.raw_pbp`
        WHERE league = 'NFL'
          AND season = @season
        ORDER BY game_date, game_id, play_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("season", "INT64", season)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = predictions["home_win"].astype(int).to_numpy()
    y_prob = predictions["home_win_probability"].astype(float).to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    actual_margin = predictions["actual_margin"].astype(float).to_numpy()
    predicted_margin = predictions["predicted_margin"].astype(float).to_numpy()

    out = {
        "games": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "ece_10": expected_calibration_error(y_true, y_prob, n_bins=10),
        "avg_pred_home_win": float(np.mean(y_prob)),
        "actual_home_win_rate": float(np.mean(y_true)),
        "spread_mae": float(mean_absolute_error(actual_margin, predicted_margin)),
        "spread_rmse": float(mean_squared_error(actual_margin, predicted_margin) ** 0.5),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NFL full-season prediction backtest metrics.")
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID", "learned-pier-478122-p7"))
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--env-file", default="data-core/.env")
    parser.add_argument("--output-csv", default="data-core/notebooks/cache/nfl_backtest_2025_v1.csv")
    parser.add_argument("--metrics-output", default="data-core/notebooks/cache/nfl_backtest_2025_v1_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    client = bigquery.Client(project=args.project)
    games = _load_completed_games(client, args.project, args.season)
    pbp = _load_pbp(client, args.project, args.season)

    predictor = GamePredictor("NFL", model_version=args.model_version)
    predictions = predictor.predict_batch(games, games, pbp)
    if predictions.empty:
        raise ValueError("No NFL predictions were generated.")
    predictions["game_date"] = pd.to_datetime(predictions["game_date"], errors="coerce")

    scored = games[
        [
            "game_id",
            "game_date",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].merge(
        predictions[
            [
                "game_date",
                "home_team",
                "away_team",
                "predicted_spread",
                "home_win_probability",
                "confidence",
                "model_disagreement",
            ]
        ],
        on=["game_date", "home_team", "away_team"],
        how="inner",
    )
    scored["home_win"] = (scored["home_score"] > scored["away_score"]).astype(int)
    scored["actual_margin"] = scored["home_score"] - scored["away_score"]
    scored["predicted_margin"] = -scored["predicted_spread"]
    scored["predicted_home_win"] = (scored["home_win_probability"] >= 0.5).astype(int)
    scored["correct_winner"] = scored["predicted_home_win"] == scored["home_win"]

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": args.project,
        "league": "NFL",
        "season": int(args.season),
        "model_version": args.model_version,
        "completed_games": int(len(games)),
        "predictions_generated": int(len(predictions)),
        "scored_games": int(len(scored)),
        "date_min": str(pd.to_datetime(scored["game_date"]).min().date()),
        "date_max": str(pd.to_datetime(scored["game_date"]).max().date()),
        "pbp_rows": int(len(pbp)),
        "metrics": _metrics(scored),
        "odds_summary": {
            "odds_rows": 0,
            "flat_roi": None,
            "status": "no_documented_nfl_bigquery_spread_odds_source",
        },
    }

    output_csv = Path(args.output_csv)
    metrics_output = Path(args.metrics_output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {metrics_output}")
    print(json.dumps(metrics["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
