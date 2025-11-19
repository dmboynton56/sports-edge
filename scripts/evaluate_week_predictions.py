#!/usr/bin/env python3
"""
Evaluate Week 11 predictions from BigQuery against actual results.

Example:
    python evaluate_week_predictions.py --project learned-pier-478122-p7 --season 2025 --week 11
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BigQuery predictions against actual results."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project ID (e.g., learned-pier-478122-p7).",
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season year (e.g., 2025).",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Week number to evaluate (e.g., 11).",
    )
    parser.add_argument(
        "--model-version",
        default="v2",
        help="Model version to evaluate (default: v2).",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        help="Optional path to write the merged evaluation CSV",
    )
    return parser.parse_args()


def fetch_predictions(
    client: bigquery.Client,
    project: str,
    season: int,
    week: int,
    model_version: str,
) -> pd.DataFrame:
    """Fetch predictions from BigQuery model_predictions table."""
    query = f"""
        WITH week_games AS (
            SELECT game_id,
                   season,
                   week AS season_week,
                   home_team,
                   away_team
            FROM `{project}.sports_edge_raw.raw_schedules`
            WHERE season = @season AND week = @week
        ),
        ranked_predictions AS (
            SELECT
                p.game_id,
                g.season,
                g.season_week,
                g.home_team,
                g.away_team,
                p.predicted_spread,
                p.home_win_prob,
                p.model_version,
                ROW_NUMBER() OVER (PARTITION BY p.game_id ORDER BY p.prediction_ts DESC) AS rn
            FROM `{project}.sports_edge_curated.model_predictions` AS p
            JOIN week_games AS g
              ON p.game_id = g.game_id
            WHERE p.model_version = @model_version
        )
        SELECT
            game_id,
            season,
            season_week,
            home_team,
            away_team,
            predicted_spread,
            home_win_prob,
            model_version
        FROM ranked_predictions
        WHERE rn = 1
        ORDER BY home_team, away_team
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("season", "INT64", season),
            bigquery.ScalarQueryParameter("week", "INT64", week),
            bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    print(f"Fetched {len(df)} predictions for season {season}, week {week}")
    return df


def fetch_actual_results(
    client: bigquery.Client,
    project: str,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Fetch actual results from BigQuery raw_schedules table."""
    query = f"""
        SELECT
            game_id,
            season,
            week,
            home_team,
            away_team,
            home_score,
            away_score
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE season = @season
          AND week = @week
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY home_team, away_team
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("season", "INT64", season),
            bigquery.ScalarQueryParameter("week", "INT64", week),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    print(f"Fetched {len(df)} actual results for season {season}, week {week}")
    return df


def evaluate(predictions: pd.DataFrame, actuals: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Join predictions with actuals and compute error metrics."""
    merged = predictions.merge(
        actuals,
        on=['game_id', 'home_team', 'away_team'],
        how='inner',
        suffixes=('', '_actual')
    )
    
    if merged.empty:
        raise ValueError("No overlapping games found between predictions and actual results.")
    
    # Calculate actual margin (home_score - away_score)
    merged['actual_margin'] = merged['home_score'] - merged['away_score']
    merged['actual_home_win'] = (merged['actual_margin'] > 0).astype(int)
    
    # Calculate prediction errors
    merged['spread_error'] = merged['predicted_spread'] - merged['actual_margin']
    merged['abs_spread_error'] = merged['spread_error'].abs()
    
    # Determine if spread prediction hit (coverage check, not just direction)
    # predicted_spread is the predicted margin from home team's perspective (negative = home favored, positive = home underdog)
    # Home covers if actualMargin is better than -predicted_spread (i.e., actualMargin > -predicted_spread)
    # For negative spreads (home favored): home covers if actualMargin >= |spread| (home wins by at least that much)
    # For positive spreads (home underdog): home covers if actualMargin > -spread (home loses by less than spread OR wins)
    def check_spread_hit(row):
        predicted_spread = row['predicted_spread']
        actual_margin = row['actual_margin']
        # Home covers if actualMargin > -predicted_spread
        # This works for both negative and positive spreads
        return 1 if actual_margin > -predicted_spread else 0
    
    merged['spread_hit'] = merged.apply(check_spread_hit, axis=1)
    
    # Brier score component
    merged['brier_component'] = (merged['home_win_prob'] - merged['actual_home_win']) ** 2
    
    metrics = {
        'games_evaluated': len(merged),
        'spread_hit_rate': merged['spread_hit'].mean(),
        'mean_abs_spread_error': merged['abs_spread_error'].mean(),
        'spread_rmse': np.sqrt((merged['spread_error'] ** 2).mean()),
        'brier_score': merged['brier_component'].mean(),
    }
    
    return merged, metrics


def print_summary(metrics: dict) -> None:
    """Pretty-print aggregate metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Games evaluated:        {metrics['games_evaluated']}")
    print(f"Spread hit rate:        {metrics['spread_hit_rate']:.1%}")
    print(f"Mean abs spread error:   {metrics['mean_abs_spread_error']:.2f} pts")
    print(f"Spread RMSE:             {metrics['spread_rmse']:.2f} pts")
    print(f"Brier score:            {metrics['brier_score']:.3f}")
    print("=" * 60)


def print_game_results(merged: pd.DataFrame) -> None:
    """Print detailed results for each game."""
    print("\n" + "=" * 60)
    print("GAME-BY-GAME RESULTS")
    print("=" * 60)
    for _, row in merged.iterrows():
        hit_miss = "✓ HIT" if row['spread_hit'] else "✗ MISS"
        print(f"\n{row['away_team']} @ {row['home_team']}")
        print(f"  Predicted spread: {row['predicted_spread']:.1f}")
        print(f"  Actual margin:    {row['actual_margin']:.1f}")
        print(f"  Error:            {row['spread_error']:.1f} pts")
        print(f"  Result:           {hit_miss}")
        print(f"  Score:            {row['away_team']} {row['away_score']} - {row['home_team']} {row['home_score']}")


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)
    
    print(f"Evaluating predictions for season {args.season}, week {args.week}")
    print(f"Model version: {args.model_version}")
    
    predictions = fetch_predictions(
        client, args.project, args.season, args.week, args.model_version
    )
    
    if predictions.empty:
        print(f"No predictions found for season {args.season}, week {args.week}")
        return
    
    actuals = fetch_actual_results(client, args.project, args.season, args.week)
    
    if actuals.empty:
        print(f"No actual results found for season {args.season}, week {args.week}")
        print("Run update_week_results.py first to fetch and update scores.")
        return
    
    merged, metrics = evaluate(predictions, actuals)
    print_summary(metrics)
    print_game_results(merged)
    
    if args.export_path:
        os.makedirs(os.path.dirname(args.export_path), exist_ok=True)
        merged.to_csv(args.export_path, index=False)
        print(f"\nDetailed evaluation saved to {args.export_path}")


if __name__ == "__main__":
    main()

