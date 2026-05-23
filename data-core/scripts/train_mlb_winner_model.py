"""
Import MLB results, build rolling features, train and test home-win models.

Example:
    PYTHONPATH=data-core python3 data-core/scripts/train_mlb_winner_model.py \
        --start-season 2021 --end-season 2025 --test-season 2025 \
        --cache-path data-core/notebooks/cache/mlb_games_2021_2025.parquet \
        --model-version v1
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.mlb_fetcher import fetch_mlb_games_for_seasons
from src.models.mlb_winner_model import (
    build_mlb_winner_features,
    save_mlb_winner_artifact,
    summarize_metrics,
    train_and_evaluate_mlb_winner,
)


def _load_cache(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported cache extension: {ext}")


def _save_cache(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported cache extension: {ext}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test an MLB home-win model.")
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--validation-season", type=int)
    parser.add_argument("--test-season", type=int)
    parser.add_argument("--cache-path", default="data-core/notebooks/cache/mlb_games.parquet")
    parser.add_argument("--refresh-cache", action="store_true", help="Refetch MLB data even if cache exists.")
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--output-model", help="Path for saved model artifact.")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_season > args.end_season:
        raise ValueError("--start-season must be <= --end-season")

    seasons = list(range(args.start_season, args.end_season + 1))
    if os.path.exists(args.cache_path) and not args.refresh_cache:
        print(f"Loading MLB games from cache: {args.cache_path}")
        games = _load_cache(args.cache_path)
    else:
        games = fetch_mlb_games_for_seasons(seasons)
        _save_cache(games, args.cache_path)
        print(f"Saved MLB game cache to {args.cache_path}")

    print(f"Building MLB features from {len(games)} games...")
    features = build_mlb_winner_features(games, min_prior_games=args.min_prior_games)
    print(f"Built {len(features)} feature rows after min_prior_games={args.min_prior_games}")

    result = train_and_evaluate_mlb_winner(
        features,
        validation_season=args.validation_season,
        test_season=args.test_season,
        random_state=args.random_state,
    )
    print()
    print(summarize_metrics(result))

    if not args.no_save_model:
        output_model = args.output_model or f"data-core/models/mlb_winner_model_{args.model_version}.pkl"
        save_mlb_winner_artifact(result, output_model, model_version=args.model_version)
        print(f"Saved MLB winner model to {output_model}")
        print(f"Saved metrics sidecar to {os.path.splitext(output_model)[0]}_metrics.json")


if __name__ == "__main__":
    main()
