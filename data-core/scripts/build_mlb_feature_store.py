"""
Build the MLB feature store and an audit sidecar from local raw caches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.features.mlb_features import build_mlb_winner_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MLB feature store.")
    parser.add_argument("--games-cache", default="data-core/notebooks/cache/mlb_games_2021_2025.parquet")
    parser.add_argument("--boxscores-cache", default="data-core/notebooks/cache/mlb_boxscores_2021_2025.parquet")
    parser.add_argument("--output", default="data-core/notebooks/cache/mlb_feature_store_2021_2025.parquet")
    parser.add_argument("--audit-output", default="data-core/notebooks/cache/mlb_feature_store_2021_2025_audit.json")
    parser.add_argument("--min-prior-games", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games = pd.read_parquet(args.games_cache)
    features = build_mlb_winner_features(games, min_prior_games=args.min_prior_games)

    boxscore_rows = 0
    if os.path.exists(args.boxscores_cache):
        boxscores = pd.read_parquet(args.boxscores_cache)
        boxscore_rows = len(boxscores)
        features = features.merge(boxscores, on="game_pk", how="left")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features.to_parquet(args.output, index=False)

    audit = {
        "games_rows": int(len(games)),
        "feature_rows": int(len(features)),
        "boxscore_rows": int(boxscore_rows),
        "seasons": sorted([int(x) for x in features["season"].dropna().unique().tolist()]),
        "min_game_date": str(pd.to_datetime(features["game_date"]).min().date()),
        "max_game_date": str(pd.to_datetime(features["game_date"]).max().date()),
        "home_win_rate": float(features["home_win"].mean()),
        "missing_home_probable_pitcher": int(features["home_probable_pitcher_id"].isna().sum()),
        "missing_away_probable_pitcher": int(features["away_probable_pitcher_id"].isna().sum()),
        "min_prior_games": int(args.min_prior_games),
    }
    with open(args.audit_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(f"Saved MLB feature store to {args.output} ({len(features)} rows)")
    print(f"Saved MLB feature audit to {args.audit_output}")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
