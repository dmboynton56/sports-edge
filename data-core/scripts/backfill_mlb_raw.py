"""
Backfill MLB raw schedule/results and optional boxscore summaries.

Caches are local artifacts by default. Parquet files are gitignored in this
repo, so full backfills can be regenerated without adding large data to git.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.mlb_boxscore_fetcher import fetch_mlb_boxscores
from src.data.mlb_fetcher import fetch_mlb_games_for_seasons


def _load_parquet_or_empty(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill MLB raw local caches.")
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--games-cache", default="data-core/notebooks/cache/mlb_games_2021_2025.parquet")
    parser.add_argument("--boxscores-cache", default="data-core/notebooks/cache/mlb_boxscores_2021_2025.parquet")
    parser.add_argument("--refresh-games", action="store_true")
    parser.add_argument("--fetch-boxscores", action="store_true")
    parser.add_argument("--limit-boxscores", type=int, help="Limit new boxscores fetched; omit for full backfill.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = list(range(args.start_season, args.end_season + 1))

    if os.path.exists(args.games_cache) and not args.refresh_games:
        games = pd.read_parquet(args.games_cache)
        print(f"Loaded {len(games)} MLB games from {args.games_cache}")
    else:
        games = fetch_mlb_games_for_seasons(seasons)
        _save(games, args.games_cache)
        print(f"Saved {len(games)} MLB games to {args.games_cache}")

    if args.fetch_boxscores:
        existing = _load_parquet_or_empty(args.boxscores_cache)
        seen = set(existing["game_pk"].astype(int)) if not existing.empty and "game_pk" in existing else set()
        pending = [int(game_pk) for game_pk in games["game_pk"].astype(int).tolist() if int(game_pk) not in seen]
        if args.limit_boxscores:
            pending = pending[: args.limit_boxscores]
        print(f"Fetching {len(pending)} new MLB boxscores ({len(seen)} already cached)...")
        if pending:
            fetched = fetch_mlb_boxscores(pending)
            combined = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched
            combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
            _save(combined, args.boxscores_cache)
            print(f"Saved {len(combined)} MLB boxscore rows to {args.boxscores_cache}")
        else:
            print("No new MLB boxscores to fetch.")


if __name__ == "__main__":
    main()
