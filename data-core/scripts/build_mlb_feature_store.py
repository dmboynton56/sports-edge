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

from src.features.mlb_features import build_mlb_market_features, build_mlb_winner_features
from src.features.mlb_market_features import OBSERVED_WEATHER_COLUMNS


V1_DEFAULTS = {
    "games_cache": "data-core/notebooks/cache/mlb_games_2021_2025.parquet",
    "boxscores_cache": "data-core/notebooks/cache/mlb_boxscores_2021_2025.parquet",
    "output": "data-core/notebooks/cache/mlb_feature_store_2021_2025.parquet",
}
V2_DEFAULTS = {
    "games_cache": "data-core/notebooks/cache/mlb_games_2021_2026.parquet",
    "boxscores_cache": "data-core/notebooks/cache/mlb_boxscores_2021_2026.parquet",
    "venue_meta": "data-core/notebooks/cache/mlb_venue_meta.json",
    "output": "data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MLB feature store.")
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--v2", action="store_true", help="Alias for --version v2.")
    parser.add_argument("--games-cache")
    parser.add_argument("--boxscores-cache")
    parser.add_argument("--venue-meta")
    parser.add_argument("--output")
    parser.add_argument("--audit-output")
    parser.add_argument("--min-prior-games", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = "v2" if args.v2 else args.version
    defaults = V2_DEFAULTS if version == "v2" else V1_DEFAULTS
    games_path = args.games_cache or defaults["games_cache"]
    boxscores_path = args.boxscores_cache or defaults["boxscores_cache"]
    output_path = args.output or defaults["output"]
    audit_path = args.audit_output or os.path.splitext(output_path)[0] + "_audit.json"

    games = pd.read_parquet(games_path)
    boxscores = pd.read_parquet(boxscores_path) if os.path.exists(boxscores_path) else pd.DataFrame()
    boxscore_rows = len(boxscores)
    if version == "v2":
        venue_meta_path = args.venue_meta or defaults["venue_meta"]
        with open(venue_meta_path, encoding="utf-8") as handle:
            venue_meta = json.load(handle)
        features = build_mlb_market_features(
            games,
            boxscores,
            venue_meta,
            min_prior_games=args.min_prior_games,
        )
    else:
        features = build_mlb_winner_features(games, min_prior_games=args.min_prior_games)
        if not boxscores.empty:
            features = features.merge(boxscores, on="game_pk", how="left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_parquet(output_path, index=False)

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
        "version": version,
    }
    if version == "v2":
        key_columns = [
            "home_starter_k9_last5",
            "away_starter_k9_last5",
            "home_starter_pitches_last3_avg",
            "away_starter_pitches_last3_avg",
            "home_team_bat_k_pg_15",
            "away_team_bat_k_pg_15",
            "combined_expected_total",
            "temp_f",
            "wind_mph",
            "elevation",
        ]
        both_history = features["home_starter_has_history"].eq(1) & features["away_starter_has_history"].eq(1)
        probable_match = pd.concat(
            [features["home_probable_matches_actual"], features["away_probable_matches_actual"]],
            ignore_index=True,
        ).dropna()
        audit.update(
            {
                "rows": int(len(features)),
                "new_column_null_rates": {
                    column: float(features[column].isna().mean()) for column in key_columns
                },
                "starter_history_both_sides_pct": float(both_history.mean() * 100.0),
                "probable_matches_actual_pct": float(probable_match.astype(bool).mean() * 100.0),
                "observed-weather": OBSERVED_WEATHER_COLUMNS,
                "observed_weather_caveat": (
                    "Weather columns are observed game-time boxscore records used as pregame forecast proxies."
                ),
            }
        )
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(f"Saved MLB feature store to {output_path} ({len(features)} rows)")
    print(f"Saved MLB feature audit to {audit_path}")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
