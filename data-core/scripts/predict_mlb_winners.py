"""
Score MLB home-win probabilities for a game date.

The script loads a saved MLB winner artifact, fetches current-season schedule
history through the requested date, builds pregame features, and writes or
prints one row per scored game.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.mlb_fetcher import fetch_mlb_schedule
from src.models.mlb_winner_model import build_mlb_prediction_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MLB game winners for a date.")
    parser.add_argument("--date", required=True, help="Game date in YYYY-MM-DD.")
    parser.add_argument("--model-path", default="data-core/models/mlb_winner_model_v2.pkl")
    parser.add_argument("--season", type=int, help="MLB season year. Defaults to --date year.")
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument("--include-final", action="store_true", help="Score games even if already final.")
    parser.add_argument("--output-csv", help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_date = pd.to_datetime(args.date).date()
    season = args.season or game_date.year

    with open(args.model_path, "rb") as f:
        artifact = pickle.load(f)

    season_start = datetime(season, 3, 1).date()
    schedule = fetch_mlb_schedule(
        season,
        start_date=season_start,
        end_date=game_date,
        include_uncompleted=True,
    )
    if schedule.empty:
        raise ValueError(f"No MLB schedule rows found for season={season} through {game_date}")

    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    target_date_mask = schedule["game_date"].dt.date == game_date
    if args.include_final:
        games_to_score = schedule[target_date_mask].copy()
    else:
        completed = schedule.get("completed", pd.Series(False, index=schedule.index)).fillna(False).astype(bool)
        games_to_score = schedule[target_date_mask & ~completed].copy()

    history = schedule[
        (schedule["game_date"].dt.date < game_date)
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()

    features = build_mlb_prediction_features(
        history,
        games_to_score,
        min_prior_games=args.min_prior_games,
    )
    if features.empty:
        raise ValueError(f"No games were scoreable for {game_date}; check schedule/finals/min-prior-games.")

    feature_cols = artifact["feature_columns"]
    probabilities = artifact["model"].predict_proba(features[feature_cols])[:, 1]
    output = features[
        [
            "game_pk",
            "game_date",
            "game_datetime",
            "away_team",
            "home_team",
            "away_probable_pitcher",
            "home_probable_pitcher",
        ]
    ].copy()
    output["home_win_prob"] = probabilities
    output["away_win_prob"] = 1.0 - probabilities
    output["predicted_winner"] = output["home_team"].where(
        output["home_win_prob"] >= 0.5,
        output["away_team"],
    )
    output = output.sort_values("game_datetime").reset_index(drop=True)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        output.to_csv(args.output_csv, index=False)
        print(f"Saved MLB predictions to {args.output_csv}")
    else:
        print(output.to_string(index=False))


if __name__ == "__main__":
    main()
