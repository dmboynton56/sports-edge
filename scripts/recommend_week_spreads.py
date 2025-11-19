#!/usr/bin/env python3
"""
Predict an NFL week, pull current book spreads, and surface high-confidence ATS plays.

Example:
    python scripts/recommend_week_spreads.py --week 12 \
        --start 2025-11-20 --end 2025-11-24 --confidence 0.60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_week import (
    load_season_schedule,
    filter_completed_games,
    collect_week_games,
    predict_games,
    load_supabase_credentials,
    create_pg_connection,
)
from src.data.pbp_loader import load_pbp

SEASON = 2025
PAYOUT_MULTIPLIER = 1.909


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend spreads for an NFL week based on model confidence."
    )
    parser.add_argument("--week", type=int, default=12, help="NFL week to score.")
    parser.add_argument(
        "--start",
        type=str,
        default="2025-11-20",
        help="Earliest game date to include (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-11-24",
        help="Latest game date to include (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.60,
        help="Confidence threshold (>= value or <= 1-value).",
    )
    return parser.parse_args()


def fetch_book_spreads(conn, week: int) -> pd.DataFrame:
    query = """
        select season,
               week,
               home_team,
               away_team,
               book_spread,
               game_time_utc::date as game_date
        from games
        where league = 'NFL'
          and season = %s
          and week = %s
          and book_spread is not null
    """
    with conn.cursor() as cur:
        cur.execute(query, (SEASON, week))
        rows = cur.fetchall()
    cols = ["season", "week", "home_team", "away_team", "book_spread", "game_date"]
    df = pd.DataFrame(rows, columns=cols)
    df["book_spread"] = df["book_spread"].astype(float)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def merge_predictions(pred_df: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    merged = books.merge(
        pred_df[
            ["home_team", "away_team", "game_date", "predicted_spread", "home_win_probability"]
        ],
        on=["home_team", "away_team", "game_date"],
        how="inner",
    )
    merged["predicted_spread"] = merged["predicted_spread"].astype(float)
    merged["home_win_probability"] = merged["home_win_probability"].astype(float)
    merged["book_spread"] = merged["book_spread"].astype(float)
    return merged


def build_recommendations(df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    recs: List[dict] = []
    for _, row in df.iterrows():
        home_prob = row["home_win_probability"]
        book_spread = row["book_spread"]
        model_spread = row["predicted_spread"]

        if home_prob >= confidence:
            bet_team = row["home_team"]
            bet_line = book_spread
            cover_prob = home_prob
        elif home_prob <= 1 - confidence:
            bet_team = row["away_team"]
            bet_line = -book_spread
            cover_prob = 1 - home_prob
        else:
            continue

        ev = cover_prob * PAYOUT_MULTIPLIER - (1 - cover_prob)
        recs.append(
            {
                "game_date": row["game_date"].date(),
                "matchup": f"{row['away_team']} @ {row['home_team']}",
                "bet_team": bet_team,
                "bet_spread": f"{bet_line:+.1f}",
                "model_spread": f"{model_spread:+.1f}",
                "book_spread": f"{book_spread:+.1f}",
                "confidence": f"{cover_prob:.1%}",
                "expected_value": f"{ev:.1%}",
            }
        )
    return pd.DataFrame(recs)


def main() -> None:
    args = parse_args()
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    schedule = load_season_schedule(SEASON)
    completed = filter_completed_games(schedule)
    try:
        pbp = load_pbp("NFL", SEASON)
    except Exception as exc:
        print(f"Warning: failed to load play-by-play data ({exc}); proceeding without it.")
        pbp = None

    week_games = collect_week_games(schedule, args.week)
    week_games["game_date"] = pd.to_datetime(week_games["game_date"])
    window_games = week_games[
        (week_games["game_date"] >= start_dt) & (week_games["game_date"] <= end_dt)
    ]
    if window_games.empty:
        print("No games found in the requested window.")
        return

    predictions = predict_games(window_games, schedule, completed, pbp)
    if not predictions:
        print("No predictions generated (missing data?).")
        return
    preds_df = pd.DataFrame(predictions)
    preds_df["game_date"] = pd.to_datetime(preds_df["game_date"])

    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    try:
        books = fetch_book_spreads(conn, args.week)
    finally:
        conn.close()

    merged = merge_predictions(preds_df, books)
    merged = merged[
        (merged["game_date"] >= start_dt) & (merged["game_date"] <= end_dt)
    ]
    if merged.empty:
        print("No book spreads matched the prediction window.")
        return

    recs = build_recommendations(merged, args.confidence)
    if recs.empty:
        print("No spreads met the confidence threshold.")
        return

    print("\nRecommended Spreads:")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
