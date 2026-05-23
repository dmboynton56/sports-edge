"""
Backfill NFL spread odds from OddsPapi into a CSV archive for ROI analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.oddspapi_odds import (
    DEFAULT_BOOKMAKER,
    OddsPapiClient,
    discover_sport_config,
    fetch_market_catalog,
    load_fixture_cache,
    resolve_and_fetch_closing_odds,
    save_fixture_cache,
)

NFL_TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def _team_name(code: object) -> str:
    token = str(code or "").upper().strip()
    return NFL_TEAM_NAMES.get(token, token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill OddsPapi NFL spreads.")
    parser.add_argument(
        "--games-path",
        default="data-core/notebooks/cache/nfl_backtest_2025_v1.csv",
    )
    parser.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER)
    parser.add_argument(
        "--output",
        default="data-core/notebooks/cache/nfl_oddspapi_spreads_2025.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="data-core/notebooks/cache/nfl_oddspapi_spreads_2025_audit.json",
    )
    parser.add_argument("--fixture-cache", default="data-core/notebooks/cache/oddspapi_fixture_map.json")
    parser.add_argument("--limit-games", type=int, default=285)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv("data-core/.env")
    load_dotenv(".env")
    api_key = os.getenv("ODDSPAPI_API_KEY")
    if not api_key:
        raise ValueError("ODDSPAPI_API_KEY is missing.")

    games = pd.read_csv(args.games_path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values("game_date", ascending=False).reset_index(drop=True)
    if args.limit_games:
        games = games.head(args.limit_games)

    client = OddsPapiClient(api_key=api_key)
    fixture_cache = load_fixture_cache(args.fixture_cache)
    config = discover_sport_config(client, "NFL")
    catalog = fetch_market_catalog(client, config["sport_id"])

    rows = []
    if args.resume and os.path.exists(args.output):
        rows = pd.read_csv(args.output).to_dict(orient="records")
    processed_ids = {str(row["game_id"]) for row in rows}
    audit_results = []

    for row in games.itertuples(index=False):
        game_id = str(row.game_id)
        if game_id in processed_ids:
            continue
        home_name = _team_name(row.home_team)
        away_name = _team_name(row.away_team)
        result = resolve_and_fetch_closing_odds(
            client,
            sport="NFL",
            home=home_name,
            away=away_name,
            game_date=row.game_date,
            bookmaker=args.bookmaker,
            fixture_cache=fixture_cache,
            market_catalog=catalog,
            config=config,
        )
        spread = result["spread"]
        matched = bool(result["matched"] and spread.get("home_spread") is not None)
        audit_results.append(
            {
                "game_id": game_id,
                "game_date": str(pd.to_datetime(row.game_date).date()),
                "matched": matched,
                "fixture_id": result.get("fixture_id"),
                "error": result.get("error"),
            }
        )
        if matched:
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": str(pd.to_datetime(row.game_date).date()),
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "book": args.bookmaker,
                    "market": "spread",
                    "line": float(spread["home_spread"]),
                    "price": spread.get("home_price"),
                    "fixture_id": result.get("fixture_id"),
                    "snapshot_ts": spread.get("snapshot_ts"),
                }
            )
        if len(audit_results) % 10 == 0:
            print(f"Processed {len(audit_results)} games; matched rows={len(rows)}")

    save_fixture_cache(args.fixture_cache, fixture_cache)
    odds_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    odds_df.to_csv(args.output, index=False)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_version": "v4",
        "bookmaker": args.bookmaker,
        "requested_games": int(len(games)),
        "matched_rows": int(len(odds_df)),
        "api_requests": client.request_count,
        "quota_remaining": client.quota_remaining,
        "output": args.output,
        "notes": [
            "OddsPapi historical coverage on this key tier is limited to recent fixtures.",
            "CSV archive grows as historical coverage expands or quota allows resume runs.",
        ],
        "results": audit_results,
    }
    with open(args.audit_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(f"Saved {len(odds_df)} spread rows to {args.output}")
    print(f"Saved audit to {args.audit_output}")


if __name__ == "__main__":
    main()
