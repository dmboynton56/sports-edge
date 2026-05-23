"""
Backfill historical MLB moneylines from OddsPapi.

Output CSV is compatible with backtest_mlb_winners.py --odds-path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


def _norm_team(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _load_games(path: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    games = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    if start_date:
        games = games[games["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        games = games[games["game_date"] <= pd.to_datetime(end_date)]
    return games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill OddsPapi MLB moneylines.")
    parser.add_argument("--games-path", default="data-core/notebooks/cache/mlb_games_2021_2026.parquet")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER)
    parser.add_argument("--output", default="data-core/notebooks/cache/mlb_oddspapi_moneylines_2025.csv")
    parser.add_argument(
        "--audit-output",
        default="data-core/notebooks/cache/mlb_oddspapi_moneylines_2025_audit.json",
    )
    parser.add_argument("--fixture-cache", default="data-core/notebooks/cache/oddspapi_fixture_map.json")
    parser.add_argument("--limit-games", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv("data-core/.env")
    load_dotenv(".env")
    api_key = os.getenv("ODDSPAPI_API_KEY")
    if not api_key:
        raise ValueError("ODDSPAPI_API_KEY is missing.")

    games = _load_games(args.games_path, args.start_date, args.end_date)
    if args.limit_games:
        games = games.head(args.limit_games)

    client = OddsPapiClient(api_key=api_key)
    fixture_cache = load_fixture_cache(args.fixture_cache)
    config = discover_sport_config(client, "MLB")
    catalog = fetch_market_catalog(client, config["sport_id"])

    existing_rows = []
    processed_game_pks: set[int] = set()
    if args.resume and os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        existing_rows = existing.to_dict(orient="records")
        processed_game_pks = set(existing["game_pk"].astype(int).tolist())

    audit = {
        "sport": "MLB",
        "games_path": args.games_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "bookmaker": args.bookmaker,
        "api_version": "v4",
        "requested_games": int(len(games)),
        "results": [],
    }

    rows = list(existing_rows)
    for row in games.itertuples(index=False):
        game_pk = int(row.game_pk)
        if game_pk in processed_game_pks:
            continue

        result = resolve_and_fetch_closing_odds(
            client,
            sport="MLB",
            home=row.home_team,
            away=row.away_team,
            game_date=row.game_date,
            bookmaker=args.bookmaker,
            fixture_cache=fixture_cache,
            market_catalog=catalog,
            config=config,
        )
        ml = result["moneylines"]
        matched = bool(
            result["matched"]
            and ml.get("home_moneyline") is not None
            and ml.get("away_moneyline") is not None
        )
        audit["results"].append(
            {
                "game_pk": game_pk,
                "game_date": str(pd.to_datetime(row.game_date).date()),
                "matched": matched,
                "fixture_id": result.get("fixture_id"),
                "error": result.get("error"),
            }
        )
        if matched:
            rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": str(pd.to_datetime(row.game_date).date()),
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "home_moneyline": ml["home_moneyline"],
                    "away_moneyline": ml["away_moneyline"],
                    "fixture_id": result.get("fixture_id"),
                    "book": args.bookmaker,
                    "snapshot_ts": ml.get("snapshot_ts"),
                }
            )

    save_fixture_cache(args.fixture_cache, fixture_cache)
    odds = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    odds.to_csv(args.output, index=False)

    audit["status"] = "ok"
    audit["output"] = args.output
    audit["api_requests"] = client.request_count
    audit["quota_remaining"] = client.quota_remaining
    audit["rows"] = int(len(odds))
    audit["matched_rows"] = int(len(odds))
    audit["matched_game_pk_rows"] = int(odds["game_pk"].notna().sum()) if not odds.empty else 0
    with open(args.audit_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(f"Saved {len(odds)} OddsPapi moneyline rows to {args.output}")
    print(f"Saved audit to {args.audit_output}")


if __name__ == "__main__":
    main()
