"""
Backfill NBA spread odds from OddsPapi into raw_nba_odds-compatible CSV/BQ rows.
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

from scripts.backfill_nba_odds import _load_dataframe, match_game_ids, schedule_team_to_odds_format
from src.data.oddspapi_odds import (
    DEFAULT_BOOKMAKER,
    OddsPapiClient,
    discover_sport_config,
    fetch_market_catalog,
    load_fixture_cache,
    resolve_and_fetch_closing_odds,
    save_fixture_cache,
)
from src.utils.team_codes import canonical_nba_abbr


def _load_games(path: str, start_date: str) -> pd.DataFrame:
    games = pd.read_csv(path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games[games["game_date"] > pd.to_datetime(start_date)].copy()
    games["home_team"] = games["home_team"].map(lambda x: canonical_nba_abbr(x) or str(x).upper())
    games["away_team"] = games["away_team"].map(lambda x: canonical_nba_abbr(x) or str(x).upper())
    games["season"] = 2025
    return games.sort_values("game_date", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill OddsPapi NBA spreads.")
    parser.add_argument(
        "--games-path",
        default="data-core/notebooks/cache/nba_backtest_2025_v3.csv",
    )
    parser.add_argument("--start-date", default="2026-02-12")
    parser.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER)
    parser.add_argument(
        "--output",
        default="data-core/notebooks/cache/nba_oddspapi_spreads_2026_tail.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="data-core/notebooks/cache/nba_oddspapi_spreads_2026_tail_audit.json",
    )
    parser.add_argument("--fixture-cache", default="data-core/notebooks/cache/oddspapi_fixture_map.json")
    parser.add_argument("--limit-games", type=int, default=50)
    parser.add_argument("--load-bq", action="store_true")
    parser.add_argument("--project", default="learned-pier-478122-p7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv("data-core/.env")
    load_dotenv(".env")
    api_key = os.getenv("ODDSPAPI_API_KEY")
    if not api_key:
        raise ValueError("ODDSPAPI_API_KEY is missing.")

    games = _load_games(args.games_path, args.start_date)
    if args.limit_games:
        games = games.head(args.limit_games)

    client = OddsPapiClient(api_key=api_key)
    fixture_cache = load_fixture_cache(args.fixture_cache)
    config = discover_sport_config(client, "NBA")
    catalog = fetch_market_catalog(client, config["sport_id"])

    rows = []
    audit_results = []
    utc_now = datetime.now(tz=timezone.utc)

    for row in games.itertuples(index=False):
        result = resolve_and_fetch_closing_odds(
            client,
            sport="NBA",
            home=row.home_team,
            away=row.away_team,
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
                "game_id": str(row.game_id),
                "game_date": str(pd.to_datetime(row.game_date).date()),
                "matched": matched,
                "fixture_id": result.get("fixture_id"),
                "error": result.get("error"),
            }
        )
        if matched:
            rows.append(
                {
                    "game_id": str(row.game_id),
                    "league": "NBA",
                    "season": int(row.season),
                    "game_date": pd.to_datetime(row.game_date).date(),
                    "home_team": schedule_team_to_odds_format(row.home_team),
                    "away_team": schedule_team_to_odds_format(row.away_team),
                    "book": args.bookmaker,
                    "market": "spread",
                    "line": float(spread["home_spread"]),
                    "price": spread.get("home_price"),
                    "whos_favored": "home" if float(spread["home_spread"]) < 0 else "away",
                    "ingested_at": utc_now,
                    "raw_record": json.dumps(
                        {
                            "fixture_id": result.get("fixture_id"),
                            "snapshot_ts": spread.get("snapshot_ts"),
                        }
                    ),
                }
            )

    save_fixture_cache(args.fixture_cache, fixture_cache)
    odds_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    odds_df.to_csv(args.output, index=False)

    if args.load_bq and not odds_df.empty:
        from google.cloud import bigquery

        bq_client = bigquery.Client(project=args.project)
        odds_df = match_game_ids(bq_client, args.project, odds_df)
        table_id = f"{args.project}.sports_edge_raw.raw_nba_odds"
        _load_dataframe(bq_client, odds_df, table_id, write_disposition="WRITE_APPEND")

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_version": "v4",
        "bookmaker": args.bookmaker,
        "requested_games": int(len(games)),
        "matched_rows": int(len(odds_df)),
        "api_requests": client.request_count,
        "quota_remaining": client.quota_remaining,
        "output": args.output,
        "loaded_bigquery": bool(args.load_bq and not odds_df.empty),
        "results": audit_results,
    }
    with open(args.audit_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(f"Saved {len(odds_df)} spread rows to {args.output}")
    print(f"Saved audit to {args.audit_output}")


if __name__ == "__main__":
    main()
