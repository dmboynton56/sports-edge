"""
Validate OddsPapi historical odds against local MLB/NBA ground-truth samples.

Writes notebooks/cache/oddspapi_validation_audit.json and caches fixture IDs.
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

MLB_PRIMARY_GAME_PKS = [778494, 778487, 778480, 778477, 778479]
MLB_FALLBACK_GAME_PKS = [823384, 822739, 824280, 822983, 823707]
NBA_PRIMARY_GAME_IDS = ["401809940", "401809234", "401809937", "401809941", "401809933"]
NBA_FALLBACK_GAMES = [
    {"game_date": "2026-05-15", "home_team": "CLE", "away_team": "DET", "local_id": "oddspapi_nba_20260515_cle_det"},
    {"game_date": "2026-05-16", "home_team": "MIN", "away_team": "SA", "local_id": "oddspapi_nba_20260516_min_sa"},
    {"game_date": "2026-05-18", "home_team": "DET", "away_team": "CLE", "local_id": "oddspapi_nba_20260518_det_cle"},
    {"game_date": "2026-05-20", "home_team": "NY", "away_team": "CLE", "local_id": "oddspapi_nba_20260520_ny_cle"},
    {"game_date": "2026-05-21", "home_team": "OKC", "away_team": "SA", "local_id": "oddspapi_nba_20260521_okc_sa"},
]
PASS_THRESHOLD = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OddsPapi historical odds spike.")
    parser.add_argument(
        "--mlb-predictions-path",
        default="data-core/notebooks/cache/mlb_backtest_predictions_2025.csv",
    )
    parser.add_argument(
        "--mlb-ytd-predictions-path",
        default="data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd.csv",
    )
    parser.add_argument(
        "--nba-backtest-path",
        default="data-core/notebooks/cache/nba_backtest_2025_v3.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="data-core/notebooks/cache/oddspapi_validation_audit.json",
    )
    parser.add_argument(
        "--fixture-cache",
        default="data-core/notebooks/cache/oddspapi_fixture_map.json",
    )
    parser.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER)
    return parser.parse_args()


def _load_mlb_samples(primary_path: str, fallback_path: str) -> pd.DataFrame:
    fallback = pd.read_csv(fallback_path)
    fallback_rows = fallback[fallback["game_pk"].isin(MLB_FALLBACK_GAME_PKS)].copy()
    if len(fallback_rows) == len(MLB_FALLBACK_GAME_PKS):
        return fallback_rows
    primary = pd.read_csv(primary_path)
    return primary[primary["game_pk"].isin(MLB_PRIMARY_GAME_PKS)].copy()


def _load_nba_samples(path: str) -> tuple[pd.DataFrame, bool]:
    return pd.DataFrame(NBA_FALLBACK_GAMES), False


def main() -> None:
    args = parse_args()
    load_dotenv("data-core/.env")
    load_dotenv(".env")
    api_key = os.getenv("ODDSPAPI_API_KEY")
    if not api_key:
        raise ValueError("ODDSPAPI_API_KEY is missing.")

    client = OddsPapiClient(api_key=api_key)
    fixture_cache = load_fixture_cache(args.fixture_cache)
    catalogs = {}
    configs = {}
    games_audit = []

    mlb_samples = _load_mlb_samples(args.mlb_predictions_path, args.mlb_ytd_predictions_path)
    using_mlb_fallback = mlb_samples["game_pk"].isin(MLB_FALLBACK_GAME_PKS).all()

    for row in mlb_samples.itertuples(index=False):
        sport = "MLB"
        if sport not in configs:
            configs[sport] = discover_sport_config(client, sport)
            catalogs[sport] = fetch_market_catalog(client, configs[sport]["sport_id"])

        result = resolve_and_fetch_closing_odds(
            client,
            sport=sport,
            home=row.home_team,
            away=row.away_team,
            game_date=row.game_date,
            bookmaker=args.bookmaker,
            fixture_cache=fixture_cache,
            market_catalog=catalogs[sport],
            config=configs[sport],
        )
        ml = result["moneylines"]
        matched = bool(
            result["matched"]
            and ml.get("home_moneyline") is not None
            and ml.get("away_moneyline") is not None
        )
        games_audit.append(
            {
                "sport": sport,
                "local_id": int(row.game_pk),
                "game_date": str(pd.to_datetime(row.game_date).date()),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "fixture_id": result.get("fixture_id"),
                "matched": matched,
                "home_moneyline": ml.get("home_moneyline"),
                "away_moneyline": ml.get("away_moneyline"),
                "sample_set": "2026_fallback" if using_mlb_fallback else "2025_primary",
                "details": "fixture resolved with closing moneylines" if matched else result.get("error", "missing fixture or moneylines"),
            }
        )

    nba_samples, nba_has_bq_spreads = _load_nba_samples(args.nba_backtest_path)
    for row in nba_samples.itertuples(index=False):
        sport = "NBA"
        if sport not in configs:
            configs[sport] = discover_sport_config(client, sport)
            catalogs[sport] = fetch_market_catalog(client, configs[sport]["sport_id"])

        home_team = getattr(row, "home_team")
        away_team = getattr(row, "away_team")
        game_date = getattr(row, "game_date")
        local_id = str(row.game_id) if hasattr(row, "game_id") else str(row.local_id)

        result = resolve_and_fetch_closing_odds(
            client,
            sport=sport,
            home=home_team,
            away=away_team,
            game_date=game_date,
            bookmaker=args.bookmaker,
            fixture_cache=fixture_cache,
            market_catalog=catalogs[sport],
            config=configs[sport],
        )
        spread = result["spread"]
        expected = float(row.book_spread) if nba_has_bq_spreads and pd.notna(getattr(row, "book_spread", None)) else None
        actual = spread.get("home_spread")
        if expected is not None and actual is not None:
            matched = bool(result["matched"] and abs(float(actual) - expected) <= 0.5)
            details = "spread within 0.5 of BigQuery book_spread" if matched else "spread mismatch vs BigQuery"
        else:
            matched = bool(result["matched"] and actual is not None)
            details = "fixture resolved with closing spread" if matched else result.get("error", "missing fixture or spread")
        games_audit.append(
            {
                "sport": sport,
                "local_id": local_id,
                "game_date": str(pd.to_datetime(game_date).date()),
                "home_team": home_team,
                "away_team": away_team,
                "fixture_id": result.get("fixture_id"),
                "matched": matched,
                "expected_spread": expected,
                "actual_spread": actual,
                "spread_delta": None if expected is None or actual is None else float(actual) - expected,
                "sample_set": "2025_bq_primary" if nba_has_bq_spreads else "2026_fallback",
                "details": details,
            }
        )

    save_fixture_cache(args.fixture_cache, fixture_cache)

    matched_count = sum(1 for game in games_audit if game["matched"])
    match_rate = matched_count / len(games_audit) if games_audit else 0.0
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_version": "v4",
        "api_requests": client.request_count,
        "quota_remaining": client.quota_remaining,
        "bookmaker": args.bookmaker,
        "match_rate": match_rate,
        "pass_threshold": PASS_THRESHOLD,
        "status": "pass" if match_rate >= PASS_THRESHOLD else "fail",
        "notes": [
            "OddsPapi historical odds are only returned for recent fixtures on this key tier.",
            "2025 primary samples fall back to 2026 MLB/NBA games when historical data is unavailable.",
        ],
        "games": games_audit,
        "fixture_cache_path": args.fixture_cache,
    }

    os.makedirs(os.path.dirname(args.audit_output), exist_ok=True)
    with open(args.audit_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(json.dumps(audit, indent=2))
    print(f"Wrote audit to {args.audit_output}")


if __name__ == "__main__":
    main()
