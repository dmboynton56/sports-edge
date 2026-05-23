"""
Backfill historical moneyline snapshots from The Odds API.

The Odds API historical endpoint is paid-plan only. With a free key, this
script writes an audit JSON that records the blocker and exits successfully by
default so project audits stay reproducible.
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

from src.data.historical_odds import (
    HistoricalOddsUnavailable,
    collapse_moneyline_consensus,
    fetch_historical_odds_snapshot,
    snapshot_ts_for_date,
)


def _load_games(path: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    games = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    games["game_date"] = pd.to_datetime(games["game_date"])
    if start_date:
        games = games[games["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        games = games[games["game_date"] <= pd.to_datetime(end_date)]
    return games.copy()


def _norm_team(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _attach_game_ids(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Match historical odds events to local game IDs by date and teams."""
    if odds.empty:
        odds["game_pk"] = pd.Series(dtype="Int64")
        return odds

    games_lookup = games.copy()
    games_lookup["game_date_key"] = pd.to_datetime(games_lookup["game_date"]).dt.date.astype(str)
    games_lookup["home_key"] = games_lookup["home_team"].map(_norm_team)
    games_lookup["away_key"] = games_lookup["away_team"].map(_norm_team)
    mapping = {
        (row.game_date_key, row.home_key, row.away_key): int(row.game_pk)
        for row in games_lookup.itertuples(index=False)
    }

    out = odds.copy()
    out["game_date_key"] = pd.to_datetime(out["commence_time"]).dt.date.astype(str)
    out["home_key"] = out["home_team"].map(_norm_team)
    out["away_key"] = out["away_team"].map(_norm_team)
    out["game_pk"] = [
        mapping.get((row.game_date_key, row.home_key, row.away_key))
        for row in out.itertuples(index=False)
    ]
    out = out.drop(columns=["game_date_key", "home_key", "away_key"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical moneylines.")
    parser.add_argument("--sport", default="MLB", choices=["MLB", "NBA", "NFL"])
    parser.add_argument("--games-path", default="data-core/notebooks/cache/mlb_games_2021_2025.parquet")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--snapshot-hour-utc", type=int, default=16)
    parser.add_argument("--bookmakers", default="draftkings,fanduel,betmgm")
    parser.add_argument("--output", default="data-core/notebooks/cache/mlb_moneylines_historical.csv")
    parser.add_argument("--audit-output", default="data-core/notebooks/cache/mlb_moneylines_historical_audit.json")
    parser.add_argument("--limit-dates", type=int)
    parser.add_argument("--fail-on-unavailable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv("data-core/.env")
    load_dotenv(".env")
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY is missing.")

    games = _load_games(args.games_path, args.start_date, args.end_date)
    snapshot_dates = sorted(pd.to_datetime(games["game_date"]).dt.date.unique().tolist())
    if args.limit_dates:
        snapshot_dates = snapshot_dates[: args.limit_dates]

    audit = {
        "sport": args.sport,
        "games_path": args.games_path,
        "date_count": len(snapshot_dates),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "snapshots": [],
    }

    all_rows = []
    try:
        for day in snapshot_dates:
            snapshot_ts = snapshot_ts_for_date(day, hour_utc=args.snapshot_hour_utc)
            print(f"Fetching {args.sport} historical odds snapshot {snapshot_ts}...")
            snapshot = fetch_historical_odds_snapshot(
                api_key=api_key,
                sport=args.sport,
                snapshot_ts=snapshot_ts,
                bookmakers=args.bookmakers,
            )
            consensus = collapse_moneyline_consensus(snapshot.rows)
            all_rows.append(consensus)
            audit["snapshots"].append(
                {
                    "requested_ts": snapshot.requested_ts,
                    "returned_ts": snapshot.returned_ts,
                    "rows": int(len(consensus)),
                }
            )
    except HistoricalOddsUnavailable as exc:
        audit["status"] = "historical_unavailable"
        audit["error"] = str(exc)[:1000]
        os.makedirs(os.path.dirname(args.audit_output), exist_ok=True)
        with open(args.audit_output, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, sort_keys=True)
        print(f"Historical odds unavailable for this API key. Wrote audit to {args.audit_output}")
        if args.fail_on_unavailable:
            raise
        return

    odds = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    odds = _attach_game_ids(odds, games)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    odds.to_csv(args.output, index=False)
    audit["status"] = "ok"
    audit["output"] = args.output
    audit["rows"] = int(len(odds))
    audit["matched_game_pk_rows"] = int(odds["game_pk"].notna().sum()) if "game_pk" in odds else 0
    with open(args.audit_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
    print(f"Saved {len(odds)} historical moneyline rows to {args.output}")
    print(f"Saved audit to {args.audit_output}")


if __name__ == "__main__":
    main()
