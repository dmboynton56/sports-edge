#!/usr/bin/env python3
"""Fetch MLB batter home run odds from The Odds API and optionally sync them."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from json_utils import dumps_strict  # noqa: E402
from src.data.mlb_fetcher import fetch_mlb_schedule  # noqa: E402
from src.data.mlb_hr_odds_fetcher import (  # noqa: E402
    DEFAULT_REGIONS,
    HR_MARKET,
    OddsApiClient,
    fetch_day_hr_odds,
    get_api_key,
)
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402


DEFAULT_CACHE_DIR = ROOT / "notebooks" / "cache"
DEFAULT_CSV_OUT = DEFAULT_CACHE_DIR / "mlb_home_run_odds.csv"
DEFAULT_AUDIT_OUT = DEFAULT_CACHE_DIR / "mlb_home_run_odds_audit.json"


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _sync_supabase(odds: pd.DataFrame) -> int:
    if odds.empty:
        return 0

    creds = load_supabase_credentials()
    missing = [
        name
        for name, value in {
            "SUPABASE_URL": creds["url"],
            "SUPABASE_DB_PASSWORD or supabaseDBpass": creds["db_password"],
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing Supabase credentials: {', '.join(missing)}")

    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    rows = []
    for _, row in odds.iterrows():
        rows.append(
            (
                _clean(row.get("game_id")),
                _clean(row.get("game_pk")),
                _clean(row.get("game_date")),
                _clean(row.get("event_time")),
                _clean(row.get("provider")) or "the_odds_api",
                _clean(row.get("provider_event_id")),
                _clean(row.get("market")),
                _clean(row.get("player_name")),
                _clean(row.get("normalized_player_name")),
                _clean(row.get("line")),
                _clean(row.get("side")),
                _clean(row.get("book")),
                _clean(row.get("book_title")),
                _clean(row.get("price")),
                _clean(row.get("implied_probability")),
                _clean(row.get("last_update")),
                _clean(row.get("snapshot_ts")),
                json.dumps(row.get("raw_record") or {}),
            )
        )

    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                insert into mlb_home_run_odds_snapshots (
                  game_id, game_pk, game_date, event_time, provider, provider_event_id,
                  market, player_name, normalized_player_name, line, side, book,
                  book_title, price, implied_probability, last_update, snapshot_ts, raw_record
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(rows)


def _write_outputs(odds: pd.DataFrame, audit: dict[str, Any], csv_out: Path, audit_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    csv_frame = odds.copy()
    if not csv_frame.empty and "raw_record" in csv_frame.columns:
        csv_frame["raw_record"] = csv_frame["raw_record"].map(lambda value: json.dumps(value or {}, sort_keys=True))
    csv_frame.to_csv(csv_out, index=False)
    audit_out.write_text(dumps_strict(audit, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch current MLB home run prop odds.")
    parser.add_argument("--date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(), default=datetime.now(timezone.utc).date())
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--market", default=HR_MARKET)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--sync-supabase", action="store_true")
    parser.add_argument("--allow-missing-key", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(ROOT / ".env", override=False)
    args = parse_args()

    try:
        api_key = get_api_key()
    except ValueError:
        if not args.allow_missing_key:
            raise
        audit = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "gameDate": args.date.isoformat(),
            "market": args.market,
            "regions": args.regions,
            "oddsRows": 0,
            "gaps": ["ODDS_API_KEY not found in environment."],
        }
        _write_outputs(pd.DataFrame(), audit, args.out_csv, args.audit_out)
        print(f"Wrote empty MLB HR odds audit to {args.audit_out}")
        return

    season = args.season or args.date.year
    schedule = fetch_mlb_schedule(season, start_date=args.date, end_date=args.date, include_uncompleted=True)
    if schedule.empty:
        audit = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "gameDate": args.date.isoformat(),
            "market": args.market,
            "regions": args.regions,
            "oddsRows": 0,
            "gaps": [f"No MLB schedule rows fetched for {args.date}."],
        }
        _write_outputs(pd.DataFrame(), audit, args.out_csv, args.audit_out)
        print(f"Wrote empty MLB HR odds audit to {args.audit_out}")
        return

    client = OddsApiClient(api_key=api_key)
    odds, audit = fetch_day_hr_odds(
        client,
        game_date=args.date,
        schedule=schedule,
        regions=args.regions,
        market=args.market,
    )
    audit["generatedAt"] = datetime.now(timezone.utc).isoformat()
    synced = _sync_supabase(odds) if args.sync_supabase else 0
    audit["supabaseRowsInserted"] = synced
    _write_outputs(odds, audit, args.out_csv, args.audit_out)
    print(f"Wrote {len(odds)} MLB HR odds rows to {args.out_csv}")
    print(f"Wrote MLB HR odds audit to {args.audit_out}")
    if args.sync_supabase:
        print(f"Synced {synced} MLB HR odds rows to Supabase")


if __name__ == "__main__":
    main()
