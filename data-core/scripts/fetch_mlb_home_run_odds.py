#!/usr/bin/env python3
"""Fetch MLB batter home run odds from The Odds API and optionally sync them."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
    HR_MARKETS,
    SLATE_TIMEZONE,
    MlbHrOddsError,
    OddsApiClient,
    fetch_day_hr_odds,
    fetch_day_hr_odds_propline,
    get_api_key,
)
from src.data.propline_client import PropLineClient, PropLineError, get_propline_api_key  # noqa: E402
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


def _odds_already_used_today(denver_date: str) -> tuple[bool, str | None]:
    """Check if The Odds API was already used for this Denver date.
    
    Returns:
        (already_used, provider_used) - True if 'the_odds_api' provider was used today
    """
    try:
        creds = load_supabase_credentials()
        if not creds.get("url") or not creds.get("db_password"):
            return False, None
        
        conn = create_pg_connection(
            supabase_url=creds["url"],
            password=creds["db_password"],
            host_override=creds.get("db_host"),
            port=creds["db_port"],
            database=creds["db_name"],
            user=creds["db_user"],
        )
        try:
            with conn.cursor() as cur:
                # Check if The Odds API was already used today by looking at odds_snapshots
                cur.execute(
                    """
                    select distinct provider
                    from mlb_home_run_odds_snapshots
                    where game_date = %s
                      and provider = 'the_odds_api'
                    limit 1
                    """,
                    (denver_date,),
                    prepare=False,
                )
                result = cur.fetchone()
                if result:
                    return True, result[0]
                return False, None
        finally:
            conn.close()
    except Exception as exc:
        print(f"Warning: Could not check Odds API usage for {denver_date}: {exc}")
        return False, None


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


def _requested_markets(args: argparse.Namespace) -> list[str] | None:
    if args.markets:
        values = args.markets.split(",")
    elif args.market:
        values = [args.market]
    else:
        return None
    return list(dict.fromkeys(value.strip() for value in values if value.strip())) or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch current MLB home run prop odds.")
    parser.add_argument(
        "--date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=datetime.now(ZoneInfo(SLATE_TIMEZONE)).date(),
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument(
        "--market",
        default=None,
        help="Optional single-market override; default fetches standard and alternate HR markets.",
    )
    parser.add_argument(
        "--markets",
        default=None,
        help="Optional comma-separated provider market keys; overrides --market.",
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--sync-supabase", action="store_true")
    parser.add_argument("--allow-missing-key", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(ROOT / ".env", override=False)
    args = parse_args()
    requested_markets = _requested_markets(args)
    audit_market = requested_markets[0] if requested_markets and len(requested_markets) == 1 else HR_MARKET

    # Check if Odds API was already used today (Denver date)
    denver_date = args.date.isoformat()
    odds_used_today, prior_provider = _odds_already_used_today(denver_date)
    
    # Try The Odds API first
    odds_api_key = None
    propline_api_key = None
    try:
        odds_api_key = get_api_key()
    except ValueError:
        pass

    try:
        propline_api_key = get_propline_api_key()
    except ValueError:
        pass

    if not odds_api_key and not propline_api_key:
        if not args.allow_missing_key:
            raise RuntimeError("Neither ODDS_API_KEY nor PROPLINE_API_KEY found in environment")
        audit = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "gameDate": args.date.isoformat(),
            "market": audit_market,
            "markets": requested_markets or list(HR_MARKETS),
            "regions": args.regions,
            "oddsRows": 0,
            "gaps": ["Neither ODDS_API_KEY nor PROPLINE_API_KEY found in environment."],
            "provider": "none",
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
            "market": audit_market,
            "markets": requested_markets or list(HR_MARKETS),
            "regions": args.regions,
            "oddsRows": 0,
            "gaps": [f"No MLB schedule rows fetched for {args.date}."],
            "provider": "none",
        }
        _write_outputs(pd.DataFrame(), audit, args.out_csv, args.audit_out)
        print(f"Wrote empty MLB HR odds audit to {args.audit_out}")
        return

    odds = pd.DataFrame()
    audit: dict[str, Any] = {}
    should_fallback = False
    fallback_reason = ""

    # Skip Odds API if already used today (conserve credits)
    if odds_used_today and odds_api_key:
        should_fallback = True
        fallback_reason = f"The Odds API already used today for {denver_date} (conserving credits: at most one Odds API call per Denver day)"
        print(fallback_reason)
    # Try The Odds API if key is available and not already used today
    elif odds_api_key:
        client = OddsApiClient(api_key=odds_api_key)
        try:
            odds, audit = fetch_day_hr_odds(
                client,
                game_date=args.date,
                schedule=schedule,
                regions=args.regions,
                markets=requested_markets,
            )
            print(f"The Odds API returned {len(odds)} priced rows, {audit.get('apiCreditsRemaining', 'unknown')} credits remaining")
            # Check for quota exhaustion or empty board
            if audit.get("apiCreditsRemaining") == "0":
                should_fallback = True
                fallback_reason = "The Odds API quota exhausted (0 credits remaining)"
            elif odds.empty or len(odds) == 0:
                should_fallback = True
                fallback_reason = "The Odds API returned 0 priced rows"
        except MlbHrOddsError as exc:
            error_str = str(exc)
            # Fallback on 401 (auth), 429 (rate limit), or quota errors
            if "401" in error_str or "429" in error_str or "quota" in error_str.lower():
                should_fallback = True
                fallback_reason = f"The Odds API error: {exc}"
            else:
                raise

    # Fallback to PropLine if needed and available
    if should_fallback and propline_api_key:
        print(f"Falling back to PropLine: {fallback_reason}")
        propline_client = PropLineClient(api_key=propline_api_key)
        try:
            odds, audit = fetch_day_hr_odds_propline(
                propline_client,
                game_date=args.date,
                schedule=schedule,
                markets=requested_markets,
            )
            audit["fallbackReason"] = fallback_reason
            if odds_used_today:
                audit["oddsAlreadyUsedToday"] = True
                audit["priorProvider"] = prior_provider
            print(f"PropLine returned {audit.get('eventsReturned', 0)} events, matched {audit.get('eventsMatched', 0)}, priced {len(odds)} rows")
        except PropLineError as exc:
            print(f"PropLine fallback also failed: {exc}")
            # Keep empty odds and original audit if both fail
            if not audit:
                audit = {
                    "gameDate": args.date.isoformat(),
                    "market": audit_market,
                    "markets": requested_markets or list(HR_MARKETS),
                    "oddsRows": 0,
                    "gaps": [f"The Odds API failed: {fallback_reason}", f"PropLine fallback failed: {exc}"],
                    "provider": "failed",
                }
    elif should_fallback and not propline_api_key:
        print(f"Would fallback to PropLine but PROPLINE_API_KEY not set: {fallback_reason}")
        audit["fallbackSkipped"] = "PROPLINE_API_KEY not configured"
        if odds_used_today:
            audit["oddsAlreadyUsedToday"] = True
            audit["priorProvider"] = prior_provider
    elif not odds_api_key and propline_api_key:
        # ODDS_API_KEY missing, go straight to PropLine
        print("ODDS_API_KEY not set, using PropLine")
        propline_client = PropLineClient(api_key=propline_api_key)
        try:
            odds, audit = fetch_day_hr_odds_propline(
                propline_client,
                game_date=args.date,
                schedule=schedule,
                markets=requested_markets,
            )
            print(f"PropLine returned {audit.get('eventsReturned', 0)} events, matched {audit.get('eventsMatched', 0)}, priced {len(odds)} rows")
        except PropLineError as exc:
            print(f"PropLine fetch failed: {exc}")
            audit = {
                "gameDate": args.date.isoformat(),
                "market": audit_market,
                "markets": requested_markets or list(HR_MARKETS),
                "oddsRows": 0,
                "gaps": [f"PropLine fetch failed: {exc}"],
                "provider": "failed",
            }

    audit["generatedAt"] = datetime.now(timezone.utc).isoformat()
    synced = _sync_supabase(odds) if args.sync_supabase else 0
    audit["supabaseRowsInserted"] = synced
    _write_outputs(odds, audit, args.out_csv, args.audit_out)
    provider = audit.get('provider', 'unknown')
    credits_info = f", {audit.get('apiCreditsRemaining', 'unknown')} credits remaining" if provider == 'the_odds_api' else ""
    print(f"Wrote {len(odds)} MLB HR odds rows to {args.out_csv} (provider: {provider}{credits_info})")
    print(f"Wrote MLB HR odds audit to {args.audit_out}")
    if args.sync_supabase:
        print(f"Synced {synced} MLB HR odds rows to Supabase")


if __name__ == "__main__":
    main()
