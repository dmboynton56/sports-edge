#!/usr/bin/env python3
"""
Fetch MLB game market odds from The Odds API.

Fetches moneyline (h2h), run-line (spreads), and totals for MLB games.
Writes results to Supabase odds_snapshots table for join in research predictions.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg.types.json import Jsonb
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mlb_fetcher import mlb_schedule_to_games_df  # noqa: E402
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials, upsert_games_pg  # noqa: E402
from src.utils.team_codes import canonical_mlb_abbr  # noqa: E402

LOGGER = logging.getLogger(__name__)

SPORT_KEY = "baseball_mlb"
PREFERRED_BOOKMAKERS = ["draftkings", "fanduel", "betmgm"]


@dataclass
class OddsResult:
    """Result of fetching and syncing odds for a single game."""

    game_pk: int
    home_team: str
    away_team: str
    event_id: str
    matched: bool
    moneyline_synced: int = 0
    runline_synced: int = 0
    totals_synced: int = 0
    error: str | None = None


def normalize_team(team_name: str) -> str:
    """Normalize team name for matching."""
    import re

    return re.sub(r"[^a-z0-9]", "", team_name.lower())


def fetch_mlb_odds(api_key: str, markets: list[str] | None = None) -> list[dict[str, Any]]:
    """Fetch live MLB odds from The Odds API."""
    if markets is None:
        markets = ["h2h", "spreads", "totals"]

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Odds API error {resp.status_code}: {resp.text}")

    data = resp.json()
    LOGGER.info(f"Fetched {len(data)} MLB events from The Odds API")
    return data


def match_game(
    event: dict[str, Any], schedule: pd.DataFrame
) -> tuple[int | None, str | None, str | None]:
    """Match Odds API event to game_pk from schedule.
    
    Args:
        event: Odds API event with home_team, away_team, commence_time
        schedule: DataFrame with game_pk, game_date, home_team, away_team
        
    Returns:
        (game_pk, home_abbr, away_abbr) or (None, home_abbr, away_abbr) if no match
    """
    home_raw = event.get("home_team", "")
    away_raw = event.get("away_team", "")
    home_abbr = canonical_mlb_abbr(home_raw)
    away_abbr = canonical_mlb_abbr(away_raw)

    if not home_abbr or not away_abbr:
        LOGGER.warning(f"Could not resolve teams: {away_raw} @ {home_raw}")
        return None, None, None

    event_time = pd.to_datetime(event["commence_time"], utc=True)

    # Ensure schedule game_date is date type for comparison
    schedule_dates = pd.to_datetime(schedule["game_date"]).dt.date
    schedule_home = schedule["home_team"].map(canonical_mlb_abbr)
    schedule_away = schedule["away_team"].map(canonical_mlb_abbr)

    matches = schedule[(schedule_home == home_abbr) & (schedule_away == away_abbr)].copy()
    if not matches.empty and "game_datetime" in matches:
        kickoffs = pd.to_datetime(matches["game_datetime"], utc=True, errors="coerce")
        differences = (kickoffs - event_time).abs()
        if differences.notna().any():
            best_index = differences.idxmin()
            if differences.loc[best_index] <= pd.Timedelta(hours=6):
                return int(matches.loc[best_index, "game_pk"]), home_abbr, away_abbr

    # Date-only fallback is used only when the schedule provider omitted a
    # kickoff. It never crosses dates, which avoids repeated-series collisions.
    event_date = event_time.date()
    date_matches = matches[schedule_dates.loc[matches.index] == event_date]
    if len(date_matches) == 1:
        return int(date_matches.iloc[0]["game_pk"]), home_abbr, away_abbr

    LOGGER.warning(
        f"No schedule match for {away_abbr} @ {home_abbr} on {event_date} (event_id={event.get('id')})"
    )
    return None, home_abbr, away_abbr


def select_best_bookmaker(bookmakers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select preferred bookmaker, trying preferred books first."""
    ordered = bookmakers_in_preference_order(bookmakers)
    return ordered[0] if ordered else None


def bookmakers_in_preference_order(
    bookmakers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer DK/FD/BetMGM, then keep the remaining books in API order."""
    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for key in PREFERRED_BOOKMAKERS:
        match = next((bm for bm in bookmakers if bm.get("key") == key), None)
        if match is not None:
            ordered.append(match)
            seen.add(id(match))
    ordered.extend(book for book in bookmakers if id(book) not in seen)
    return ordered


def first_paired_moneyline(
    bookmakers: list[dict[str, Any]], home_team: str, away_team: str
) -> tuple[int | None, int | None, str | None]:
    """Return the first complete home/away moneyline across preferred books."""
    for book in bookmakers_in_preference_order(bookmakers):
        home_price, away_price, book_key = extract_moneyline(book, home_team, away_team)
        if home_price is not None and away_price is not None:
            return home_price, away_price, book_key
    return None, None, None


def first_paired_runline(
    bookmakers: list[dict[str, Any]], home_team: str, away_team: str
) -> tuple[float | None, int | None, int | None, str | None]:
    """Return the first complete run-line pair across preferred books."""
    for book in bookmakers_in_preference_order(bookmakers):
        home_line, home_price, away_price, book_key = extract_runline(
            book, home_team, away_team
        )
        if home_line is not None and home_price is not None and away_price is not None:
            return home_line, home_price, away_price, book_key
    return None, None, None, None


def first_paired_totals(
    bookmakers: list[dict[str, Any]],
) -> tuple[float | None, int | None, int | None, str | None]:
    """Return the first complete totals pair across preferred books."""
    for book in bookmakers_in_preference_order(bookmakers):
        total_line, over_price, under_price, book_key = extract_totals(book)
        if total_line is not None and over_price is not None and under_price is not None:
            return total_line, over_price, under_price, book_key
    return None, None, None, None


def extract_moneyline(
    bookmaker: dict[str, Any], home_team: str, away_team: str
) -> tuple[int | None, int | None, str | None]:
    """Extract home/away moneyline prices.
    
    Args:
        bookmaker: Bookmaker object from Odds API
        home_team: Home team name from event (e.g., "New York Yankees")
        away_team: Away team name from event
    """
    market = next((m for m in bookmaker.get("markets", []) if m["key"] == "h2h"), None)
    if not market or len(market.get("outcomes", [])) < 2:
        return None, None, None

    outcomes = market["outcomes"]
    home = next((o for o in outcomes if o.get("name") == home_team), None)
    away = next((o for o in outcomes if o.get("name") == away_team), None)

    home_price = int(home["price"]) if home and home.get("price") is not None else None
    away_price = int(away["price"]) if away and away.get("price") is not None else None
    book_key = bookmaker.get("key")

    return home_price, away_price, book_key


def extract_runline(
    bookmaker: dict[str, Any], home_team: str, away_team: str
) -> tuple[float | None, int | None, int | None, str | None]:
    """Extract run-line (spread) for home team.
    
    Args:
        bookmaker: Bookmaker object from Odds API
        home_team: Home team name from event (e.g., "New York Yankees")
        away_team: Away team name from event
    """
    market = next((m for m in bookmaker.get("markets", []) if m["key"] == "spreads"), None)
    if not market or len(market.get("outcomes", [])) < 2:
        return None, None, None, None

    outcomes = market["outcomes"]
    home = next((o for o in outcomes if o.get("name") == home_team), None)
    away = next((o for o in outcomes if o.get("name") == away_team), None)

    if not home or not away:
        return None, None, None, None

    home_line = float(home["point"]) if home.get("point") is not None else None
    home_price = int(home["price"]) if home.get("price") is not None else None
    away_price = int(away["price"]) if away.get("price") is not None else None
    book_key = bookmaker.get("key")

    return home_line, home_price, away_price, book_key


def extract_totals(
    bookmaker: dict[str, Any],
) -> tuple[float | None, int | None, int | None, str | None]:
    """Extract totals line and over/under prices."""
    market = next((m for m in bookmaker.get("markets", []) if m["key"] == "totals"), None)
    if not market or len(market.get("outcomes", [])) < 2:
        return None, None, None, None

    outcomes = market["outcomes"]
    over = next((o for o in outcomes if o.get("name") == "Over"), None)
    under = next((o for o in outcomes if o.get("name") == "Under"), None)

    if not over or not under:
        return None, None, None, None

    total_line = float(over["point"]) if over.get("point") is not None else None
    over_price = int(over["price"]) if over.get("price") is not None else None
    under_price = int(under["price"]) if under.get("price") is not None else None
    book_key = bookmaker.get("key")

    return total_line, over_price, under_price, book_key


def sync_event_odds(
    conn,
    event: dict[str, Any],
    game_pk: int,
    home_abbr: str,
    away_abbr: str,
    snapshot_ts: datetime,
) -> OddsResult:
    """Sync odds for a single event to Supabase."""
    result = OddsResult(
        game_pk=game_pk,
        home_team=home_abbr,
        away_team=away_abbr,
        event_id=event.get("id", ""),
        matched=True,
    )

    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        result.error = "no_bookmakers"
        return result

    # Extract event-level team names for matching outcomes
    event_home_team = event.get("home_team", "")
    event_away_team = event.get("away_team", "")
    commence_time = pd.Timestamp(event["commence_time"]).to_pydatetime()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, home_team, away_team, game_time_utc
            FROM games
            WHERE league = 'MLB'
              AND game_time_utc BETWEEN %s - INTERVAL '12 hours' AND %s + INTERVAL '12 hours'
            """,
            (commence_time, commence_time),
            prepare=False,
        )
        candidates = [
            row for row in cur.fetchall()
            if canonical_mlb_abbr(row[1]) == home_abbr
            and canonical_mlb_abbr(row[2]) == away_abbr
        ]
    if not candidates:
        result.matched = False
        result.error = "no_supabase_game_match"
        LOGGER.warning(
            "No Supabase games row for %s @ %s game_pk=%s commence=%s",
            away_abbr,
            home_abbr,
            game_pk,
            commence_time,
        )
        return result
    game_id = min(
        candidates,
        key=lambda row: abs((pd.Timestamp(row[3]) - pd.Timestamp(commence_time)).total_seconds()),
    )[0]

    with conn.cursor() as cur:
        # Moneyline — scan preferred books first, then the rest. Evening slates
        # often keep totals/spreads on DK after that book has already dropped h2h.
        home_ml, away_ml, ml_book = first_paired_moneyline(
            bookmakers, event_home_team, event_away_team
        )
        if home_ml is not None and away_ml is not None:
            ml_title = _book_title(bookmakers, ml_book)
            cur.executemany(
                """
                INSERT INTO odds_snapshots (
                  game_id, book, market, selection, line, price, snapshot_ts,
                  provider_event_id, commence_time_utc, metadata
                ) VALUES (%s,%s,'moneyline',%s,NULL,%s,%s,%s,%s,%s)
                """,
                [
                    (game_id, ml_book, side, price, snapshot_ts, event.get("id"), commence_time,
                     Jsonb({"game_pk": game_pk, "book_title": ml_title}))
                    for side, price in (("home", home_ml), ("away", away_ml))
                ],
            )
            result.moneyline_synced = 2

        # Run-line
        rl_line, rl_home_price, rl_away_price, rl_book = first_paired_runline(
            bookmakers, event_home_team, event_away_team
        )
        if rl_line is not None and rl_home_price is not None and rl_away_price is not None:
            rl_title = _book_title(bookmakers, rl_book)
            cur.executemany(
                """
                INSERT INTO odds_snapshots (
                  game_id, book, market, selection, line, price, snapshot_ts,
                  provider_event_id, commence_time_utc, metadata
                ) VALUES (%s,%s,'spread',%s,%s,%s,%s,%s,%s,%s)
                """,
                [
                    (game_id, rl_book, side, line, price, snapshot_ts, event.get("id"), commence_time,
                     Jsonb({"game_pk": game_pk, "book_title": rl_title}))
                    for side, line, price in (
                        ("home", rl_line, rl_home_price),
                        ("away", -rl_line, rl_away_price),
                    )
                ],
            )
            result.runline_synced = 2

        # Totals
        total_line, over_price, under_price, totals_book = first_paired_totals(bookmakers)
        if total_line is not None and over_price is not None and under_price is not None:
            totals_title = _book_title(bookmakers, totals_book)
            cur.executemany(
                """
                INSERT INTO odds_snapshots (
                  game_id, book, market, selection, line, price, snapshot_ts,
                  provider_event_id, commence_time_utc, metadata
                ) VALUES (%s,%s,'total',%s,%s,%s,%s,%s,%s,%s)
                """,
                [
                    (game_id, totals_book, side, total_line, price, snapshot_ts, event.get("id"),
                     commence_time, Jsonb({"game_pk": game_pk, "book_title": totals_title}))
                    for side, price in (("over", over_price), ("under", under_price))
                ],
            )
            result.totals_synced = 2

    missing_markets = [
        name
        for name, synced in (
            ("moneyline", result.moneyline_synced),
            ("run_line", result.runline_synced),
            ("total", result.totals_synced),
        )
        if not synced
    ]
    if missing_markets:
        LOGGER.warning(
            "No paired %s from any book for %s @ %s game_pk=%s books=%s",
            ",".join(missing_markets),
            away_abbr,
            home_abbr,
            game_pk,
            ",".join(str(book.get("key") or "?") for book in bookmakers) or "none",
        )
        if result.error is None:
            result.error = "missing_markets:" + ",".join(missing_markets)

    return result


def _book_title(bookmakers: list[dict[str, Any]], book_key: str | None) -> str | None:
    if not book_key:
        return None
    match = next((book for book in bookmakers if book.get("key") == book_key), None)
    return (match or {}).get("title") or book_key


def sync_mlb_odds(
    conn,
    odds_events: list[dict[str, Any]],
    schedule: pd.DataFrame,
    snapshot_ts: datetime,
) -> list[OddsResult]:
    """Sync all MLB odds events to Supabase."""
    results = []

    for event in odds_events:
        game_pk, home_abbr, away_abbr = match_game(event, schedule)
        if game_pk is None:
            results.append(
                OddsResult(
                    game_pk=0,
                    home_team=home_abbr or "",
                    away_team=away_abbr or "",
                    event_id=event.get("id", ""),
                    matched=False,
                    error="no_schedule_match",
                )
            )
            continue

        result = sync_event_odds(conn, event, game_pk, home_abbr, away_abbr, snapshot_ts)
        results.append(result)

    conn.commit()
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Fetch MLB game market odds.")
    parser.add_argument("--date", help="Game date (YYYY-MM-DD); defaults to today.")
    parser.add_argument("--season", type=int, help="Season year; defaults to date year.")
    parser.add_argument("--markets", default="h2h,spreads,totals", help="Comma-separated markets.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but do not write to DB.")
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY not found in environment")

    game_date = pd.to_datetime(args.date).date() if args.date else pd.Timestamp.now().date()
    season = args.season or game_date.year

    # Fetch schedule
    from src.data.mlb_fetcher import fetch_mlb_schedule

    schedule = fetch_mlb_schedule(
        season, start_date=game_date, end_date=game_date, include_uncompleted=True
    )
    if schedule.empty:
        LOGGER.warning(f"No MLB games scheduled for {game_date}")
        return

    LOGGER.info(f"Found {len(schedule)} games in schedule for {game_date}")

    # Fetch odds
    markets = [m.strip() for m in args.markets.split(",")]
    odds_events = fetch_mlb_odds(api_key, markets)

    if args.dry_run:
        LOGGER.info(f"Dry-run mode: fetched {len(odds_events)} events, skipping DB write")
        return

    # Sync to Supabase
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
        snapshot_ts = datetime.now(timezone.utc)
        games_df = mlb_schedule_to_games_df(schedule, season=season)
        game_ids = upsert_games_pg(conn, games_df)
        LOGGER.info("Upserted %s MLB slate games into serving table", len(game_ids))
        results = sync_mlb_odds(conn, odds_events, schedule, snapshot_ts)

        matched = [r for r in results if r.matched]
        ml_synced = sum(r.moneyline_synced for r in results)
        rl_synced = sum(r.runline_synced for r in results)
        tot_synced = sum(r.totals_synced for r in results)

        LOGGER.info(
            f"Synced odds for {len(matched)}/{len(odds_events)} events: "
            f"{ml_synced} moneyline, {rl_synced} run-line, {tot_synced} totals"
        )
        for market, attr in (
            ("moneyline", "moneyline_synced"),
            ("run_line", "runline_synced"),
            ("total", "totals_synced"),
        ):
            missing = [r for r in matched if not getattr(r, attr)]
            if missing:
                LOGGER.warning(
                    "Matched events missing paired %s: %s",
                    market,
                    ", ".join(f"{r.away_team}@{r.home_team}" for r in missing),
                )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
