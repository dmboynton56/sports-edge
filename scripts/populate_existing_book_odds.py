#!/usr/bin/env python3
"""
Backfill book spreads for existing games in Supabase when book_spread is NULL.

Usage:
    python scripts/populate_existing_book_odds.py --league NFL --start-date 2025-11-10 --end-date 2025-11-17 --bookmakers fanduel

Environment:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    ODDS_API_KEY (used by src.data.odds_fetcher)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd
import psycopg
from dotenv import load_dotenv

# Ensure project root on sys.path for src imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import odds_fetcher
from scripts.predict_week import (
    create_pg_connection,
    load_supabase_credentials,
)

# Mapping of NFL abbreviations to common Odds API team strings for matching.
NFL_TEAM_ALIASES: Dict[str, list[str]] = {
    "ARI": ["arizona cardinals"],
    "ATL": ["atlanta falcons"],
    "BAL": ["baltimore ravens"],
    "BUF": ["buffalo bills"],
    "CAR": ["carolina panthers"],
    "CHI": ["chicago bears"],
    "CIN": ["cincinnati bengals"],
    "CLE": ["cleveland browns"],
    "DAL": ["dallas cowboys"],
    "DEN": ["denver broncos"],
    "DET": ["detroit lions"],
    "GB": ["green bay packers", "green bay"],
    "HOU": ["houston texans"],
    "IND": ["indianapolis colts"],
    "JAX": ["jacksonville jaguars"],
    "KC": ["kansas city chiefs", "kansas city"],
    "LAC": ["los angeles chargers", "la chargers"],
    "LAR": ["los angeles rams", "la rams"],
    # Some data sources store just "LA" without specifying Rams/Chargers; handle both.
    "LA": ["los angeles rams", "los angeles chargers", "la rams", "la chargers"],
    "LV": ["las vegas raiders", "oakland raiders", "raiders"],
    "MIA": ["miami dolphins"],
    "MIN": ["minnesota vikings"],
    "NE": ["new england patriots", "new england"],
    "NO": ["new orleans saints"],
    "NYG": ["new york giants", "ny giants"],
    "NYJ": ["new york jets", "ny jets"],
    "PHI": ["philadelphia eagles"],
    "PIT": ["pittsburgh steelers"],
    "SEA": ["seattle seahawks"],
    "SF": ["san francisco 49ers", "san francisco"],
    "TB": ["tampa bay buccaneers", "tampa bay", "buccaneers"],
    "TEN": ["tennessee titans"],
    "WAS": ["washington commanders", "washington"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate Supabase book_spread for games missing it.")
    parser.add_argument("--league", choices=["NFL", "NBA"], required=True, help="League to process.")
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Earliest game_date (default: today UTC).",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Latest game_date (default: start-date + 7 days).",
    )
    parser.add_argument(
        "--bookmakers",
        default=None,
        help="Comma-separated bookmaker keys to prefer (e.g., 'fanduel,draftkings').",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="If set, update games.book_spread without inserting odds_snapshots (useful when RLS blocks inserts).",
    )
    return parser.parse_args()


def _date_bounds(args: argparse.Namespace):
    start = args.start_date or datetime.now(tz=timezone.utc).date()
    end = args.end_date or (start + timedelta(days=7))
    if end < start:
        raise ValueError("end-date must be on or after start-date")
    return start, end


def fetch_games_missing_spread_pg(conn: psycopg.Connection, league: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """Pull games with null book_spread in the date window using direct PG connection (bypasses RLS)."""
    query = """
        select *
        from games
        where league = %s
          and book_spread is null
          and game_time_utc between %s and %s
    """
    df = pd.read_sql_query(
        query,
        conn,
        params=(
            league,
            f"{start_date} 00:00:00+00",
            f"{end_date} 23:59:59+00",
        ),
    )
    if not df.empty:
        df["game_time_utc"] = pd.to_datetime(df["game_time_utc"], utc=True, errors="coerce")
        df["game_date"] = df["game_time_utc"].dt.date
    return df


def _canonical_team(code_or_name: str, league: str) -> Optional[str]:
    """Map Odds API team name or code to our canonical code (NFL only for now)."""
    if not isinstance(code_or_name, str):
        return None
    token = code_or_name.strip().lower().replace(" ", "").replace(".", "").replace("-", "")
    # Direct code match
    for code in NFL_TEAM_ALIASES:
        if token == code.lower():
            return code
    # Alias match
    for code, aliases in NFL_TEAM_ALIASES.items():
        for alias in aliases:
            alias_token = alias.replace(" ", "").replace(".", "").replace("-", "")
            if token == alias_token:
                return code
    return None


def pick_home_spread(
    odds_df: pd.DataFrame,
    home_team_code: str,
    away_team_code: str,
    bookmaker: Optional[str],
) -> Optional[Dict]:
    """Return a dict with book, line, price for the home team spread (home perspective)."""
    odds_df = odds_df.copy()
    odds_df["home_code"] = odds_df["home_team"].apply(lambda x: _canonical_team(x, "NFL"))
    odds_df["away_code"] = odds_df["away_team"].apply(lambda x: _canonical_team(x, "NFL"))

    spreads = odds_df[odds_df["market"] == "spreads"]
    if bookmaker:
        spreads = spreads[spreads["book"] == bookmaker]
    # Allow LA ambiguity: match either Rams or Chargers if the code is "LA"
    home_candidates = ["LAR", "LAC"] if home_team_code == "LA" else [home_team_code]
    away_candidates = ["LAR", "LAC"] if away_team_code == "LA" else [away_team_code]
    spreads = spreads[
        (spreads["home_code"].isin(home_candidates))
        & (spreads["away_code"].isin(away_candidates))
    ]
    if spreads.empty:
        # Fallback: substring match on raw team names when canonical codes fail
        spreads = odds_df[odds_df["market"] == "spreads"]
        spreads = spreads[
            spreads["home_team"].str.lower().str.contains(home_team_code.lower(), na=False)
            & spreads["away_team"].str.lower().str.contains(away_team_code.lower(), na=False)
        ]
    if spreads.empty:
        return None
    # Try to find the home-team outcome explicitly
    home_rows = spreads[
        spreads["outcome_name"].str.lower().str.contains(home_team_code.lower(), na=False)
    ]
    if not home_rows.empty:
        row = home_rows.iloc[0]
        line = row.get("line")
        price = row.get("price")
        book = row.get("book")
        if pd.isna(line):
            return None
        return {"book": book, "line": float(line), "price": float(price) if pd.notna(price) else None}

    # If no home outcome, try away outcome and negate the line to home perspective
    away_rows = spreads[
        spreads["outcome_name"].str.lower().str.contains(away_team_code.lower(), na=False)
    ]
    if not away_rows.empty:
        row = away_rows.iloc[0]
        line = row.get("line")
        price = row.get("price")
        book = row.get("book")
        if pd.isna(line):
            return None
        return {"book": book, "line": float(-line), "price": float(price) if pd.notna(price) else None}

    return None


def update_game_and_snapshot_pg(conn: psycopg.Connection, game_id: str, spread: Dict, skip_snapshots: bool) -> bool:
    """Update games.book_spread and insert odds_snapshot using direct PG connection."""
    now_ts = datetime.now(tz=timezone.utc)
    updated = False
    with conn.cursor() as cur:
        cur.execute(
            "update games set book_spread = %s where id = %s",
            (spread["line"], game_id),
        )
        updated = cur.rowcount > 0
        if not skip_snapshots:
            cur.execute(
                """
                insert into odds_snapshots (game_id, book, market, line, price, snapshot_ts)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (
                    game_id,
                    spread["book"] or "unknown",
                    "spread",
                    spread["line"],
                    spread.get("price"),
                    now_ts,
                ),
            )
    conn.commit()
    return updated


def main():
    args = parse_args()
    start_date, end_date = _date_bounds(args)
    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )

    games = fetch_games_missing_spread_pg(conn, args.league, start_date, end_date)
    if games.empty:
        print("No games with missing book_spread in the requested window.")
        conn.close()
        return

    print(f"Found {len(games)} games missing book_spread between {start_date} and {end_date}.")
    updated_count = 0
    skipped_no_spread = 0
    # Fetch odds per date to limit API calls
    for game_date, day_games in games.groupby("game_date"):
        odds_df = odds_fetcher.fetch_odds(
            args.league,
            date=game_date.isoformat(),
            markets="spreads",
            bookmakers=args.bookmakers,
        )
        if odds_df.empty:
            # Fallback: try without date filters (Odds API sometimes rejects far-future dates).
            odds_df = odds_fetcher.fetch_odds(
                args.league,
                date=None,
                markets="spreads",
                bookmakers=args.bookmakers,
            )
        if odds_df.empty:
            print(f"  No odds returned for {game_date}, skipping {len(day_games)} games.")
            continue
        for _, game in day_games.iterrows():
            spread = pick_home_spread(odds_df, game["home_team"], game["away_team"], args.bookmakers)
            if not spread:
                # Fallback: re-fetch without bookmaker filter for this game/date
                alt_df = odds_fetcher.fetch_odds(
                    args.league,
                    date=game_date.isoformat(),
                    markets="spreads",
                    bookmakers=None,
                )
                if alt_df.empty:
                    alt_df = odds_fetcher.fetch_odds(
                        args.league,
                        date=None,
                        markets="spreads",
                        bookmakers=None,
                    )
                spread = pick_home_spread(alt_df, game["home_team"], game["away_team"], None)
            if not spread:
                print(f"  No spread found for {game['away_team']} @ {game['home_team']} on {game_date}.")
                skipped_no_spread += 1
                continue
            wrote = update_game_and_snapshot_pg(conn, game["id"], spread, args.skip_snapshots)
            if not wrote:
                print(f"  Attempted update but Supabase returned no rows for {game['away_team']} @ {game['home_team']} ({game_date}). Check credentials/RLS.")
            else:
                updated_count += 1
                print(f"  Updated {game['away_team']} @ {game['home_team']} ({game_date}) -> {spread['line']} ({spread['book']})")
    conn.close()
    print(f"\nSummary: updated {updated_count} games, skipped {skipped_no_spread} with no spread returned.")


if __name__ == "__main__":
    main()
