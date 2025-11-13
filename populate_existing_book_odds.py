"""
Populate missing book odds for future games stored in Supabase.

Example:
    python populate_existing_book_odds.py --markets spreads
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import pandas as pd

from src.data.odds_fetcher import fetch_odds
from predict_WEEK_11 import load_supabase_credentials, create_pg_connection

NFL_TEAM_MAP = {
    "ari": "arizona cardinals",
    "arizona cardinals": "arizona cardinals",
    "atl": "atlanta falcons",
    "atlanta falcons": "atlanta falcons",
    "bal": "baltimore ravens",
    "baltimore ravens": "baltimore ravens",
    "buf": "buffalo bills",
    "buffalo bills": "buffalo bills",
    "car": "carolina panthers",
    "carolina panthers": "carolina panthers",
    "chi": "chicago bears",
    "chicago bears": "chicago bears",
    "cin": "cincinnati bengals",
    "cincinnati bengals": "cincinnati bengals",
    "cle": "cleveland browns",
    "cleveland browns": "cleveland browns",
    "dal": "dallas cowboys",
    "dallas cowboys": "dallas cowboys",
    "den": "denver broncos",
    "denver broncos": "denver broncos",
    "det": "detroit lions",
    "detroit lions": "detroit lions",
    "gb": "green bay packers",
    "green bay packers": "green bay packers",
    "hou": "houston texans",
    "houston texans": "houston texans",
    "ind": "indianapolis colts",
    "indianapolis colts": "indianapolis colts",
    "jax": "jacksonville jaguars",
    "jacksonville jaguars": "jacksonville jaguars",
    "kc": "kansas city chiefs",
    "kansas city chiefs": "kansas city chiefs",
    "lv": "las vegas raiders",
    "las vegas raiders": "las vegas raiders",
    "lac": "los angeles chargers",
    "los angeles chargers": "los angeles chargers",
    "la chargers": "los angeles chargers",
    "lar": "los angeles rams",
    "los angeles rams": "los angeles rams",
    "la rams": "los angeles rams",
    "la": "los angeles rams",
    "mia": "miami dolphins",
    "miami dolphins": "miami dolphins",
    "min": "minnesota vikings",
    "minnesota vikings": "minnesota vikings",
    "ne": "new england patriots",
    "new england patriots": "new england patriots",
    "no": "new orleans saints",
    "new orleans saints": "new orleans saints",
    "nyg": "new york giants",
    "new york giants": "new york giants",
    "nyj": "new york jets",
    "new york jets": "new york jets",
    "phi": "philadelphia eagles",
    "philadelphia eagles": "philadelphia eagles",
    "pit": "pittsburgh steelers",
    "pittsburgh steelers": "pittsburgh steelers",
    "sf": "san francisco 49ers",
    "san francisco 49ers": "san francisco 49ers",
    "sea": "seattle seahawks",
    "seattle seahawks": "seattle seahawks",
    "tb": "tampa bay buccaneers",
    "tampa bay buccaneers": "tampa bay buccaneers",
    "ten": "tennessee titans",
    "tennessee titans": "tennessee titans",
    "wft": "washington football team",
    "washington football team": "washington commanders",
    "was": "washington commanders",
    "wsh": "washington commanders",
    "washington commanders": "washington commanders",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Odds API spreads for upcoming games and update games.book_spread."
    )
    parser.add_argument(
        "--markets",
        type=str,
        default="spreads",
        help="Comma-separated list of Odds API markets to fetch (default: spreads).",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="us",
        help="Comma-separated Odds API regions (default: us).",
    )
    parser.add_argument(
        "--bookmakers",
        type=str,
        default=None,
        help="Comma-separated list of bookmaker keys to request from The Odds API (e.g., 'draftkings,fanduel').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed matching information for debugging.",
    )
    return parser.parse_args()


def normalize_team(name: str, league: str) -> str:
    key = (name or "").strip().lower()
    if league.upper() == "NFL":
        key = key.replace(".", "")
        return NFL_TEAM_MAP.get(key, key)
    return key


def fetch_target_games(conn) -> List[Dict]:
    """Return future games whose book_spread is NULL."""
    query = """
        select g.id,
               g.league,
               g.season,
               g.week,
               g.home_team,
               g.away_team,
               g.game_time_utc::date as game_date
        from games g
        where g.game_time_utc >= now()
          and g.book_spread is null
        order by g.game_time_utc
    """
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    columns = ["id", "league", "season", "week", "home_team", "away_team", "game_date"]
    return [dict(zip(columns, row)) for row in rows]


def build_game_lookup(games: List[Dict]) -> Dict[Tuple[str, str, str, datetime.date], Dict]:
    lookup = {}
    for game in games:
        league = game["league"].upper()
        home_norm = normalize_team(game["home_team"], league)
        away_norm = normalize_team(game["away_team"], league)
        game["norm_home"] = home_norm
        game["norm_away"] = away_norm
        key = (league, home_norm, away_norm, game["game_date"])
        lookup[key] = game
    return lookup


def update_game_book_spreads(conn, spreads: Dict[str, float]) -> int:
    if not spreads:
        return 0
    update_sql = """
        update games
        set book_spread = %s
        where id = %s
    """
    payload = [(line, game_id) for game_id, line in spreads.items()]
    with conn.cursor() as cur:
        cur.executemany(update_sql, payload)
    conn.commit()
    return len(payload)


def sanitize_market(market: str) -> Optional[str]:
    if not market:
        return None
    market = market.lower()
    if market.endswith('s'):
        # Odds API returns plural keys; Supabase table uses singular
        if market in ('spreads', 'spread'):
            return 'spread'
        if market in ('totals', 'total'):
            return 'total'
        if market in ('moneylines', 'moneyline'):
            return 'moneyline'
    if market in ('spread', 'total', 'moneyline'):
        return market
    return None


def populate_odds(conn, games: List[Dict], markets: str, regions: str, bookmakers: Optional[str], verbose: bool = False) -> int:
    if not games:
        print("No games require book spread updates.")
        return 0

    lookup = build_game_lookup(games)
    games_by_league_date = defaultdict(list)
    pair_date_map = defaultdict(list)
    for game in games:
        league = game["league"].upper()
        games_by_league_date[(league, game["game_date"])].append(game)
        pair_date_map[(league, game["norm_home"], game["norm_away"])].append(game["game_date"])

    home_spread_map: Dict[str, float] = {}
    unmatched_rows: List[Tuple[str, str, str, datetime.date, str]] = []
    missing_games: List[Tuple[str, datetime.date]] = []

    for (league, game_date), league_games in games_by_league_date.items():
        date_str = game_date.isoformat()
        print(f"Fetching odds for {league} games on {date_str} ({len(league_games)} matchups)...")
        odds_df = fetch_odds(
            league,
            date=date_str,
            markets=markets,
            regions=regions,
            bookmakers=bookmakers,
        )
        if odds_df.empty:
            print(f"  No odds returned for {league} {date_str}.")
            missing_games.append((league, game_date))
            continue

        odds_df["norm_home"] = odds_df["home_team"].apply(lambda n: normalize_team(n, league))
        odds_df["norm_away"] = odds_df["away_team"].apply(lambda n: normalize_team(n, league))
        odds_df["game_date"] = pd.to_datetime(odds_df["commence_time"]).dt.date

        for _, row in odds_df.iterrows():
            key = (league, row["norm_home"], row["norm_away"], row["game_date"])
            game = lookup.get(key)
            if not game:
                unmatched_rows.append((league, row["home_team"], row["away_team"], row["game_date"], row["book"]))
                continue
            market = sanitize_market(row["market"])
            if not market:
                unmatched_rows.append((league, row["home_team"], row["away_team"], row["game_date"], f"{row['book']} (market={row['market']})"))
                continue
            line = float(row["line"]) if pd.notna(row["line"]) else None
            if market == "spread" and line is not None:
                outcome_norm = normalize_team(row.get("outcome_name", ""), league)
                if outcome_norm == game.get("norm_home"):
                    home_spread_map[game["id"]] = line

    if verbose:
        for game_id, line in home_spread_map.items():
            game = next((g for g in games if g["id"] == game_id), None)
            if game:
                print(f"  -> {game['away_team']} @ {game['home_team']} ({game['game_date']}): {line:+.1f}")

    updated_spreads = update_game_book_spreads(conn, home_spread_map)
    if updated_spreads:
        print(f"Updated book_spread for {updated_spreads} games.")
    else:
        print("No book_spread values updated.")

    if missing_games:
        print("\nGames with no odds returned:")
        for league, game_date in missing_games:
            print(f"  - {league} on {game_date}")
    if unmatched_rows:
        print("\nOdds rows that did not match any Supabase game (team normalization mismatch?):")
        for league, home, away, game_date, book in unmatched_rows[:20]:
            key = (league, normalize_team(home, league), normalize_team(away, league))
            alt_dates = pair_date_map.get(key, [])
            hint = ""
            if alt_dates:
                alt_str = ", ".join(sorted({str(d) for d in alt_dates}))
                hint = f" (closest Supabase dates: {alt_str})"
            print(f"  - {league} {game_date}: {away} @ {home} ({book}){hint}")
        if len(unmatched_rows) > 20:
            print(f"  ... {len(unmatched_rows) - 20} more unmatched rows omitted")

    return updated_spreads


def main():
    args = parse_args()
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
        games = fetch_target_games(conn)
        if args.verbose:
            print(f"Found {len(games)} future games missing book_spread.")
        populate_odds(conn, games, args.markets, args.regions, args.bookmakers, verbose=args.verbose)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
