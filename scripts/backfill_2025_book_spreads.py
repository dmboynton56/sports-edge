#!/usr/bin/env python3
"""
Backfill NULL games.book_spread entries for an NFL season by scraping
SportsOddsHistory spreads and pushing them to Supabase.

Usage:
    python scripts/backfill_2025_book_spreads.py --season 2025 --dry-run
    python scripts/backfill_2025_book_spreads.py --season 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, FrozenSet
import re

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_WEEK_11 import load_supabase_credentials, create_pg_connection  # noqa: E402

NFL_TEAM_MAP: Dict[str, str] = {
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
    "kan": "kansas city chiefs",
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
    "sfo": "san francisco 49ers",
    "san francisco 49ers": "san francisco 49ers",
    "sea": "seattle seahawks",
    "seattle seahawks": "seattle seahawks",
    "tb": "tampa bay buccaneers",
    "tampa bay buccaneers": "tampa bay buccaneers",
    "ten": "tennessee titans",
    "tennessee titans": "tennessee titans",
    "was": "washington commanders",
    "wsh": "washington commanders",
    "washington commanders": "washington commanders",
    "washington football team": "washington commanders",
}


@dataclass
class ScrapedSpread:
    season: int
    week: int
    game_date: date
    favorite: str
    underdog: str
    favorite_spread: float
    raw_spread: str
    favorite_norm: str
    underdog_norm: str


def normalize_team(name: str) -> str:
    key = (name or "").strip().lower().replace(".", "")
    return NFL_TEAM_MAP.get(key, key)


def parse_spread(raw_value: str) -> Optional[float]:
    raw = (raw_value or "").strip()
    if not raw or raw.lower() in {"nan", "pk", "pick", "pick'em"}:
        return 0.0 if raw.lower().startswith("p") else None
    raw = raw.replace("½", ".5").replace("¼", ".25").replace("¾", ".75")
    parts = raw.split()
    candidate = parts[-1]
    if candidate.lower().startswith("pk") or candidate.lower().startswith("pick"):
        return 0.0
    try:
        return float(candidate)
    except ValueError:
        logging.warning("Unable to parse spread value %s", raw_value)
        return None


def load_sportsoddshistory_html(season: int, html_file: Optional[str] = None) -> str:
    if html_file:
        path = Path(html_file).expanduser()
        logging.info("Loading spreads HTML from %s", path)
        return path.read_text(encoding="utf-8")
    url = f"https://www.sportsoddshistory.com/nfl-game-season/?y={season}"
    logging.info("Fetching spreads from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def _clean(cell) -> str:
    return cell.get_text(" ", strip=True) if cell else ""


def _parse_week_table(table, season: int, week: int) -> List[ScrapedSpread]:
    rows: List[ScrapedSpread] = []
    body = table.find("tbody") or table
    for row in body.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 9:
            continue
        favorite = _clean(cells[4])
        underdog = _clean(cells[8])
        spread_text = _clean(cells[6])
        date_text = _clean(cells[1])
        if not favorite or not underdog or not spread_text or not date_text:
            continue
        if favorite.lower() == "favorite" or underdog.lower() == "underdog":
            continue
        spread_value = parse_spread(spread_text)
        if spread_value is None:
            continue
        try:
            game_date = datetime.strptime(date_text, "%b %d, %Y").date()
        except ValueError:
            logging.debug("Skipping row with unparseable date %s", date_text)
            continue
        rows.append(
            ScrapedSpread(
                season=season,
                week=week,
                game_date=game_date,
                favorite=favorite,
                underdog=underdog,
                favorite_spread=spread_value,
                raw_spread=spread_text,
                favorite_norm=normalize_team(favorite),
                underdog_norm=normalize_team(underdog),
            )
        )
    return rows


WEEK_PATTERN = re.compile(r"Week\s+(\d+)", re.IGNORECASE)


def fetch_sportsoddshistory_spreads(season: int, html_file: Optional[str] = None) -> List[ScrapedSpread]:
    html = load_sportsoddshistory_html(season, html_file=html_file)
    soup = BeautifulSoup(html, "html.parser")
    spreads: List[ScrapedSpread] = []
    for header in soup.find_all("h3"):
        text = header.get_text(" ", strip=True)
        match = WEEK_PATTERN.search(text)
        if not match:
            continue
        week_num = int(match.group(1))
        table = header.find_next("table", class_="soh1")
        if not table:
            continue
        rows = _parse_week_table(table, season, week_num)
        spreads.extend(rows)
    week_count = len({spread.week for spread in spreads})
    logging.info("Parsed %d spread rows across %d weeks", len(spreads), week_count)
    return spreads


def fetch_supabase_games(conn, season: int) -> List[Dict]:
    query = """
        select g.id,
               g.week,
               g.home_team,
               g.away_team,
               (g.game_time_utc at time zone 'US/Eastern')::date as game_date_et
        from games g
        where g.league = 'NFL'
          and g.season = %s
          and g.book_spread is null
        order by g.game_time_utc
    """
    with conn.cursor() as cur:
        cur.execute(query, (season,))
        rows = cur.fetchall()
    columns = ["id", "week", "home_team", "away_team", "game_date_et"]
    games = [dict(zip(columns, row)) for row in rows]
    logging.info("Found %d Supabase games missing book_spread", len(games))
    return games


def enrich_games(games: Iterable[Dict]) -> List[Dict]:
    enriched_games: List[Dict] = []
    for game in games:
        game_date = game["game_date_et"]
        if game_date is None:
            logging.warning("Skipping game %s with null game_time_utc", game["id"])
            continue
        home_norm = normalize_team(game["home_team"])
        away_norm = normalize_team(game["away_team"])
        enriched_games.append(
            {
                **game,
                "home_norm": home_norm,
                "away_norm": away_norm,
            }
        )
    return enriched_games


def pair_games_with_spreads(
    scraped_spreads: List[ScrapedSpread],
    games: List[Dict],
) -> Tuple[List[Tuple[Dict, ScrapedSpread]], List[Dict]]:
    spreads_by_date: Dict[Tuple[date, FrozenSet[str]], ScrapedSpread] = {}
    spreads_by_week: Dict[Tuple[int, FrozenSet[str]], ScrapedSpread] = {}
    for spread in scraped_spreads:
        team_key = frozenset({spread.favorite_norm, spread.underdog_norm})
        spreads_by_date[(spread.game_date, team_key)] = spread
        spreads_by_week[(spread.week, team_key)] = spread

    matched: List[Tuple[Dict, ScrapedSpread]] = []
    unmatched_games: List[Dict] = []
    for game in games:
        team_key = frozenset({game["home_norm"], game["away_norm"]})
        spread = spreads_by_date.get((game["game_date_et"], team_key))
        if not spread:
            week_val = game.get("week")
            try:
                week_int = int(week_val) if week_val is not None else None
            except (TypeError, ValueError):
                week_int = None
            if week_int is not None:
                spread = spreads_by_week.get((week_int, team_key))
        if not spread:
            unmatched_games.append(game)
            continue
        matched.append((game, spread))
    return matched, unmatched_games


def build_updates(pairs: Iterable[Tuple[Dict, ScrapedSpread]]) -> List[Dict]:
    updates: List[Dict] = []
    for game, spread in pairs:
        favorite_is_home = spread.favorite_norm == game["home_norm"]
        home_spread = spread.favorite_spread if favorite_is_home else -spread.favorite_spread
        updates.append(
            {
                "game_id": game["id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "game_date": game["game_date_et"],
                "home_spread": home_spread,
                "favorite": spread.favorite,
                "underdog": spread.underdog,
                "raw_spread": spread.raw_spread,
            }
        )
    return updates


def apply_updates(conn, updates: List[Dict], dry_run: bool = False) -> int:
    if not updates:
        return 0
    if dry_run:
        for upd in updates:
            logging.info(
                "[DRY RUN] Would set book_spread=%s for %s vs %s on %s (favorite=%s, raw=%s)",
                upd["home_spread"],
                upd["home_team"],
                upd["away_team"],
                upd["game_date"],
                upd["favorite"],
                upd["raw_spread"],
            )
        return 0
    with conn.cursor() as cur:
        for upd in updates:
            cur.execute(
                "update games set book_spread = %s where id = %s",
                (upd["home_spread"], upd["game_id"]),
            )
    conn.commit()
    return len(updates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill games.book_spread via SportsOddsHistory spreads."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season to backfill (default: 2025).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show updates without writing to Supabase.",
    )
    parser.add_argument(
        "--html-file",
        type=str,
        default=None,
        help="Optional path to a saved SportsOddsHistory HTML document for offline parsing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    scraped = fetch_sportsoddshistory_spreads(args.season, html_file=args.html_file)
    if not scraped:
        logging.error("No spreads scraped for season %s", args.season)
        sys.exit(1)

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
        games_raw = fetch_supabase_games(conn, args.season)
        games = enrich_games(games_raw)
        pairs, unmatched = pair_games_with_spreads(scraped, games)
        updates = build_updates(pairs)
        updated = apply_updates(conn, updates, dry_run=args.dry_run)
        logging.info(
            "Prepared %d updates, applied %d. Supabase games without a spread: %d",
            len(updates),
            updated,
            len(unmatched),
        )
        if unmatched:
            logging.warning(
                "Still missing spreads for %d games (first few: %s)",
                len(unmatched),
                ", ".join(
                    f"{g['home_team']} vs {g['away_team']} on {g['game_date_et']}"
                    for g in unmatched[:10]
                ),
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
