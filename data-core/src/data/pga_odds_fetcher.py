"""
PGA golf outright odds fetcher using The Odds API v4.

Supports all four majors. Returns structured DataFrames with per-book odds
and consensus no-vig implied probabilities for the full field.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.odds_math import (
    american_to_decimal,
    american_to_implied,
    best_available,
    consensus_no_vig,
    overround,
    remove_vig,
)

load_dotenv()

LOG = logging.getLogger(__name__)

GOLF_SPORT_KEYS = {
    "masters": "golf_masters_tournament_winner",
    "pga_championship": "golf_pga_championship_winner",
    "us_open": "golf_us_open_winner",
    "the_open": "golf_the_open_championship_winner",
}

TOURNAMENT_DISPLAY = {
    "masters": "Masters Tournament",
    "pga_championship": "PGA Championship",
    "us_open": "US Open",
    "the_open": "The Open Championship",
}

BASE_URL = "https://api.the-odds-api.com/v4"


def _normalize_name(name: str) -> str:
    """
    Normalize player name for fuzzy matching.
    Strips accents, lowercases, removes suffixes (Jr., III, etc.), collapses whitespace.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.lower().strip()
    ascii_name = re.sub(r"\b(jr\.?|sr\.?|iii|ii|iv)\b", "", ascii_name)
    ascii_name = re.sub(r"[^a-z\s]", "", ascii_name)
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip()
    return ascii_name


def build_name_index(names: List[str]) -> Dict[str, str]:
    """Map normalized names back to original names for matching."""
    return {_normalize_name(n): n for n in names}


def match_player_name(
    api_name: str,
    name_index: Dict[str, str],
) -> Optional[str]:
    """
    Match an API player name to our prediction player names.
    Tries exact normalized match, then last-name + first-initial match.
    """
    norm = _normalize_name(api_name)
    if norm in name_index:
        return name_index[norm]

    parts = norm.split()
    if len(parts) >= 2:
        last = parts[-1]
        first_init = parts[0][0] if parts[0] else ""
        for idx_norm, original in name_index.items():
            idx_parts = idx_norm.split()
            if len(idx_parts) >= 2:
                if idx_parts[-1] == last and idx_parts[0] and idx_parts[0][0] == first_init:
                    return original

    return None


def get_api_key() -> str:
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise ValueError("ODDS_API_KEY not found in environment")
    return key


def fetch_active_golf_sports(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return list of currently active golf sport objects from The Odds API."""
    api_key = api_key or get_api_key()
    resp = requests.get(f"{BASE_URL}/sports", params={"apiKey": api_key})
    resp.raise_for_status()
    return [s for s in resp.json() if s.get("group", "").lower() == "golf" and s.get("active")]


def detect_active_tournament(api_key: Optional[str] = None) -> Optional[str]:
    """
    Auto-detect which major tournament is currently active.
    Returns our internal key (e.g., 'masters') or None.
    """
    active = fetch_active_golf_sports(api_key)
    active_keys = {s["key"] for s in active}
    for our_key, api_key_val in GOLF_SPORT_KEYS.items():
        if api_key_val in active_keys:
            return our_key
    return None


def fetch_outrights(
    tournament: str,
    regions: str = "us",
    odds_format: str = "american",
    api_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Fetch outright winner odds for a golf major.

    Args:
        tournament: internal key like 'masters', 'pga_championship', etc.
        regions: API regions param (default 'us')
        odds_format: 'american' or 'decimal'
        api_key: override for ODDS_API_KEY env var

    Returns:
        (raw_json_list, response_headers_dict)
    """
    api_key = api_key or get_api_key()
    sport_key = GOLF_SPORT_KEYS.get(tournament)
    if not sport_key:
        raise ValueError(f"Unknown tournament '{tournament}'. Valid: {list(GOLF_SPORT_KEYS)}")

    resp = requests.get(
        f"{BASE_URL}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": "outrights",
            "oddsFormat": odds_format,
        },
    )
    resp.raise_for_status()

    headers = {
        "requests_remaining": resp.headers.get("x-requests-remaining", ""),
        "requests_used": resp.headers.get("x-requests-used", ""),
    }
    return resp.json(), headers


def parse_outrights(
    raw: List[Dict[str, Any]],
    odds_format: str = "american",
) -> pd.DataFrame:
    """
    Flatten The Odds API outright response into a tidy DataFrame.

    Columns: player, book, book_title, price, implied_prob, last_update
    """
    rows = []
    for event in raw:
        commence = event.get("commence_time", "")
        for bm in event.get("bookmakers", []):
            book_key = bm.get("key", "")
            book_title = bm.get("title", "")
            last_update = bm.get("last_update", "")
            for market in bm.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price", 0)
                    if odds_format == "american":
                        imp = american_to_implied(price)
                    else:
                        imp = 1.0 / price if price > 0 else 0.0
                    rows.append(
                        {
                            "player": outcome.get("name", ""),
                            "book": book_key,
                            "book_title": book_title,
                            "price": price,
                            "decimal_odds": american_to_decimal(price) if odds_format == "american" else price,
                            "implied_prob": imp,
                            "last_update": last_update,
                            "commence_time": commence,
                        }
                    )
    return pd.DataFrame(rows)


def build_player_odds_summary(
    df: pd.DataFrame,
    prediction_names: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Build per-player odds summary with consensus implied prob, best price, and per-book breakdown.

    Args:
        df: parsed outrights DataFrame from parse_outrights()
        prediction_names: optional list of player names from our model to match against

    Returns:
        (player_odds_list, unmatched_api_names_dict)
    """
    if df.empty:
        return [], {}

    name_index = build_name_index(prediction_names) if prediction_names else {}

    books_in_data = sorted(df["book"].unique().tolist())

    per_book_implied: Dict[str, Dict[str, float]] = {}
    for book in books_in_data:
        bdf = df[df["book"] == book]
        per_book_implied[book] = dict(zip(bdf["player"], bdf["implied_prob"]))

    book_dicts = [per_book_implied[b] for b in books_in_data]
    consensus = consensus_no_vig(book_dicts)

    all_players = sorted(df["player"].unique(), key=lambda p: consensus.get(p, 0), reverse=True)

    unmatched: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []

    for api_name in all_players:
        matched_name = match_player_name(api_name, name_index) if name_index else api_name
        if prediction_names and not matched_name:
            unmatched[api_name] = "no match in predictions"
            matched_name = api_name

        player_rows = df[df["player"] == api_name]
        book_odds: Dict[str, int] = {}
        book_details: Dict[str, Dict[str, Any]] = {}
        for _, row in player_rows.iterrows():
            b = row["book"]
            book_odds[b] = int(row["price"])
            book_details[b] = {
                "price": int(row["price"]),
                "implied": round(row["implied_prob"], 5),
                "decimal": round(row["decimal_odds"], 3),
            }

        bp, bb = best_available(book_odds)

        results.append(
            {
                "player": matched_name,
                "apiName": api_name,
                "bestPrice": bp,
                "bestBook": bb,
                "consensusImplied": round(consensus.get(api_name, 0.0), 5),
                "bookOdds": book_details,
            }
        )

    return results, unmatched


def fetch_and_summarize(
    tournament: str,
    prediction_names: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end: fetch odds, parse, summarize, return dict ready for JSON export.

    Returns dict with keys: tournament, fetchedAt, books, overrounds, playerOdds, unmatched
    """
    raw, headers = fetch_outrights(tournament, api_key=api_key)
    if not raw:
        LOG.warning("No odds data returned for %s", tournament)
        return {
            "tournament": TOURNAMENT_DISPLAY.get(tournament, tournament),
            "fetchedAt": datetime.now(timezone.utc).isoformat(),
            "books": [],
            "overrounds": {},
            "playerOdds": [],
            "unmatched": {},
            "apiCreditsRemaining": headers.get("requests_remaining", ""),
        }

    df = parse_outrights(raw)
    player_odds, unmatched = build_player_odds_summary(df, prediction_names)

    books = sorted(df["book"].unique().tolist())
    book_overrounds = {}
    for b in books:
        bdf = df[df["book"] == b]
        book_overrounds[b] = round(overround(dict(zip(bdf["player"], bdf["implied_prob"]))), 4)

    if unmatched:
        LOG.warning("Unmatched API players: %s", list(unmatched.keys()))

    return {
        "tournament": TOURNAMENT_DISPLAY.get(tournament, tournament),
        "fetchedAt": datetime.now(timezone.utc).isoformat(),
        "commenceTime": df["commence_time"].iloc[0] if not df.empty else "",
        "books": books,
        "overrounds": book_overrounds,
        "playerOdds": player_odds,
        "unmatched": unmatched,
        "apiCreditsRemaining": headers.get("requests_remaining", ""),
    }
