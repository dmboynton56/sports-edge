"""Tests for fetch_mlb_game_odds script."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts to path
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fetch_mlb_game_odds as odds_fetcher
from src.utils.team_codes import canonical_mlb_abbr


def test_canonical_mlb_abbr_full_names():
    """Test team abbreviation resolution from full names."""
    assert canonical_mlb_abbr("New York Yankees") == "NYY"
    assert canonical_mlb_abbr("Los Angeles Dodgers") == "LAD"
    assert canonical_mlb_abbr("Boston Red Sox") == "BOS"
    assert canonical_mlb_abbr("Chicago Cubs") == "CHC"
    assert canonical_mlb_abbr("San Francisco Giants") == "SF"
    assert canonical_mlb_abbr("Oakland Athletics") == "ATH"


def test_canonical_mlb_abbr_already_abbr():
    """Test that abbreviations pass through or are canonicalized."""
    assert canonical_mlb_abbr("NYY") == "NYY"
    assert canonical_mlb_abbr("LAD") == "LAD"
    assert canonical_mlb_abbr("BOS") == "BOS"
    assert canonical_mlb_abbr("OAK") == "ATH"  # OAK canonicalizes to ATH
    assert canonical_mlb_abbr("ATH") == "ATH"


def test_match_game_exact():
    """Test exact game matching by date and teams."""
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 12345,
                "game_date": pd.to_datetime("2026-08-26").date(),
                "home_team": "NYY",
                "away_team": "BOS",
            },
            {
                "game_pk": 12346,
                "game_date": pd.to_datetime("2026-08-26").date(),
                "home_team": "LAD",
                "away_team": "SF",
            },
        ]
    )

    event = {
        "id": "test_event_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": "2026-08-26T19:00:00Z",
    }

    game_pk, home_abbr, away_abbr = odds_fetcher.match_game(event, schedule)
    assert game_pk == 12345
    assert home_abbr == "NYY"
    assert away_abbr == "BOS"


def test_match_game_with_athletics():
    """Test Oakland Athletics matching (ATH vs OAK)."""
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 12345,
                "game_date": pd.to_datetime("2026-08-26").date(),
                "home_team": "ATH",  # Schedule uses canonical ATH
                "away_team": "TEX",
            }
        ]
    )

    # Odds API may use "Oakland Athletics"
    event = {
        "id": "test_event_1",
        "home_team": "Oakland Athletics",
        "away_team": "Texas Rangers",
        "commence_time": "2026-08-26T19:00:00Z",
    }

    game_pk, home_abbr, away_abbr = odds_fetcher.match_game(event, schedule)
    assert game_pk == 12345
    assert home_abbr == "ATH"  # Canonicalized to ATH
    assert away_abbr == "TEX"


def test_match_game_uses_utc_kickoff_across_date_boundary():
    """A Denver slate game can begin on the next UTC date."""
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 12345,
                "game_date": pd.to_datetime("2026-08-27").date(),
                "game_datetime": "2026-08-27T00:30:00Z",
                "home_team": "NYY",
                "away_team": "BOS",
            }
        ]
    )

    # Event time in UTC might map to different date in local TZ
    event = {
        "id": "test_event_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": "2026-08-27T00:30:00Z",
    }

    game_pk, home_abbr, away_abbr = odds_fetcher.match_game(event, schedule)
    assert game_pk == 12345


def test_match_game_no_match():
    """Test no match returns None."""
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 12345,
                "game_date": pd.to_datetime("2026-08-26").date(),
                "home_team": "LAD",
                "away_team": "SF",
            }
        ]
    )

    event = {
        "id": "test_event_1",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": "2026-08-26T19:00:00Z",
    }

    game_pk, home_abbr, away_abbr = odds_fetcher.match_game(event, schedule)
    assert game_pk is None
    assert home_abbr == "NYY"
    assert away_abbr == "BOS"


def test_select_best_bookmaker():
    """Test bookmaker selection prefers draftkings > fanduel > betmgm."""
    bookmakers = [
        {"key": "bovada", "title": "Bovada"},
        {"key": "fanduel", "title": "FanDuel"},
        {"key": "draftkings", "title": "DraftKings"},
    ]

    best = odds_fetcher.select_best_bookmaker(bookmakers)
    assert best["key"] == "draftkings"

    # Test fallback when preferred not present
    bookmakers_no_preferred = [
        {"key": "bovada", "title": "Bovada"},
        {"key": "williamhill", "title": "William Hill"},
    ]

    best = odds_fetcher.select_best_bookmaker(bookmakers_no_preferred)
    assert best["key"] == "bovada"  # First available


def test_extract_moneyline_realistic_payload():
    """Test moneyline extraction with realistic Odds API payload.
    
    In real payloads, bookmaker does NOT have home_team/away_team keys.
    Team names are on the event, not on bookmaker.
    """
    # Realistic bookmaker object (no home_team/away_team)
    bookmaker = {
        "key": "draftkings",
        "title": "DraftKings",
        "last_update": "2026-08-26T13:00:00Z",
        "markets": [
            {
                "key": "h2h",
                "last_update": "2026-08-26T13:00:00Z",
                "outcomes": [
                    {"name": "New York Yankees", "price": -150},
                    {"name": "Boston Red Sox", "price": 130},
                ],
            }
        ],
    }

    # Team names come from the event
    home_team = "New York Yankees"
    away_team = "Boston Red Sox"

    home_price, away_price, book_key = odds_fetcher.extract_moneyline(
        bookmaker, home_team, away_team
    )
    assert home_price == -150
    assert away_price == 130
    assert book_key == "draftkings"


def test_extract_runline_realistic_payload():
    """Test run-line extraction with realistic Odds API payload."""
    # Realistic bookmaker object (no home_team/away_team)
    bookmaker = {
        "key": "fanduel",
        "title": "FanDuel",
        "last_update": "2026-08-26T13:00:00Z",
        "markets": [
            {
                "key": "spreads",
                "last_update": "2026-08-26T13:00:00Z",
                "outcomes": [
                    {"name": "New York Yankees", "point": -1.5, "price": -120},
                    {"name": "Boston Red Sox", "point": 1.5, "price": 100},
                ],
            }
        ],
    }

    # Team names come from the event
    home_team = "New York Yankees"
    away_team = "Boston Red Sox"

    home_line, home_price, away_price, book_key = odds_fetcher.extract_runline(
        bookmaker, home_team, away_team
    )
    assert home_line == -1.5
    assert home_price == -120
    assert away_price == 100
    assert book_key == "fanduel"


def test_extract_totals():
    """Test totals extraction."""
    bookmaker = {
        "key": "betmgm",
        "markets": [
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 8.5, "price": -110},
                    {"name": "Under", "point": 8.5, "price": -110},
                ],
            }
        ],
    }

    total_line, over_price, under_price, book_key = odds_fetcher.extract_totals(bookmaker)
    assert total_line == 8.5
    assert over_price == -110
    assert under_price == -110
    assert book_key == "betmgm"


def test_extract_moneyline_returns_none_when_missing():
    """Test extract_moneyline returns None when team names don't match."""
    bookmaker = {
        "key": "draftkings",
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": "Team A", "price": -150},
                    {"name": "Team B", "price": 130},
                ],
            }
        ],
    }

    # Wrong team names
    home_price, away_price, book_key = odds_fetcher.extract_moneyline(
        bookmaker, "Wrong Home", "Wrong Away"
    )
    assert home_price is None
    assert away_price is None
    assert book_key == "draftkings"  # Still returns book key
