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


def _book(key: str, title: str, markets: list[dict]) -> dict:
    return {"key": key, "title": title, "markets": markets}


def test_first_paired_moneyline_falls_back_when_preferred_book_dropped_h2h():
    """Evening slates often keep DK totals after that book has already dropped ML."""
    bookmakers = [
        _book(
            "draftkings",
            "DraftKings",
            [
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 8.5, "price": -110},
                        {"name": "Under", "point": 8.5, "price": -110},
                    ],
                }
            ],
        ),
        _book(
            "fanduel",
            "FanDuel",
            [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "New York Yankees", "price": -145},
                        {"name": "Boston Red Sox", "price": 125},
                    ],
                }
            ],
        ),
    ]

    home_price, away_price, book_key = odds_fetcher.first_paired_moneyline(
        bookmakers, "New York Yankees", "Boston Red Sox"
    )
    assert home_price == -145
    assert away_price == 125
    assert book_key == "fanduel"


def test_first_paired_runline_requires_both_sides():
    bookmakers = [
        _book(
            "draftkings",
            "DraftKings",
            [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "New York Yankees", "point": -1.5, "price": -115},
                    ],
                }
            ],
        ),
        _book(
            "betmgm",
            "BetMGM",
            [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "New York Yankees", "point": -1.5, "price": -120},
                        {"name": "Boston Red Sox", "point": 1.5, "price": 100},
                    ],
                }
            ],
        ),
    ]

    home_line, home_price, away_price, book_key = odds_fetcher.first_paired_runline(
        bookmakers, "New York Yankees", "Boston Red Sox"
    )
    assert home_line == -1.5
    assert home_price == -120
    assert away_price == 100
    assert book_key == "betmgm"


def test_first_paired_moneyline_returns_none_when_no_book_has_pair():
    bookmakers = [
        _book(
            "draftkings",
            "DraftKings",
            [
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 8.5, "price": -110},
                        {"name": "Under", "point": 8.5, "price": -110},
                    ],
                }
            ],
        )
    ]

    home_price, away_price, book_key = odds_fetcher.first_paired_moneyline(
        bookmakers, "New York Yankees", "Boston Red Sox"
    )
    assert home_price is None
    assert away_price is None
    assert book_key is None


def test_extract_totals_skips_team_and_period_markets():
    """PropLine can mix game totals with team totals and F5 slices on the same key."""
    bookmaker = {
        "key": "draftkings",
        "markets": [
            {
                "key": "totals",
                "team": "New York Yankees",
                "outcomes": [
                    {"name": "Over", "point": 4.5, "price": -110},
                    {"name": "Under", "point": 4.5, "price": -110},
                ],
            },
            {
                "key": "totals",
                "period": "f5",
                "outcomes": [
                    {"name": "Over", "point": 5.5, "price": -105},
                    {"name": "Under", "point": 5.5, "price": -105},
                ],
            },
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 8.5, "price": -115},
                    {"name": "Under", "point": 8.5, "price": -105},
                ],
            },
        ],
    }

    total_line, over_price, under_price, book_key = odds_fetcher.extract_totals(bookmaker)
    assert total_line == 8.5
    assert over_price == -115
    assert under_price == -105
    assert book_key == "draftkings"


def test_research_uses_propline_even_when_shared_budget_already_spent():
    """PropLine-first must still succeed after HR has claimed the Odds budget."""
    called = {"odds": 0}
    events, provider, reason = odds_fetcher.fetch_mlb_game_odds_events(
        markets=["h2h"],
        odds_api_key="odds-key",
        propline_api_key="pl-key",
        already_used_today=True,
        fetch_odds_api=lambda _key, _markets: called.__setitem__("odds", 1) or [{"id": "odds"}],
        fetch_propline=lambda _markets: [{"id": "pl1"}],
    )
    assert events == [{"id": "pl1"}]
    assert provider == "propline"
    assert called["odds"] == 0
    assert "PropLine-first" in reason


def test_research_prefers_propline_and_does_not_call_odds():
    """Research is PropLine-first so morning Daily cannot burn Odds credits."""
    called = {"odds": 0, "claim": 0}

    def odds_api(_key, _markets):
        called["odds"] += 1
        return [{"id": "odds"}]

    propline_events = [{"id": "pl1", "home_team": "New York Yankees"}]
    events, provider, reason = odds_fetcher.fetch_mlb_game_odds_events(
        markets=["h2h", "spreads", "totals"],
        odds_api_key="odds-key",
        propline_api_key="pl-key",
        fetch_odds_api=odds_api,
        fetch_propline=lambda _markets: propline_events,
        claim_odds_usage=lambda: called.__setitem__("claim", called["claim"] + 1) or True,
    )

    assert events == propline_events
    assert provider == "propline"
    assert called["odds"] == 0
    assert called["claim"] == 0
    assert "PropLine-first" in reason


def test_research_uses_odds_when_propline_empty_and_budget_free():
    called = {"odds": 0, "claim": 0}

    def odds_api(_key, _markets):
        called["odds"] += 1
        return [{"id": "odds"}]

    events, provider, reason = odds_fetcher.fetch_mlb_game_odds_events(
        markets=["h2h"],
        odds_api_key="odds-key",
        propline_api_key="pl-key",
        already_used_today=False,
        fetch_odds_api=odds_api,
        fetch_propline=lambda _markets: [],
        claim_odds_usage=lambda: called.__setitem__("claim", called["claim"] + 1) or True,
    )

    assert events == [{"id": "odds"}]
    assert provider == "the_odds_api"
    assert called["odds"] == 1
    assert called["claim"] == 1
    assert "0 MLB game-line events" in reason


def test_research_skips_odds_when_propline_empty_and_shared_budget_used():
    called = {"odds": 0}

    def odds_api(_key, _markets):
        called["odds"] += 1
        return [{"id": "odds"}]

    with pytest.raises(RuntimeError, match="shared daily budget"):
        odds_fetcher.fetch_mlb_game_odds_events(
            markets=["h2h"],
            odds_api_key="odds-key",
            propline_api_key="pl-key",
            already_used_today=True,
            force_odds_api=False,
            fetch_odds_api=odds_api,
            fetch_propline=lambda _markets: [],
        )

    assert called["odds"] == 0


def test_research_skips_odds_when_propline_fails_and_hr_already_used_today():
    """If HR already stamped the shared budget, research must not call Odds."""
    called = {"odds": 0}

    def odds_api(_key, _markets):
        called["odds"] += 1
        return [{"id": "odds"}]

    with pytest.raises(RuntimeError, match="already used today"):
        odds_fetcher.fetch_mlb_game_odds_events(
            markets=["h2h"],
            odds_api_key="odds-key",
            propline_api_key="pl-key",
            already_used_today=True,
            fetch_odds_api=odds_api,
            fetch_propline=lambda _markets: (_ for _ in ()).throw(RuntimeError("PropLine down")),
        )

    assert called["odds"] == 0


def test_force_odds_api_bypasses_shared_budget_after_propline_miss():
    events, provider, reason = odds_fetcher.fetch_mlb_game_odds_events(
        markets=["h2h"],
        odds_api_key="odds-key",
        propline_api_key="pl-key",
        already_used_today=True,
        force_odds_api=True,
        fetch_odds_api=lambda _key, _markets: [{"id": "odds"}],
        fetch_propline=lambda _markets: [],
        claim_odds_usage=lambda: False,
    )
    assert events == [{"id": "odds"}]
    assert provider == "the_odds_api"
    assert "0 MLB game-line events" in reason


def test_lost_shared_budget_claim_skips_odds():
    called = {"odds": 0}

    def odds_api(_key, _markets):
        called["odds"] += 1
        return [{"id": "odds"}]

    with pytest.raises(RuntimeError, match="shared daily budget"):
        odds_fetcher.fetch_mlb_game_odds_events(
            markets=["h2h"],
            odds_api_key="odds-key",
            propline_api_key=None,
            already_used_today=False,
            fetch_odds_api=odds_api,
            claim_odds_usage=lambda: False,
        )

    assert called["odds"] == 0


def test_non_quota_odds_error_does_not_fallback():
    def boom(_key, _markets):
        raise RuntimeError("Odds API error 500: upstream exploded")

    with pytest.raises(RuntimeError, match="500"):
        odds_fetcher.fetch_mlb_game_odds_events(
            markets=["h2h"],
            odds_api_key="odds-key",
            propline_api_key=None,
            fetch_odds_api=boom,
        )


def test_quota_without_propline_fails_closed():
    def boom(_key, _markets):
        raise RuntimeError("Odds API error 401: OUT_OF_USAGE_CREDITS")

    with pytest.raises(RuntimeError, match="ODDS_API_KEY is not set|OUT_OF_USAGE_CREDITS"):
        odds_fetcher.fetch_mlb_game_odds_events(
            markets=["h2h"],
            odds_api_key="odds-key",
            propline_api_key=None,
            fetch_odds_api=boom,
        )


def test_both_providers_empty_fails_closed():
    with pytest.raises(RuntimeError, match="returned 0 events"):
        odds_fetcher.fetch_mlb_game_odds_events(
            markets=["h2h"],
            odds_api_key="odds-key",
            propline_api_key="pl-key",
            fetch_odds_api=lambda _key, _markets: [],
            fetch_propline=lambda _markets: [],
            claim_odds_usage=lambda: True,
        )


def test_fetch_propline_mlb_game_odds_uses_bulk_game_line_endpoint():
    from unittest.mock import MagicMock

    from src.data.propline_client import fetch_propline_mlb_game_odds

    client = MagicMock()
    client.get.return_value = [{"id": "e1"}]
    events = fetch_propline_mlb_game_odds(client, markets=["h2h", "spreads", "totals"])
    assert events == [{"id": "e1"}]
    client.get.assert_called_once_with(
        "/sports/baseball_mlb/odds",
        {"markets": "h2h,spreads,totals"},
    )
