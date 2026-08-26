"""Tests for PropLine fallback in MLB HR odds fetcher."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.mlb_hr_odds_fetcher import (
    MlbHrOddsError,
    fetch_day_hr_odds,
    fetch_day_hr_odds_propline,
    match_events_to_schedule,
    normalize_name,
    normalize_team,
)
from src.data.propline_client import PropLineClient, PropLineError


def test_normalize_name():
    """Test player name normalization removes suffixes and special chars."""
    assert normalize_name("José Ramírez Jr.") == "jose ramirez"
    assert normalize_name("Fernando Tatis III") == "fernando tatis"
    assert normalize_name("Mike Trout") == "mike trout"


def test_normalize_team():
    """Test team name normalization for matching."""
    assert normalize_team("New York Yankees") == "newyorkyankees"
    assert normalize_team("Los Angeles Dodgers") == "losangelesdodgers"
    assert normalize_team("St. Louis Cardinals") == "stlouiscardinals"


def test_match_events_to_schedule_by_teams_and_time():
    """Test event matching joins by team names and commence_time."""
    schedule = pd.DataFrame(
        {
            "game_pk": [123456],
            "game_date": ["2026-08-26"],
            "game_datetime": ["2026-08-26T19:05:00Z"],
            "home_team": ["LAD"],
            "away_team": ["NYY"],
            "home_team_abbr": ["LAD"],
            "away_team_abbr": ["NYY"],
        }
    )
    events = [
        {
            "id": "prop123",
            "home_team": "Los Angeles Dodgers",
            "away_team": "New York Yankees",
            "commence_time": "2026-08-26T19:05:00Z",
        }
    ]
    matched = match_events_to_schedule(events, schedule)
    assert "prop123" in matched
    assert matched["prop123"]["game_pk"] == 123456


def test_match_events_no_schedule_match():
    """Test unmatched events return empty when schedule has no matching game."""
    schedule = pd.DataFrame(
        {
            "game_pk": [999999],
            "game_date": ["2026-08-27"],
            "game_datetime": ["2026-08-27T19:05:00Z"],
            "home_team": ["SF"],
            "away_team": ["SD"],
            "home_team_abbr": ["SF"],
            "away_team_abbr": ["SD"],
        }
    )
    events = [
        {
            "id": "prop456",
            "home_team": "Los Angeles Dodgers",
            "away_team": "New York Yankees",
            "commence_time": "2026-08-26T19:05:00Z",
        }
    ]
    matched = match_events_to_schedule(events, schedule)
    assert "prop456" not in matched


@patch("src.data.mlb_hr_odds_fetcher.fetch_mlb_events")
@patch("src.data.mlb_hr_odds_fetcher.fetch_event_hr_odds")
def test_fetch_day_hr_odds_sets_provider_field(mock_fetch_event, mock_fetch_events):
    """Test that fetch_day_hr_odds sets provider='the_odds_api' in audit and rows."""
    mock_fetch_events.return_value = [
        {
            "id": "odds123",
            "home_team": "Los Angeles Dodgers",
            "away_team": "New York Yankees",
            "commence_time": "2026-08-26T19:05:00Z",
        }
    ]
    mock_fetch_event.return_value = {
        "id": "odds123",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "outcomes": [
                            {
                                "description": "Aaron Judge",
                                "name": "Over",
                                "price": 150,
                                "point": 0.5,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    schedule = pd.DataFrame(
        {
            "game_pk": [123456],
            "game_date": ["2026-08-26"],
            "game_datetime": ["2026-08-26T19:05:00Z"],
            "home_team": ["LAD"],
            "away_team": ["NYY"],
            "home_team_abbr": ["LAD"],
            "away_team_abbr": ["NYY"],
        }
    )
    client = MagicMock()
    client.request_count = 2
    client.response_meta.requests_remaining = "498"
    client.response_meta.requests_used = "2"
    client.response_meta.requests_last = "1"

    odds, audit = fetch_day_hr_odds(client, game_date=date(2026, 8, 26), schedule=schedule)

    assert audit["provider"] == "the_odds_api"
    assert not odds.empty
    assert odds.iloc[0]["provider"] == "the_odds_api"


@patch("src.data.propline_client.fetch_propline_mlb_events")
@patch("src.data.propline_client.fetch_propline_event_odds")
def test_fetch_day_hr_odds_propline_sets_provider_field(mock_fetch_event, mock_fetch_events):
    """Test that fetch_day_hr_odds_propline sets provider='propline' in audit and rows."""
    mock_fetch_events.return_value = [
        {
            "id": "prop789",
            "home_team": "Los Angeles Dodgers",
            "away_team": "New York Yankees",
            "commence_time": "2026-08-26T19:05:00Z",
        }
    ]
    mock_fetch_event.return_value = {
        "id": "prop789",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "outcomes": [
                            {
                                "description": "Shohei Ohtani",
                                "name": "Over",
                                "price": 200,
                                "point": 0.5,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    schedule = pd.DataFrame(
        {
            "game_pk": [654321],
            "game_date": ["2026-08-26"],
            "game_datetime": ["2026-08-26T19:05:00Z"],
            "home_team": ["LAD"],
            "away_team": ["NYY"],
            "home_team_abbr": ["LAD"],
            "away_team_abbr": ["NYY"],
        }
    )
    client = MagicMock()
    client.request_count = 2

    odds, audit = fetch_day_hr_odds_propline(client, game_date=date(2026, 8, 26), schedule=schedule)

    assert audit["provider"] == "propline"
    assert not odds.empty
    assert odds.iloc[0]["provider"] == "propline"


def test_propline_client_uses_header_auth():
    """Test PropLine client passes API key via X-API-Key header, not query param."""
    with patch("src.data.propline_client.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "[]"
        mock_response.json.return_value = []
        mock_response.headers = {}
        mock_get.return_value = mock_response

        client = PropLineClient(api_key="test-key-123", min_request_interval_sec=0)
        client.get("/sports/baseball_mlb/events")

        # Check that header was set, not query param
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["X-API-Key"] == "test-key-123"
        assert "apiKey" not in (call_kwargs.get("params") or {})
