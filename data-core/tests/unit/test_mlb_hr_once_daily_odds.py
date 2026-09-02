"""Tests for once-per-day Odds API credit conservation in MLB HR odds fetcher."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import the function we're testing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fetch_mlb_home_run_odds import _odds_already_used_today


def test_odds_already_used_today_returns_true_when_the_odds_api_used():
    """Test that _odds_already_used_today returns True when the_odds_api provider was used today."""
    denver_date = "2026-09-02"
    
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = ("the_odds_api",)
    
    with patch("fetch_mlb_home_run_odds.load_supabase_credentials") as mock_creds:
        mock_creds.return_value = {
            "url": "http://test",
            "db_password": "test",
            "db_host": None,
            "db_port": 5432,
            "db_name": "postgres",
            "db_user": "postgres",
        }
        with patch("fetch_mlb_home_run_odds.create_pg_connection", return_value=mock_conn):
            already_used, provider = _odds_already_used_today(denver_date)
    
    assert already_used is True
    assert provider == "the_odds_api"
    mock_cursor.execute.assert_called_once()
    sql = mock_cursor.execute.call_args[0][0]
    assert "mlb_home_run_odds_snapshots" in sql
    assert "provider = 'the_odds_api'" in sql


def test_odds_already_used_today_returns_false_when_no_odds_today():
    """Test that _odds_already_used_today returns False when no Odds API calls were made today."""
    denver_date = "2026-09-02"
    
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = None
    
    with patch("fetch_mlb_home_run_odds.load_supabase_credentials") as mock_creds:
        mock_creds.return_value = {
            "url": "http://test",
            "db_password": "test",
            "db_host": None,
            "db_port": 5432,
            "db_name": "postgres",
            "db_user": "postgres",
        }
        with patch("fetch_mlb_home_run_odds.create_pg_connection", return_value=mock_conn):
            already_used, provider = _odds_already_used_today(denver_date)
    
    assert already_used is False
    assert provider is None


def test_odds_already_used_today_handles_missing_credentials():
    """Test that _odds_already_used_today handles missing credentials gracefully."""
    denver_date = "2026-09-02"
    
    with patch("fetch_mlb_home_run_odds.load_supabase_credentials") as mock_creds:
        mock_creds.return_value = {"url": None, "db_password": None}
        already_used, provider = _odds_already_used_today(denver_date)
    
    assert already_used is False
    assert provider is None


def test_odds_already_used_today_handles_db_exception():
    """Test that _odds_already_used_today handles database exceptions gracefully."""
    denver_date = "2026-09-02"
    
    with patch("fetch_mlb_home_run_odds.load_supabase_credentials") as mock_creds:
        mock_creds.return_value = {"url": "http://test", "db_password": "test"}
        with patch("fetch_mlb_home_run_odds.create_pg_connection", side_effect=Exception("DB error")):
            already_used, provider = _odds_already_used_today(denver_date)
    
    assert already_used is False
    assert provider is None


@patch("fetch_mlb_home_run_odds._odds_already_used_today")
@patch("fetch_mlb_home_run_odds.fetch_mlb_schedule")
@patch("fetch_mlb_home_run_odds.get_api_key")
@patch("fetch_mlb_home_run_odds.get_propline_api_key")
@patch("fetch_mlb_home_run_odds.fetch_day_hr_odds_propline")
@patch("fetch_mlb_home_run_odds._write_outputs")
@patch("fetch_mlb_home_run_odds._sync_supabase")
def test_main_skips_odds_api_when_already_used_today(
    mock_sync,
    mock_write,
    mock_propline_fetch,
    mock_propline_key,
    mock_odds_key,
    mock_schedule,
    mock_already_used,
):
    """Test that main() skips Odds API and uses PropLine when already used today."""
    mock_already_used.return_value = (True, "the_odds_api")
    mock_odds_key.return_value = "test-odds-key"
    mock_propline_key.return_value = "test-propline-key"
    mock_schedule.return_value = pd.DataFrame(
        {
            "game_pk": [123456],
            "game_date": ["2026-09-02"],
            "game_datetime": ["2026-09-02T19:05:00Z"],
            "home_team": ["Los Angeles Dodgers"],
            "away_team": ["New York Yankees"],
            "home_team_abbr": ["LAD"],
            "away_team_abbr": ["NYY"],
        }
    )
    mock_propline_fetch.return_value = (
        pd.DataFrame({"provider": ["propline"], "player_name": ["Test Player"]}),
        {
            "provider": "propline",
            "oddsRows": 1,
            "eventsReturned": 1,
            "eventsMatched": 1,
        },
    )
    mock_sync.return_value = 1
    
    # Import and run main
    from fetch_mlb_home_run_odds import main
    
    with patch("sys.argv", ["fetch_mlb_home_run_odds.py", "--date", "2026-09-02"]):
        main()
    
    # Verify PropLine was called (not Odds API)
    mock_propline_fetch.assert_called_once()
    
    # Verify audit contains the skip reason
    audit_arg = mock_write.call_args[0][1]
    assert audit_arg.get("oddsAlreadyUsedToday") is True
    assert audit_arg.get("priorProvider") == "the_odds_api"
    assert "fallbackReason" in audit_arg


@patch("fetch_mlb_home_run_odds._odds_already_used_today")
@patch("fetch_mlb_home_run_odds.fetch_mlb_schedule")
@patch("fetch_mlb_home_run_odds.get_api_key")
@patch("fetch_mlb_home_run_odds.get_propline_api_key")
@patch("fetch_mlb_home_run_odds.fetch_day_hr_odds")
@patch("fetch_mlb_home_run_odds._write_outputs")
@patch("fetch_mlb_home_run_odds._sync_supabase")
def test_main_uses_odds_api_when_not_yet_used_today(
    mock_sync,
    mock_write,
    mock_odds_fetch,
    mock_propline_key,
    mock_odds_key,
    mock_schedule,
    mock_already_used,
):
    """Test that main() uses Odds API when not yet used today."""
    mock_already_used.return_value = (False, None)
    mock_odds_key.return_value = "test-odds-key"
    mock_propline_key.return_value = "test-propline-key"
    mock_schedule.return_value = pd.DataFrame(
        {
            "game_pk": [123456],
            "game_date": ["2026-09-02"],
            "game_datetime": ["2026-09-02T19:05:00Z"],
            "home_team": ["Los Angeles Dodgers"],
            "away_team": ["New York Yankees"],
            "home_team_abbr": ["LAD"],
            "away_team_abbr": ["NYY"],
        }
    )
    mock_odds_fetch.return_value = (
        pd.DataFrame({"provider": ["the_odds_api"], "player_name": ["Test Player"]}),
        {
            "provider": "the_odds_api",
            "oddsRows": 1,
            "eventsReturned": 1,
            "eventsMatched": 1,
            "apiCreditsRemaining": "499",
        },
    )
    mock_sync.return_value = 1
    
    # Import and run main
    from fetch_mlb_home_run_odds import main
    
    with patch("sys.argv", ["fetch_mlb_home_run_odds.py", "--date", "2026-09-02"]):
        main()
    
    # Verify Odds API was called (not PropLine)
    mock_odds_fetch.assert_called_once()
    
    # Verify audit shows Odds API was used
    audit_arg = mock_write.call_args[0][1]
    assert audit_arg["provider"] == "the_odds_api"
    assert audit_arg.get("oddsAlreadyUsedToday") is not True
