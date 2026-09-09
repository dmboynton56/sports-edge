"""Tests for once-per-day Odds API credit conservation in MLB HR odds fetcher."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from fetch_mlb_home_run_odds import _odds_already_used_today, _should_skip_odds_api


def test_force_odds_api_only_bypasses_the_once_daily_guard_when_explicit():
    assert _should_skip_odds_api(True, False) is True
    assert _should_skip_odds_api(True, True) is False
    assert _should_skip_odds_api(False, False) is False


def test_odds_already_used_today_delegates_to_shared_budget():
    """HR and research share odds_api_usage; the script wrapper must not drift."""
    with patch("fetch_mlb_home_run_odds.odds_already_used_today", return_value=(True, "mlb_research")) as mock_used:
        already_used, source = _odds_already_used_today("2026-09-09")

    assert already_used is True
    assert source == "mlb_research"
    mock_used.assert_called_once_with("2026-09-09")


def test_odds_already_used_today_returns_false_when_no_odds_today():
    with patch("fetch_mlb_home_run_odds.odds_already_used_today", return_value=(False, None)):
        already_used, provider = _odds_already_used_today("2026-09-02")

    assert already_used is False
    assert provider is None


@patch("fetch_mlb_home_run_odds._odds_already_used_today")
@patch("fetch_mlb_home_run_odds.fetch_mlb_schedule")
@patch("fetch_mlb_home_run_odds.get_api_key")
@patch("fetch_mlb_home_run_odds.get_propline_api_key")
@patch("fetch_mlb_home_run_odds.fetch_day_hr_odds_propline")
@patch("fetch_mlb_home_run_odds._write_outputs")
@patch("fetch_mlb_home_run_odds._sync_supabase")
def test_main_skips_odds_api_when_research_already_used_today(
    mock_sync,
    mock_write,
    mock_propline_fetch,
    mock_propline_key,
    mock_odds_key,
    mock_schedule,
    mock_already_used,
):
    """If research claimed the shared Denver-day budget, HR must not call Odds."""
    mock_already_used.return_value = (True, "mlb_research")
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

    from fetch_mlb_home_run_odds import main

    with patch("sys.argv", ["fetch_mlb_home_run_odds.py", "--date", "2026-09-02"]):
        main()

    mock_propline_fetch.assert_called_once()
    audit_arg = mock_write.call_args[0][1]
    assert audit_arg.get("oddsAlreadyUsedToday") is True
    assert audit_arg.get("priorProvider") == "mlb_research"
    assert "shared daily budget" in audit_arg.get("fallbackReason", "")


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


@patch("fetch_mlb_home_run_odds.claim_odds_api_usage")
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
    mock_claim,
):
    """Test that main() uses Odds API when not yet used today."""
    mock_already_used.return_value = (False, None)
    mock_claim.return_value = True
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
    
    mock_odds_fetch.assert_called_once()
    mock_claim.assert_called_once()
    assert mock_claim.call_args.args[1] == "mlb_hr"

    # Verify audit shows Odds API was used
    audit_arg = mock_write.call_args[0][1]
    assert audit_arg["provider"] == "the_odds_api"
    assert audit_arg.get("oddsAlreadyUsedToday") is not True
