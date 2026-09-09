"""Tests for the shared MLB Odds API daily budget."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data.odds_api_budget import (
    SKIP_REASON_PROPLINE_FIRST,
    SKIP_REASON_SHARED_BUDGET,
    SOURCE_HR,
    SOURCE_RESEARCH,
    claim_odds_api_usage,
    odds_already_used_today,
    should_skip_odds_api,
)


def _conn_with_fetchone(value):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = value
    return mock_conn, mock_cursor


def test_should_skip_odds_api_respects_force_flag():
    assert should_skip_odds_api(True, False) is True
    assert should_skip_odds_api(True, True) is False
    assert should_skip_odds_api(False, False) is False


def test_odds_already_used_today_reads_shared_usage_table():
    mock_conn, mock_cursor = _conn_with_fetchone((SOURCE_HR,))

    already_used, source = odds_already_used_today("2026-09-09", conn=mock_conn)

    assert already_used is True
    assert source == SOURCE_HR
    sql = mock_cursor.execute.call_args[0][0]
    assert "odds_api_usage" in sql
    assert "denver_date" in sql


def test_odds_already_used_today_falls_back_to_hr_snapshots():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.side_effect = [None, ("the_odds_api",), ("2026-09-09",)]

    already_used, source = odds_already_used_today("2026-09-09", conn=mock_conn)

    assert already_used is True
    assert source == SOURCE_HR
    sqls = [call.args[0] for call in mock_cursor.execute.call_args_list]
    assert any("odds_api_usage" in sql for sql in sqls)
    assert any("mlb_home_run_odds_snapshots" in sql for sql in sqls)


def test_odds_already_used_today_falls_back_to_research_snapshots():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.side_effect = [None, None, (1,), ("2026-09-09",)]

    already_used, source = odds_already_used_today("2026-09-09", conn=mock_conn)

    assert already_used is True
    assert source == SOURCE_RESEARCH
    sqls = [call.args[0] for call in mock_cursor.execute.call_args_list]
    assert any("odds_snapshots" in sql for sql in sqls)
    assert any("the_odds_api" in sql for sql in sqls)


def test_odds_already_used_today_returns_false_when_unused():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = None

    already_used, source = odds_already_used_today("2026-09-09", conn=mock_conn)

    assert already_used is False
    assert source is None


def test_odds_already_used_today_handles_missing_credentials():
    with patch("src.data.odds_api_budget.load_supabase_credentials") as mock_creds:
        mock_creds.return_value = {"url": None, "db_password": None}
        already_used, source = odds_already_used_today("2026-09-09")

    assert already_used is False
    assert source is None


def test_claim_odds_api_usage_inserts_and_returns_true():
    mock_conn, mock_cursor = _conn_with_fetchone(("2026-09-09",))

    claimed = claim_odds_api_usage("2026-09-09", SOURCE_RESEARCH, notes="test", conn=mock_conn)

    assert claimed is True
    sql = mock_cursor.execute.call_args[0][0]
    assert "insert into odds_api_usage" in sql.lower()
    assert "on conflict (denver_date) do nothing" in sql.lower()


def test_claim_odds_api_usage_returns_false_when_row_exists():
    mock_conn, mock_cursor = _conn_with_fetchone(None)

    claimed = claim_odds_api_usage("2026-09-09", SOURCE_HR, conn=mock_conn)

    assert claimed is False


def test_claim_rejects_invalid_source():
    with pytest.raises(ValueError, match="invalid odds_api_usage source"):
        claim_odds_api_usage("2026-09-09", "nba", conn=MagicMock())


def test_skip_reason_constants_are_explicit():
    assert "shared daily budget" in SKIP_REASON_SHARED_BUDGET
    assert "PropLine-first" in SKIP_REASON_PROPLINE_FIRST
