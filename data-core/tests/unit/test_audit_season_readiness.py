"""Unit tests for audit_season_readiness helpers."""

from datetime import date, timedelta

from scripts.audit_season_readiness import _date_window


def test_date_window():
    start, end = _date_window(date(2026, 9, 9), 7)
    assert start == date(2026, 9, 9)
    assert end == date(2026, 9, 16)
