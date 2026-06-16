from datetime import date
from types import SimpleNamespace

import pandas as pd

from scripts.build_feature_snapshots import _filter_schedules_for_window, _resolve_date_window


def test_resolve_date_window_from_explicit_dates():
    args = SimpleNamespace(
        start_date=date(2026, 9, 10),
        end_date=date(2026, 9, 20),
        lookback_days=None,
        lookahead_days=None,
        date=None,
    )

    assert _resolve_date_window(args) == (date(2026, 9, 10), date(2026, 9, 20))


def test_resolve_date_window_from_anchor_and_offsets():
    args = SimpleNamespace(
        start_date=None,
        end_date=None,
        lookback_days=1,
        lookahead_days=3,
        date=date(2026, 6, 8),
    )

    assert _resolve_date_window(args) == (date(2026, 6, 7), date(2026, 6, 11))


def test_filter_schedules_for_window_keeps_only_target_games():
    schedules = pd.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "game_date": pd.to_datetime(["2026-06-07", "2026-06-08", "2026-06-12"]),
        }
    )

    filtered = _filter_schedules_for_window(schedules, (date(2026, 6, 8), date(2026, 6, 11)))

    assert filtered["game_id"].tolist() == ["B"]
