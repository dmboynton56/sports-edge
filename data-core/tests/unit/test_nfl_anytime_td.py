from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.refresh_nfl_anytime_td import (
    _should_refresh_game,
    american_implied_probability,
    build_future_player_rows,
    normalize_player_name,
)
from src.models.nfl_anytime_td import build_feature_frame


def test_feature_frame_lags_touchdown_outcomes_before_future_game():
    stats = pd.DataFrame(
        [
            {"game_id": "g1", "season": 2025, "week": 1, "player_id": "p1", "player_display_name": "Runner", "position": "RB", "team": "A", "opponent_team": "B", "carries": 10, "targets": 2, "rushing_tds": 1, "receiving_tds": 0},
            {"game_id": "g2", "season": 2025, "week": 2, "player_id": "p1", "player_display_name": "Runner", "position": "RB", "team": "A", "opponent_team": "B", "carries": 8, "targets": 1, "rushing_tds": 0, "receiving_tds": 0},
            {"game_id": "g3", "season": 2026, "week": 1, "player_id": "p1", "player_display_name": "Runner", "position": "RB", "team": "A", "opponent_team": "B", "is_future": True},
        ]
    )
    schedule = pd.DataFrame(
        [
            {"game_id": "g1", "game_date": date(2025, 9, 1), "home_team": "A", "away_team": "B", "total_line": 44},
            {"game_id": "g2", "game_date": date(2025, 9, 8), "home_team": "B", "away_team": "A", "total_line": 45},
            {"game_id": "g3", "game_date": date(2026, 9, 10), "home_team": "A", "away_team": "B", "total_line": 46},
        ]
    )

    features = build_feature_frame(stats, schedule)
    future = features[features["game_id"].eq("g3")].iloc[0]

    assert future["has_td"] != future["has_td"]  # NaN target is never filled for inference.
    assert future["rolling_td_rate_3"] == pytest.approx(0.5)
    assert future["rolling_opportunities_3"] == pytest.approx(10.5)
    assert future["career_games_before"] == 2


def test_name_fallback_adds_depth_role_flags():
    rows, flags = build_future_player_rows(
        [{"game_id": "g", "game_date": date(2026, 9, 10), "home_team": "LA", "away_team": "SF"}],
        {"00-1": {"full_name": "Depth Runner Jr.", "position": "RB", "team": "LAR", "status": "ACT"}},
        {"s1": {"full_name": "Depth Runner Jr.", "position": "RB", "team": "LAR", "depth_chart_order": 4}},
        season=2026,
        week=1,
    )

    assert rows.iloc[0]["team"] == "LA"
    assert flags["00-1"] == ["secondary_depth_role", "deep_depth_chart"]


def test_odds_helpers_and_cache_window():
    assert normalize_player_name("Deebo Samuel Sr.") == "deebo samuel"
    assert american_implied_probability(200) == pytest.approx(1 / 3)
    assert american_implied_probability(-150) == pytest.approx(0.6)

    now = datetime(2026, 9, 3, tzinfo=timezone.utc)
    far_game = {
        "game_time_utc": now + timedelta(days=6),
        "td_odds_ts": now - timedelta(days=1),
    }
    near_game = {
        "game_time_utc": now + timedelta(days=2),
        "td_odds_ts": now - timedelta(days=1),
    }
    assert not _should_refresh_game(far_game, now=now, near_kickoff_hours=72, minimum_refresh_hours=6)
    assert _should_refresh_game(near_game, now=now, near_kickoff_hours=72, minimum_refresh_hours=6)
