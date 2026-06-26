from __future__ import annotations

from datetime import date

import pandas as pd

from src.models.mlb_hr_recency import NO_HR_IN_HISTORY_FLAG, games_since_last_hr, recency_quality_flags


def _batting(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_homered_last_game_is_zero_games_since() -> None:
    batting = _batting(
        [
            {"player_id": 1, "game_pk": 100, "game_date": "2026-06-24", "plate_appearances": 4, "home_runs": 0},
            {"player_id": 1, "game_pk": 101, "game_date": "2026-06-25", "plate_appearances": 4, "home_runs": 1},
        ]
    )
    out = games_since_last_hr(batting, date(2026, 6, 26))
    row = out.loc[out["player_id"] == 1].iloc[0]
    assert row["games_since_last_hr"] == 0
    assert row["last_hr_date"] == "2026-06-25"


def test_counts_games_after_last_hr() -> None:
    batting = _batting(
        [
            {"player_id": 2, "game_pk": 200, "game_date": "2026-06-20", "plate_appearances": 4, "home_runs": 1},
            {"player_id": 2, "game_pk": 201, "game_date": "2026-06-22", "plate_appearances": 4, "home_runs": 0},
            {"player_id": 2, "game_pk": 202, "game_date": "2026-06-24", "plate_appearances": 4, "home_runs": 0},
        ]
    )
    out = games_since_last_hr(batting, date(2026, 6, 26))
    row = out.loc[out["player_id"] == 2].iloc[0]
    assert row["games_since_last_hr"] == 2
    assert row["last_hr_date"] == "2026-06-20"


def test_doubleheader_same_day_counts_second_game() -> None:
    batting = _batting(
        [
            {"player_id": 3, "game_pk": 300, "game_date": "2026-06-25", "plate_appearances": 4, "home_runs": 1},
            {"player_id": 3, "game_pk": 301, "game_date": "2026-06-25", "plate_appearances": 4, "home_runs": 0},
        ]
    )
    out = games_since_last_hr(batting, date(2026, 6, 26))
    row = out.loc[out["player_id"] == 3].iloc[0]
    assert row["games_since_last_hr"] == 1
    assert row["last_hr_date"] == "2026-06-25"


def test_no_hr_in_history_window_is_null() -> None:
    batting = _batting(
        [
            {"player_id": 4, "game_pk": 400, "game_date": "2026-06-24", "plate_appearances": 4, "home_runs": 0},
        ]
    )
    out = games_since_last_hr(batting, date(2026, 6, 26))
    row = out.loc[out["player_id"] == 4].iloc[0]
    assert pd.isna(row["games_since_last_hr"])
    assert row["last_hr_date"] is None
    assert recency_quality_flags(row["games_since_last_hr"]) == [NO_HR_IN_HISTORY_FLAG]
