from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.mlb_totals_model import over_label, train_and_evaluate_mlb_totals
from src.models.mlb_winner_model import default_feature_columns


@pytest.fixture(scope="module")
def synthetic_result() -> dict:
    rows = []
    game_pk = 1
    for season in range(2021, 2027):
        for game_number in range(8):
            total = 6 + ((game_number + season) % 7)
            game_date = pd.Timestamp(season, 5, game_number + 1)
            rows.append(
                {
                    "game_pk": game_pk,
                    "season": season,
                    "game_date": game_date,
                    "game_datetime": game_date + pd.Timedelta(hours=18),
                    "home_team": f"H{game_number}",
                    "away_team": f"A{game_number}",
                    "home_score": total // 2,
                    "away_score": total - total // 2,
                    "total_runs": total,
                    "signal": float(total) + (season - 2021) * 0.01,
                    "venue_total_runs_per_game": 8.0 + game_number * 0.2,
                    "temp_f": 55.0 + total * 2.0,
                    "wind_mph": 5.0,
                    "wind_out": float(game_number % 3),
                    "wind_in": float((game_number + 1) % 2),
                    "wind_cross": 1.0,
                    "is_dome_or_closed": 0.0,
                    "is_day_game": 0.0,
                    "elevation": 500.0,
                    "venue_prior_games": float(game_number + 10),
                    "venue_home_win_pct": 0.52,
                }
            )
            game_pk += 1
    return train_and_evaluate_mlb_totals(pd.DataFrame(rows), validation_season=2025, test_season=2026)


def test_resolved_features_exclude_outcome_labels(synthetic_result: dict) -> None:
    resolved = synthetic_result["feature_columns"]
    assert {"total_runs", "home_score", "away_score"}.isdisjoint(resolved)
    direct = default_feature_columns(
        pd.DataFrame(
            {
                "total_runs": [9],
                "home_score": [5],
                "away_score": [4],
                "pregame_numeric": [1.0],
            }
        )
    )
    assert direct == ["pregame_numeric"]


def test_time_split_never_trains_on_test_season(synthetic_result: dict) -> None:
    splits = synthetic_result["splits"]
    assert splits["train_seasons"] == [2021, 2022, 2023, 2024]
    assert splits["validation_season"] == 2025
    assert splits["test_season"] == 2026
    assert max(splits["final_train_seasons"]) < splits["test_season"]


def test_over_label_threshold_arithmetic() -> None:
    np.testing.assert_array_equal(over_label([8, 9], 8.5), [0, 1])
    np.testing.assert_array_equal(over_label([9, 10], 9.5), [0, 1])
    with pytest.raises(ValueError, match="push"):
        over_label([8, 9], 8.0)

