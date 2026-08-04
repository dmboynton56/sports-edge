import numpy as np
import pandas as pd

from src.models.mlb_runline_model import (
    calibration_table,
    cover_probability_from_residuals,
    resolved_feature_columns,
    split_runline_frame,
)


def test_runline_feature_selection_excludes_outcomes():
    frame = pd.DataFrame(
        {
            "pregame_signal": [0.1, 0.2],
            "home_cover_15": [0, 1],
            "run_diff": [-1, 2],
            "home_score": [3, 5],
            "away_score": [4, 3],
            "total_runs": [7, 8],
        }
    )
    assert resolved_feature_columns(frame) == ["pregame_signal"]


def test_time_split_keeps_test_season_out_of_training():
    frame = pd.DataFrame({"season": [2021, 2024, 2025, 2026], "value": range(4)})
    train, validation, test, refit = split_runline_frame(frame, 2025, 2026)
    assert set(train["season"]) == {2021, 2024}
    assert set(validation["season"]) == {2025}
    assert set(test["season"]) == {2026}
    assert refit["season"].max() == 2025
    assert 2026 not in set(refit["season"])


def test_half_run_cover_probability_has_no_push_boundary():
    predicted = np.array([1.0, 2.0, 3.0])
    residuals = np.array([-1.0, 0.0, 1.0])
    probability = cover_probability_from_residuals(predicted, residuals)
    assert probability[0] < probability[1] < probability[2]
    assert np.all((probability > 0) & (probability < 1))
    assert np.allclose(probability + (1 - probability), 1.0)


def test_calibration_table_accounts_for_every_row():
    table = calibration_table(np.array([0, 0, 1, 1]), np.array([0.05, 0.25, 0.75, 1.0]))
    assert len(table) == 10
    assert sum(row["count"] for row in table) == 4
    assert table[-1]["count"] == 1
