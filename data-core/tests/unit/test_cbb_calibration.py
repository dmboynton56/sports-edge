import numpy as np

from src.models.cbb_train_matchup_model import check_upset_calibration


def test_upset_calibration_deduplicates_mirrored_matchups():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.95, 0.05, 0.2, 0.8])
    seed_diffs = np.array([-15, 15, 13, -13])
    game_keys = np.array(
        [
            (2025, 1, 16),
            (2025, 1, 16),
            (2025, 2, 15),
            (2025, 2, 15),
        ],
        dtype=object,
    )

    rows = check_upset_calibration(y_true, y_prob, seed_diffs, game_keys)
    by_matchup = {row["matchup"]: row for row in rows}

    assert by_matchup["1v16"]["games"] == 1
    assert by_matchup["1v16"]["actual_upsets"] == 0
    assert abs(by_matchup["1v16"]["avg_underdog_prob"] - 0.05) < 1e-12
    assert by_matchup["2v15"]["games"] == 1
    assert by_matchup["2v15"]["actual_upsets"] == 1
    assert abs(by_matchup["2v15"]["avg_underdog_prob"] - 0.2) < 1e-12
