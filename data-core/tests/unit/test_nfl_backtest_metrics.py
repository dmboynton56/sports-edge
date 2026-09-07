import pandas as pd
import pytest

from scripts.export_nfl_backtest_history import _metrics


def test_metrics_exposes_home_probability_and_margin_bias():
    predictions = pd.DataFrame(
        {
            "home_win": [1, 0, 1, 0],
            "home_win_probability": [0.9, 0.8, 0.8, 0.7],
            "home_win_prob_from_model": [0.85, 0.75, 0.75, 0.65],
            "win_prob_from_spread": [0.95, 0.85, 0.85, 0.75],
            "actual_margin": [7.0, -3.0, 4.0, -6.0],
            "predicted_margin": [10.0, 6.0, 8.0, 4.0],
        }
    )

    result = _metrics(predictions)

    assert result["actual_home_win_rate"] == 0.5
    assert result["avg_pred_home_win"] == 0.8
    assert result["home_probability_bias"] == pytest.approx(0.3)
    assert result["avg_actual_margin"] == 0.5
    assert result["avg_predicted_margin"] == 7.0
    assert result["home_margin_bias"] == 6.5
    assert result["predicted_home_favorite_rate"] == 1.0
    assert set(result["components"]) == {"direct_model", "spread_link"}


def test_metrics_reports_negative_skill_when_model_loses_to_coin_flip():
    predictions = pd.DataFrame(
        {
            "home_win": [1, 0, 1, 0],
            "home_win_probability": [0.1, 0.9, 0.1, 0.9],
            "actual_margin": [3.0, -3.0, 7.0, -7.0],
            "predicted_margin": [-3.0, 3.0, -7.0, 7.0],
        }
    )

    result = _metrics(predictions)

    assert result["brier"] > result["constant_50_brier"]
    assert result["brier_skill_vs_50"] < 0
    assert "components" not in result
