import pandas as pd

from src.models.nfl_shadow import evaluate_shadow_predictions


GATES = {"max_ece": 0.05, "max_home_probability_bias": 0.03, "max_home_margin_bias": 0.75}


def _shadow_rows(count=64, weeks=4):
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(count)],
            "week": [(i % weeks) + 1 for i in range(count)],
            "kickoff_ts": ["2026-09-10T20:00:00Z"] * count,
            "prediction_ts": ["2026-09-10T18:00:00Z"] * count,
            "data_fingerprint": ["abc"] * count,
            "calibrated_probability": [0.5] * count,
            "predicted_margin": [0.0] * count,
            "home_win": [i % 2 for i in range(count)],
            "home_margin": [1 if i % 2 else -1 for i in range(count)],
            "freshness_ok": [True] * count,
            "critical_integrity_failure": [False] * count,
        }
    )


def test_shadow_requires_both_four_weeks_and_64_games():
    assert evaluate_shadow_predictions(_shadow_rows(), GATES)["promote"] is True
    too_few_games = evaluate_shadow_predictions(_shadow_rows(count=63), GATES)
    assert too_few_games["promote"] is False
    too_few_weeks = evaluate_shadow_predictions(_shadow_rows(weeks=3), GATES)
    assert too_few_weeks["promote"] is False


def test_shadow_fails_closed_on_lineage_freshness_or_post_kickoff_prediction():
    frame = _shadow_rows()
    frame.loc[0, "data_fingerprint"] = ""
    frame.loc[1, "freshness_ok"] = False
    frame.loc[2, "prediction_ts"] = frame.loc[2, "kickoff_ts"]
    result = evaluate_shadow_predictions(frame, GATES)
    assert result["promote"] is False
    assert len(result["integrity_issues"]) == 3
