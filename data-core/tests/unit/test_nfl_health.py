from __future__ import annotations

import inspect

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.nfl_health import (
    book_ats_result,
    metrics_block,
    prior_decay_mix,
    score_prediction_frame,
    segmented_health,
)
from src.models.nfl_live import (
    DEFAULT_ARTIFACT,
    LIVE_MODEL_VERSION,
    PARENT_LIVE_MODEL_VERSION,
    apply_live_serving_stamp,
    load_live_artifact,
    posthoc_calibration_enabled,
)
from src.models.nfl_recalibration import (
    fit_temperature,
    ship_decision,
    walk_forward_predictions,
)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "week": [1, 1, 2, 2],
            "game_date": pd.to_datetime(["2026-09-10", "2026-09-10", "2026-09-17", "2026-09-17"]),
            "home_team": ["KC", "BUF", "PHI", "DAL"],
            "away_team": ["BAL", "NYJ", "GB", "NYG"],
            "home_score": [27, 17, 24, 20],
            "away_score": [20, 20, 17, 27],
            "my_spread": [-3.0, 1.0, -6.5, -3.0],
            "my_home_win_prob": [0.62, 0.44, 0.78, 0.58],
            "closing_spread": [-6.5, 2.5, -3.5, -1.5],
            "home_moneyline": [-280, 120, -180, -125],
            "away_moneyline": [230, -140, 155, 105],
        }
    )


def test_book_ats_bets_the_model_side_of_the_closing_spread():
    # Model is less on home than the book (-3 vs -6.5) → bet away. Home won by 7, covers -6.5, so away loses.
    assert book_ats_result(7, -3.0, -6.5) == "loss"
    # Model is more on home (-6.5 vs -3.5) → bet home. Home won by 7, covers -3.5.
    assert book_ats_result(7, -6.5, -3.5) == "win"
    assert book_ats_result(3.0, -3.0, -3.0) == "no_bet"
    assert book_ats_result(3.0, -3.0, -3.0 + 1e-12) is not None


def test_score_frame_and_metrics_match_manual_ats_ml():
    scored = score_prediction_frame(_games())
    assert scored.loc[0, "winner_result"] == "win"
    assert scored.loc[1, "winner_result"] == "win"
    assert scored.loc[0, "book_ats"] == "loss"
    assert scored.loc[2, "book_ats"] == "win"
    metrics = metrics_block(scored, bootstrap=False)
    assert metrics["rows"] == 4
    assert metrics["winner"]["wins"] == 3
    assert metrics["winner"]["losses"] == 1
    assert metrics["book_ats"]["bets"] == 4
    assert metrics["moneyline"]["bets"] == 4
    assert metrics["margin"]["mae"] > 0
    assert "brier" in metrics["probability"]
    assert metrics["market_probability"]["brier"] > 0


def test_segments_break_out_week_favorite_and_confidence():
    scored = score_prediction_frame(_games())
    segments = segmented_health(scored)
    weeks = {row["segment"]: row["rows"] for row in segments["week"]}
    assert weeks == {"week_1": 2, "week_2": 2}
    labels = {row["segment"] for row in segments["favorite"]}
    assert "model_favorite" in labels
    assert "model_dog" in labels
    conf = {row["segment"] for row in segments["confidence"]}
    assert conf


def test_prior_decay_matches_documented_week3_mix():
    mix = prior_decay_mix(3, window=3, prior_season_games=17)
    assert mix["current_season"] == pytest.approx(3 / 7)
    assert mix["last_season"] == pytest.approx((4 / 7) * (17 / 25))
    assert mix["league_average"] == pytest.approx((4 / 7) * (8 / 25))
    later = prior_decay_mix(10, window=10, prior_season_games=17)
    assert later["current_season"] > mix["current_season"]


def test_walk_forward_does_not_fit_on_the_scored_week():
    frame = pd.DataFrame(
        {
            "week": [1, 1, 1, 1, 2, 2],
            "game_date": pd.to_datetime(["2026-09-10"] * 4 + ["2026-09-17"] * 2),
            "game_id": list("abcdef"),
            "value": [0.0, 0.0, 0.0, 0.0, 99.0, 99.0],
        }
    )

    seen_train_means = []

    def fit_fn(train):
        seen_train_means.append(float(train["value"].mean()))
        return float(train["value"].mean())

    def apply_fn(fitted, test):
        return np.full(len(test), fitted)

    mask, values, folds = walk_forward_predictions(frame, fit_fn=fit_fn, apply_fn=apply_fn, min_train_rows=4)
    assert folds and folds[0]["test_week"] == 2
    assert seen_train_means == [0.0]
    assert mask.tolist() == [False, False, False, False, True, True]
    assert values[mask].tolist() == [0.0, 0.0]


def test_temperature_greater_than_one_shrinks_confident_probabilities():
    # Overconfident home-favorite prices against a 50/50 outcome stream.
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    raw = np.array([0.95, 0.90, 0.92, 0.88, 0.91, 0.87, 0.93, 0.85, 0.94, 0.86])
    calibrator = fit_temperature(y, raw)
    calibrated = calibrator.transform(raw)
    assert calibrator.params["temperature"] > 1.0
    assert abs(calibrated[0] - 0.5) < abs(raw[0] - 0.5)


def test_ship_decision_requires_brier_and_log_loss_gains():
    improved = {
        "method": "temperature",
        "status": "ok",
        "oos_rows": 50,
        "candidate": {"brier": 0.23, "log_loss": 0.65},
        "delta": {"brier": -0.01, "log_loss": -0.02},
    }
    decision = ship_decision([improved], [])
    assert decision["ship"] is True
    assert decision["probability_method"] == "temperature"

    mixed = dict(improved)
    mixed["delta"] = {"brier": -0.01, "log_loss": 0.01}
    assert ship_decision([mixed], [])["ship"] is False
    tiny = dict(improved)
    tiny["delta"] = {"brier": -0.0005, "log_loss": -0.01}
    assert ship_decision([tiny], [])["ship"] is False


def test_posthoc_calibration_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("NFL_V2_POSTHOC_CALIBRATION", raising=False)
    monkeypatch.delenv("NFL_V2_DISABLE_POSTHOC_CALIBRATION", raising=False)
    assert posthoc_calibration_enabled({}) is False
    assert posthoc_calibration_enabled({"NFL_V2_POSTHOC_CALIBRATION": "on"}) is True
    assert posthoc_calibration_enabled(
        {"NFL_V2_POSTHOC_CALIBRATION": "on", "NFL_V2_DISABLE_POSTHOC_CALIBRATION": "1"}
    ) is False
    assert LIVE_MODEL_VERSION == "nfl-v2-live-20261020"


def test_health_expected_version_equals_predict_written_version(monkeypatch):
    """Health, Daily Refresh, and the loaded live artifact must share one stamp."""
    pickle = joblib.load(DEFAULT_ARTIFACT)
    assert pickle["model_version"] == PARENT_LIVE_MODEL_VERSION

    artifact = load_live_artifact()
    assert artifact["model_version"] == LIVE_MODEL_VERSION
    assert artifact.get("parent_model_version") == PARENT_LIVE_MODEL_VERSION
    assert (artifact.get("win_prob") or {}).get("method") == "margin_normal"

    monkeypatch.setattr("sys.argv", ["nfl_model_health.py"])
    from scripts.nfl_model_health import _parse_args as health_args
    from scripts.nfl_model_health import reconstruct_live

    from scripts.grade_nfl_v2_live import _parse_args as grade_args
    from scripts.predict_nfl_upcoming import LIVE_MODEL_VERSION as predict_version
    from src.pipeline.refresh_nfl import LIVE_MODEL_VERSION as refresh_version

    assert health_args().model_version == predict_version == refresh_version == LIVE_MODEL_VERSION
    monkeypatch.setattr("sys.argv", ["grade_nfl_v2_live.py"])
    assert grade_args().model_version == LIVE_MODEL_VERSION
    source = inspect.getsource(reconstruct_live)
    assert "score_live_features" in source
    assert 'artifact["probability_model"].predict_probability' not in source


def test_live_stamp_fails_closed_without_margin_normal_sidecar(monkeypatch):
    monkeypatch.setattr("src.models.nfl_margin_probability.load_win_prob_sidecar", lambda *args, **kwargs: {})
    artifact = {"model_version": PARENT_LIVE_MODEL_VERSION}
    with pytest.raises(ValueError, match="margin_normal"):
        apply_live_serving_stamp(artifact, expected=LIVE_MODEL_VERSION, sidecar=True)
