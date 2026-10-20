from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.nfl_live import LIVE_MODEL_VERSION, predict_live_games
from src.models.nfl_margin_probability import (
    BlendStack,
    choose_published_probability,
    count_sign_contradictions,
    fit_blend_stack,
    is_sign_contradiction,
    margin_implied_home_win_prob,
    margin_winprob_gate,
    probability_side,
    reconcile_published_probability,
    residual_sigma,
    spread_side,
)
from src.models.nfl_v2 import FEATURE_COLUMNS
from tests.unit.test_nfl_live import ConstantMarginModel, ConstantProbabilityModel, _schedules, _stats


class ConfigurableMarginModel:
    def __init__(self, value: float):
        self.value = value

    def predict_margin(self, frame):
        return np.full(len(frame), self.value)


def test_phi_matches_negative_spread_over_sigma():
    # Home favored by 7.2 → P(home) well above a coin flip.
    p = float(margin_implied_home_win_prob(7.2, 13.21))
    assert p == pytest.approx(0.707, abs=0.01)
    dog = float(margin_implied_home_win_prob(-7.2, 13.21))
    assert dog == pytest.approx(1.0 - p, abs=1e-9)


def test_residual_sigma_is_rmse_and_floored():
    assert residual_sigma([12.0, -12.0, 0.0]) == pytest.approx(np.sqrt((144 + 144) / 3))
    assert residual_sigma([1.0, -1.0]) == 8.0  # MIN_SIGMA floor
    assert residual_sigma([]) == pytest.approx(13.21)


def test_sign_contradiction_ignores_toss_up_band():
    assert is_sign_contradiction(0.494, -7.2) is False  # toss-up, even with a big home favorite
    assert is_sign_contradiction(0.40, -7.2) is True  # away p, home favored, outside band
    assert is_sign_contradiction(0.76, -7.8) is False  # both home
    assert is_sign_contradiction(0.36, 9.0) is False  # both away
    assert is_sign_contradiction(0.65, 2.3) is True  # home p, away favored
    assert spread_side(-7.2) == "home"
    assert spread_side(2.3) == "away"
    assert spread_side(0.0) is None
    assert probability_side(0.494) is None
    assert probability_side(0.40) == "away"


def test_count_sign_contradictions_on_week7_pattern():
    p = [0.494, 0.764, 0.348, 0.364, 0.494]
    spread = [-7.19, -7.83, 2.34, 8.98, 0.29]
    # GB@DET and NE@CHI sit in the toss-up band; the others agree on side.
    assert count_sign_contradictions(p, spread) == 0
    assert count_sign_contradictions([0.40, 0.65], [-7.2, 2.3]) == 2


def test_blend_stack_roundtrip_and_predict():
    rng = np.random.default_rng(0)
    margin_p = np.clip(0.5 + 0.2 * rng.normal(size=400), 0.05, 0.95)
    logits = rng.normal(size=400)
    y = (margin_p + 0.05 * logits > 0.5).astype(int)
    fitted = fit_blend_stack(y, margin_p, logits)
    restored = BlendStack.from_dict(fitted.to_dict())
    assert restored is not None
    pred = restored.predict(margin_p, logits)
    assert pred.min() > 0
    assert pred.max() < 1
    assert ((pred >= 0.5) == y).mean() > 0.8


def test_reconcile_falls_back_to_margin_p_and_logs(caplog):
    published, margin_p = choose_published_probability(
        classifier_p=[0.70],
        margin=[-7.0],
        method="isotonic",
        sigma=13.21,
    )
    assert is_sign_contradiction(published[0], 7.0)
    with caplog.at_level("WARNING", logger="nfl_margin_probability"):
        fixed, reconciled = reconcile_published_probability(
            published, [-7.0], margin_p, game_ids=["GB@DET"]
        )
    assert reconciled[0]
    assert not is_sign_contradiction(fixed[0], 7.0)
    assert "GB@DET" in caplog.text
    assert fixed[0] == pytest.approx(float(margin_p[0]))


def _comparison(brier, log_loss, contradictions, ci_low, ci_high):
    return {
        "candidate": {
            "brier": brier,
            "log_loss": log_loss,
            "sign_contradictions": contradictions,
        },
        "delta_brier_ci": {"lower_95": ci_low, "upper_95": ci_high},
        "delta_log_loss_ci": {"lower_95": ci_low, "upper_95": ci_high},
    }


def test_gate_ships_margin_normal_when_not_worse_and_clean():
    walk_margin = _comparison(0.226, 0.64, 0, -0.004, 0.002)
    walk_blend = _comparison(0.228, 0.65, 0, -0.001, 0.003)
    decision = margin_winprob_gate(walk_margin=walk_margin, walk_blend=walk_blend)
    assert decision["ship"] is True
    assert decision["selected_method"] == "margin_normal"


def test_gate_prefers_better_brier_when_both_eligible():
    walk_margin = _comparison(0.227, 0.65, 0, -0.002, 0.001)
    walk_blend = _comparison(0.224, 0.64, 0, -0.005, 0.001)
    decision = margin_winprob_gate(walk_margin=walk_margin, walk_blend=walk_blend)
    assert decision["selected_method"] == "blend"


def test_gate_rejects_when_contradictions_remain_or_clearly_worse():
    worse = _comparison(0.24, 0.70, 0, 0.001, 0.01)
    dirty = _comparison(0.22, 0.64, 4, -0.01, 0.0)
    decision = margin_winprob_gate(walk_margin=worse, walk_blend=dirty)
    assert decision["ship"] is False
    assert decision["selected_method"] is None


def _artifact(path: Path, probability: float = 0.6, margin: float = 3.5, win_prob=None) -> None:
    payload = {
        "model_version": LIVE_MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "probability_model": ConstantProbabilityModel(probability),
        "margin_model": ConstantMarginModel(),
        "trained_through": 2025,
        "dataset_fingerprint": "test",
    }
    if win_prob is not None:
        payload["win_prob"] = win_prob
    if margin != 3.5:
        payload["margin_model"] = ConfigurableMarginModel(margin)
    joblib.dump(payload, path)


def test_live_guard_replaces_contradicting_classifier_p(tmp_path):
    artifact = tmp_path / "model.joblib"
    _artifact(artifact, probability=0.70, margin=-7.0)
    result = predict_live_games(_schedules(), _stats(), ["live"], artifact_path=artifact)
    assert result.iloc[0]["predicted_spread"] == pytest.approx(7.0)
    assert result.iloc[0]["home_win_probability_classifier"] == pytest.approx(0.70)
    assert bool(result.iloc[0]["win_prob_reconciled"])
    published = float(result.iloc[0]["home_win_probability"])
    assert published < 0.5
    assert not is_sign_contradiction(published, 7.0)
    assert result.iloc[0]["moneyline_pick"] == "DEN"


def test_live_margin_method_uses_phi_and_drives_toss_up(tmp_path):
    artifact = tmp_path / "model.joblib"
    _artifact(
        artifact,
        probability=0.494,
        margin=7.2,
        win_prob={"method": "margin_normal", "residual_sigma": 13.21},
    )
    result = predict_live_games(_schedules(), _stats(), ["live"], artifact_path=artifact)
    published = float(result.iloc[0]["home_win_probability"])
    assert published == pytest.approx(float(margin_implied_home_win_prob(7.2, 13.21)))
    assert published > 0.55
    assert not bool(result.iloc[0]["is_toss_up"])
    assert result.iloc[0]["moneyline_pick"] == "KC"
    assert result.iloc[0]["home_win_probability_classifier"] == pytest.approx(0.494)
    assert result.iloc[0]["win_prob_method"] == "margin_normal"


def test_rollback_env_keeps_classifier_p(tmp_path, monkeypatch):
    monkeypatch.setenv("NFL_V2_LIVE_VERSION", "nfl-v2-live-20260906")
    artifact = tmp_path / "model.joblib"
    _artifact(
        artifact,
        probability=0.494,
        margin=7.2,
        win_prob={"method": "margin_normal", "residual_sigma": 13.21},
    )
    result = predict_live_games(_schedules(), _stats(), ["live"], artifact_path=artifact)
    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.494)
    assert bool(result.iloc[0]["is_toss_up"])
    assert result.iloc[0]["win_prob_method"] == "isotonic"


def test_live_agreeing_rows_keep_classifier_p(tmp_path):
    artifact = tmp_path / "model.joblib"
    _artifact(artifact, probability=0.6, margin=3.5)
    result = predict_live_games(_schedules(), _stats(), ["live"], artifact_path=artifact)
    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.6)
    assert not bool(result.iloc[0]["win_prob_reconciled"])
