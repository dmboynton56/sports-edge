from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.nfl_calibrator_promotion import (
    CANDIDATE_LIVE_MODEL_VERSION,
    LIVE_VERSION_ENV,
    promotion_gate,
    toss_up_shift,
    wrap_frozen_artifact,
)
from src.models.nfl_experiment import ProbabilityCalibrator
from src.models.nfl_live import LIVE_MODEL_VERSION, load_live_artifact
from src.models.nfl_toss_up import is_nfl_toss_up


def _side(*, brier, log_loss, ece_10, mae, unique=20, n=92):
    return {
        "n": n,
        "baseline": {
            "brier": 0.24,
            "log_loss": 0.68,
            "ece_10": 0.09,
            "mae": 9.6,
            "unique_probabilities": 17,
        },
        "candidate": {
            "brier": brier,
            "log_loss": log_loss,
            "ece_10": ece_10,
            "mae": mae,
            "unique_probabilities": unique,
        },
        "delta_brier_ci": {"mean": brier - 0.24, "lower_95": brier - 0.24 - 0.01, "upper_95": brier - 0.24 + 0.01},
        "delta_log_loss_ci": {
            "mean": log_loss - 0.68,
            "lower_95": log_loss - 0.68 - 0.02,
            "upper_95": log_loss - 0.68 + 0.02,
        },
    }


def test_gate_ships_when_holdout_brier_and_log_loss_improve():
    holdout = _side(brier=0.22, log_loss=0.64, ece_10=0.07, mae=9.6)
    holdout["delta_brier_ci"] = {"mean": -0.02, "lower_95": -0.03, "upper_95": -0.01}
    holdout["delta_log_loss_ci"] = {"mean": -0.04, "lower_95": -0.06, "upper_95": -0.01}
    walk = _side(brier=0.23, log_loss=0.66, ece_10=0.08, mae=9.5, n=1359)
    walk["delta_brier_ci"] = {"mean": -0.01, "lower_95": -0.02, "upper_95": -0.005}
    walk["delta_log_loss_ci"] = {"mean": -0.02, "lower_95": -0.03, "upper_95": -0.01}
    decision = promotion_gate(holdout, walk, selected_method="sigmoid")
    assert decision["ship"] is True


def test_gate_ships_when_holdout_is_noise_and_walk_forward_is_clear():
    holdout = _side(brier=0.239, log_loss=0.679, ece_10=0.088, mae=9.6)
    holdout["delta_brier_ci"] = {"mean": -0.001, "lower_95": -0.01, "upper_95": 0.008}
    holdout["delta_log_loss_ci"] = {"mean": -0.001, "lower_95": -0.02, "upper_95": 0.015}
    walk = _side(brier=0.22, log_loss=0.64, ece_10=0.07, mae=9.5, n=1359)
    walk["delta_brier_ci"] = {"mean": -0.02, "lower_95": -0.03, "upper_95": -0.01}
    walk["delta_log_loss_ci"] = {"mean": -0.04, "lower_95": -0.05, "upper_95": -0.02}
    decision = promotion_gate(holdout, walk, selected_method="temperature")
    assert decision["ship"] is True
    assert decision["holdout_within_noise"] is True
    assert decision["walk_forward_clearly_better"] is True


def test_gate_rejects_isotonic_even_if_metrics_improve():
    holdout = _side(brier=0.22, log_loss=0.64, ece_10=0.07, mae=9.6)
    holdout["delta_brier_ci"] = {"mean": -0.02, "lower_95": -0.03, "upper_95": -0.01}
    holdout["delta_log_loss_ci"] = {"mean": -0.04, "lower_95": -0.06, "upper_95": -0.01}
    walk = _side(brier=0.23, log_loss=0.66, ece_10=0.08, mae=9.5, n=1359)
    walk["delta_brier_ci"] = {"mean": -0.01, "lower_95": -0.02, "upper_95": -0.005}
    walk["delta_log_loss_ci"] = {"mean": -0.02, "lower_95": -0.03, "upper_95": -0.01}
    decision = promotion_gate(holdout, walk, selected_method="isotonic")
    assert decision["ship"] is False


def test_gate_rejects_when_ece_or_mae_worsens_or_holdout_is_clearly_worse():
    worse_ece = _side(brier=0.22, log_loss=0.64, ece_10=0.12, mae=9.6)
    worse_ece["delta_brier_ci"] = {"mean": -0.02, "lower_95": -0.03, "upper_95": -0.01}
    worse_ece["delta_log_loss_ci"] = {"mean": -0.04, "lower_95": -0.06, "upper_95": -0.01}
    walk = _side(brier=0.22, log_loss=0.64, ece_10=0.07, mae=9.5, n=1359)
    walk["delta_brier_ci"] = {"mean": -0.02, "lower_95": -0.03, "upper_95": -0.01}
    walk["delta_log_loss_ci"] = {"mean": -0.04, "lower_95": -0.05, "upper_95": -0.02}
    assert promotion_gate(worse_ece, walk, selected_method="sigmoid")["ship"] is False

    worse_holdout = _side(brier=0.26, log_loss=0.72, ece_10=0.07, mae=9.6)
    worse_holdout["delta_brier_ci"] = {"mean": 0.02, "lower_95": 0.01, "upper_95": 0.04}
    worse_holdout["delta_log_loss_ci"] = {"mean": 0.04, "lower_95": 0.01, "upper_95": 0.07}
    assert promotion_gate(worse_holdout, walk, selected_method="sigmoid")["ship"] is False


def test_gate_rejects_noisy_2026_when_walk_forward_brier_is_not_clear():
    holdout = _side(brier=0.239, log_loss=0.676, ece_10=0.075, mae=9.6)
    holdout["delta_brier_ci"] = {"mean": -0.001, "lower_95": -0.008, "upper_95": 0.006}
    holdout["delta_log_loss_ci"] = {"mean": -0.004, "lower_95": -0.02, "upper_95": 0.011}
    walk = _side(brier=0.227, log_loss=0.647, ece_10=0.031, mae=10.25, n=1359)
    walk["delta_brier_ci"] = {"mean": -0.0004, "lower_95": -0.003, "upper_95": 0.002}
    walk["delta_log_loss_ci"] = {"mean": -0.033, "lower_95": -0.072, "upper_95": -0.003}
    decision = promotion_gate(holdout, walk, selected_method="sigmoid")
    assert decision["ship"] is False
    assert decision["holdout_within_noise"] is True
    assert decision["walk_forward_clearly_better"] is False


def test_toss_up_shift_counts_enter_and_exit():
    baseline = [0.50, 0.62, 0.48, 0.80]
    candidate = [0.61, 0.51, 0.48, 0.80]
    shift = toss_up_shift(baseline, candidate)
    assert shift["entered"] == 1
    assert shift["exited"] == 1
    assert shift["stayed"] == 1
    assert shift["net"] == 0
    assert is_nfl_toss_up(0.45)
    assert not is_nfl_toss_up(0.56)


def test_wrap_frozen_artifact_keeps_hgb_and_ridge():
    frozen = load_live_artifact()
    calibrator = ProbabilityCalibrator(method="none")
    wrapped = wrap_frozen_artifact(
        frozen,
        calibrator=calibrator,
        calibration="none",
        selection={"selected": "none", "reason": "test", "candidates": []},
    )
    assert wrapped["model_version"] == CANDIDATE_LIVE_MODEL_VERSION
    assert wrapped["parent_model_version"] == LIVE_MODEL_VERSION
    assert wrapped["margin_model"] is frozen["margin_model"]
    assert wrapped["probability_model"].model is frozen["probability_model"].model
    assert wrapped["probability_model"].calibrator.method == "none"
    assert wrapped["selection"]["calibration_method"] == "none"
    assert LIVE_VERSION_ENV == "NFL_V2_LIVE_VERSION"
