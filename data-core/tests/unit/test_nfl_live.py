from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.grade_nfl_v2_live import build_report
from src.models.nfl_live import LIVE_MODEL_VERSION, predict_live_games
from src.models.nfl_v2 import FEATURE_COLUMNS


class ConstantProbabilityModel:
    def predict_probability(self, frame):
        return np.full(len(frame), 0.6)


class ConstantMarginModel:
    def predict_margin(self, frame):
        return np.full(len(frame), 3.5)


def _artifact(path: Path) -> None:
    joblib.dump(
        {
            "model_version": LIVE_MODEL_VERSION,
            "feature_columns": FEATURE_COLUMNS,
            "probability_model": ConstantProbabilityModel(),
            "margin_model": ConstantMarginModel(),
            "trained_through": 2025,
            "dataset_fingerprint": "test",
        },
        path,
    )


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": "old", "league": "NFL", "season": 2025, "week": 18, "gameday": "2026-01-01", "home_team": "DEN", "away_team": "KC", "home_score": 24, "away_score": 21},
            {"game_id": "live", "league": "NFL", "season": 2026, "week": 1, "gameday": "2026-09-09", "home_team": "KC", "away_team": "DEN", "home_score": None, "away_score": None},
        ]
    )


def _stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": "old", "team": team, "off_epa": 0.1, "def_epa_allowed": 0.0, "off_success": 0.5, "def_success_allowed": 0.45, "off_explosive": 0.1, "def_explosive_allowed": 0.1, "primary_qb_id": f"{team}-QB"}
            for team in ("DEN", "KC")
        ]
    )


def test_live_prediction_uses_betting_spread_convention_and_lineage(tmp_path):
    artifact = tmp_path / "model.joblib"
    _artifact(artifact)
    result = predict_live_games(_schedules(), _stats(), ["live"], artifact_path=artifact)
    assert result.iloc[0]["home_win_probability"] == pytest.approx(0.6)
    assert result.iloc[0]["predicted_spread"] == pytest.approx(-3.5)
    assert len(result.iloc[0]["input_hash"]) == 64


def test_live_prediction_fails_closed_on_incomplete_pbp(tmp_path):
    artifact = tmp_path / "model.joblib"
    _artifact(artifact)
    with pytest.raises(ValueError, match="PBP coverage"):
        predict_live_games(_schedules(), _stats().iloc[:1], ["live"], artifact_path=artifact)


def test_live_report_tracks_probability_margin_and_home_bias():
    frame = pd.DataFrame(
        {
            "game_id": ["a", "b"], "season": [2026, 2026], "week": [1, 1],
            "home_score": [24, 17], "away_score": [20, 21],
            "my_home_win_prob": [0.6, 0.4], "my_spread": [-3.0, 2.0],
        }
    )
    report = build_report(frame, LIVE_MODEL_VERSION, 2026)
    assert report["graded_games"] == 2
    assert report["metrics"]["probability"]["brier"] == pytest.approx(0.16)
    assert report["metrics"]["margin"]["home_margin_bias"] == pytest.approx(0.5)
    assert report["sample_gate"]["reached"] is False
