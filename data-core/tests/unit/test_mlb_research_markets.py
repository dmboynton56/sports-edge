"""Unit tests for MLB research markets scoring module."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.mlb_research_markets import (
    load_mlb_moneyline_v3,
    load_mlb_runline_v1,
    load_mlb_totals_v1,
    score_mlb_moneyline_v3,
    score_mlb_runline_v1,
    score_mlb_totals_v1,
)


@pytest.fixture
def mock_schedule():
    """Minimal MLB schedule for testing."""
    return pd.DataFrame(
        {
            "game_pk": [123456, 123457],
            "game_date": ["2026-08-26", "2026-08-26"],
            "game_datetime": ["2026-08-26 19:05:00", "2026-08-26 19:10:00"],
            "home_team": ["NYY", "BOS"],
            "away_team": ["BAL", "TOR"],
            "home_score": [5.0, 3.0],
            "away_score": [3.0, 4.0],
            "season": [2026, 2026],
        }
    )


def test_load_moneyline_artifact():
    """Test that we can load the v3 moneyline artifact."""
    model_path = Path(__file__).parents[2] / "models" / "mlb_winner_model_v3.pkl"
    if not model_path.exists():
        pytest.skip("mlb_winner_model_v3.pkl not found")
    
    artifact = load_mlb_moneyline_v3(model_path)
    assert "model" in artifact
    assert "feature_columns" in artifact
    assert artifact.get("model_version") in {"v3", None}  # May not be set in older artifacts


def test_load_totals_artifact():
    """Test that we can load the v1 totals artifact."""
    model_path = Path(__file__).parents[2] / "models" / "mlb_totals_model_v1.pkl"
    if not model_path.exists():
        pytest.skip("mlb_totals_model_v1.pkl not found")
    
    artifact = load_mlb_totals_v1(model_path)
    assert "model" in artifact
    assert "feature_columns" in artifact
    assert "probability_method" in artifact


def test_load_runline_artifact():
    """Test that we can load the v1 run-line artifact."""
    model_path = Path(__file__).parents[2] / "models" / "mlb_runline_model_v1.pkl"
    if not model_path.exists():
        pytest.skip("mlb_runline_model_v1.pkl not found")
    
    artifact = load_mlb_runline_v1(model_path)
    assert "classifier" in artifact
    assert "feature_columns" in artifact


def test_moneyline_v3_uses_v1_features():
    """Assert that moneyline v3 pickle uses v1 schedule-only features (no weather/starters)."""
    model_path = Path(__file__).parents[2] / "models" / "mlb_winner_model_v3.pkl"
    if not model_path.exists():
        pytest.skip("mlb_winner_model_v3.pkl not found")
    
    artifact = load_mlb_moneyline_v3(model_path)
    feature_cols = set(artifact["feature_columns"])
    
    # V2 markers that should NOT be in moneyline v3
    v2_markers = {
        "temp_f", "wind_mph", "wind_out", "wind_in", "wind_cross", "elevation",
        "is_dome_or_closed", "is_day_game",
    }
    starter_markers = {
        col for col in feature_cols if "starter_k9" in col or "starter_bb9" in col or "starter_era" in col
    }
    
    found_v2 = v2_markers.intersection(feature_cols)
    if found_v2 or starter_markers:
        pytest.fail(
            f"Moneyline v3 should use v1 schedule-only features, but found v2 weather/starter columns: "
            f"{found_v2.union(starter_markers)}"
        )


def test_totals_v1_uses_v2_features():
    """Assert that totals v1 pickle uses v2 features (weather, starters, venue)."""
    model_path = Path(__file__).parents[2] / "models" / "mlb_totals_model_v1.pkl"
    if not model_path.exists():
        pytest.skip("mlb_totals_model_v1.pkl not found")
    
    artifact = load_mlb_totals_v1(model_path)
    feature_cols = set(artifact["feature_columns"])
    
    # V2 markers that SHOULD be in totals v1
    expected_v2 = {"temp_f", "wind_mph", "elevation"}
    missing = expected_v2.difference(feature_cols)
    if missing:
        pytest.fail(
            f"Totals v1 should use v2 features (weather, starters), but missing: {missing}. "
            "This means the scoring function will KeyError on prediction."
        )


def test_runline_v1_uses_v2_features():
    """Assert that run-line v1 pickle uses v2 features (weather, starters, venue)."""
    model_path = Path(__file__).parents[2] / "models" / "mlb_runline_model_v1.pkl"
    if not model_path.exists():
        pytest.skip("mlb_runline_model_v1.pkl not found")
    
    artifact = load_mlb_runline_v1(model_path)
    feature_cols = set(artifact["feature_columns"])
    
    # V2 markers that SHOULD be in run-line v1
    expected_v2 = {"temp_f", "wind_mph", "elevation"}
    missing = expected_v2.difference(feature_cols)
    if missing:
        pytest.fail(
            f"Run-line v1 should use v2 features (weather, starters), but missing: {missing}. "
            "This means the scoring function will KeyError on prediction."
        )


def test_score_moneyline_empty_schedule():
    """Test scoring moneyline on an empty schedule."""
    model_path = Path(__file__).parents[2] / "models" / "mlb_winner_model_v3.pkl"
    if not model_path.exists():
        pytest.skip("mlb_winner_model_v3.pkl not found")
    
    artifact = load_mlb_moneyline_v3(model_path)
    empty_schedule = pd.DataFrame(columns=["game_pk", "game_date", "game_datetime", "home_team", "away_team", "home_score", "away_score"])
    
    result = score_mlb_moneyline_v3(
        artifact=artifact,
        schedule=empty_schedule,
        game_date=date(2026, 8, 26),
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert "home_win_prob" in result.columns
    assert "game_datetime" in result.columns

