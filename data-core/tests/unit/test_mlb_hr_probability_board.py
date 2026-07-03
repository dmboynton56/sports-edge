from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.predict_mlb_home_runs as hr_predictions
from scripts.predict_mlb_home_runs import (
    _build_probability_board,
    _build_statcast_blend_predictions,
    _candidate_key_set,
    _filter_to_candidate_set,
    _predictions_to_rows,
    _statcast_health_payload,
)
from src.models.mlb_home_run_model import FEATURE_COLUMNS


def _prediction_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "game_id": "MLB_1",
        "game_date": "2026-06-26",
        "event_time": "2026-06-26T23:00:00+00:00",
        "player_name": "Test Player",
        "team": "AAA",
        "opponent": "BBB",
        "venue": "Test Park",
        "lineup_slot": 1,
        "lineup_status": "projected",
        "opposing_probable_pitcher": "Pitcher",
        "baseline_probability": 0.04,
        "games_since_last_hr": None,
        "last_hr_date": None,
        "confidence": 0.7,
        "model_version": "mlb-hr-v1",
        "prediction_ts": "2026-06-26T14:00:00+00:00",
        "quality_flags": "[]",
        "top_features": "[]",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _candidate_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "game_id": "MLB_1",
        "game_pk": 1,
        "game_date": "2026-06-26",
        "event_time": "2026-06-26T23:00:00+00:00",
        "player_name": "Test Player",
        "team": "AAA",
        "opponent": "BBB",
        "team_id": 1,
        "opponent_id": 2,
        "opposing_starter_id": 99,
        "venue": "Test Park",
        "lineup_slot": 1,
        "lineup_status": "projected",
        "opposing_probable_pitcher": "Pitcher",
        "probable_pitcher_known": True,
        "baseline_probability": 0.04,
        "heuristic_probability": 0.05,
        "games_since_last_hr": None,
        "last_hr_date": None,
    }
    defaults.update({col: 1.0 for col in FEATURE_COLUMNS})
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_probability_board_labels_model_agreement() -> None:
    v1 = _prediction_frame(
        [
            {"player_id": 1, "player_name": "Consensus Bat", "hr_probability": 0.25, "rank": 1},
            {"player_id": 2, "player_name": "Fade Bat", "hr_probability": 0.22, "rank": 2},
            {"player_id": 3, "player_name": "Missing Bat", "hr_probability": 0.18, "rank": 3},
            {"player_id": 4, "player_name": "Boost Bat", "hr_probability": 0.08, "rank": 10},
        ]
    )
    statcast = _prediction_frame(
        [
            {"player_id": 1, "hr_probability": 0.24, "rank": 1, "model_version": "statcast"},
            {"player_id": 2, "hr_probability": 0.09, "rank": 8, "model_version": "statcast"},
            {
                "player_id": 3,
                "hr_probability": 0.18,
                "rank": 3,
                "model_version": "statcast",
                "quality_flags": json.dumps(["statcast_features_unavailable"]),
            },
            {"player_id": 4, "hr_probability": 0.21, "rank": 4, "model_version": "statcast"},
        ]
    )

    board = _build_probability_board(v1, statcast_predictions=statcast)
    labels = dict(zip(board["player_id"], board["model_agreement"], strict=False))

    assert labels[1] == "Consensus"
    assert labels[2] == "Statcast fade"
    assert labels[3] == "Missing Statcast"
    assert labels[4] == "Statcast boost"
    assert board.loc[board["player_id"] == 3, "statcast_available"].item() is False


def test_probability_board_without_statcast_marks_v1_only() -> None:
    v1 = _prediction_frame(
        [
            {"player_id": 1, "hr_probability": 0.25, "rank": 1},
            {"player_id": 2, "hr_probability": 0.22, "rank": 2},
        ]
    )

    board = _build_probability_board(v1, statcast_predictions=None)

    assert set(board["model_agreement"]) == {"V1 only"}
    assert board["statcast_probability"].isna().all()


def test_legacy_model_feeds_filter_to_same_candidate_set() -> None:
    v1 = _prediction_frame(
        [
            {"player_id": 1, "hr_probability": 0.25, "rank": 1},
            {"player_id": 2, "hr_probability": 0.22, "rank": 2},
            {"player_id": 3, "hr_probability": 0.18, "rank": 3},
        ]
    )
    statcast = _prediction_frame(
        [
            {"player_id": 3, "hr_probability": 0.30, "rank": 1, "model_version": "statcast"},
            {"player_id": 1, "hr_probability": 0.24, "rank": 2, "model_version": "statcast"},
            {"player_id": 2, "hr_probability": 0.10, "rank": 3, "model_version": "statcast"},
        ]
    )
    board = _build_probability_board(v1, statcast_predictions=statcast, top_n=2)

    v1_filtered = _filter_to_candidate_set(v1, board)
    statcast_filtered = _filter_to_candidate_set(statcast, board)
    board_keys = set(zip(board["game_id"], board["player_id"], strict=False))

    assert set(zip(v1_filtered["game_id"], v1_filtered["player_id"], strict=False)) == board_keys
    assert set(zip(statcast_filtered["game_id"], statcast_filtered["player_id"], strict=False)) == board_keys


def test_prediction_rows_include_probability_board_fields() -> None:
    v1 = _prediction_frame([{"player_id": 1, "hr_probability": 0.25, "rank": 1}])
    statcast = _prediction_frame(
        [{"player_id": 1, "hr_probability": 0.24, "rank": 1, "model_version": "statcast"}]
    )
    board = _build_probability_board(v1, statcast_predictions=statcast)

    row = _predictions_to_rows(board)[0]

    assert row["modelProbability"] == row["v1Probability"]
    assert row["v1Rank"] == 1
    assert row["statcastRank"] == 1
    assert row["statcastAvailable"] is True
    assert row["modelAgreement"] == "Consensus"
    assert "consensusScore" in row


def test_statcast_blend_fills_missing_enriched_candidates(monkeypatch) -> None:
    candidates = _candidate_frame(
        [
            {"player_id": 1, "player_name": "Ready Bat", "heuristic_probability": 0.06},
            {"player_id": 2, "player_name": "Not Ready Bat", "heuristic_probability": 0.05},
            {"player_id": 3, "player_name": "Missing Bat", "heuristic_probability": 0.04},
        ]
    )
    v1 = _prediction_frame(
        [
            {"player_id": 1, "player_name": "Ready Bat", "hr_probability": 0.11, "rank": 1},
            {"player_id": 2, "player_name": "Not Ready Bat", "hr_probability": 0.09, "rank": 2},
            {"player_id": 3, "player_name": "Missing Bat", "hr_probability": 0.07, "rank": 3},
        ]
    )

    def fake_build_torch_candidate_features(frame, as_of, *, statcast_cache, refresh_statcast, **_kwargs):
        enriched = frame.iloc[:2].copy()
        enriched["statcast_feature_ready"] = [1.0, 0.0]
        enriched["statcast_feature_quality"] = ["full", "missing"]
        return enriched

    monkeypatch.setattr(hr_predictions, "build_torch_candidate_features", fake_build_torch_candidate_features)
    monkeypatch.setattr(hr_predictions, "predict_torch_probs", lambda artifact, frame: np.array([0.22]))
    monkeypatch.setattr(hr_predictions, "apply_heuristic_blend", lambda artifact, torch_probs, heuristic_probs: torch_probs)

    statcast, gaps = _build_statcast_blend_predictions(
        candidates,
        v1,
        date(2026, 6, 26),
        torch_artifact={},
        statcast_cache=Path("/tmp/unused-statcast-cache.csv"),
    )

    assert _candidate_key_set(statcast) == _candidate_key_set(v1)
    by_player = statcast.set_index("player_id")
    assert by_player.loc[3, "hr_probability"] == 0.07
    assert "statcast_features_unavailable" in json.loads(by_player.loc[3, "quality_flags"])
    assert any("returned no row for 1 candidates" in gap for gap in gaps)


def test_statcast_health_payload_reports_coverage_and_agreement_distribution() -> None:
    statcast = _prediction_frame(
        [
            {"player_id": 1, "hr_probability": 0.25, "rank": 1, "statcast_feature_ready": 1.0},
            {
                "player_id": 2,
                "hr_probability": 0.10,
                "rank": 2,
                "statcast_feature_ready": 0.0,
                "quality_flags": json.dumps(["statcast_features_unavailable"]),
            },
        ]
    )
    board = statcast.copy()
    board["model_agreement"] = ["Consensus", "Missing Statcast"]

    health = _statcast_health_payload(
        statcast,
        board,
        enabled=True,
        artifact_loaded=True,
        artifact_path=Path("/tmp/model.pt"),
        artifact_error=None,
        gaps=[],
        min_batter_bbe=5,
        min_pitcher_bbe=5,
        allow_partial=False,
    )

    assert health["coverage"] == 0.5
    assert health["readyRows"] == 1
    assert health["totalRows"] == 2
    assert health["modelAgreement"]["Consensus"] == 1
