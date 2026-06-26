from __future__ import annotations

import json

import pandas as pd

from scripts.predict_mlb_home_runs import (
    _build_probability_board,
    _filter_to_candidate_set,
    _predictions_to_rows,
)


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
