from __future__ import annotations

from scripts.sync_player_markets_to_supabase import _comparison_fields, _comparison_lookup


def test_comparison_fields_use_probability_board_overlay_for_each_model() -> None:
    board_prediction = {
        "gameId": "MLB_1",
        "playerId": "123",
        "v1Probability": 0.21,
        "v1Rank": 2,
        "statcastProbability": 0.25,
        "statcastRank": 1,
        "statcastAvailable": True,
        "modelAgreement": "Statcast boost",
        "consensusScore": 124,
        "marketSignalRank": 121,
    }
    payload = {"predictions": [board_prediction]}
    comparison = _comparison_lookup(payload)[("MLB_1", "123")]
    statcast_pred = {
        "gameId": "MLB_1",
        "playerId": "123",
        "modelProbability": 0.25,
        "rank": 1,
        "qualityFlags": [],
    }

    fields = _comparison_fields(
        statcast_pred,
        comparison,
        "mlb-hr-torch-statcast-v1-blend",
    )

    assert fields == (0.21, 2, 0.25, 1, True, "Statcast boost", 124, 121)


def test_comparison_fields_marks_statcast_unavailable_when_external_features_missing() -> None:
    statcast_pred = {
        "gameId": "MLB_1",
        "playerId": "123",
        "modelProbability": 0.21,
        "rank": 1,
        "qualityFlags": ["statcast_features_unavailable"],
    }

    fields = _comparison_fields(
        statcast_pred,
        None,
        "mlb-hr-torch-statcast-v1-blend",
    )

    assert fields[2] == 0.21
    assert fields[3] == 1
    assert fields[4] is False
    assert fields[5] == "Missing Statcast"
