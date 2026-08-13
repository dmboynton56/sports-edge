import json

from scripts.validate_mlb_hr_board_input import validate


def test_validate_mlb_hr_board_input_rejects_duplicate_invalid_rows(tmp_path):
    path = tmp_path / "mlb.json"
    path.write_text(json.dumps({
        "models": {"mlb-hr-v1": {"predictions": [
            {"gameId": "g1", "playerId": "p1", "gameDate": "2026-08-11", "eventTime": "2026-08-11T23:00:00Z", "modelProbability": 0.2},
            {"gameId": "g1", "playerId": "p1", "gameDate": "2026-08-10", "eventTime": "bad", "modelProbability": 1.2},
        ]}},
    }), encoding="utf-8")
    failures = validate(path, "2026-08-11")
    assert any("duplicate" in failure for failure in failures)
    assert any("model probability" in failure for failure in failures)
    assert any("eventTime" in failure for failure in failures)


def test_validate_mlb_hr_board_input_allows_confirmed_empty_slate(tmp_path):
    path = tmp_path / "mlb.json"
    path.write_text(json.dumps({"models": {"mlb-hr-v1": {"predictions": []}}}), encoding="utf-8")
    assert validate(path, "2026-08-11", allow_empty=True) == []
