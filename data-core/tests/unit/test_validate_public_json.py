from __future__ import annotations

from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from validate_public_json import validate_payload  # noqa: E402


def test_validate_mlb_home_runs_rejects_stale_game_date() -> None:
    payload = {
        "generatedAt": "2026-06-18T13:00:00+00:00",
        "predictions": [
            {"gameDate": "2026-06-18", "eventTime": "2026-06-19T01:40:00+00:00"},
            {"gameDate": "2026-06-17", "eventTime": "2026-06-18T01:40:00+00:00"},
        ],
    }

    failures = validate_payload(Path("mlb_home_runs.json"), payload, mlb_hr_date="2026-06-18")

    assert any("expected only 2026-06-18" in failure for failure in failures)


def test_validate_mlb_home_runs_requires_explicit_game_date() -> None:
    payload = {
        "generatedAt": "2026-06-18T13:00:00+00:00",
        "predictions": [
            {"eventTime": "2026-06-18T23:10:00+00:00"},
        ],
    }

    failures = validate_payload(Path("mlb_home_runs.json"), payload, mlb_hr_date="2026-06-18")

    assert any("missing gameDate" in failure for failure in failures)


def test_validate_mlb_home_runs_accepts_current_game_date() -> None:
    payload = {
        "generatedAt": "2026-06-18T13:00:00+00:00",
        "predictions": [
            {"gameDate": "2026-06-18", "eventTime": "2026-06-19T01:40:00+00:00"},
        ],
    }

    assert validate_payload(Path("mlb_home_runs.json"), payload, mlb_hr_date="2026-06-18") == []
