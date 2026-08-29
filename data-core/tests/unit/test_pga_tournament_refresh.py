from datetime import date
from datetime import date
from pathlib import Path
import subprocess
import sys

import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pga.live_leaderboard import (
    active_players_for_round_state,
    EspnScoreboardError,
    fetch_core_event,
    fetch_leaderboard_event,
    fetch_live_leaderboard,
    fetch_scoreboard,
    rounds_completed_from_leaderboard,
)
from src.pga import live_leaderboard as live_leaderboard_module
from src.pga.tournament_registry import (
    PgaRegistry,
    PgaTournament,
    event_status_for_phase,
    infer_phase,
    resolve_active_tournament,
)
from scripts.refresh_pga_tournament import (
    _field_fetch_command,
    _live_event_unmatched,
    _should_preserve_last_good_output,
    _should_skip_leaderboard,
    ensure_field,
)


def _tournament(key: str, priority: int = 0) -> PgaTournament:
    return PgaTournament(
        key=key,
        name="U.S. Open Championship",
        season=2026,
        course="Shinnecock Hills Golf Club",
        par=70,
        start_date=date(2026, 6, 18),
        end_date=date(2026, 6, 21),
        espn_match=("U.S. Open",),
        cut_size=2,
        cut_after_round=2,
        priority=priority,
    )


def test_field_fetch_command_dispatches_espn_source():
    tournament = PgaTournament(
        key="john_deere_classic_2026",
        name="John Deere Classic",
        season=2026,
        course="TPC Deere Run",
        par=71,
        start_date=date(2026, 7, 2),
        end_date=date(2026, 7, 5),
        espn_event_id="401811954",
        field_source="espn",
    )

    cmd = _field_fetch_command(tournament)

    assert cmd is not None
    assert "scripts/fetch_pga_field.py" in cmd
    assert "--event-id" in cmd
    assert "401811954" in cmd


def test_leaderboard_is_optional_outside_live_phase():
    assert _should_skip_leaderboard("pre", False) is True
    assert _should_skip_leaderboard("post", False) is True
    assert _should_skip_leaderboard("live", False) is False
    assert _should_skip_leaderboard("live", True) is True


def test_scoreboard_does_not_retry_non_retryable_espn_403(monkeypatch):
    response = requests.Response()
    response.status_code = 403
    error = requests.exceptions.HTTPError("403 Forbidden", response=response)
    calls = 0

    def fake_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(live_leaderboard_module.requests, "get", fake_get)

    with pytest.raises(EspnScoreboardError):
        fetch_scoreboard(max_attempts=3, backoff_seconds=0)

    assert calls == 1


def test_missing_pretournament_field_is_deferred(monkeypatch):
    tournament = _tournament("field_deferred")
    monkeypatch.setattr("scripts.refresh_pga_tournament._field_fetch_command", lambda _: ["fetch-field"])

    def fail_fetch(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["fetch-field"])

    monkeypatch.setattr("scripts.refresh_pga_tournament._run", fail_fetch)

    assert ensure_field(
        tournament,
        force=False,
        skip_fetch=False,
        dry_run=False,
        allow_unavailable=True,
    ) is False


def test_live_event_unmatched_detects_registry_drift():
    registry = PgaRegistry(season=2026, tournaments=(_tournament("us_open_2026"),))
    scoreboard = {
        "events": [
            {
                "name": "John Deere Classic",
                "competitions": [
                    {"status": {"type": {"state": "in", "completed": False}}, "competitors": []}
                ],
            }
        ]
    }

    assert _live_event_unmatched(registry, scoreboard) == "John Deere Classic"


def test_resolve_active_tournament_uses_pre_window_and_priority():
    low = _tournament("low", priority=1)
    high = _tournament("high", priority=100)
    registry = PgaRegistry(season=2026, tournaments=(low, high))

    assert resolve_active_tournament(registry, as_of=date(2026, 6, 16)).key == "high"
    assert resolve_active_tournament(registry, tournament_key="low", as_of=date(2026, 6, 16)).key == "low"
    assert resolve_active_tournament(registry, as_of=date(2026, 5, 1)) is None


def test_resolve_active_tournament_prefers_scoreboard_match_within_window():
    us_open = _tournament("us_open_2026", priority=100)
    other = PgaTournament(
        key="travelers_2026",
        name="Travelers Championship",
        season=2026,
        course="TPC River Highlands",
        par=70,
        start_date=date(2026, 6, 18),
        end_date=date(2026, 6, 21),
        espn_match=("Travelers Championship",),
        priority=1,
    )
    scoreboard = {"events": [{"name": "Travelers Championship", "competitions": [{"status": {"type": {}}, "competitors": []}]}]}
    registry = PgaRegistry(season=2026, tournaments=(us_open, other))

    assert resolve_active_tournament(registry, as_of=date(2026, 6, 19), scoreboard=scoreboard).key == "travelers_2026"


def test_infer_phase_maps_pre_live_post_and_completed_leaderboard():
    tournament = _tournament("us_open_2026")

    assert infer_phase(tournament, as_of=date(2026, 6, 17)) == "pre"
    assert infer_phase(tournament, as_of=date(2026, 6, 19)) == "live"
    assert infer_phase(tournament, as_of=date(2026, 6, 22)) == "post"
    assert (
        infer_phase(
            tournament,
            as_of=date(2026, 6, 21),
            leaderboard={"isCompleted": True, "currentRound": 4},
        )
        == "post"
    )
    assert event_status_for_phase("live") == "in_progress"


def test_fetch_live_leaderboard_selects_matched_espn_event():
    scoreboard = {
        "events": [
            {
                "name": "Travelers Championship",
                "competitions": [{"status": {"period": 1, "type": {"description": "In Progress"}}, "competitors": []}],
            },
            {
                "name": "U.S. Open Championship",
                "date": "2026-06-18T12:00Z",
                "competitions": [
                    {
                        "status": {"period": 2, "type": {"description": "In Progress", "state": "in", "completed": False}},
                        "competitors": [
                            {
                                "athlete": {"displayName": "Test Player"},
                                "score": "-3",
                                "linescores": [{"period": 1, "value": 67}],
                                "status": {"displayThru": "4", "type": {"description": "Active"}},
                            }
                        ],
                    }
                ],
            },
        ]
    }

    leaderboard = fetch_live_leaderboard(espn_match=("US Open",), scoreboard=scoreboard)

    assert leaderboard is not None
    assert leaderboard["event"] == "U.S. Open Championship"
    assert leaderboard["players"][0]["player"] == "Test Player"
    assert leaderboard["players"][0]["positionDisplay"] == "1"
    assert leaderboard["players"][0]["roundHoles"][1] == 0


def test_core_event_adapter_normalizes_status_scores_rounds_and_names(monkeypatch):
    root_url = "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga/events/42?lang=en&region=us"
    payloads = {
        root_url: {
            "id": "42",
            "name": "Core Open",
            "date": "2026-06-18T04:00Z",
            "endDate": "2026-06-21T04:00Z",
            "competitions": [
                {
                    "id": "42",
                    "date": "2026-06-18T04:00Z",
                    "status": {"period": 2, "type": {"state": "in", "completed": False, "description": "In Progress"}},
                    "competitors": [
                        {
                            "id": "1",
                            "athlete": {"$ref": "https://core.test/athlete/1"},
                            "status": {"$ref": "https://core.test/status/1"},
                            "score": {"$ref": "https://core.test/score/1"},
                            "linescores": {"$ref": "https://core.test/lines/1"},
                        },
                        {
                            "id": "2",
                            "athlete": {"$ref": "https://core.test/athlete/2"},
                            "status": {"$ref": "https://core.test/status/2"},
                            "score": {"$ref": "https://core.test/score/2"},
                            "linescores": {"$ref": "https://core.test/lines/2"},
                        },
                    ],
                }
            ],
        },
        "https://core.test/athlete/1": {"displayName": "First Player", "fullName": "First Player"},
        "https://core.test/athlete/2": {"displayName": "Second Player", "fullName": "Second Player"},
        "https://core.test/status/1": {
            "thru": 18,
            "position": {"displayName": "1"},
            "type": {"state": "in", "completed": False, "description": "Active"},
        },
        "https://core.test/status/2": {
            "thru": 18,
            "position": {"displayName": "2"},
            "type": {"state": "in", "completed": False, "description": "Active"},
        },
        "https://core.test/score/1": {"displayValue": "-3", "completedRoundsDisplayValue": "-3"},
        "https://core.test/score/2": {"displayValue": "E", "completedRoundsDisplayValue": "E"},
        "https://core.test/lines/1": {"items": [{"period": 1, "value": 67, "linescores": [{}] * 18}]},
        "https://core.test/lines/2": {"items": [{"period": 1, "value": 70, "linescores": [{}] * 18}]},
    }

    monkeypatch.setattr(live_leaderboard_module, "_fetch_json", lambda url, *, timeout: payloads[url])

    event = fetch_core_event("42", min_players=2, max_workers=2)
    leaderboard = live_leaderboard_module.parse_leaderboard_event(event)

    assert event["name"] == "Core Open"
    assert leaderboard is not None
    assert leaderboard["currentRound"] == 2
    assert leaderboard["players"][0]["player"] == "First Player"
    assert leaderboard["players"][0]["toPar"] == "-3"
    assert leaderboard["players"][0]["rounds"] == {1: 67}
    assert leaderboard["players"][0]["thru"] == "18"
    assert rounds_completed_from_leaderboard(leaderboard) == 1


def test_site_403_falls_back_to_core_event(monkeypatch):
    core_event = {"name": "Core Open", "competitions": [{"competitors": [{"athlete": {"displayName": "Player"}}]}]}
    core_leaderboard = {"event": "Core Open", "players": [{"player": "Player"}]}

    monkeypatch.setattr(
        live_leaderboard_module,
        "fetch_scoreboard",
        lambda **kwargs: (_ for _ in ()).throw(EspnScoreboardError("403 Forbidden")),
    )
    monkeypatch.setattr(
        live_leaderboard_module,
        "fetch_core_leaderboard",
        lambda event_id, **kwargs: (core_event, core_leaderboard),
    )

    event, leaderboard = fetch_leaderboard_event(espn_match=("Core Open",), espn_event_id="42")

    assert event == core_event
    assert leaderboard == core_leaderboard


def test_total_espn_outage_preserves_last_good_output():
    assert _should_preserve_last_good_output("live", None) is True
    assert _should_preserve_last_good_output("post", None) is True
    assert _should_preserve_last_good_output("pre", None) is False
    assert _should_preserve_last_good_output("live", {"players": []}) is False


def test_rounds_completed_detection_handles_in_progress_and_complete_statuses():
    full_round_players = [
        {"player": "A", "toPar": "-2", "rounds": {1: 68, 2: 70, 3: 71, 4: 72}, "roundHoles": {1: 18, 2: 18, 3: 18, 4: 18}},
        {"player": "B", "toPar": "+1", "rounds": {1: 71, 2: 70, 3: 70, 4: 72}, "roundHoles": {1: 18, 2: 18, 3: 18, 4: 18}},
    ]
    assert rounds_completed_from_leaderboard(
        {"currentRound": 2, "status": "In Progress", "players": full_round_players},
        total_rounds=4,
    ) == 1
    assert rounds_completed_from_leaderboard(
        {"currentRound": 2, "status": "Round 2 Complete", "players": full_round_players},
        total_rounds=4,
    ) == 2
    assert rounds_completed_from_leaderboard(
        {"currentRound": 4, "isCompleted": True, "players": full_round_players},
        total_rounds=4,
    ) == 4


def test_rounds_completed_waits_for_full_completed_round_scores():
    leaderboard = {
        "currentRound": 2,
        "status": "In Progress",
        "players": [
            {"player": "A", "toPar": "-2", "rounds": {1: 68}, "roundHoles": {1: 18}},
            {"player": "B", "toPar": "+7", "rounds": {1: 46}, "roundHoles": {1: 10}},
        ],
    }

    assert rounds_completed_from_leaderboard(leaderboard, total_rounds=4) == 0


def test_cut_is_applied_only_after_configured_cut_round():
    players = [
        {"player": "A", "toPar": "-3", "totalStrokes": 137},
        {"player": "B", "toPar": "-1", "totalStrokes": 139},
        {"player": "C", "toPar": "-1", "totalStrokes": 139},
        {"player": "D", "toPar": "+2", "totalStrokes": 142},
    ]

    active_r1, out_r1, cut_line_r1, cut_applied_r1 = active_players_for_round_state(
        players,
        rounds_completed=1,
        cut_after_round=2,
        cut_size=2,
    )
    assert [row["player"] for row in active_r1] == ["A", "B", "C", "D"]
    assert out_r1 == []
    assert cut_line_r1 is None
    assert cut_applied_r1 is False

    active_r2, out_r2, cut_line_r2, cut_applied_r2 = active_players_for_round_state(
        players,
        rounds_completed=2,
        cut_after_round=2,
        cut_size=2,
    )
    assert [row["player"] for row in active_r2] == ["A", "B", "C"]
    assert [row["player"] for row in out_r2] == ["D"]
    assert cut_line_r2 == -1
    assert cut_applied_r2 is True


def test_no_cut_event_keeps_all_players_active_through_all_rounds():
    """Regression test: no-cut events (like Tour Championship) should keep all players active after any round."""
    players = [
        {"player": "Player 1", "toPar": "-10", "totalStrokes": 130},
        {"player": "Player 2", "toPar": "-5", "totalStrokes": 135},
        {"player": "Player 30", "toPar": "+8", "totalStrokes": 148},
    ]
    
    # No-cut event: cut_after_round > total_rounds (e.g. 999)
    for round_no in [1, 2, 3]:
        active, out, cut_line, cut_applied = active_players_for_round_state(
            players,
            rounds_completed=round_no,
            cut_after_round=999,
            cut_size=30,
        )
        assert len(active) == 3, f"Round {round_no}: all players should remain active"
        assert len(out) == 0, f"Round {round_no}: no players should be cut"
        assert cut_line is None, f"Round {round_no}: no cut line should exist"
        assert cut_applied is False, f"Round {round_no}: cut should never be applied"
