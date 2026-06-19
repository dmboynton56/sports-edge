from datetime import date
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pga.live_leaderboard import (
    active_players_for_round_state,
    fetch_live_leaderboard,
    rounds_completed_from_leaderboard,
)
from src.pga.tournament_registry import (
    PgaRegistry,
    PgaTournament,
    event_status_for_phase,
    infer_phase,
    resolve_active_tournament,
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
