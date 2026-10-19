from datetime import datetime, timezone

import pytest

from src.data.nfl_injury_news import (
    GameNews,
    PlayerStatus,
    PostedStarter,
    build_game_news,
    collect_live_news,
    detect_changed_games,
    fingerprint_game_news,
    is_unavailable,
    parse_espn_injuries,
    qb_ranks_from_depth_chart,
    resolve_starter,
    window_name_for,
)
from src.models.nfl_toss_up import is_nfl_toss_up, nfl_moneyline_pick


def test_parse_espn_injuries_and_skip_out_qb():
    payload = {
        "injuries": [
            {
                "displayName": "Baltimore Ravens",
                "injuries": [
                    {
                        "status": "Out",
                        "type": {"description": "out"},
                        "athlete": {
                            "id": "1",
                            "displayName": "Lamar Jackson",
                            "position": {"abbreviation": "QB"},
                        },
                    },
                    {
                        "status": "Active",
                        "type": {"description": "active"},
                        "athlete": {
                            "id": "2",
                            "displayName": "Tyler Huntley",
                            "position": {"abbreviation": "QB"},
                        },
                    },
                ],
            }
        ]
    }
    rows = parse_espn_injuries(payload)
    assert [row.name for row in rows] == ["Lamar Jackson", "Tyler Huntley"]
    assert is_unavailable("Out")
    assert not is_unavailable("questionable")


def test_resolve_starter_walks_past_out_depth_chart_qb():
    injuries = [
        PlayerStatus(team="BAL", name="Lamar Jackson", position="QB", status="out", espn_id="1"),
        PlayerStatus(team="BAL", name="Tyler Huntley", position="QB", status="active", espn_id="2"),
    ]
    sleeper = {
        "a": {"full_name": "Tyler Huntley", "team": "BAL", "gsis_id": "00-huntley", "position": "QB"},
        "b": {"full_name": "Lamar Jackson", "team": "BAL", "gsis_id": "00-lamar", "position": "QB", "injury_status": "Out"},
    }
    starter = resolve_starter(
        "BAL",
        espn_ranks=["1", "2"],
        injuries=injuries,
        sleeper_players=sleeper,
        names_by_espn_id={"1": "Lamar Jackson", "2": "Tyler Huntley"},
    )
    assert starter.name == "Tyler Huntley"
    assert starter.gsis_id == "00-huntley"
    assert starter.depth_rank == 2
    assert starter.skipped[0]["name"] == "Lamar Jackson"


def test_sleeper_fallback_when_espn_ranks_empty():
    sleeper = {
        "1": {
            "full_name": "Tyler Huntley",
            "team": "BAL",
            "gsis_id": "00-huntley",
            "position": "QB",
            "depth_chart_order": 1,
            "status": "Active",
        },
        "2": {
            "full_name": "Lamar Jackson",
            "team": "BAL",
            "gsis_id": "00-lamar",
            "position": "QB",
            "depth_chart_order": 2,
            "injury_status": "Out",
        },
    }
    starter = resolve_starter(
        "BAL",
        espn_ranks=[],
        injuries=[],
        sleeper_players=sleeper,
        names_by_espn_id={},
        source="sleeper",
    )
    assert starter.name == "Tyler Huntley"
    assert starter.source == "sleeper"


def test_qb_ranks_from_depth_chart():
    payload = {
        "items": [
            {"name": "Base 3-4 D", "positions": {"lde": {}}},
            {
                "name": "3WR 1TE",
                "positions": {
                    "qb": {
                        "athletes": [
                            {"rank": 1, "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2026/athletes/10"}},
                            {"rank": 2, "athlete": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2026/athletes/20"}},
                        ]
                    }
                },
            },
        ]
    }
    assert qb_ranks_from_depth_chart(payload) == ["10", "20"]


def _news(*, home="Huntley", away="Darnold", spread=2.5, injuries=(), home_id="h1", away_id="a1"):
    return GameNews(
        home_team="DEN",
        away_team="SEA",
        game_date="2026-10-15",
        provider_game_id="2026_06_SEA_DEN",
        supabase_game_id="11111111-1111-1111-1111-111111111111",
        home_starter=PostedStarter("DEN", home, home_id, None, 1, "active"),
        away_starter=PostedStarter("SEA", away, away_id, None, 1, "active"),
        key_injuries=tuple(injuries),
        espn_home_spread=spread,
        source="espn",
        fallback_used=False,
        fetched_at="2026-10-15T13:15:00+00:00",
    )


def test_change_detection_qb_and_line_and_stable():
    current = _news()
    previous = {
        "11111111-1111-1111-1111-111111111111": {
            "status_fingerprint": current.status_fingerprint,
            "home_starter_qb_id": "h1",
            "away_starter_qb_id": "a1",
            "espn_home_spread": 2.5,
        }
    }
    assert detect_changed_games([current], previous) == []

    huntley = _news(home="Huntley", home_id="huntley")
    lamar_prior = {
        "11111111-1111-1111-1111-111111111111": {
            "status_fingerprint": "old",
            "home_starter_qb_id": "lamar",
            "home_starter_qb_name": "Lamar Jackson",
            "away_starter_qb_id": "a1",
            "espn_home_spread": 2.5,
        }
    }
    qb = detect_changed_games([huntley], lamar_prior)
    assert qb[0].reasons == ("qb_home",)

    moved = _news(spread=4.5)
    line = detect_changed_games([moved], previous)
    assert line[0].reasons == ("line_move",)

    small = _news(spread=3.5)
    assert detect_changed_games([small], previous) == []

    injured = _news(injuries=(PlayerStatus(team="DEN", name="Courtland Sutton", position="WR", status="out"),))
    injury = detect_changed_games([injured], previous)
    assert injury[0].reasons == ("key_injury",)


def test_initial_snapshot_is_dirty():
    news = _news()
    decisions = detect_changed_games([news], {})
    assert decisions[0].reasons == ("initial",)


def test_window_name_denver():
    # 2026-10-15/18 are MDT. TNF inactives window is 23:00 UTC = 5:00 PM MT, not 21:00.
    assert window_name_for(datetime(2026, 10, 15, 13, 15, tzinfo=timezone.utc)) == "thu-am"
    assert window_name_for(datetime(2026, 10, 15, 23, 0, tzinfo=timezone.utc)) == "thu-tnf"
    assert window_name_for(datetime(2026, 10, 16, 0, 0, tzinfo=timezone.utc)) == "thu-tnf"
    assert window_name_for(datetime(2026, 10, 18, 13, 15, tzinfo=timezone.utc)) == "sun-am"
    assert window_name_for(datetime(2026, 10, 18, 15, 45, tzinfo=timezone.utc)) == "sun-inactives"
    assert window_name_for(datetime(2026, 10, 18, 19, 0, tzinfo=timezone.utc)) == "sun-late"
    assert window_name_for(datetime(2026, 10, 18, 23, 15, tzinfo=timezone.utc)) == "sun-snf"
    assert window_name_for(datetime(2026, 10, 19, 23, 0, tzinfo=timezone.utc)) == "mon-mnf"
    # MST companion: 2026-11-05 00:00 UTC = Thu 5:00 PM MST after the fall-back.
    assert window_name_for(datetime(2026, 11, 6, 0, 0, tzinfo=timezone.utc)) == "thu-tnf"


def test_toss_up_band():
    assert is_nfl_toss_up(0.494)
    assert is_nfl_toss_up(0.55) is False
    assert nfl_moneyline_pick("DEN", "SEA", 0.494) is None
    assert nfl_moneyline_pick("DEN", "SEA", 0.364) == "SEA"


def test_build_game_news_attaches_key_injuries():
    games = [{"home_team": "BAL", "away_team": "CLE", "game_date": "2026-10-18", "provider_game_id": "g1", "id": "u1"}]
    injuries = [
        PlayerStatus(team="BAL", name="Lamar Jackson", position="QB", status="out", espn_id="1", gsis_id="00-l"),
        PlayerStatus(team="BAL", name="Zay Flowers", position="WR", status="questionable", espn_id="3"),
    ]
    rows = build_game_news(
        games,
        injuries=injuries,
        depth_ranks={"BAL": ["1", "2"], "CLE": []},
        names_by_espn_id={"1": "Lamar Jackson", "2": "Tyler Huntley"},
        spreads={("CLE", "BAL"): 3.0},
        sleeper_players={
            "x": {"full_name": "Tyler Huntley", "team": "BAL", "gsis_id": "00-h", "position": "QB", "depth_chart_order": 2}
        },
        source="espn",
        fallback_used=False,
        fetched_at=datetime(2026, 10, 18, tzinfo=timezone.utc),
    )
    assert rows[0].home_starter.name == "Tyler Huntley"
    assert [row.name for row in rows[0].key_injuries] == ["Lamar Jackson"]
    assert fingerprint_game_news(rows[0]) == rows[0].status_fingerprint
    assert rows[0].provider_game_id == "g1"


def test_supabase_uuid_is_not_treated_as_provider_game_id():
    games = [{
        "home_team": "BAL",
        "away_team": "CLE",
        "game_date": "2026-10-18",
        "game_id": "11111111-1111-1111-1111-111111111111",
    }]
    rows = build_game_news(
        games,
        injuries=[],
        depth_ranks={},
        names_by_espn_id={},
        spreads={},
        sleeper_players={},
        source="espn",
        fallback_used=False,
        fetched_at=datetime(2026, 10, 18, tzinfo=timezone.utc),
    )
    assert rows[0].provider_game_id is None
    assert rows[0].supabase_game_id == "11111111-1111-1111-1111-111111111111"


def test_resolve_starter_skips_unnamed_depth_rank():
    starter = resolve_starter(
        "BAL",
        espn_ranks=["1", "2"],
        injuries=[],
        sleeper_players={},
        names_by_espn_id={"2": "Tyler Huntley"},
    )
    assert starter.name == "Tyler Huntley"
    assert starter.depth_rank == 2


def test_sleeper_skips_inactive_roster_qb():
    sleeper = {
        "1": {
            "full_name": "Practice Squad",
            "team": "BAL",
            "gsis_id": "00-g",
            "position": "QB",
            "depth_chart_order": 1,
            "status": "Inactive",
        },
        "2": {
            "full_name": "Tyler Huntley",
            "team": "BAL",
            "gsis_id": "00-h",
            "position": "QB",
            "depth_chart_order": 2,
        },
    }
    starter = resolve_starter(
        "BAL",
        espn_ranks=[],
        injuries=[],
        sleeper_players=sleeper,
        names_by_espn_id={},
        source="sleeper",
    )
    assert starter.name == "Tyler Huntley"
    assert is_unavailable("inactive")


def test_collect_live_news_fails_closed_when_espn_and_sleeper_down():
    def boom(*_args, **_kwargs):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="failed closed"):
        collect_live_news(
            [{"home_team": "BAL", "away_team": "CLE"}],
            season=2026,
            get=boom,
            sleeper_players=None,
        )
