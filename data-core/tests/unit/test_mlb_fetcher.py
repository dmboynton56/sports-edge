from datetime import date, timezone

import pandas as pd

from src.data.mlb_fetcher import MLB_SLATE_GAME_TYPES, fetch_mlb_schedule, mlb_schedule_to_games_df
from src.utils.supabase_pg import upsert_games_pg
from tests.unit.test_supabase_pg import FakeConnection


def test_mlb_schedule_to_games_df_keeps_official_date_for_late_utc_kickoff():
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "game_date": "2026-09-04",
                "game_datetime": "2026-09-05T02:10:00Z",
                "home_team": "San Diego Padres",
                "away_team": "New York Yankees",
                "home_probable_pitcher": "Home Arm",
                "away_probable_pitcher": "Away Arm",
            }
        ]
    )

    games = mlb_schedule_to_games_df(schedule)

    assert len(games) == 1
    assert games.loc[0, "league"] == "MLB"
    assert games.loc[0, "season"] == 2026
    assert games.loc[0, "game_date"] == date(2026, 9, 4)
    assert games.loc[0, "home_team"] == "San Diego Padres"
    assert games.loc[0, "away_team"] == "New York Yankees"
    assert games.loc[0, "home_probable_pitcher"] == "Home Arm"
    kickoff = pd.Timestamp(games.loc[0, "game_time_utc"])
    assert kickoff.tzinfo is not None
    assert kickoff.astimezone(timezone.utc) == pd.Timestamp("2026-09-05T02:10:00Z")


def test_mlb_schedule_to_games_df_accepts_scored_research_rows():
    scored = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-09-04",
                "game_datetime": "2026-09-04T23:10:00",
                "home_team": "New York Mets",
                "away_team": "San Francisco Giants",
                "home_win_prob": 0.55,
            }
        ]
    )

    games = mlb_schedule_to_games_df(scored, season=2026)

    assert list(games["season"]) == [2026]
    assert games.loc[0, "game_date"] == date(2026, 9, 4)
    assert games.loc[0, "week"] is None


def test_mlb_schedule_to_games_df_empty_input():
    games = mlb_schedule_to_games_df(pd.DataFrame())
    assert games.empty
    assert "game_time_utc" in games.columns


def test_mlb_schedule_to_games_df_upserts_missing_serving_rows():
    schedule = pd.DataFrame(
        [
            {
                "season": 2026,
                "game_date": "2026-09-04",
                "game_datetime": "2026-09-05T02:10:00Z",
                "home_team": "San Diego Padres",
                "away_team": "New York Yankees",
                "home_probable_pitcher": "Home Arm",
                "away_probable_pitcher": "Away Arm",
            }
        ]
    )
    conn = FakeConnection(
        columns=[
            ("id",),
            ("league",),
            ("season",),
            ("week",),
            ("game_date",),
            ("home_team",),
            ("away_team",),
            ("game_time_utc",),
            ("book_spread",),
            ("home_probable_pitcher",),
            ("away_probable_pitcher",),
            ("created_at",),
        ],
        existing_game_id=None,
    )

    game_ids = upsert_games_pg(conn, mlb_schedule_to_games_df(schedule))

    assert len(conn.inserts) == 1
    assert conn.inserts[0][0] == "MLB"
    assert conn.inserts[0][3] == date(2026, 9, 4)
    assert conn.inserts[0][4] == "San Diego Padres"
    assert game_ids == {"2026-09-04_New York Yankees_San Diego Padres": "inserted-game-id"}


class _FakeMlbResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _alcs_payload():
    return {
        "dates": [
            {
                "date": "2026-10-16",
                "games": [
                    {
                        "gamePk": 778123,
                        "season": "2026",
                        "gameType": "L",
                        "officialDate": "2026-10-16",
                        "gameDate": "2026-10-17T00:08:00Z",
                        "status": {"codedGameState": "S", "detailedState": "Scheduled"},
                        "venue": {"id": 5, "name": "Progressive Field"},
                        "teams": {
                            "home": {
                                "team": {
                                    "id": 114,
                                    "name": "Cleveland Guardians",
                                    "abbreviation": "CLE",
                                }
                            },
                            "away": {
                                "team": {
                                    "id": 139,
                                    "name": "Tampa Bay Rays",
                                    "abbreviation": "TB",
                                }
                            },
                        },
                    }
                ],
            }
        ]
    }


def test_fetch_mlb_schedule_includes_postseason_game_types(monkeypatch):
    requested = []

    def fake_get(url, params=None, timeout=None):
        requested.append(params["gameType"])
        if params["gameType"] != "L":
            return _FakeMlbResponse({"dates": []})
        return _FakeMlbResponse(_alcs_payload())

    monkeypatch.setattr("src.data.mlb_fetcher.requests.get", fake_get)

    frame = fetch_mlb_schedule(
        2026,
        start_date="2026-10-16",
        end_date="2026-10-16",
        include_uncompleted=True,
    )

    assert requested == list(MLB_SLATE_GAME_TYPES)
    assert len(frame) == 1
    assert frame.loc[0, "game_type"] == "L"
    assert int(frame.loc[0, "game_pk"]) == 778123
    assert frame.loc[0, "home_team"] == "Cleveland Guardians"
    assert frame.loc[0, "away_team"] == "Tampa Bay Rays"


def test_fetch_mlb_schedule_regular_season_only_does_not_request_postseason(monkeypatch):
    requested = []

    def fake_get(url, params=None, timeout=None):
        requested.append(params["gameType"])
        return _FakeMlbResponse({"dates": []})

    monkeypatch.setattr("src.data.mlb_fetcher.requests.get", fake_get)

    frame = fetch_mlb_schedule(
        2026,
        game_types=("R",),
        start_date="2026-10-16",
        end_date="2026-10-16",
        include_uncompleted=True,
    )

    assert requested == ["R"]
    assert frame.empty
