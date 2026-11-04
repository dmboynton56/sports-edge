from datetime import date
from unittest.mock import patch

import pandas as pd

from src.data.nba_fetcher import (
    ensure_nba_schedule_covers_dates,
    fetch_nba_games_for_date,
    nba_season_start_year,
)


def test_nba_season_start_year_uses_october_boundary():
    assert nba_season_start_year("2026-11-04") == 2026
    assert nba_season_start_year("2026-06-15") == 2025
    assert nba_season_start_year(pd.Timestamp("2026-11-05", tz="UTC")) == 2026


def _espn_scoreboard_payload(*, events: list[dict]) -> dict:
    return {
        "leagues": [
            {
                "season": {
                    "year": 2027,
                    "displayName": "2026-27",
                }
            }
        ],
        "events": events,
    }


def test_fetch_nba_games_for_date_tags_start_year_from_et_tip(monkeypatch):
    payload = _espn_scoreboard_payload(
        events=[
            {
                "id": "401809111",
                "date": "2026-11-05T00:00Z",
                "status": {"type": {"name": "STATUS_SCHEDULED"}},
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "DET"}, "score": None},
                            {"homeAway": "away", "team": {"abbreviation": "PHI"}, "score": None},
                        ]
                    }
                ],
            }
        ]
    )

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr("src.data.nba_fetcher.requests.get", lambda *args, **kwargs: DummyResponse())

    games = fetch_nba_games_for_date("2026-11-04")

    assert len(games) == 1
    assert games.iloc[0]["season"] == 2026
    assert pd.Timestamp(games.iloc[0]["game_date"]).date() == date(2026, 11, 4)
    assert games.iloc[0]["home_team"] == "DET"
    assert games.iloc[0]["away_team"] == "PHI"


def test_fetch_nba_games_for_date_empty_events_does_not_use_espn_year():
    payload = _espn_scoreboard_payload(events=[])

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    with patch("src.data.nba_fetcher.requests.get", return_value=DummyResponse()):
        games = fetch_nba_games_for_date("2026-11-03")

    assert games.empty


def test_ensure_nba_schedule_covers_dates_fetches_missing_days(monkeypatch):
    existing = pd.DataFrame(
        {
            "game_id": ["old"],
            "game_date": [pd.Timestamp("2026-11-02")],
            "home_team": ["BOS"],
            "away_team": ["NYK"],
            "league": ["NBA"],
        }
    )
    fetched = pd.DataFrame(
        {
            "game_id": ["401809111"],
            "game_date": [pd.Timestamp("2026-11-04 19:00:00")],
            "home_team": ["DET"],
            "away_team": ["PHI"],
            "season": [2026],
        }
    )

    def fake_fetch(day: str, raise_on_error: bool = False):
        assert raise_on_error is True
        if day == "2026-11-04":
            return fetched
        return pd.DataFrame()

    monkeypatch.setattr("src.data.nba_fetcher.fetch_nba_games_for_date", fake_fetch)

    out = ensure_nba_schedule_covers_dates(
        existing,
        date(2026, 11, 3),
        date(2026, 11, 4),
        raise_on_error=True,
    )

    assert set(out["game_id"]) == {"old", "401809111"}
    assert date(2026, 11, 4) in set(pd.to_datetime(out["game_date"]).dt.date)
