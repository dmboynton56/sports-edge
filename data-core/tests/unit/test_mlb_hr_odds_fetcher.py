from datetime import date
from types import SimpleNamespace

import pandas as pd

from src.data.mlb_hr_odds_fetcher import (
    flatten_event_hr_odds,
    match_events_to_schedule,
    normalize_name,
    fetch_day_hr_odds,
    fetch_event_hr_odds,
    slate_day_bounds,
)


def _event_payload() -> dict:
    return {
        "id": "event-1",
        "sport_key": "baseball_mlb",
        "commence_time": "2026-06-18T23:05:00Z",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "last_update": "2026-06-18T17:00:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "price": 320, "point": 0.5},
                            {"name": "Under", "description": "Aaron Judge", "price": -450, "point": 0.5},
                            {"name": "Over", "description": "Rafael Devers", "price": 410, "point": 0.5},
                        ],
                    }
                ],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "batter_home_runs",
                        "last_update": "2026-06-18T17:01:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "price": 300, "point": 0.5},
                            {"name": "Under", "description": "Aaron Judge", "price": -430, "point": 0.5},
                        ],
                    }
                ],
            },
            {
                "key": "bovada",
                "title": "Bovada",
                "markets": [
                    {
                        "key": "batter_home_runs_alternate",
                        "last_update": "2026-06-18T17:02:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Aaron Judge", "price": 350, "point": 0.5},
                            {"name": "Over", "description": "Aaron Judge", "price": 900, "point": 1.5},
                        ],
                    }
                ],
            },
        ],
    }


class _RecordingClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.request_count = 0
        self.response_meta = SimpleNamespace(
            requests_remaining=None,
            requests_used=None,
            requests_last=None,
        )

    def get(self, path: str, params: dict):
        self.calls.append((path, params))
        self.request_count += 1
        if path.endswith("/events"):
            return [_event_payload()]
        return _event_payload()


def test_normalize_name_strips_suffix():
    assert normalize_name("Jose Ramirez Jr.") == "jose ramirez"


def test_flatten_event_hr_odds_player_prop_shape():
    frame = flatten_event_hr_odds(
        _event_payload(),
        game_meta={
            "game_id": "MLB_123",
            "game_pk": 123,
            "game_date": "2026-06-18",
            "event_time": "2026-06-18T23:05:00Z",
        },
        snapshot_ts="2026-06-18T17:02:00Z",
    )

    assert len(frame) == 7
    assert set(frame["side"]) == {"Over", "Under"}
    assert set(frame["book"]) == {"draftkings", "fanduel", "bovada"}
    assert set(frame["market"]) == {"batter_home_runs", "batter_home_runs_alternate"}
    judge_over = frame[
        (frame["normalized_player_name"] == "aaron judge")
        & (frame["book"] == "draftkings")
        & (frame["side"] == "Over")
    ].iloc[0]
    assert judge_over["line"] == 0.5
    assert judge_over["price"] == 320
    assert round(judge_over["implied_probability"], 5) == round(100 / 420, 5)


def test_flatten_can_retain_only_the_standard_market():
    frame = flatten_event_hr_odds(_event_payload(), target_market="batter_home_runs")
    assert len(frame) == 5
    assert set(frame["market"]) == {"batter_home_runs"}


def test_slate_day_bounds_use_denver_and_handle_dst():
    assert slate_day_bounds(date(2026, 8, 13)) == (
        "2026-08-13T06:00:00Z",
        "2026-08-14T06:00:00Z",
    )
    assert slate_day_bounds(date(2026, 3, 8)) == (
        "2026-03-08T07:00:00Z",
        "2026-03-09T06:00:00Z",
    )


def test_event_odds_default_requests_standard_and_alternate_markets():
    client = _RecordingClient()
    fetch_event_hr_odds(client, event_id="event-1")
    assert client.calls[-1][1]["markets"] == "batter_home_runs,batter_home_runs_alternate"


def test_day_audit_reports_market_specific_coverage():
    client = _RecordingClient()
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 123,
                "game_date": pd.Timestamp("2026-06-18"),
                "game_datetime": "2026-06-18T23:05:00Z",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "home_team_abbr": "NYY",
                "away_team_abbr": "BOS",
            }
        ]
    )
    odds, audit = fetch_day_hr_odds(client, game_date=date(2026, 6, 18), schedule=schedule)

    assert len(odds) == 7
    assert audit["markets"] == ["batter_home_runs", "batter_home_runs_alternate"]
    assert audit["eventsWithMarket"] == {"batter_home_runs": 1, "batter_home_runs_alternate": 1}
    assert audit["eventsMissingMarket"] == {"batter_home_runs": [], "batter_home_runs_alternate": []}


def test_match_events_to_schedule_by_teams():
    schedule = pd.DataFrame(
        [
            {
                "game_pk": 123,
                "game_date": pd.Timestamp("2026-06-18"),
                "game_datetime": "2026-06-18T23:05:00Z",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "home_team_abbr": "NYY",
                "away_team_abbr": "BOS",
            }
        ]
    )
    matched = match_events_to_schedule([_event_payload()], schedule)
    assert matched["event-1"]["game_id"] == "MLB_123"
    assert matched["event-1"]["game_pk"] == 123
