import pandas as pd

from src.data.mlb_hr_odds_fetcher import (
    flatten_event_hr_odds,
    match_events_to_schedule,
    normalize_name,
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
        ],
    }


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

    assert len(frame) == 5
    assert set(frame["side"]) == {"Over", "Under"}
    assert set(frame["book"]) == {"draftkings", "fanduel"}
    judge_over = frame[
        (frame["normalized_player_name"] == "aaron judge")
        & (frame["book"] == "draftkings")
        & (frame["side"] == "Over")
    ].iloc[0]
    assert judge_over["line"] == 0.5
    assert judge_over["price"] == 320
    assert round(judge_over["implied_probability"], 5) == round(100 / 420, 5)


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
