from datetime import date

from scripts.audit_cfb_readiness import readiness
from scripts.refresh_cfb_markets import american_implied_probability, match_odds_event, normalize_team
from scripts.train_cfb_market_model import (
    ESPN_SCOREBOARD_LIMIT,
    fetch_games,
    game_date_denver,
    month_tokens,
    scoreboard_params,
)
from src.models.cfb_market import normal_probability_above, parse_espn_scoreboard


def test_parse_espn_scoreboard_preserves_pregame_contract():
    payload = {
        "events": [{
            "id": "401",
            "date": "2026-09-03T22:00Z",
            "season": {"year": 2026},
            "week": {"number": 1},
            "status": {"type": {"completed": False, "name": "STATUS_SCHEDULED"}},
            "competitions": [{
                "neutralSite": False,
                "competitors": [
                    {"homeAway": "home", "team": {"id": "1", "displayName": "Rutgers Scarlet Knights"}},
                    {"homeAway": "away", "team": {"id": "2", "displayName": "Massachusetts Minutemen"}},
                ],
            }],
        }],
    }
    game = parse_espn_scoreboard(payload)[0]
    assert game["event_id"] == "401"
    assert game["home_team"] == "Rutgers Scarlet Knights"
    assert game["away_score"] is None
    assert game["completed"] is False


def test_event_matching_handles_umass_alias():
    game = {
        "game_time_utc": "2026-09-03T22:00Z",
        "home_team": "Rutgers Scarlet Knights",
        "away_team": "Massachusetts Minutemen",
    }
    event = {
        "id": "odds-1",
        "commence_time": "2026-09-03T22:02:03Z",
        "home_team": "Rutgers Scarlet Knights",
        "away_team": "UMass Minutemen",
    }
    assert normalize_team("Massachusetts Minutemen") == normalize_team("UMass Minutemen")
    assert match_odds_event(game, [event]) == event


def test_probability_and_implied_odds_math():
    assert 0.49 < normal_probability_above(10, 10, 5) < 0.51
    assert american_implied_probability(150) == 0.4
    assert american_implied_probability(-150) == 0.6


def test_readiness_requires_fresh_predictions_and_market_coverage():
    summary = {
        "scheduled_games": 8,
        "model_supportable_outcomes": True,
        "predicted_games": 8,
        "fresh_prediction_games": 8,
        "fresh_moneyline_games": 6,
        "fresh_spread_games": 8,
        "fresh_total_games": 8,
        "stale_recommendations": 0,
        "guardrail_violations": 0,
    }
    assert readiness(summary)
    assert not readiness({**summary, "fresh_total_games": 7})


def test_month_tokens_cover_inclusive_range_without_hyphenated_dates():
    assert month_tokens(date(2024, 8, 1), date(2024, 9, 15)) == ["202408", "202409"]
    assert month_tokens(date(2024, 12, 15), date(2025, 1, 31)) == ["202412", "202501"]
    assert month_tokens(date(2026, 9, 16), date(2026, 9, 19)) == ["202609"]
    params = scoreboard_params("202408")
    assert params["dates"] == "202408"
    assert "-" not in str(params["dates"])
    assert params["limit"] == ESPN_SCOREBOARD_LIMIT == 200
    assert params["limit"] != 1000


def test_game_date_denver_uses_mountain_calendar_day():
    # 2024-09-16 02:00 UTC is still 2024-09-15 evening in Denver.
    assert game_date_denver({"game_time_utc": "2024-09-16T02:00:00Z"}) == date(2024, 9, 15)
    assert game_date_denver({"game_time_utc": "2024-09-16T12:00:00Z"}) == date(2024, 9, 16)


def _espn_event(event_id: str, when: str) -> dict:
    return {
        "id": event_id,
        "date": when,
        "season": {"year": 2024},
        "week": {"number": 1},
        "status": {"type": {"completed": True, "name": "STATUS_FINAL"}},
        "competitions": [{
            "neutralSite": False,
            "competitors": [
                {"homeAway": "home", "team": {"id": "1", "displayName": "Home"}, "score": "21"},
                {"homeAway": "away", "team": {"id": "2", "displayName": "Away"}, "score": "17"},
            ],
        }],
    }


def test_fetch_games_queries_yyyy_mm_months_and_filters_window(monkeypatch):
    payloads = {
        "202408": {"events": [
            _espn_event("aug-31", "2024-08-31T19:00Z"),
            _espn_event("sep-spill", "2024-09-01T19:00Z"),
        ]},
        "202409": {"events": [
            _espn_event("sep-01", "2024-09-01T19:00Z"),
            _espn_event("sep-14", "2024-09-14T19:00Z"),
            _espn_event("sep-20", "2024-09-20T19:00Z"),
        ]},
    }
    calls: list[dict] = []

    class FakeResponse:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            calls.append({"url": url, "params": dict(params), "timeout": timeout})
            return FakeResponse(payloads[params["dates"]])

    monkeypatch.setattr("scripts.train_cfb_market_model.requests.Session", FakeSession)

    games = fetch_games(date(2024, 8, 1), date(2024, 9, 15))
    ids = {game["event_id"] for game in games}

    assert [call["params"]["dates"] for call in calls] == ["202408", "202409"]
    assert all("-" not in call["params"]["dates"] for call in calls)
    assert all(call["params"]["limit"] == 200 for call in calls)
    assert ids == {"aug-31", "sep-spill", "sep-01", "sep-14"}
    assert "sep-20" not in ids
