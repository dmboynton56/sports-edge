from datetime import date

from scripts.audit_cfb_readiness import readiness
from scripts.refresh_cfb_markets import american_implied_probability, match_odds_event, normalize_team
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
