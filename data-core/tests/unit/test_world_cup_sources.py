import json

import pandas as pd

from src.data.world_cup_sources import (
    build_team_rating_inputs,
    derive_round_of_32_slots,
    extract_espn_team_form,
    normalize_fixtures_for_model,
    parse_world_football_elo_tsv,
    parse_espn_world_cup_scoreboard,
)
from src.models.world_cup import Fixture, TeamRating, WorldCupRatingModel, WorldCupTournamentSimulator
from scripts.predict_world_cup import build_payload


def test_parse_espn_scoreboard_payload_extracts_completed_fixture():
    payload = {
        "events": [
            {
                "id": "401",
                "date": "2026-06-12T01:00Z",
                "competitions": [
                    {
                        "date": "2026-06-12T01:00Z",
                        "status": {"type": {"state": "post", "completed": True, "description": "Final"}},
                        "venue": {"fullName": "Estadio Test"},
                        "altGameNote": "FIFA World Cup, Group A",
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": "2",
                                "form": "WWDLW",
                                "team": {"displayName": "United States"},
                            },
                            {
                                "homeAway": "away",
                                "score": "1",
                                "form": "LDLWW",
                                "team": {"displayName": "Canada"},
                            },
                        ],
                    }
                ],
            }
        ]
    }

    fixtures = parse_espn_world_cup_scoreboard(payload)

    assert fixtures.loc[0, "match_id"] == "401"
    assert fixtures.loc[0, "home_team"] == "United States"
    assert fixtures.loc[0, "away_team"] == "Canada"
    assert fixtures.loc[0, "status"] == "final"
    assert fixtures.loc[0, "group"] == "A"
    assert fixtures.loc[0, "home_score"] == 2
    assert fixtures.loc[0, "venue"] == "Estadio Test"
    assert json.loads(fixtures.loc[0, "raw_record"])["id"] == "401"

    forms = extract_espn_team_form(fixtures)
    usa = forms.set_index("team").loc["United States"]
    assert usa["form_points_per_game"] == 2.0


def test_parse_espn_scoreboard_payload_drops_scheduled_zero_zero_scores():
    payload = {
        "events": [
            {
                "id": "402",
                "date": "2026-06-13T01:00Z",
                "competitions": [
                    {
                        "date": "2026-06-13T01:00Z",
                        "status": {"type": {"state": "pre", "completed": False, "description": "Scheduled"}},
                        "altGameNote": "FIFA World Cup, Group B",
                        "competitors": [
                            {"homeAway": "home", "score": "0", "team": {"displayName": "Mexico"}},
                            {"homeAway": "away", "score": "0", "team": {"displayName": "Canada"}},
                        ],
                    }
                ],
            }
        ]
    }

    fixtures = parse_espn_world_cup_scoreboard(payload)

    assert fixtures.loc[0, "status"] == "scheduled"
    assert pd.isna(fixtures.loc[0, "home_score"])
    assert pd.isna(fixtures.loc[0, "away_score"])


def test_parse_espn_scoreboard_payload_infers_knockout_placeholder_stage():
    payload = {
        "events": [
            {
                "id": "403",
                "date": "2026-07-03T01:00Z",
                "season": {"slug": "group-stage"},
                "competitions": [
                    {
                        "date": "2026-07-03T01:00Z",
                        "status": {"type": {"state": "pre", "completed": False, "description": "Scheduled"}},
                        "competitors": [
                            {"homeAway": "home", "team": {"displayName": "Group A Winner"}},
                            {"homeAway": "away", "team": {"displayName": "Third Place Group C/E/F/H/I"}},
                        ],
                    }
                ],
            }
        ]
    }

    fixtures = parse_espn_world_cup_scoreboard(payload)

    assert fixtures.loc[0, "stage"] == "round_of_32"


def test_parse_world_football_elo_current_tsv_maps_country_codes():
    ratings_tsv = "1\t1\tES\t2157\t1\n2\t2\tUS\t1811\t1\n"
    teams_tsv = "ES\tSpain\nUS\tUnited States\n"

    ratings = parse_world_football_elo_tsv(ratings_tsv, teams_tsv)

    rows = ratings.set_index("team")
    assert rows.loc["Spain", "elo"] == 2157
    assert rows.loc["United States", "elo_rank"] == 2


def test_build_team_rating_inputs_blends_strength_form_history_and_players():
    teams = pd.DataFrame({"team": ["United States", "Canada", "Türkiye"], "group": ["A", "A", "B"]})
    fixtures = pd.DataFrame(
        {
            "match_id": ["1"],
            "stage": ["group"],
            "home_team": ["United States"],
            "away_team": ["Canada"],
        }
    )
    fifa = pd.DataFrame({"team": ["United States", "Canada"], "rank": [14, 31]})
    elo = pd.DataFrame({"team": ["United States", "Canada", "Turkey"], "elo": [1810, 1675, 1750]})
    recent = pd.DataFrame(
        {
            "date": ["2026-05-01", "2026-05-05"],
            "home_team": ["United States", "Canada"],
            "away_team": ["Canada", "United States"],
            "home_score": [3, 0],
            "away_score": [1, 0],
        }
    )
    history = pd.DataFrame(
        {
            "team": ["United States", "Canada"],
            "season": [2022, 2022],
            "stage": ["round_of_16", "group"],
        }
    )
    players = pd.DataFrame(
        {
            "team": ["United States", "United States", "Canada"],
            "minutes": [1800, 1200, 900],
            "goals": [12, 5, 2],
            "assists": [6, 4, 1],
            "rating": [7.6, 7.2, 6.9],
            "availability": [1.0, 1.0, 0.5],
        }
    )

    ratings = build_team_rating_inputs(
        teams=teams,
        fixtures=fixtures,
        fifa_rankings=fifa,
        world_elo=elo,
        recent_results=recent,
        world_cup_history=history,
        player_form=players,
        season=2026,
    )
    usa = ratings.set_index("team").loc["United States"]

    assert usa["group"] == "A"
    assert usa["fifa_rank"] == 14
    assert usa["elo"] == 1810
    assert usa["host_boost"] > 0
    assert usa["world_cup_experience_score"] > ratings.set_index("team").loc["Canada", "world_cup_experience_score"]
    assert usa["star_player_score"] > ratings.set_index("team").loc["Canada", "star_player_score"]
    assert ratings.set_index("team").loc["Türkiye", "elo"] == 1750


def test_build_team_rating_inputs_filters_fixture_placeholder_teams():
    fixtures = pd.DataFrame(
        {
            "match_id": ["1", "2"],
            "stage": ["group", "round_of_32"],
            "group": ["A", None],
            "home_team": ["United States", "Winner Group A"],
            "away_team": ["Canada", "Runner-up Group B"],
        }
    )

    ratings = build_team_rating_inputs(fixtures=fixtures)

    assert set(ratings["team"]) == {"United States", "Canada"}
    assert ratings.set_index("team").loc["United States", "group"] == "A"


def test_derive_round_of_32_slots_from_placeholder_fixtures():
    fixtures = pd.DataFrame(
        {
            "match_id": [str(i) for i in range(16)],
            "stage": ["round_of_32"] * 16,
            "kickoff_utc": [f"2026-07-{3 + i // 4:02d}T00:00:00Z" for i in range(16)],
            "home_team": ["Group A Winner"] + ["Group B Winner"] * 15,
            "away_team": ["Third Place Group C/E/F/H/I"] + ["Group A 2nd Place"] * 15,
        }
    )

    slots = derive_round_of_32_slots(fixtures)

    assert len(slots) == 16
    assert slots[0] == ("1A", "3*")
    assert slots[1] == ("1B", "2A")


def test_prediction_payload_preserves_completed_fixture_scores():
    teams = [
        TeamRating(team="United States", group="A", elo=1800),
        TeamRating(team="Canada", group="A", elo=1650),
        TeamRating(team="Mexico", group="B", elo=1700),
        TeamRating(team="Japan", group="B", elo=1680),
    ]
    fixtures = [
        Fixture(
            match_id="1",
            stage="group",
            group="A",
            kickoff_utc="2026-06-12T01:00:00Z",
            home_team="United States",
            away_team="Canada",
            status="final",
            home_score=2,
            away_score=1,
        ),
    ]

    model = WorldCupRatingModel(teams, model_version="test")
    row = WorldCupTournamentSimulator(model, fixtures).predict_matches()[0]
    payload = build_payload(teams=teams, fixtures=fixtures, model_version="test", n_sims=20, seed=1, round_of_32_slots=[])

    assert row["status"] == "final"
    assert row["home_score"] == 2
    assert payload["matches"][0]["away_score"] == 1


def test_normalize_fixtures_for_model_supports_external_match_id():
    raw = pd.DataFrame(
        {
            "external_match_id": ["ext-1"],
            "stage": ["group"],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )

    fixtures = normalize_fixtures_for_model(raw)

    assert fixtures.loc[0, "match_id"] == "ext-1"
    assert fixtures.loc[0, "status"] == "scheduled"
