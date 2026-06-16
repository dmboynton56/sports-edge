import math

from src.models.world_cup import (
    Fixture,
    TeamRating,
    WorldCupRatingModel,
    WorldCupTournamentSimulator,
)
from scripts.predict_world_cup import build_payload
from scripts.sync_world_cup_to_supabase import sync_payload


def _teams():
    return [
        TeamRating(team="Alpha", group="A", fifa_rank=1, elo=2050, form_points_per_game=2.4, star_player_score=1.2),
        TeamRating(team="Beta", group="A", fifa_rank=18, elo=1780, form_points_per_game=1.8),
        TeamRating(team="Gamma", group="A", fifa_rank=42, elo=1600, form_points_per_game=1.3),
        TeamRating(team="Delta", group="A", fifa_rank=75, elo=1450, form_points_per_game=0.9),
        TeamRating(team="Epsilon", group="B", fifa_rank=7, elo=1910, form_points_per_game=2.1),
        TeamRating(team="Zeta", group="B", fifa_rank=25, elo=1690, form_points_per_game=1.6),
        TeamRating(team="Eta", group="B", fifa_rank=55, elo=1520, form_points_per_game=1.1),
        TeamRating(team="Theta", group="B", fifa_rank=95, elo=1375, form_points_per_game=0.7),
    ]


def _fixtures():
    return [
        Fixture(match_id="A1", stage="group", group="A", kickoff_utc=None, home_team="Alpha", away_team="Beta"),
        Fixture(match_id="A2", stage="group", group="A", kickoff_utc=None, home_team="Gamma", away_team="Delta"),
        Fixture(match_id="A3", stage="group", group="A", kickoff_utc=None, home_team="Alpha", away_team="Gamma"),
        Fixture(match_id="A4", stage="group", group="A", kickoff_utc=None, home_team="Beta", away_team="Delta"),
        Fixture(match_id="A5", stage="group", group="A", kickoff_utc=None, home_team="Alpha", away_team="Delta"),
        Fixture(match_id="A6", stage="group", group="A", kickoff_utc=None, home_team="Beta", away_team="Gamma"),
        Fixture(match_id="B1", stage="group", group="B", kickoff_utc=None, home_team="Epsilon", away_team="Zeta"),
        Fixture(match_id="B2", stage="group", group="B", kickoff_utc=None, home_team="Eta", away_team="Theta"),
        Fixture(match_id="B3", stage="group", group="B", kickoff_utc=None, home_team="Epsilon", away_team="Eta"),
        Fixture(match_id="B4", stage="group", group="B", kickoff_utc=None, home_team="Zeta", away_team="Theta"),
        Fixture(match_id="B5", stage="group", group="B", kickoff_utc=None, home_team="Epsilon", away_team="Theta"),
        Fixture(match_id="B6", stage="group", group="B", kickoff_utc=None, home_team="Zeta", away_team="Eta"),
    ]


def test_match_probabilities_sum_to_one():
    model = WorldCupRatingModel(_teams(), model_version="test")
    prediction = model.predict_pair("Alpha", "Delta")

    total = prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob
    assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert prediction.home_win_prob > prediction.away_win_prob
    assert prediction.home_knockout_win_prob > prediction.home_win_prob


def test_completed_group_result_affects_rank_probability():
    teams = _teams()
    fixtures = _fixtures()
    fixtures[0] = Fixture(
        match_id="A1",
        stage="group",
        group="A",
        kickoff_utc=None,
        home_team="Alpha",
        away_team="Beta",
        home_score=0,
        away_score=3,
        status="final",
    )
    model = WorldCupRatingModel(teams, model_version="test")
    simulator = WorldCupTournamentSimulator(model, fixtures)

    result = simulator.simulate(n_sims=500, seed=7)
    group_a = {row["team"]: row for row in result.group_rank_probabilities["A"]}

    assert group_a["Beta"]["rank_1"] > 0.05
    assert group_a["Alpha"]["rank_4"] < 0.5


def test_stronger_team_has_higher_champion_probability_than_weak_team():
    model = WorldCupRatingModel(_teams(), model_version="test")
    simulator = WorldCupTournamentSimulator(model, _fixtures())

    result = simulator.simulate(n_sims=800, seed=11)
    probs = {row["team"]: row for row in result.team_probabilities}

    assert result.bracket_source == "power_seeded_fallback"
    assert probs["Alpha"]["group"] == "A"
    assert probs["Alpha"]["group_prob"] == 1.0
    assert probs["Alpha"]["champion"] > probs["Theta"]["champion"]
    assert probs["Alpha"]["round_of_32"] >= probs["Delta"]["round_of_32"]


def test_payload_builder_matches_supabase_dry_run_counts():
    payload = build_payload(
        teams=_teams(),
        fixtures=_fixtures(),
        model_version="test",
        n_sims=200,
        seed=12,
        round_of_32_slots=[],
    )

    assert payload["modelVersion"] == "test"
    assert len(payload["matches"]) == len(_fixtures())
    assert len(payload["teamProbabilities"]) == len(_teams())

    counts = sync_payload(None, payload, dry_run=True)
    assert counts["matches"] == len(_fixtures())
    assert counts["predictions"] == len(_fixtures())
    assert counts["team_probabilities"] == len(_teams())
