from datetime import date, datetime, timezone
import json

import pandas as pd

from scripts.refresh_world_cup import (
    DEFAULT_MODEL_VERSION,
    build_bigquery_frames,
    build_output_paths,
    tournament_window_for_season,
)


def test_tournament_window_for_2026_uses_official_fixture_window():
    assert tournament_window_for_season(2026) == (date(2026, 6, 11), date(2026, 7, 19))


def test_build_output_paths_sanitizes_model_version(tmp_path):
    paths = build_output_paths(tmp_path, "world cup/v0 live")

    assert paths.teams_csv == tmp_path / "world_cup_team_ratings.csv"
    assert paths.predictions_json == tmp_path / "world_cup_predictions_world_cup_v0_live.json"

    default_paths = build_output_paths(tmp_path, "///")
    assert default_paths.predictions_json == tmp_path / f"world_cup_predictions_{DEFAULT_MODEL_VERSION}.json"


def test_build_bigquery_frames_maps_payload_to_warehouse_tables():
    run_ts = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    fixtures_raw = pd.DataFrame(
        {
            "match_id": ["401"],
            "season": [2026],
            "tournament": ["FIFA World Cup"],
            "stage": ["group"],
            "group": ["A"],
            "kickoff_utc": ["2026-06-12T01:00:00Z"],
            "home_team": ["United States"],
            "away_team": ["Canada"],
            "status": ["final"],
            "home_score": [2],
            "away_score": [1],
            "neutral_site": [True],
            "venue": ["Estadio Test"],
            "source": ["espn_scoreboard"],
            "raw_record": ['{"id":"401"}'],
        }
    )
    team_ratings = pd.DataFrame(
        {
            "team": ["United States"],
            "group": ["A"],
            "fifa_rank": [14],
            "elo": [1810],
            "form_points_per_game": [2.0],
            "form_goal_diff_per_game": [0.6],
            "world_cup_experience_score": [3.2],
            "star_player_score": [1.1],
            "host_boost": [0.18],
            "market_rating": [0.2],
        }
    )
    payload = {
        "season": 2026,
        "modelVersion": "world-cup-v0-live",
        "updatedAt": "2026-06-15T12:00:00Z",
        "simulations": 50000,
        "bracketSource": "configured_round_of_32_slots",
        "matches": [
            {
                "match_id": "401",
                "stage": "group",
                "group": "A",
                "kickoff_utc": "2026-06-12T01:00:00Z",
                "home_team": "United States",
                "away_team": "Canada",
                "home_win_prob": 0.55,
                "draw_prob": 0.25,
                "away_win_prob": 0.20,
                "home_knockout_win_prob": 0.63,
                "away_knockout_win_prob": 0.37,
                "projected_home_goals": 1.4,
                "projected_away_goals": 0.9,
                "prediction_ts": "2026-06-15T12:00:00Z",
            }
        ],
        "teamProbabilities": [
            {
                "team": "United States",
                "group": "A",
                "rating": 0.41,
                "group_prob": 1.0,
                "round_of_32": 0.86,
                "round_of_16": 0.49,
                "quarterfinal": 0.24,
                "semifinal": 0.11,
                "final": 0.05,
                "champion": 0.02,
            }
        ],
        "groupRankProbabilities": {
            "A": [
                {
                    "team": "United States",
                    "rank_1": 0.44,
                    "rank_2": 0.29,
                    "rank_3": 0.18,
                    "rank_4": 0.09,
                }
            ]
        },
    }

    frames = build_bigquery_frames(
        fixtures_raw=fixtures_raw,
        team_ratings=team_ratings,
        payload=payload,
        world_elo=pd.DataFrame({"team": ["United States"], "elo": [1810], "elo_rank": [20]}),
        fifa_rankings=pd.DataFrame({"team": ["United States"], "rank": [14]}),
        run_ts=run_ts,
    )

    raw_fixture = frames[("sports_edge_raw", "raw_wc_fixtures")].iloc[0]
    assert raw_fixture["external_match_id"] == "401"
    assert raw_fixture["group_name"] == "A"
    assert raw_fixture["ingested_at"] == run_ts

    rating = frames[("sports_edge_curated", "wc_team_ratings")].iloc[0]
    assert rating["group_name"] == "A"
    assert rating["model_version"] == "world-cup-v0-live"

    prediction = frames[("sports_edge_curated", "wc_match_predictions")].iloc[0]
    assert prediction["model_name"] == "sports_edge_world_cup"
    assert prediction["home_win_prob"] == 0.55

    team_probability = frames[("sports_edge_curated", "wc_team_probabilities")].iloc[0]
    assert json.loads(team_probability["group_rank_probs"])["rank_1"] == 0.44
    assert team_probability["champion_prob"] == 0.02

    assert len(frames[("sports_edge_raw", "raw_world_elo")]) == 1
    assert len(frames[("sports_edge_raw", "raw_fifa_rankings")]) == 1
    assert len(frames[("sports_edge_curated", "wc_simulation_runs")]) == 1
