import pandas as pd

from src.models.mlb_winner_model import build_mlb_winner_features


def test_mlb_features_use_only_prior_games():
    games = pd.DataFrame(
        [
            {
                "game_pk": 1,
                "season": 2025,
                "game_date": "2025-04-01",
                "game_datetime": "2025-04-01T17:00:00",
                "home_team_id": 1,
                "home_team": "Home",
                "home_probable_pitcher_id": 101,
                "home_probable_pitcher": "Home Starter",
                "away_team_id": 2,
                "away_team": "Away",
                "away_probable_pitcher_id": 202,
                "away_probable_pitcher": "Away Starter",
                "venue_id": 10,
                "venue_name": "Park",
                "home_score": 10,
                "away_score": 2,
                "home_win": 1,
                "run_diff": 8,
            },
            {
                "game_pk": 2,
                "season": 2025,
                "game_date": "2025-04-02",
                "game_datetime": "2025-04-02T17:00:00",
                "home_team_id": 2,
                "home_team": "Away",
                "home_probable_pitcher_id": 202,
                "home_probable_pitcher": "Away Starter",
                "away_team_id": 1,
                "away_team": "Home",
                "away_probable_pitcher_id": 101,
                "away_probable_pitcher": "Home Starter",
                "venue_id": 10,
                "venue_name": "Park",
                "home_score": 3,
                "away_score": 4,
                "home_win": 0,
                "run_diff": -1,
            },
        ]
    )

    features = build_mlb_winner_features(games, min_prior_games=0)

    first = features.iloc[0]
    assert first["home_games_played"] == 0
    assert first["away_games_played"] == 0
    assert first["home_run_diff_per_game"] == 0
    assert first["home_starter_prior_starts"] == 0
    assert first["venue_prior_games"] == 0

    second = features.iloc[1]
    assert second["home_games_played"] == 1
    assert second["away_games_played"] == 1
    assert second["home_win_pct"] == 0
    assert second["away_win_pct"] == 1
    assert second["home_run_diff_per_game"] == -8
    assert second["away_run_diff_per_game"] == 8
    assert second["home_starter_prior_starts"] == 1
    assert second["away_starter_prior_starts"] == 1
    assert second["home_starter_runs_allowed_per_start"] == 10
    assert second["away_starter_runs_allowed_per_start"] == 2
    assert second["venue_prior_games"] == 1
    assert second["venue_total_runs_per_game"] == 12
