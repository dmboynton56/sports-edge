import pandas as pd

from src.features.rest_schedule import add_rest_features


def test_same_team_multiple_games_on_same_date_preserves_game_grain():
    games = pd.DataFrame(
        [
            {"game_id": "a", "season": 2025, "game_date": "2026-02-15", "home_team": "STARS", "away_team": "WORLD"},
            {"game_id": "b", "season": 2025, "game_date": "2026-02-15", "home_team": "STRIPES", "away_team": "STARS"},
            {"game_id": "c", "season": 2025, "game_date": "2026-02-15", "home_team": "STRIPES", "away_team": "WORLD"},
        ]
    )

    result = add_rest_features(games, games)

    assert len(result) == 3
    assert result["game_id"].is_unique
