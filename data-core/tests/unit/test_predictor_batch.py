import types

import pandas as pd

from src.models.predictor import GamePredictor


def test_predict_batch_builds_the_slate_in_one_call():
    predictor = object.__new__(GamePredictor)
    observed_sizes: list[int] = []

    def fake_predict(self, game_rows, *_args, **_kwargs):
        observed_sizes.append(len(game_rows))
        return [
            {
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_win_probability": 0.5,
            }
            for row in game_rows.itertuples(index=False)
        ]

    predictor.predict = types.MethodType(fake_predict, predictor)
    games = pd.DataFrame(
        [
            {"home_team": "SEA", "away_team": "NE"},
            {"home_team": "LAR", "away_team": "SF"},
        ]
    )

    predictions = predictor.predict_batch(games, pd.DataFrame([{"season": 2025}]))

    assert observed_sizes == [2]
    assert len(predictions) == 2
    assert predictions["home_team"].tolist() == ["SEA", "LAR"]


def test_team_strength_excludes_future_unfinished_and_previous_season_games():
    predictor = object.__new__(GamePredictor)
    history = pd.DataFrame([
        ("A", "B", "2025-01-01", 2025, 20, 10),
        ("B", "A", "2025-01-02", 2025, 30, 10),
        ("A", "B", "2025-01-03", 2025, None, None),
        ("A", "B", "2025-01-04", 2025, 100, 0),
        ("A", "B", "2024-12-31", 2024, 100, 0),
    ], columns=["home_team", "away_team", "game_date", "season", "home_score", "away_score"])
    games = pd.DataFrame([dict(home_team="A", away_team="B", game_date="2025-01-04", season=2025)])
    row = predictor._add_team_strength_features(games, history).iloc[0]
    assert row.home_team_win_pct == row.away_team_win_pct == 0.5
    assert row.home_team_point_diff == -5
    assert row.away_team_point_diff == 5
    assert row.home_team_point_diff_std == row.away_team_point_diff_std == 15
    assert row.home_team_win_pct_at_home == 1
    assert row.away_team_win_pct_on_road == 0


def test_team_strength_preserves_unknown_teams_and_single_game_defaults():
    predictor = object.__new__(GamePredictor)
    history = pd.DataFrame([dict(home_team="B", away_team="A", game_date="2025-01-01", season=2025, home_score=10, away_score=10)])
    games = pd.DataFrame([
        dict(home_team="A", away_team="B", game_date="2025-01-02", season=2025),
        dict(home_team="X", away_team="Y", game_date="2025-01-02", season=2025),
    ])
    result = predictor._add_team_strength_features(games, history)
    assert result.iloc[0].home_team_win_pct == result.iloc[0].away_team_win_pct == 0
    assert result.iloc[0].home_team_point_diff_std == result.iloc[0].away_team_point_diff_std == 10
    assert result.iloc[0].home_team_win_pct_at_home == 0.5
    assert result.iloc[0].away_team_win_pct_on_road == 0.4
    assert pd.isna(result.iloc[1].home_team_win_pct)
    assert pd.isna(result.iloc[1].away_team_point_diff)
