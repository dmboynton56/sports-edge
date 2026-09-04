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
