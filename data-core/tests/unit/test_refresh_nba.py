from datetime import date

import pandas as pd
import pytest

from src.pipeline.refresh_nba import _prediction_window, _query_games


def test_prediction_window_covers_serving_lookahead():
    assert _prediction_window(date(2026, 11, 3), 0, 1) == (date(2026, 11, 3), date(2026, 11, 4))


def test_query_games_uses_inclusive_date_window():
    captured = {}

    class QueryResult:
        @staticmethod
        def to_dataframe():
            return pd.DataFrame(
                {
                    "game_id": ["401809111"],
                    "game_date": [date(2026, 11, 4)],
                    "league": ["NBA"],
                }
            )

    class Client:
        def query(self, query, job_config):
            captured["query"] = query
            captured["params"] = {
                parameter.name: parameter.value for parameter in job_config.query_parameters
            }
            return QueryResult()

    df = _query_games(Client(), "project", date(2026, 11, 3), date(2026, 11, 4))

    assert "BETWEEN @start_date AND @end_date" in captured["query"]
    assert captured["params"] == {
        "start_date": date(2026, 11, 3),
        "end_date": date(2026, 11, 4),
    }
    assert df.iloc[0]["game_id"] == "401809111"
    assert pd.Timestamp(df.iloc[0]["game_date"]).date() == date(2026, 11, 4)


def test_prediction_window_rejects_negative_offsets():
    with pytest.raises(ValueError, match="non-negative"):
        _prediction_window(date(2026, 11, 3), -1, 0)
