from datetime import date
from types import SimpleNamespace

import pandas as pd

import pytest

from scripts.build_feature_snapshots import (
    _assert_feature_grain,
    _assert_schedule_league,
    _fetch_table,
    _filter_schedules_for_window,
    _ensure_staging_schema,
    _resolve_date_window,
)


def test_resolve_date_window_from_explicit_dates():
    args = SimpleNamespace(
        start_date=date(2026, 9, 10),
        end_date=date(2026, 9, 20),
        lookback_days=None,
        lookahead_days=None,
        date=None,
    )

    assert _resolve_date_window(args) == (date(2026, 9, 10), date(2026, 9, 20))


def test_resolve_date_window_from_anchor_and_offsets():
    args = SimpleNamespace(
        start_date=None,
        end_date=None,
        lookback_days=1,
        lookahead_days=3,
        date=date(2026, 6, 8),
    )

    assert _resolve_date_window(args) == (date(2026, 6, 7), date(2026, 6, 11))


def test_filter_schedules_for_window_keeps_only_target_games():
    schedules = pd.DataFrame(
        {
            "game_id": ["A", "B", "C"],
            "game_date": pd.to_datetime(["2026-06-07", "2026-06-08", "2026-06-12"]),
        }
    )

    filtered = _filter_schedules_for_window(schedules, (date(2026, 6, 8), date(2026, 6, 11)))

    assert filtered["game_id"].tolist() == ["B"]


def test_assert_schedule_league_accepts_isolated_rows():
    schedules = pd.DataFrame({"game_id": ["A", "B"], "league": ["NFL", "nfl"]})

    _assert_schedule_league(schedules, "NFL")


def test_assert_schedule_league_rejects_cross_sport_rows():
    schedules = pd.DataFrame({"game_id": ["A", "B"], "league": ["NFL", "MLB"]})

    with pytest.raises(ValueError, match="unexpected=\\['MLB'\\]"):
        _assert_schedule_league(schedules, "NFL")


def test_assert_schedule_league_rejects_missing_league_values():
    schedules = pd.DataFrame({"game_id": ["A", "B"], "league": ["NFL", None]})

    with pytest.raises(ValueError, match="missing=1"):
        _assert_schedule_league(schedules, "NFL")


def test_assert_feature_grain_accepts_one_row_per_game():
    schedules = pd.DataFrame({"game_id": ["A", "B"]})
    features = pd.DataFrame({"game_id": ["B", "A"], "value": [1.0, 2.0]})

    _assert_feature_grain(features, schedules)


def test_assert_feature_grain_rejects_join_blowup():
    schedules = pd.DataFrame({"game_id": ["A", "B"]})
    features = pd.DataFrame({"game_id": ["A", "A", "B"]})

    with pytest.raises(ValueError, match="duplicates=\\['A'\\]"):
        _assert_feature_grain(features, schedules)


def test_assert_feature_grain_rejects_missing_or_unexpected_games():
    schedules = pd.DataFrame({"game_id": ["A", "B"]})
    features = pd.DataFrame({"game_id": ["A", "C"]})

    with pytest.raises(ValueError, match="missing=\\['B'\\].*unexpected=\\['C'\\]"):
        _assert_feature_grain(features, schedules)


def test_fetch_table_pushes_league_filter_into_bigquery():
    class QueryResult:
        @staticmethod
        def to_dataframe():
            return pd.DataFrame({"game_id": ["A"], "league": ["NFL"]})

    class Client:
        query_text = ""
        job_config = None

        def query(self, query, job_config):
            self.query_text = query
            self.job_config = job_config
            return QueryResult()

    client = Client()
    result = _fetch_table(
        client,
        "project",
        "sports_edge_raw",
        "raw_schedules",
        [2025],
        league="NFL",
    )

    parameters = {
        parameter.name: getattr(parameter, "value", getattr(parameter, "values", None))
        for parameter in client.job_config.query_parameters
    }
    assert "AND league = @league" in client.query_text
    assert parameters == {"seasons": [2025], "league": "NFL"}
    assert result["game_id"].tolist() == ["A"]


def test_ensure_staging_schema_clones_canonical_schema_when_missing():
    class Job:
        @staticmethod
        def result():
            return None

    class Client:
        query_text = None

        @staticmethod
        def get_table(_table):
            from google.api_core import exceptions

            raise exceptions.NotFound("missing")

        def query(self, query):
            self.query_text = query
            return Job()

    client = Client()
    _ensure_staging_schema(
        client,
        destination_table="project.dataset.stage",
        canonical_table="project.dataset.canonical",
    )
    assert client.query_text == "CREATE TABLE `project.dataset.stage` LIKE `project.dataset.canonical`"
