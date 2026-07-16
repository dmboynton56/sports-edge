from datetime import datetime, timezone

import scripts.sync_evaluation_history_to_supabase as sync_module
from scripts.sync_evaluation_history_to_supabase import (
    EvaluationPayload,
    build_evaluation_payloads,
    sync_payloads,
)


FIXED_GENERATED_AT = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _normalized_sql(statement):
    return " ".join(statement.split())


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = -1
        self._returned_id = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement, params=None, prepare=None):
        normalized = _normalized_sql(statement)
        self.conn.executions.append((normalized, params, prepare))
        self.rowcount = 1

        if normalized.startswith("INSERT INTO model_evaluation_runs"):
            self.conn.next_id += 1
            self._returned_id = f"evaluation-{self.conn.next_id}"
        elif normalized.startswith("INSERT INTO model_evaluation_history"):
            identity = params[:5]
            if identity in self.conn.history_identities:
                self.rowcount = 0
            else:
                self.conn.history_identities.add(identity)

    def fetchone(self):
        return (self._returned_id,)


class FakeConnection:
    def __init__(self):
        self.executions = []
        self.history_identities = set()
        self.next_id = 0
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def _evaluation(evaluation_name="performance_history_nba_2025-26_spread"):
    return EvaluationPayload(
        league="NBA",
        model_name="sports_edge",
        model_version="v3",
        evaluation_name=evaluation_name,
        metrics={"sample": {"completed_games": 1175}},
        calibration={"brier": 0.24},
        artifact_refs=["scripts/backtest_nba_spread.py"],
        notes=None,
    )


def _freeze_generated_at(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_GENERATED_AT

    monkeypatch.setattr(sync_module, "datetime", FixedDatetime)


def test_build_evaluation_payloads_creates_model_and_strategy_rows():
    history = {
        "sports": [
            {
                "sport": "NBA",
                "model_version": "v3",
                "season": "2025-26",
                "market": "spread",
                "data_source": "BigQuery backtest + Supabase ATS",
                "odds_status": "partial_historical_spread_odds",
                "sample": {
                    "completed_games": 1175,
                    "supabase_graded_games": 64,
                },
                "metrics": {
                    "bigquery_default_bets": 591,
                    "bigquery_default_wins": 301,
                    "bigquery_default_accuracy": 0.509,
                    "bigquery_default_roi": -0.005,
                    "bigquery_best_sweep_bets": 112,
                    "bigquery_best_sweep_edge_threshold": 0.5,
                    "bigquery_best_sweep_min_confidence": 0.4,
                    "best_reported_sweep_roi": 0.083,
                    "supabase_ats_wins": 31,
                    "supabase_ats_losses": 33,
                    "supabase_ats_pushes": 0,
                    "supabase_ats_roi": -0.075,
                    "bigquery_brier": 0.24,
                },
                "artifact_refs": ["scripts/backtest_nba_spread.py"],
                "gaps": ["odds gap"],
            },
            {
                "sport": "MLB",
                "model_version": "v3",
                "season": "2026 YTD",
                "market": "moneyline",
                "data_source": "MLB Stats API",
                "odds_status": "free_moneyline_history",
                "sample": {"odds_joined_games": 673},
                "metrics": {
                    "brier": 0.247,
                    "log_loss": 0.688,
                    "roc_auc": 0.543,
                    "ece_10": 0.012,
                    "flat_roi": -0.031,
                },
                "artifact_refs": ["scripts/backtest_mlb_winners.py"],
                "gaps": [],
            },
        ]
    }

    evaluations, strategies = build_evaluation_payloads(history)

    assert [row.league for row in evaluations] == ["NBA", "MLB"]
    assert evaluations[0].evaluation_name == "performance_history_nba_2025-26_spread"
    assert evaluations[0].model_name == "sports_edge"
    assert evaluations[0].metrics["sample"]["completed_games"] == 1175
    assert evaluations[0].calibration["bigquery_brier"] == 0.24
    assert evaluations[1].calibration == {
        "brier": 0.247,
        "log_loss": 0.688,
        "roc_auc": 0.543,
        "ece_10": 0.012,
    }

    strategy_ids = [row.strategy_id for row in strategies]
    assert strategy_ids == [
        "bigquery_default_edge_1_conf_0",
        "bigquery_best_reported_sweep",
        "supabase_ats_flat_minus_110",
        "moneyline_flat_pick",
    ]
    nba_default = strategies[0]
    assert nba_default.league == "NBA"
    assert nba_default.bets == 591
    assert nba_default.wins == 301
    assert nba_default.roi == -0.005
    assert nba_default.edge_threshold == 1.0
    assert nba_default.min_confidence == 0.0

    mlb_strategy = strategies[-1]
    assert mlb_strategy.league == "MLB"
    assert mlb_strategy.bets == 673
    assert mlb_strategy.roi == -0.031


def test_build_evaluation_payloads_skips_strategy_when_no_roi_metric():
    history = {
        "sports": [
            {
                "sport": "CBB",
                "model_version": "manual matchup artifacts",
                "season": "CV 2016-2025",
                "market": "tournament winner probability",
                "data_source": "cache",
                "odds_status": "no_sportsbook_odds",
                "sample": {"folds": 9},
                "metrics": {"log_loss": 0.575, "brier": 0.198, "auc": 0.758},
                "artifact_refs": [],
                "gaps": [],
            }
        ]
    }

    evaluations, strategies = build_evaluation_payloads(history)

    assert len(evaluations) == 1
    assert evaluations[0].league == "CBB"
    assert evaluations[0].calibration == {
        "log_loss": 0.575,
        "brier": 0.198,
        "auc": 0.758,
    }
    assert strategies == []


def test_sync_payloads_appends_one_history_row_per_evaluation_and_keeps_latest_statements(
    monkeypatch,
):
    _freeze_generated_at(monkeypatch)
    conn = FakeConnection()
    evaluations = [_evaluation(), _evaluation("performance_history_nba_2025-26_moneyline")]

    result = sync_payloads(conn, evaluations, [])

    assert result == (2, 2, 0)
    assert conn.commits == 1

    history_inserts = [
        execution
        for execution in conn.executions
        if execution[0].startswith("INSERT INTO model_evaluation_history")
    ]
    assert len(history_inserts) == len(evaluations)
    assert all(
        "ON CONFLICT (league, model_name, model_version, evaluation_name, generated_at) DO NOTHING"
        in statement
        for statement, _, _ in history_inserts
    )
    assert all(params[4] == FIXED_GENERATED_AT for _, params, _ in history_inserts)

    latest_statements = [
        statement
        for statement, _, _ in conn.executions
        if "model_evaluation_runs" in statement
    ]
    assert latest_statements == [
        (
            "DELETE FROM model_evaluation_runs WHERE league = %s AND model_name = %s "
            "AND model_version = %s AND evaluation_name = %s"
        ),
        (
            "INSERT INTO model_evaluation_runs ( league, model_name, model_version, "
            "evaluation_name, generated_at, metrics, calibration, artifact_refs, status, notes ) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, 'candidate', %s) "
            "RETURNING id"
        ),
        (
            "DELETE FROM model_evaluation_runs WHERE league = %s AND model_name = %s "
            "AND model_version = %s AND evaluation_name = %s"
        ),
        (
            "INSERT INTO model_evaluation_runs ( league, model_name, model_version, "
            "evaluation_name, generated_at, metrics, calibration, artifact_refs, status, notes ) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, 'candidate', %s) "
            "RETURNING id"
        ),
    ]


def test_sync_payloads_repeated_generated_at_does_not_append_duplicate_history(monkeypatch):
    _freeze_generated_at(monkeypatch)
    conn = FakeConnection()
    evaluations = [_evaluation()]

    first_result = sync_payloads(conn, evaluations, [])
    second_result = sync_payloads(conn, evaluations, [])

    assert first_result == (1, 1, 0)
    assert second_result == (1, 0, 0)
    assert len(conn.history_identities) == 1
    assert conn.commits == 2
