from __future__ import annotations

import pytest
from google.cloud import bigquery

import json

from scripts.sync_player_markets_to_bigquery import TABLES, _ensure_table, _load, build_mlb_rows


class FakeTableClient:
    def __init__(self, table: bigquery.Table) -> None:
        self.table = table
        self.updated_fields: list[str] | None = None

    def get_table(self, table_id: str) -> bigquery.Table:
        expected = f"{self.table.project}.{self.table.dataset_id}.{self.table.table_id}"
        assert table_id == expected
        return self.table

    def update_table(self, table: bigquery.Table, fields: list[str]) -> bigquery.Table:
        self.table = table
        self.updated_fields = fields
        return table


class FakeLoadJob:
    def result(self) -> None:
        return None


class FakeLoadClient:
    def __init__(self) -> None:
        self.job_config: bigquery.LoadJobConfig | None = None

    def load_table_from_dataframe(
        self,
        frame,
        table_id: str,
        *,
        job_config: bigquery.LoadJobConfig,
    ) -> FakeLoadJob:
        self.job_config = job_config
        return FakeLoadJob()


def test_ensure_table_adds_missing_nullable_fields() -> None:
    schema = [
        field
        for field in TABLES["mlb_home_run_predictions"]["schema"]
        if field.name
        not in {
            "games_since_last_hr",
            "last_hr_date",
            "v1_probability",
            "v1_rank",
            "statcast_probability",
            "statcast_rank",
            "statcast_available",
            "model_agreement",
            "consensus_score",
            "market_signal_rank",
        }
    ]
    table = bigquery.Table("project.dataset.mlb_home_run_predictions", schema=schema)
    client = FakeTableClient(table)

    _ensure_table(client, "project.dataset.mlb_home_run_predictions", TABLES["mlb_home_run_predictions"])

    names = {field.name for field in client.table.schema}
    assert {
        "games_since_last_hr",
        "last_hr_date",
        "v1_probability",
        "v1_rank",
        "statcast_probability",
        "statcast_rank",
        "statcast_available",
        "model_agreement",
        "consensus_score",
        "market_signal_rank",
    } <= names
    assert client.updated_fields == ["schema"]


def test_ensure_table_rejects_missing_required_fields() -> None:
    schema = [
        field
        for field in TABLES["mlb_home_run_predictions"]["schema"]
        if field.name != "game_id"
    ]
    table = bigquery.Table("project.dataset.mlb_home_run_predictions", schema=schema)
    client = FakeTableClient(table)

    with pytest.raises(ValueError, match="missing required fields"):
        _ensure_table(client, "project.dataset.mlb_home_run_predictions", TABLES["mlb_home_run_predictions"])


def test_ensure_table_accepts_bigquery_type_aliases() -> None:
    spec = {
        "schema": [
            bigquery.SchemaField("season", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("win_prob", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("is_active", "BOOLEAN", mode="NULLABLE"),
        ]
    }
    table = bigquery.Table(
        "project.dataset.alias_table",
        schema=[
            bigquery.SchemaField("season", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("win_prob", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("is_active", "BOOL", mode="NULLABLE"),
        ],
    )
    client = FakeTableClient(table)

    _ensure_table(client, "project.dataset.alias_table", spec)

    assert client.updated_fields is None


def test_load_allows_nullable_field_addition() -> None:
    client = FakeLoadClient()

    _load(
        client,
        "project.dataset.mlb_home_run_predictions",
        [{"game_id": "MLB_1", "player_id": "1"}],
        TABLES["mlb_home_run_predictions"]["schema"],
    )

    assert client.job_config is not None
    assert bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION in client.job_config.schema_update_options


def test_build_mlb_rows_preserves_probability_board_comparison_fields(tmp_path) -> None:
    payload = {
        "generatedAt": "2026-06-30T13:00:00+00:00",
        "predictions": [
            {
                "gameId": "MLB_1",
                "gameDate": "2026-06-30",
                "eventTime": "2026-06-30T23:00:00+00:00",
                "playerId": "123",
                "player": "Test Bat",
                "modelProbability": 0.21,
                "modelVersion": "mlb-hr-v1",
                "v1Probability": 0.21,
                "v1Rank": 2,
                "statcastProbability": 0.25,
                "statcastRank": 1,
                "statcastAvailable": True,
                "modelAgreement": "Statcast boost",
                "consensusScore": 124,
                "marketSignalRank": 121,
            }
        ],
        "models": {
            "mlb-hr-v1": {
                "modelVersion": "mlb-hr-v1",
                "predictions": [
                    {
                        "gameId": "MLB_1",
                        "gameDate": "2026-06-30",
                        "eventTime": "2026-06-30T23:00:00+00:00",
                        "playerId": "123",
                        "player": "Test Bat",
                        "modelProbability": 0.21,
                        "modelVersion": "mlb-hr-v1",
                        "rank": 2,
                        "qualityFlags": [],
                    }
                ],
            },
            "mlb-hr-torch-statcast-v1-blend": {
                "modelVersion": "mlb-hr-torch-statcast-v1-blend",
                "predictions": [
                    {
                        "gameId": "MLB_1",
                        "gameDate": "2026-06-30",
                        "eventTime": "2026-06-30T23:00:00+00:00",
                        "playerId": "123",
                        "player": "Test Bat",
                        "modelProbability": 0.25,
                        "modelVersion": "mlb-hr-torch-statcast-v1-blend",
                        "rank": 1,
                        "qualityFlags": [],
                    }
                ],
            },
        },
    }
    path = tmp_path / "mlb_home_runs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    batches = build_mlb_rows(path)
    rows = [row for batch, _, _ in batches for row in batch]

    assert len(rows) == 2
    assert {row["model_version"] for row in rows} == {
        "mlb-hr-v1",
        "mlb-hr-torch-statcast-v1-blend",
    }
    assert all(row["v1_probability"] == 0.21 for row in rows)
    assert all(row["statcast_probability"] == 0.25 for row in rows)
    assert all(row["statcast_available"] is True for row in rows)
    assert all(row["model_agreement"] == "Statcast boost" for row in rows)
    assert all(row["consensus_score"] == 124 for row in rows)
