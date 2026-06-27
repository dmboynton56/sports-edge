from __future__ import annotations

import pytest
from google.cloud import bigquery

from scripts.sync_player_markets_to_bigquery import TABLES, _ensure_table, _load


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
        if field.name not in {"games_since_last_hr", "last_hr_date"}
    ]
    table = bigquery.Table("project.dataset.mlb_home_run_predictions", schema=schema)
    client = FakeTableClient(table)

    _ensure_table(client, "project.dataset.mlb_home_run_predictions", TABLES["mlb_home_run_predictions"])

    names = {field.name for field in client.table.schema}
    assert {"games_since_last_hr", "last_hr_date"} <= names
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
