#!/usr/bin/env python3
"""
Create BigQuery tables for the Sports-Edge pipeline.

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    python gcp_setup/create_bigquery_tables.py --project learned-pier-478122-p7
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List

from google.cloud import bigquery
from dotenv import load_dotenv


DATASETS: Dict[str, Dict[str, str]] = {
    "sports_edge_raw": {"location": "US"},
    "sports_edge_curated": {"location": "US"},
}


def _schema(fields: Iterable[tuple[str, str, str | None]]) -> List[bigquery.SchemaField]:
    """Convert `(name, type, mode)` tuples into SchemaField objects."""
    schema = []
    for name, field_type, mode in fields:
        schema.append(bigquery.SchemaField(name=name, field_type=field_type, mode=mode or "NULLABLE"))
    return schema


TABLE_SPECS = [
    {
        "dataset": "sports_edge_raw",
        "table": "raw_pbp",
        "schema": _schema(
            [
                ("game_id", "STRING", "REQUIRED"),
                ("play_id", "INT64", "REQUIRED"),
                ("season", "INT64", "REQUIRED"),
                ("week", "INT64", "NULLABLE"),
                ("game_date", "DATE", "NULLABLE"),
                ("home_team", "STRING", "NULLABLE"),
                ("away_team", "STRING", "NULLABLE"),
                ("posteam", "STRING", "NULLABLE"),
                ("defteam", "STRING", "NULLABLE"),
                ("play_type", "STRING", "NULLABLE"),
                ("yards_gained", "FLOAT64", "NULLABLE"),
                ("epa", "FLOAT64", "NULLABLE"),
                ("raw_record", "JSON", "NULLABLE"),
            ]
        ),
        "partition_field": "game_date",
        "description": "Raw play-by-play rows exactly as delivered by nflreadpy (extra columns land in raw_record JSON).",
    },
    {
        "dataset": "sports_edge_raw",
        "table": "raw_schedules",
        "schema": _schema(
            [
                ("game_id", "STRING", "REQUIRED"),
                ("season", "INT64", "REQUIRED"),
                ("week", "INT64", "NULLABLE"),
                ("game_date", "DATE", "NULLABLE"),
                ("home_team", "STRING", "REQUIRED"),
                ("away_team", "STRING", "REQUIRED"),
                ("home_score", "INT64", "NULLABLE"),
                ("away_score", "INT64", "NULLABLE"),
                ("venue", "STRING", "NULLABLE"),
                ("result", "STRING", "NULLABLE"),
                ("raw_record", "JSON", "NULLABLE"),
            ]
        ),
        "partition_field": "game_date",
        "description": "Schedules and final scores from nflreadpy.",
    },
    {
        "dataset": "sports_edge_raw",
        "table": "raw_team_stats",
        "schema": _schema(
            [
                ("team", "STRING", "REQUIRED"),
                ("season", "INT64", "REQUIRED"),
                ("week", "INT64", "REQUIRED"),
                ("points_for", "FLOAT64", "NULLABLE"),
                ("points_against", "FLOAT64", "NULLABLE"),
                ("epa_off", "FLOAT64", "NULLABLE"),
                ("epa_def", "FLOAT64", "NULLABLE"),
                ("win_pct", "FLOAT64", "NULLABLE"),
                ("raw_record", "JSON", "NULLABLE"),
            ]
        ),
        "partition_field": None,
        "description": "Weekly team summary stats stored verbatim plus JSON extras.",
    },
    {
        "dataset": "sports_edge_curated",
        "table": "feature_snapshots",
        "schema": _schema(
            [
                ("game_id", "STRING", "REQUIRED"),
                ("league", "STRING", "REQUIRED"),
                ("season", "INT64", "REQUIRED"),
                ("game_date", "DATE", "REQUIRED"),
                ("as_of_ts", "TIMESTAMP", "REQUIRED"),
                ("home_team", "STRING", "REQUIRED"),
                ("away_team", "STRING", "REQUIRED"),
                ("home_win", "BOOL", "NULLABLE"),
                ("home_margin", "FLOAT64", "NULLABLE"),
                ("rest_home", "FLOAT64", "NULLABLE"),
                ("rest_away", "FLOAT64", "NULLABLE"),
                ("b2b_home", "BOOL", "NULLABLE"),
                ("b2b_away", "BOOL", "NULLABLE"),
                ("opp_strength_home_season", "FLOAT64", "NULLABLE"),
                ("opp_strength_away_season", "FLOAT64", "NULLABLE"),
                ("home_team_win_pct", "FLOAT64", "NULLABLE"),
                ("away_team_win_pct", "FLOAT64", "NULLABLE"),
                ("home_team_point_diff", "FLOAT64", "NULLABLE"),
                ("away_team_point_diff", "FLOAT64", "NULLABLE"),
                ("rest_differential", "FLOAT64", "NULLABLE"),
                ("win_pct_differential", "FLOAT64", "NULLABLE"),
                ("point_diff_differential", "FLOAT64", "NULLABLE"),
                ("opp_strength_differential", "FLOAT64", "NULLABLE"),
                ("week_number", "INT64", "NULLABLE"),
                ("month", "INT64", "NULLABLE"),
                ("is_playoff", "BOOL", "NULLABLE"),
                ("form_home_epa_off_3", "FLOAT64", "NULLABLE"),
                ("form_home_epa_off_5", "FLOAT64", "NULLABLE"),
                ("form_home_epa_off_10", "FLOAT64", "NULLABLE"),
                ("form_home_epa_def_3", "FLOAT64", "NULLABLE"),
                ("form_home_epa_def_5", "FLOAT64", "NULLABLE"),
                ("form_home_epa_def_10", "FLOAT64", "NULLABLE"),
                ("form_away_epa_off_3", "FLOAT64", "NULLABLE"),
                ("form_away_epa_off_5", "FLOAT64", "NULLABLE"),
                ("form_away_epa_off_10", "FLOAT64", "NULLABLE"),
                ("form_away_epa_def_3", "FLOAT64", "NULLABLE"),
                ("form_away_epa_def_5", "FLOAT64", "NULLABLE"),
                ("form_away_epa_def_10", "FLOAT64", "NULLABLE"),
                ("form_epa_off_diff_3", "FLOAT64", "NULLABLE"),
                ("form_epa_off_diff_5", "FLOAT64", "NULLABLE"),
                ("form_epa_off_diff_10", "FLOAT64", "NULLABLE"),
                ("form_epa_def_diff_3", "FLOAT64", "NULLABLE"),
                ("form_epa_def_diff_5", "FLOAT64", "NULLABLE"),
                ("form_epa_def_diff_10", "FLOAT64", "NULLABLE"),
                ("feature_version", "STRING", "NULLABLE"),
            ]
        ),
        "partition_field": "game_date",
        "cluster_fields": ["league", "season"],
        "description": "Feature contract used for model training/scoring.",
    },
    {
        "dataset": "sports_edge_curated",
        "table": "model_predictions",
        "schema": _schema(
            [
                ("prediction_id", "STRING", "REQUIRED"),
                ("game_id", "STRING", "REQUIRED"),
                ("league", "STRING", "REQUIRED"),
                ("model_version", "STRING", "REQUIRED"),
                ("predicted_spread", "FLOAT64", "NULLABLE"),
                ("home_win_prob", "FLOAT64", "NULLABLE"),
                ("prediction_ts", "TIMESTAMP", "REQUIRED"),
                ("input_hash", "STRING", "NULLABLE"),
            ]
        ),
        "partition_field": "prediction_ts",
        "description": "Outputs from BQML or Python models before syncing to Supabase.",
    },
    {
        "dataset": "sports_edge_curated",
        "table": "model_runs",
        "schema": _schema(
            [
                ("run_id", "STRING", "REQUIRED"),
                ("started_at", "TIMESTAMP", "REQUIRED"),
                ("finished_at", "TIMESTAMP", "NULLABLE"),
                ("league", "STRING", "NULLABLE"),
                ("rows_written", "INT64", "NULLABLE"),
                ("status", "STRING", "NULLABLE"),
                ("error_text", "STRING", "NULLABLE"),
            ]
        ),
        "partition_field": "started_at",
        "description": "Audit log for each pipeline execution (training or scoring).",
    },
]


def ensure_dataset(client: bigquery.Client, project: str, dataset_id: str, location: str) -> None:
    dataset_ref = bigquery.Dataset(f"{project}.{dataset_id}")
    dataset_ref.location = location
    try:
        client.create_dataset(dataset_ref)
        print(f"Created dataset {dataset_id}")
    except Exception as exc:  # noqa: BLE001
        if "Already Exists" in str(exc):
            print(f"Dataset {dataset_id} already exists; skipping.")
        else:
            raise


def ensure_table(client: bigquery.Client, project: str, spec: Dict) -> None:
    table_ref = bigquery.Table(f"{project}.{spec['dataset']}.{spec['table']}", schema=spec["schema"])
    if spec.get("partition_field"):
        table_ref.time_partitioning = bigquery.TimePartitioning(field=spec["partition_field"])
    if spec.get("cluster_fields"):
        table_ref.clustering_fields = spec["cluster_fields"]
    if spec.get("description"):
        table_ref.description = spec["description"]

    try:
        client.create_table(table_ref)
        print(f"Created table {spec['dataset']}.{spec['table']}")
    except Exception as exc:  # noqa: BLE001
        if "Already Exists" in str(exc):
            print(f"Table {spec['dataset']}.{spec['table']} already exists; skipping.")
        else:
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BigQuery datasets/tables for Sports Edge.")
    parser.add_argument("--project", required=True, help="GCP project ID (e.g., learned-pier-478122-p7).")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    client = bigquery.Client(project=args.project)

    for dataset_id, options in DATASETS.items():
        ensure_dataset(client, args.project, dataset_id, options["location"])

    for spec in TABLE_SPECS:
        ensure_table(client, args.project, spec)

    print("All datasets/tables ensured.")


if __name__ == "__main__":
    main()
