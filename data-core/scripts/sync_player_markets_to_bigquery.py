#!/usr/bin/env python3
"""Load player-market artifacts into BigQuery curated tables."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PGA_JSON = ROOT.parent / "web" / "public" / "data" / "pga_tournaments" / "us_open_2026.json"
DEFAULT_MLB_JSON = ROOT.parent / "web" / "public" / "data" / "mlb_home_runs.json"


def _schema(fields: list[tuple[str, str, str]]) -> list[bigquery.SchemaField]:
    return [bigquery.SchemaField(name, field_type, mode=mode) for name, field_type, mode in fields]


TABLES = {
    "pga_tournaments": {
        "schema": _schema(
            [
                ("event_key", "STRING", "REQUIRED"),
                ("season", "INT64", "REQUIRED"),
                ("name", "STRING", "REQUIRED"),
                ("start_date", "DATE", "REQUIRED"),
                ("end_date", "DATE", "REQUIRED"),
                ("course", "STRING", "NULLABLE"),
                ("par", "INT64", "NULLABLE"),
                ("field_size", "INT64", "NULLABLE"),
                ("status", "STRING", "NULLABLE"),
                ("source", "STRING", "NULLABLE"),
                ("raw_record", "STRING", "NULLABLE"),
                ("updated_at", "TIMESTAMP", "REQUIRED"),
            ]
        ),
        "partition_field": "start_date",
        "cluster_fields": ["event_key", "season"],
    },
    "pga_player_predictions": {
        "schema": _schema(
            [
                ("event_key", "STRING", "REQUIRED"),
                ("player_name", "STRING", "REQUIRED"),
                ("player_id", "STRING", "NULLABLE"),
                ("exp_sg_per_round", "FLOAT64", "NULLABLE"),
                ("make_cut_prob", "FLOAT64", "NULLABLE"),
                ("top5_prob", "FLOAT64", "NULLABLE"),
                ("top10_prob", "FLOAT64", "NULLABLE"),
                ("top20_prob", "FLOAT64", "NULLABLE"),
                ("win_prob", "FLOAT64", "NULLABLE"),
                ("projected_total_strokes", "FLOAT64", "NULLABLE"),
                ("projected_score_to_par", "FLOAT64", "NULLABLE"),
                ("model_version", "STRING", "REQUIRED"),
                ("prediction_ts", "TIMESTAMP", "REQUIRED"),
                ("simulation_count", "INT64", "NULLABLE"),
                ("confidence", "FLOAT64", "NULLABLE"),
                ("quality_flags", "STRING", "NULLABLE"),
                ("feature_snapshot", "STRING", "NULLABLE"),
            ]
        ),
        "partition_field": "prediction_ts",
        "cluster_fields": ["event_key", "model_version"],
    },
    "mlb_home_run_predictions": {
        "schema": _schema(
            [
                ("game_id", "STRING", "REQUIRED"),
                ("game_date", "DATE", "REQUIRED"),
                ("event_time", "TIMESTAMP", "NULLABLE"),
                ("player_id", "STRING", "REQUIRED"),
                ("player_name", "STRING", "REQUIRED"),
                ("team", "STRING", "NULLABLE"),
                ("opponent", "STRING", "NULLABLE"),
                ("venue", "STRING", "NULLABLE"),
                ("lineup_slot", "INT64", "NULLABLE"),
                ("lineup_status", "STRING", "NULLABLE"),
                ("opposing_probable_pitcher", "STRING", "NULLABLE"),
                ("hr_probability", "FLOAT64", "REQUIRED"),
                ("baseline_probability", "FLOAT64", "NULLABLE"),
                ("rank", "INT64", "NULLABLE"),
                ("confidence", "FLOAT64", "NULLABLE"),
                ("model_version", "STRING", "REQUIRED"),
                ("prediction_ts", "TIMESTAMP", "REQUIRED"),
                ("quality_flags", "STRING", "NULLABLE"),
                ("top_features", "STRING", "NULLABLE"),
            ]
        ),
        "partition_field": "game_date",
        "cluster_fields": ["model_version", "team"],
    },
}


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _ensure_table(client: bigquery.Client, table_id: str, spec: dict[str, Any]) -> None:
    try:
        client.get_table(table_id)
        return
    except Exception:  # noqa: BLE001
        pass
    table = bigquery.Table(table_id, schema=spec["schema"])
    partition_field = spec.get("partition_field")
    if partition_field:
        table.time_partitioning = bigquery.TimePartitioning(field=partition_field)
    if spec.get("cluster_fields"):
        table.clustering_fields = spec["cluster_fields"]
    client.create_table(table)
    print(f"Created BigQuery table {table_id}")


def _delete(client: bigquery.Client, table_id: str, query: str, params: list[bigquery.ScalarQueryParameter]) -> None:
    client.query(query, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()


def _load(client: bigquery.Client, table_id: str, rows: list[dict[str, Any]], schema: list[bigquery.SchemaField]) -> int:
    if not rows:
        return 0
    frame = pd.DataFrame(rows)
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
    client.load_table_from_dataframe(frame, table_id, job_config=job_config).result()
    return len(frame)


def build_pga_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    event = payload["event"]
    preds = payload.get("predictions", [])
    prediction_ts = payload.get("generatedAt") or datetime.now(timezone.utc).isoformat()
    meta = payload.get("predictionMeta", {})
    model_version = str(meta.get("model_version") or "pga-baseline-v0")
    tournament_rows = [
        {
            "event_key": event["eventKey"],
            "season": int(event["season"]),
            "name": event["name"],
            "start_date": event["startDate"],
            "end_date": event["endDate"],
            "course": event.get("course"),
            "par": event.get("par"),
            "field_size": meta.get("n_players") or len(preds),
            "status": event.get("status", "scheduled"),
            "source": "pga_tournament_dashboard_json",
            "raw_record": json.dumps(event, sort_keys=True),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    prediction_rows = []
    for pred in preds:
        prediction_rows.append(
            {
                "event_key": event["eventKey"],
                "player_name": pred.get("player"),
                "player_id": str(pred.get("player_id")) if pred.get("player_id") is not None else None,
                "exp_sg_per_round": _clean(pred.get("exp_sg_per_round")),
                "make_cut_prob": _clean(pred.get("best_calibrated_target_made_cut_prob")),
                "top5_prob": _clean((pred.get("sim_top5_pct") or 0) / 100 if pred.get("sim_top5_pct") is not None else None),
                "top10_prob": _clean(pred.get("best_calibrated_target_top10_prob")),
                "top20_prob": _clean(pred.get("best_calibrated_target_top20_prob")),
                "win_prob": _clean(pred.get("best_calibrated_target_win_prob")),
                "projected_total_strokes": _clean(pred.get("projected_total_strokes")),
                "projected_score_to_par": _clean(pred.get("projected_score_to_par")),
                "model_version": model_version,
                "prediction_ts": prediction_ts,
                "simulation_count": meta.get("n_sims"),
                "confidence": _clean(pred.get("confidence")),
                "quality_flags": json.dumps(pred.get("quality_flags") or []),
                "feature_snapshot": json.dumps({"source": pred.get("source"), "starts_before": pred.get("starts_before")}),
            }
        )
    return tournament_rows, prediction_rows, event["eventKey"], model_version


def build_mlb_rows(path: Path) -> tuple[list[dict[str, Any]], str | None, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for pred in payload.get("predictions", []):
        rows.append(
            {
                "game_id": pred.get("gameId"),
                "game_date": pred.get("eventTime", "")[:10],
                "event_time": pred.get("eventTime"),
                "player_id": str(pred.get("playerId") or pred.get("player_id") or pred.get("player") or pred.get("id")),
                "player_name": pred.get("player"),
                "team": pred.get("team"),
                "opponent": pred.get("opponent"),
                "venue": pred.get("venue"),
                "lineup_slot": pred.get("lineupSlot"),
                "lineup_status": pred.get("lineupStatus") or "projected",
                "opposing_probable_pitcher": pred.get("opposingProbablePitcher"),
                "hr_probability": pred.get("modelProbability"),
                "baseline_probability": pred.get("baselineProbability"),
                "rank": pred.get("rank"),
                "confidence": pred.get("confidence"),
                "model_version": pred.get("modelVersion") or payload.get("modelVersion"),
                "prediction_ts": pred.get("updatedAt") or payload.get("generatedAt"),
                "quality_flags": json.dumps(pred.get("qualityFlags") or []),
                "top_features": json.dumps(pred.get("topFeatures") or []),
            }
        )
    if not rows:
        return rows, None, None
    return rows, rows[0]["game_date"], rows[0]["model_version"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync player-market artifacts to BigQuery.")
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID"))
    parser.add_argument("--dataset", default="sports_edge_curated")
    parser.add_argument("--pga-json", type=Path, default=DEFAULT_PGA_JSON)
    parser.add_argument("--mlb-json", type=Path, default=DEFAULT_MLB_JSON)
    parser.add_argument("--skip-pga", action="store_true")
    parser.add_argument("--skip-mlb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.project:
        raise SystemExit("--project or GCP_PROJECT_ID is required")
    client = bigquery.Client(project=args.project)
    dataset_id = f"{args.project}.{args.dataset}"
    client.create_dataset(bigquery.Dataset(dataset_id), exists_ok=True)

    table_ids = {name: f"{dataset_id}.{name}" for name in TABLES}
    for name, table_id in table_ids.items():
        _ensure_table(client, table_id, TABLES[name])

    if not args.skip_pga and args.pga_json.exists():
        tournament_rows, prediction_rows, event_key, model_version = build_pga_rows(args.pga_json)
        _delete(
            client,
            table_ids["pga_tournaments"],
            f"delete from `{table_ids['pga_tournaments']}` where event_key = @event_key",
            [bigquery.ScalarQueryParameter("event_key", "STRING", event_key)],
        )
        _delete(
            client,
            table_ids["pga_player_predictions"],
            f"delete from `{table_ids['pga_player_predictions']}` where event_key = @event_key and model_version = @model_version",
            [
                bigquery.ScalarQueryParameter("event_key", "STRING", event_key),
                bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
            ],
        )
        tournaments = _load(client, table_ids["pga_tournaments"], tournament_rows, TABLES["pga_tournaments"]["schema"])
        predictions = _load(client, table_ids["pga_player_predictions"], prediction_rows, TABLES["pga_player_predictions"]["schema"])
        print(f"Synced {tournaments} PGA tournaments and {predictions} PGA predictions to BigQuery")

    if not args.skip_mlb and args.mlb_json.exists():
        rows, game_date, model_version = build_mlb_rows(args.mlb_json)
        if rows and game_date and model_version:
            _delete(
                client,
                table_ids["mlb_home_run_predictions"],
                f"delete from `{table_ids['mlb_home_run_predictions']}` where game_date = @game_date and model_version = @model_version",
                [
                    bigquery.ScalarQueryParameter("game_date", "DATE", game_date),
                    bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
                ],
            )
            loaded = _load(client, table_ids["mlb_home_run_predictions"], rows, TABLES["mlb_home_run_predictions"]["schema"])
            print(f"Synced {loaded} MLB home run predictions to BigQuery")


if __name__ == "__main__":
    main()
