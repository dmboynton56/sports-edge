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
from google.api_core import exceptions as google_exceptions
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PGA_JSON = ROOT.parent / "web" / "public" / "data" / "pga_tournaments" / "current.json"
DEFAULT_MLB_JSON = ROOT.parent / "web" / "public" / "data" / "mlb_home_runs.json"


def _schema(fields: list[tuple[str, str, str]]) -> list[bigquery.SchemaField]:
    return [bigquery.SchemaField(name, field_type, mode=mode) for name, field_type, mode in fields]


def _normalized_field_type(field_type: str) -> str:
    aliases = {
        "BOOL": "BOOLEAN",
        "FLOAT": "FLOAT64",
        "INTEGER": "INT64",
    }
    normalized = field_type.upper()
    return aliases.get(normalized, normalized)


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
                ("v1_probability", "FLOAT64", "NULLABLE"),
                ("v1_rank", "INT64", "NULLABLE"),
                ("statcast_probability", "FLOAT64", "NULLABLE"),
                ("statcast_rank", "INT64", "NULLABLE"),
                ("statcast_available", "BOOL", "NULLABLE"),
                ("model_agreement", "STRING", "NULLABLE"),
                ("consensus_score", "FLOAT64", "NULLABLE"),
                ("market_signal_rank", "INT64", "NULLABLE"),
                ("games_since_last_hr", "INT64", "NULLABLE"),
                ("last_hr_date", "DATE", "NULLABLE"),
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


def _player_id(pred: dict[str, Any]) -> str:
    return str(pred.get("playerId") or pred.get("player_id") or pred.get("player") or pred.get("id"))


def _prediction_key(pred: dict[str, Any]) -> tuple[str, str]:
    return (str(pred.get("gameId")), _player_id(pred))


def _comparison_lookup(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        _prediction_key(pred): pred
        for pred in payload.get("predictions", [])
        if pred.get("gameId") is not None
    }


def _quality_flags(pred: dict[str, Any]) -> list[str]:
    value = pred.get("qualityFlags") or []
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            return [value]
    return list(value)


def _model_is_v1(model_version: str | None) -> bool:
    return (model_version or "").startswith("mlb-hr-v1")


def _model_is_statcast(model_version: str | None) -> bool:
    return "statcast" in (model_version or "")


def _comparison_value(pred: dict[str, Any], comparison: dict[str, Any] | None, key: str) -> Any:
    if comparison and key in comparison:
        return comparison.get(key)
    return pred.get(key)


def _comparison_fields(
    pred: dict[str, Any],
    comparison: dict[str, Any] | None,
    model_version: str,
) -> dict[str, Any]:
    v1_probability = _comparison_value(pred, comparison, "v1Probability")
    if v1_probability is None and _model_is_v1(model_version):
        v1_probability = pred.get("modelProbability")
    v1_rank = _comparison_value(pred, comparison, "v1Rank")
    if v1_rank is None and _model_is_v1(model_version):
        v1_rank = pred.get("rank")

    statcast_probability = _comparison_value(pred, comparison, "statcastProbability")
    if statcast_probability is None and _model_is_statcast(model_version):
        statcast_probability = pred.get("modelProbability")
    statcast_rank = _comparison_value(pred, comparison, "statcastRank")
    if statcast_rank is None and _model_is_statcast(model_version):
        statcast_rank = pred.get("rank")

    statcast_available = _comparison_value(pred, comparison, "statcastAvailable")
    if statcast_available is None and _model_is_statcast(model_version):
        statcast_available = "statcast_features_unavailable" not in _quality_flags(pred)

    model_agreement = _comparison_value(pred, comparison, "modelAgreement")
    if model_agreement is None and _model_is_v1(model_version):
        model_agreement = "V1 only"
    elif model_agreement is None and statcast_available is False:
        model_agreement = "Missing Statcast"

    return {
        "v1_probability": _clean(v1_probability),
        "v1_rank": _clean(v1_rank),
        "statcast_probability": _clean(statcast_probability),
        "statcast_rank": _clean(statcast_rank),
        "statcast_available": _clean(statcast_available),
        "model_agreement": _clean(model_agreement),
        "consensus_score": _clean(_comparison_value(pred, comparison, "consensusScore")),
        "market_signal_rank": _clean(_comparison_value(pred, comparison, "marketSignalRank")),
    }


def _ensure_table(client: bigquery.Client, table_id: str, spec: dict[str, Any]) -> None:
    try:
        table = client.get_table(table_id)
    except google_exceptions.NotFound:
        table = None
    if table is not None:
        existing_by_name = {field.name: field for field in table.schema}
        fields_to_add: list[bigquery.SchemaField] = []
        mismatches: list[str] = []
        required_missing: list[str] = []
        for field in spec["schema"]:
            existing = existing_by_name.get(field.name)
            if existing is None:
                if field.mode == "REQUIRED":
                    required_missing.append(field.name)
                else:
                    fields_to_add.append(field)
                continue
            if _normalized_field_type(existing.field_type) != _normalized_field_type(field.field_type):
                mismatches.append(
                    f"{field.name}: existing {existing.field_type} != expected {field.field_type}"
                )
        if required_missing:
            raise ValueError(
                f"BigQuery table {table_id} is missing required fields that cannot be auto-added: "
                f"{', '.join(required_missing)}"
            )
        if mismatches:
            raise ValueError(f"BigQuery table {table_id} schema mismatch: {'; '.join(mismatches)}")
        if fields_to_add:
            table.schema = [*table.schema, *fields_to_add]
            client.update_table(table, ["schema"])
            print(f"Added BigQuery columns to {table_id}: {', '.join(field.name for field in fields_to_add)}")
        return
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
    for field in schema:
        if field.name not in frame.columns:
            continue
        if field.field_type == "DATE":
            values = pd.to_datetime(frame[field.name], errors="coerce")
            frame[field.name] = values.map(lambda value: value.date() if pd.notna(value) else None)
        elif field.field_type in {"TIMESTAMP", "DATETIME"}:
            frame[field.name] = pd.to_datetime(frame[field.name], errors="coerce", utc=True)
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        write_disposition="WRITE_APPEND",
    )
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


def build_mlb_rows(path: Path) -> list[tuple[list[dict[str, Any]], str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    batches: list[tuple[list[dict[str, Any]], str, str]] = []
    comparisons = _comparison_lookup(payload)
    models = payload.get("models")
    if isinstance(models, dict) and models:
        for model_key, model_payload in models.items():
            if not isinstance(model_payload, dict):
                continue
            rows = []
            for pred in model_payload.get("predictions", []):
                model_version = pred.get("modelVersion") or model_payload.get("modelVersion") or model_key
                comparison = comparisons.get(_prediction_key(pred))
                rows.append(
                    {
                        "game_id": pred.get("gameId"),
                        "game_date": pred.get("gameDate") or pred.get("eventTime", "")[:10],
                        "event_time": pred.get("eventTime"),
                        "player_id": _player_id(pred),
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
                        **_comparison_fields(pred, comparison, model_version),
                        "games_since_last_hr": pred.get("gamesSinceLastHr"),
                        "last_hr_date": pred.get("lastHrDate"),
                        "confidence": pred.get("confidence"),
                        "model_version": model_version,
                        "prediction_ts": pred.get("updatedAt") or payload.get("generatedAt"),
                        "quality_flags": json.dumps(pred.get("qualityFlags") or []),
                        "top_features": json.dumps(pred.get("topFeatures") or []),
                    }
                )
            if rows:
                batches.append((rows, rows[0]["game_date"], rows[0]["model_version"]))
        return batches

    rows = []
    for pred in payload.get("predictions", []):
        model_version = pred.get("modelVersion") or payload.get("modelVersion")
        rows.append(
            {
                "game_id": pred.get("gameId"),
                "game_date": pred.get("gameDate") or pred.get("eventTime", "")[:10],
                "event_time": pred.get("eventTime"),
                "player_id": _player_id(pred),
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
                **_comparison_fields(pred, pred, model_version or ""),
                "games_since_last_hr": pred.get("gamesSinceLastHr"),
                "last_hr_date": pred.get("lastHrDate"),
                "confidence": pred.get("confidence"),
                "model_version": model_version,
                "prediction_ts": pred.get("updatedAt") or payload.get("generatedAt"),
                "quality_flags": json.dumps(pred.get("qualityFlags") or []),
                "top_features": json.dumps(pred.get("topFeatures") or []),
            }
        )
    if not rows:
        return []
    return [(rows, rows[0]["game_date"], rows[0]["model_version"])]


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
        batches = build_mlb_rows(args.mlb_json)
        loaded_total = 0
        for rows, game_date, model_version in batches:
            _delete(
                client,
                table_ids["mlb_home_run_predictions"],
                f"delete from `{table_ids['mlb_home_run_predictions']}` where game_date = @game_date and model_version = @model_version",
                [
                    bigquery.ScalarQueryParameter("game_date", "DATE", game_date),
                    bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
                ],
            )
            loaded_total += _load(
                client,
                table_ids["mlb_home_run_predictions"],
                rows,
                TABLES["mlb_home_run_predictions"]["schema"],
            )
        if loaded_total:
            print(f"Synced {loaded_total} MLB home run predictions to BigQuery")


if __name__ == "__main__":
    main()
