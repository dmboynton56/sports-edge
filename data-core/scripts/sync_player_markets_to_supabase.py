#!/usr/bin/env python3
"""Sync PGA tournament and MLB home run artifacts into Supabase."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402


DEFAULT_PGA_JSON = ROOT.parent / "web" / "public" / "data" / "pga_tournaments" / "current.json"
DEFAULT_MLB_JSON = ROOT.parent / "web" / "public" / "data" / "mlb_home_runs.json"


def _json_list(value: Any) -> str:
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps([value])
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "[]"
    return json.dumps(value)


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
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
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

    return (
        _clean(v1_probability),
        _clean(v1_rank),
        _clean(statcast_probability),
        _clean(statcast_rank),
        _clean(statcast_available),
        _clean(model_agreement),
        _clean(_comparison_value(pred, comparison, "consensusScore")),
        _clean(_comparison_value(pred, comparison, "marketSignalRank")),
    )


def sync_pga(conn, path: Path) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    event = payload["event"]
    preds = payload.get("predictions", [])
    prediction_ts = payload.get("generatedAt") or datetime.now(timezone.utc).isoformat()
    meta = payload.get("predictionMeta", {})
    model_version = str(meta.get("model_version") or "pga-baseline-v0")
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into pga_tournaments (event_key, season, name, start_date, end_date, course, par, field_size, status, source, raw_record, updated_at)
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
            on conflict (event_key) do update set
              season = excluded.season,
              name = excluded.name,
              start_date = excluded.start_date,
              end_date = excluded.end_date,
              course = excluded.course,
              par = excluded.par,
              field_size = excluded.field_size,
              status = excluded.status,
              source = excluded.source,
              raw_record = excluded.raw_record,
              updated_at = now()
            """,
            (
                event["eventKey"],
                int(event["season"]),
                event["name"],
                event["startDate"],
                event["endDate"],
                event.get("course"),
                event.get("par"),
                meta.get("n_players") or len(preds),
                event.get("status", "scheduled"),
                "pga_tournament_dashboard_json",
                json.dumps(event),
            ),
            prepare=False,
        )
        cur.execute("delete from pga_player_predictions where event_key = %s and model_version = %s", (event["eventKey"], model_version), prepare=False)
        rows = []
        for pred in preds:
            rows.append(
                (
                    event["eventKey"],
                    pred.get("player"),
                    pred.get("player_id"),
                    _clean(pred.get("exp_sg_per_round")),
                    _clean(pred.get("best_calibrated_target_made_cut_prob")),
                    _clean((pred.get("sim_top5_pct") or 0) / 100 if pred.get("sim_top5_pct") is not None else None),
                    _clean(pred.get("best_calibrated_target_top10_prob")),
                    _clean(pred.get("best_calibrated_target_top20_prob")),
                    _clean(pred.get("best_calibrated_target_win_prob")),
                    _clean(pred.get("projected_total_strokes")),
                    _clean(pred.get("projected_score_to_par")),
                    model_version,
                    prediction_ts,
                    meta.get("n_sims"),
                    _clean(pred.get("confidence")),
                    _json_list(pred.get("quality_flags")),
                    json.dumps({"source": pred.get("source"), "starts_before": pred.get("starts_before")}),
                )
            )
        cur.executemany(
            """
            insert into pga_player_predictions (
              event_key, player_name, player_id, exp_sg_per_round, make_cut_prob, top5_prob,
              top10_prob, top20_prob, win_prob, projected_total_strokes, projected_score_to_par,
              model_version, prediction_ts, simulation_count, confidence, quality_flags, feature_snapshot
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            rows,
        )
    conn.commit()
    return 1, len(rows)


def _iter_mlb_prediction_payloads(payload: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]], str | None]]:
    models = payload.get("models")
    if isinstance(models, dict) and models:
        out: list[tuple[str, list[dict[str, Any]], str | None]] = []
        for model_key, model_payload in models.items():
            if not isinstance(model_payload, dict):
                continue
            out.append(
                (
                    str(model_payload.get("modelVersion") or model_key),
                    model_payload.get("predictions") or [],
                    model_payload.get("modelVersion") or model_key,
                )
            )
        return out
    return [
        (
            str(payload.get("modelVersion") or "mlb-hr-v1-heuristic"),
            payload.get("predictions") or [],
            payload.get("modelVersion"),
        )
    ]


def sync_mlb(conn, path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparisons = _comparison_lookup(payload)
    total = 0
    for model_version, predictions, _ in _iter_mlb_prediction_payloads(payload):
        rows = []
        for pred in predictions:
            sync_model_version = pred.get("modelVersion") or model_version
            comparison = comparisons.get(_prediction_key(pred))
            rows.append(
                (
                    pred.get("gameId"),
                    pred.get("gameDate") or pred.get("eventTime", "")[:10],
                    pred.get("eventTime"),
                    _player_id(pred),
                    pred.get("player"),
                    pred.get("team"),
                    pred.get("opponent"),
                    pred.get("venue"),
                    pred.get("lineupSlot"),
                    pred.get("lineupStatus") or "projected",
                    pred.get("opposingProbablePitcher"),
                    pred.get("modelProbability"),
                    pred.get("baselineProbability"),
                    pred.get("rank"),
                    *_comparison_fields(pred, comparison, sync_model_version),
                    pred.get("gamesSinceLastHr"),
                    pred.get("lastHrDate"),
                    pred.get("confidence"),
                    sync_model_version,
                    pred.get("updatedAt") or payload.get("generatedAt"),
                    json.dumps(pred.get("qualityFlags") or []),
                    json.dumps(pred.get("topFeatures") or []),
                )
            )
        if not rows:
            continue
        sync_model_version = rows[0][25]
        game_date = rows[0][1]
        with conn.cursor() as cur:
            cur.execute(
                "delete from mlb_home_run_predictions where game_date = %s and model_version = %s",
                (game_date, sync_model_version),
                prepare=False,
            )
            cur.executemany(
                """
                insert into mlb_home_run_predictions (
                  game_id, game_date, event_time, player_id, player_name, team, opponent, venue,
                  lineup_slot, lineup_status, opposing_probable_pitcher, hr_probability,
                  baseline_probability, rank, v1_probability, v1_rank, statcast_probability,
                  statcast_rank, statcast_available, model_agreement, consensus_score,
                  market_signal_rank, games_since_last_hr, last_hr_date, confidence,
                  model_version, prediction_ts, quality_flags, top_features
                )
                values (
                  %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s::jsonb, %s::jsonb
                )
                """,
                rows,
            )
        conn.commit()
        total += len(rows)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync player-market artifacts to Supabase.")
    parser.add_argument("--pga-json", type=Path, default=DEFAULT_PGA_JSON)
    parser.add_argument("--mlb-json", type=Path, default=DEFAULT_MLB_JSON)
    parser.add_argument("--skip-pga", action="store_true")
    parser.add_argument("--skip-mlb", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    try:
        if not args.skip_pga and args.pga_json.exists():
            tournaments, predictions = sync_pga(conn, args.pga_json)
            print(f"Synced {tournaments} PGA tournaments and {predictions} PGA predictions")
        if not args.skip_mlb and args.mlb_json.exists():
            rows = sync_mlb(conn, args.mlb_json)
            print(f"Synced {rows} MLB home run predictions")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
