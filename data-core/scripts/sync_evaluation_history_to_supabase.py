#!/usr/bin/env python3
"""Sync cached performance-history artifacts into Supabase evaluation tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_performance_history import build_history
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


MODEL_NAME = "sports_edge"


@dataclass(frozen=True)
class EvaluationPayload:
    league: str
    model_name: str
    model_version: str
    evaluation_name: str
    metrics: dict[str, Any]
    calibration: dict[str, Any]
    artifact_refs: list[str]
    notes: str | None


@dataclass(frozen=True)
class StrategyPayload:
    league: str
    model_name: str
    model_version: str
    strategy_id: str
    market: str
    odds_source: str | None
    edge_threshold: float | None
    min_confidence: float | None
    sample_size: int | None
    bets: int | None
    wins: int | None
    losses: int | None
    pushes: int | None
    units: float | None
    roi: float | None
    metrics: dict[str, Any]


CALIBRATION_KEYS = {
    "auc",
    "avg_pred_home_win",
    "baseline_brier",
    "baseline_log_loss",
    "bigquery_auc",
    "bigquery_brier",
    "bigquery_log_loss",
    "brier",
    "ece",
    "ece_10",
    "log_loss",
    "roc_auc",
    "win_auc",
    "win_brier",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _evaluation_name(sport: dict[str, Any]) -> str:
    season = str(sport["season"]).lower().replace(" ", "_").replace(">=", "gte")
    market = str(sport["market"]).lower().replace("/", "_").replace(" ", "_")
    return f"performance_history_{sport['sport'].lower()}_{season}_{market}"


def build_evaluation_payloads(history: dict[str, Any]) -> tuple[list[EvaluationPayload], list[StrategyPayload]]:
    evaluations: list[EvaluationPayload] = []
    strategies: list[StrategyPayload] = []

    for sport in history.get("sports", []):
        league = str(sport["sport"]).upper()
        metrics = {
            "market": sport.get("market"),
            "season": sport.get("season"),
            "data_source": sport.get("data_source"),
            "odds_status": sport.get("odds_status"),
            "sample": sport.get("sample", {}),
            "metrics": sport.get("metrics", {}),
            "gaps": sport.get("gaps", []),
        }
        calibration = {
            key: value
            for key, value in sport.get("metrics", {}).items()
            if key in CALIBRATION_KEYS
        }
        evaluation = EvaluationPayload(
            league=league,
            model_name=MODEL_NAME,
            model_version=str(sport["model_version"]),
            evaluation_name=_evaluation_name(sport),
            metrics=metrics,
            calibration=calibration,
            artifact_refs=list(sport.get("artifact_refs", [])),
            notes="; ".join(sport.get("gaps", [])) or None,
        )
        evaluations.append(evaluation)

        sport_metrics = sport.get("metrics", {})
        sample = sport.get("sample", {})
        market = str(sport.get("market", "unknown"))
        odds_source = sport.get("odds_status")

        if league == "NBA":
            strategies.append(
                StrategyPayload(
                    league=league,
                    model_name=MODEL_NAME,
                    model_version=evaluation.model_version,
                    strategy_id="bigquery_default_edge_1_conf_0",
                    market=market,
                    odds_source=odds_source,
                    edge_threshold=_as_float(sport_metrics.get("bigquery_default_edge_threshold", 1.0)),
                    min_confidence=_as_float(sport_metrics.get("bigquery_default_min_confidence", 0.0)),
                    sample_size=_as_int(sample.get("completed_games")),
                    bets=_as_int(sport_metrics.get("bigquery_default_bets")),
                    wins=_as_int(sport_metrics.get("bigquery_default_wins")),
                    losses=None,
                    pushes=None,
                    units=None,
                    roi=_as_float(sport_metrics.get("bigquery_default_roi")),
                    metrics={"accuracy": sport_metrics.get("bigquery_default_accuracy")},
                )
            )
            strategies.append(
                StrategyPayload(
                    league=league,
                    model_name=MODEL_NAME,
                    model_version=evaluation.model_version,
                    strategy_id="bigquery_best_reported_sweep",
                    market=market,
                    odds_source=odds_source,
                    edge_threshold=_as_float(sport_metrics.get("bigquery_best_sweep_edge_threshold")),
                    min_confidence=_as_float(sport_metrics.get("bigquery_best_sweep_min_confidence")),
                    sample_size=_as_int(sample.get("completed_games")),
                    bets=_as_int(sport_metrics.get("bigquery_best_sweep_bets")),
                    wins=None,
                    losses=None,
                    pushes=None,
                    units=None,
                    roi=_as_float(sport_metrics.get("best_reported_sweep_roi")),
                    metrics={},
                )
            )

        if "supabase_ats_roi" in sport_metrics:
            wins = _as_int(sport_metrics.get("supabase_ats_wins"))
            losses = _as_int(sport_metrics.get("supabase_ats_losses"))
            pushes = _as_int(sport_metrics.get("supabase_ats_pushes"))
            strategies.append(
                StrategyPayload(
                    league=league,
                    model_name=MODEL_NAME,
                    model_version=evaluation.model_version,
                    strategy_id="supabase_ats_flat_minus_110",
                    market=market,
                    odds_source=odds_source,
                    edge_threshold=None,
                    min_confidence=None,
                    sample_size=_as_int(sample.get("supabase_graded_games")),
                    bets=(wins or 0) + (losses or 0) + (pushes or 0),
                    wins=wins,
                    losses=losses,
                    pushes=pushes,
                    units=None,
                    roi=_as_float(sport_metrics.get("supabase_ats_roi")),
                    metrics={},
                )
            )

        if league == "MLB" and sport_metrics.get("flat_roi") is not None:
            strategies.append(
                StrategyPayload(
                    league=league,
                    model_name=MODEL_NAME,
                    model_version=evaluation.model_version,
                    strategy_id="moneyline_flat_pick",
                    market=market,
                    odds_source=odds_source,
                    edge_threshold=None,
                    min_confidence=None,
                    sample_size=_as_int(sample.get("odds_joined_games")),
                    bets=_as_int(sample.get("odds_joined_games")),
                    wins=None,
                    losses=None,
                    pushes=None,
                    units=None,
                    roi=_as_float(sport_metrics.get("flat_roi")),
                    metrics={
                        "brier": sport_metrics.get("brier"),
                        "log_loss": sport_metrics.get("log_loss"),
                        "roc_auc": sport_metrics.get("roc_auc"),
                    },
                )
            )

    return evaluations, strategies


def _load_history(cache_dir: Path, history_json: Path | None) -> dict[str, Any]:
    if history_json and history_json.exists():
        return json.loads(history_json.read_text(encoding="utf-8"))
    return build_history(cache_dir)


def sync_payloads(conn, evaluations: list[EvaluationPayload], strategies: list[StrategyPayload]) -> tuple[int, int]:
    strategies_by_eval = {}
    for strategy in strategies:
        key = (strategy.league, strategy.model_name, strategy.model_version)
        strategies_by_eval.setdefault(key, []).append(strategy)

    inserted_evaluations = 0
    inserted_strategies = 0
    generated_at = datetime.now(timezone.utc)

    with conn.cursor() as cur:
        for evaluation in evaluations:
            cur.execute(
                """
                DELETE FROM model_evaluation_runs
                WHERE league = %s
                  AND model_name = %s
                  AND model_version = %s
                  AND evaluation_name = %s
                """,
                (
                    evaluation.league,
                    evaluation.model_name,
                    evaluation.model_version,
                    evaluation.evaluation_name,
                ),
                prepare=False,
            )
            cur.execute(
                """
                INSERT INTO model_evaluation_runs (
                    league,
                    model_name,
                    model_version,
                    evaluation_name,
                    generated_at,
                    metrics,
                    calibration,
                    artifact_refs,
                    status,
                    notes
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, 'candidate', %s)
                RETURNING id
                """,
                (
                    evaluation.league,
                    evaluation.model_name,
                    evaluation.model_version,
                    evaluation.evaluation_name,
                    generated_at,
                    json.dumps(evaluation.metrics, default=_json_safe),
                    json.dumps(evaluation.calibration, default=_json_safe),
                    evaluation.artifact_refs,
                    evaluation.notes,
                ),
                prepare=False,
            )
            evaluation_id = cur.fetchone()[0]
            inserted_evaluations += 1

            for strategy in strategies_by_eval.get(
                (evaluation.league, evaluation.model_name, evaluation.model_version),
                [],
            ):
                cur.execute(
                    """
                    INSERT INTO strategy_backtest_results (
                        evaluation_run_id,
                        league,
                        model_name,
                        model_version,
                        strategy_id,
                        market,
                        odds_source,
                        edge_threshold,
                        min_confidence,
                        sample_size,
                        bets,
                        wins,
                        losses,
                        pushes,
                        units,
                        roi,
                        metrics
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        evaluation_id,
                        strategy.league,
                        strategy.model_name,
                        strategy.model_version,
                        strategy.strategy_id,
                        strategy.market,
                        strategy.odds_source,
                        strategy.edge_threshold,
                        strategy.min_confidence,
                        strategy.sample_size,
                        strategy.bets,
                        strategy.wins,
                        strategy.losses,
                        strategy.pushes,
                        strategy.units,
                        strategy.roi,
                        json.dumps(strategy.metrics, default=_json_safe),
                    ),
                    prepare=False,
                )
                inserted_strategies += 1

    conn.commit()
    return inserted_evaluations, inserted_strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync performance-history evaluation rows to Supabase.")
    parser.add_argument("--cache-dir", default=str(ROOT / "notebooks" / "cache"))
    parser.add_argument("--history-json", default=str(ROOT / "notebooks" / "cache" / "performance_history.json"))
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = _load_history(Path(args.cache_dir), Path(args.history_json) if args.history_json else None)
    evaluations, strategies = build_evaluation_payloads(history)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "evaluations": [payload.__dict__ for payload in evaluations],
                    "strategies": [payload.__dict__ for payload in strategies],
                },
                indent=2,
                default=_json_safe,
            )
        )
        return

    load_dotenv(args.env_file)
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
        inserted_evaluations, inserted_strategies = sync_payloads(conn, evaluations, strategies)
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "inserted_evaluations": inserted_evaluations,
                "inserted_strategies": inserted_strategies,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
