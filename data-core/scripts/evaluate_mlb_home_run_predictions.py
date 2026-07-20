#!/usr/bin/env python3
"""Join MLB HR prediction rows to completed outcomes and score the board."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402

DEFAULT_PREDICTIONS = ROOT / "notebooks" / "cache" / "mlb_home_run_predictions.csv"
DEFAULT_CACHE = ROOT / "notebooks" / "cache" / "mlb_home_run_outcome_boxscores.jsonl"
DEFAULT_OUT_CSV = ROOT / "notebooks" / "cache" / "mlb_home_run_predictions_evaluated.csv"
DEFAULT_METRICS = ROOT / "notebooks" / "cache" / "mlb_home_run_prediction_outcome_metrics.json"
MLB_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"


def _game_pk_from_id(value: Any) -> int | None:
    text = str(value or "")
    if text.startswith("MLB_"):
        text = text.split("MLB_", 1)[1]
    try:
        return int(text)
    except ValueError:
        return None


def _read_cached_boxscores(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "game_pk" in row:
            out[int(row["game_pk"])] = row
    return out


def _write_cached_boxscores(path: Path, payloads: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(payloads[key], sort_keys=True) for key in sorted(payloads)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _fetch_boxscore(game_pk: int, timeout: int) -> dict[str, Any]:
    response = requests.get(MLB_BOXSCORE_URL.format(game_pk=int(game_pk)), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    payload["game_pk"] = int(game_pk)
    return payload


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _extract_outcomes(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    game_pk = int(payload.get("game_pk"))
    for side in ("home", "away"):
        team = (payload.get("teams") or {}).get(side) or {}
        for player_id in team.get("batters") or []:
            player = (team.get("players") or {}).get(f"ID{int(player_id)}") or {}
            stats = (player.get("stats") or {}).get("batting") or {}
            home_runs = _safe_int(stats.get("homeRuns"))
            plate_appearances = _safe_int(stats.get("plateAppearances"))
            rows.append(
                {
                    "game_pk": game_pk,
                    "player_id": int(player_id),
                    "actual_home_runs": home_runs,
                    "actual_home_run": int(home_runs > 0),
                    "actual_plate_appearances": plate_appearances,
                }
            )
    return pd.DataFrame(rows)


def _clip_prob(values: pd.Series) -> np.ndarray:
    return np.clip(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float), 1e-6, 1 - 1e-6)


def _top_k_hit_rate(frame: pd.DataFrame, prob_col: str, k: int) -> float:
    top = (
        frame.sort_values(["game_date", prob_col], ascending=[True, False])
        .groupby("game_date", group_keys=False)
        .head(k)
    )
    return float(top["actual_home_run"].mean()) if len(top) else float("nan")


def _score(frame: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    y = pd.to_numeric(frame["actual_home_run"], errors="coerce").fillna(0).astype(int)
    p = _clip_prob(frame[prob_col])
    out: dict[str, Any] = {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "top_10_hit_rate": _top_k_hit_rate(frame, prob_col, 10),
        "top_25_hit_rate": _top_k_hit_rate(frame, prob_col, 25),
    }
    if y.nunique() > 1:
        out["auc"] = float(roc_auc_score(y, p))
    return out


def _as_nullable_int(series: pd.Series) -> pd.Series:
    """Normalize merge keys so Supabase/CSV strings and API ints join cleanly."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def evaluate_predictions(predictions: pd.DataFrame, cache_path: Path, *, timeout: int, sleep: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = predictions.copy()
    if "game_pk" not in predictions.columns:
        predictions["game_pk"] = predictions["game_id"].map(_game_pk_from_id)
    predictions["game_pk"] = _as_nullable_int(predictions["game_pk"])
    if "player_id" in predictions.columns:
        predictions["player_id"] = _as_nullable_int(predictions["player_id"])
    cached = _read_cached_boxscores(cache_path)
    changed = False
    for game_pk in sorted(predictions["game_pk"].dropna().astype(int).unique()):
        if game_pk not in cached:
            try:
                cached[game_pk] = _fetch_boxscore(int(game_pk), timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                cached[game_pk] = {"game_pk": int(game_pk), "error": str(exc)}
            changed = True
            time.sleep(sleep)
    if changed:
        _write_cached_boxscores(cache_path, cached)

    outcomes = []
    for payload in cached.values():
        if payload.get("error"):
            continue
        outcomes.append(_extract_outcomes(payload))
    if not outcomes:
        evaluated = predictions.copy()
        evaluated["actual_home_run"] = np.nan
        evaluated["actual_home_runs"] = np.nan
        evaluated["actual_plate_appearances"] = np.nan
    else:
        outcome_frame = pd.concat(outcomes, ignore_index=True).drop_duplicates(["game_pk", "player_id"])
        outcome_frame["game_pk"] = _as_nullable_int(outcome_frame["game_pk"])
        outcome_frame["player_id"] = _as_nullable_int(outcome_frame["player_id"])
        evaluated = predictions.merge(outcome_frame, on=["game_pk", "player_id"], how="left")

    scored = evaluated[evaluated["actual_home_run"].notna()].copy()
    metrics: dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "evaluated" if len(scored) else "pending_outcomes",
        "prediction_rows": int(len(predictions)),
        "evaluated_rows": int(len(scored)),
        "missing_outcome_rows": int(evaluated["actual_home_run"].isna().sum()),
        "evaluated_dates": sorted(scored["game_date"].astype(str).unique().tolist()) if len(scored) else [],
        "model_versions": sorted(predictions.get("model_version", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
    }
    if len(scored):
        metrics["model_probability"] = _score(scored, "hr_probability")
        if "baseline_probability" in scored.columns:
            metrics["baseline_probability"] = _score(scored, "baseline_probability")
    return evaluated, metrics


def load_predictions_from_supabase(game_date: str) -> pd.DataFrame:
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
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  game_id,
                  game_date,
                  event_time,
                  player_id,
                  player_name,
                  team,
                  opponent,
                  venue,
                  lineup_slot,
                  hr_probability,
                  baseline_probability,
                  rank,
                  confidence,
                  model_version,
                  prediction_ts,
                  quality_flags,
                  top_features
                from mlb_home_run_predictions
                where game_date = %s
                order by model_version, rank nulls last, player_name
                """,
                (game_date,),
                prepare=False,
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MLB HR prediction rows against completed boxscores.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--from-supabase", action="store_true", help="Load predictions for --date from Supabase.")
    parser.add_argument("--date", help="Prediction game_date to evaluate, YYYY-MM-DD.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.from_supabase:
        if not args.date:
            raise SystemExit("--date is required with --from-supabase")
        predictions = load_predictions_from_supabase(args.date)
        if predictions.empty:
            raise SystemExit(f"No MLB HR predictions found in Supabase for {args.date}")
    else:
        predictions = pd.read_csv(args.predictions)
        if args.date and "game_date" in predictions.columns:
            predictions = predictions[predictions["game_date"].astype(str).str.slice(0, 10) == args.date].copy()
        if predictions.empty:
            raise SystemExit("No MLB HR prediction rows to evaluate.")
    evaluated, metrics = evaluate_predictions(predictions, args.cache, timeout=args.timeout, sleep=args.sleep)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(args.out_csv, index=False)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Wrote evaluated predictions to {args.out_csv}")
    print(f"Wrote metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
