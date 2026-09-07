#!/usr/bin/env python3
"""Grade timestamp-safe NFL v2 live predictions from the Supabase serving path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.nfl_live import LIVE_MODEL_VERSION
from src.models.nfl_v2 import margin_metrics, probability_metrics, write_json
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


DEFAULT_OUTPUT = ROOT / "notebooks" / "cache" / "nfl_v2_live_performance.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=LIVE_MODEL_VERSION)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    return parser.parse_args()


def fetch_graded_predictions(conn, model_version: str, season: int) -> pd.DataFrame:
    query = """
        SELECT DISTINCT ON (g.id)
          g.id::text AS game_id,
          g.season,
          g.week,
          g.game_time_utc,
          g.home_team,
          g.away_team,
          g.home_score,
          g.away_score,
          p.my_home_win_prob,
          p.my_spread,
          p.asof_ts
        FROM games g
        JOIN model_predictions p ON p.game_id = g.id
        WHERE g.league = 'NFL'
          AND g.season = %s
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND p.model_version = %s
          AND p.asof_ts < g.game_time_utc
        ORDER BY g.id, p.asof_ts DESC
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (season, model_version), prepare=False)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def build_report(frame: pd.DataFrame, model_version: str, season: int) -> dict:
    generated_at = datetime.now(timezone.utc).isoformat()
    if frame.empty:
        return {
            "generated_at": generated_at,
            "model_version": model_version,
            "season": season,
            "status": "awaiting_graded_games",
            "graded_games": 0,
            "metrics": {},
            "segments": [],
            "gaps": ["No completed games with a timestamp-safe v2 prediction are available yet."],
        }

    numeric = frame.copy()
    for column in ("home_score", "away_score", "my_home_win_prob", "my_spread"):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(subset=["home_score", "away_score", "my_home_win_prob", "my_spread"])
    if numeric.empty:
        return {
            "generated_at": generated_at,
            "model_version": model_version,
            "season": season,
            "status": "integrity_failure",
            "graded_games": 0,
            "metrics": {},
            "segments": [],
            "alerts": ["Completed prediction rows exist but required scores or model outputs are missing."],
            "gaps": ["Required grading fields are incomplete."],
        }
    numeric["home_win"] = (numeric["home_score"] > numeric["away_score"]).astype(int)
    numeric["home_margin"] = numeric["home_score"] - numeric["away_score"]
    numeric["predicted_margin"] = -numeric["my_spread"]

    probability = probability_metrics(numeric["home_win"], numeric["my_home_win_prob"])
    margin = margin_metrics(numeric["home_margin"], numeric["predicted_margin"])
    segments = []
    for label, subset in (
        ("weeks_1_4", numeric[pd.to_numeric(numeric["week"], errors="coerce") <= 4]),
        ("week_5_plus", numeric[pd.to_numeric(numeric["week"], errors="coerce") >= 5]),
    ):
        if subset.empty:
            continue
        segment = probability_metrics(subset["home_win"], subset["my_home_win_prob"])
        segments.append({"segment": label, **{key: segment.get(key) for key in ("rows", "brier", "log_loss", "ece_10", "home_probability_bias", "auc")}})

    weeks = int(pd.to_numeric(numeric["week"], errors="coerce").nunique())
    eligible_sample = len(numeric) >= 64 and weeks >= 4
    alerts = []
    if eligible_sample and probability["ece_10"] > 0.05:
        alerts.append("ECE exceeds 0.05")
    if eligible_sample and abs(probability["home_probability_bias"]) > 0.03:
        alerts.append("home-probability bias exceeds +/-3 percentage points")
    if eligible_sample and abs(margin["home_margin_bias"]) > 0.75:
        alerts.append("home-margin bias exceeds +/-0.75 points")
    for segment in segments:
        if segment["rows"] >= 40 and abs(segment["home_probability_bias"]) > 0.08:
            alerts.append(f"{segment['segment']} home-probability bias exceeds +/-8 points")

    return {
        "generated_at": generated_at,
        "model_version": model_version,
        "season": season,
        "status": "alert" if alerts else "monitoring" if not eligible_sample else "sample_gate_reached",
        "graded_games": int(len(numeric)),
        "graded_weeks": weeks,
        "sample_gate": {"minimum_games": 64, "minimum_weeks": 4, "reached": eligible_sample},
        "metrics": {"probability": probability, "margin": margin},
        "segments": segments,
        "alerts": alerts,
        "gaps": [] if eligible_sample else ["Four weeks and 64 graded games are required before a live-rollout decision."],
    }


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file)
    credentials = load_supabase_credentials()
    if not credentials["db_password"]:
        raise SystemExit("Supabase database credentials are required to grade NFL v2.")
    conn = create_pg_connection(
        credentials["url"], credentials["db_password"], credentials["db_host"],
        credentials["db_port"], credentials["db_name"], credentials["db_user"],
    )
    try:
        report = build_report(fetch_graded_predictions(conn, args.model_version, args.season), args.model_version, args.season)
    finally:
        conn.close()
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
