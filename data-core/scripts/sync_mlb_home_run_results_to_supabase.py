#!/usr/bin/env python3
"""Sync evaluated MLB home run prediction outcomes to Supabase."""

from __future__ import annotations

import argparse
import json
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


DEFAULT_EVALUATED = ROOT / "notebooks" / "cache" / "mlb_home_run_predictions_evaluated.csv"


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _top_k_bucket(rank: Any) -> str | None:
    try:
        value = int(rank)
    except (TypeError, ValueError):
        return None
    if value <= 10:
        return "top_10"
    if value <= 25:
        return "top_25"
    if value <= 50:
        return "top_50"
    return "field"


def sync_results(path: Path) -> int:
    frame = pd.read_csv(path)
    if frame.empty:
        return 0
    frame = frame[frame["actual_home_run"].notna()].copy()
    if frame.empty:
        return 0

    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    evaluated_at = datetime.now(timezone.utc)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            (
                _clean(row.get("game_id")),
                _clean(str(row.get("game_date"))[:10]),
                _clean(str(row.get("player_id"))),
                _clean(row.get("player_name")),
                _clean(row.get("team")),
                _clean(row.get("opponent")),
                _clean(row.get("model_version")),
                _clean(row.get("prediction_ts")),
                _clean(row.get("rank")),
                _top_k_bucket(row.get("rank")),
                _clean(row.get("hr_probability")),
                bool(int(float(row.get("actual_home_run")))),
                _clean(row.get("actual_home_runs")),
                _clean(row.get("actual_plate_appearances")),
                evaluated_at,
                json.dumps(row.where(pd.notna(row), None).to_dict(), default=str),
            )
        )

    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                insert into mlb_home_run_results (
                  game_id, game_date, player_id, player_name, team, opponent,
                  model_version, prediction_ts, rank, top_k_bucket,
                  model_probability, actual_home_run, actual_home_runs,
                  actual_plate_appearances, evaluated_at, raw_record
                )
                values (
                  %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s::jsonb
                )
                on conflict (game_id, player_id, model_version, prediction_ts)
                do update set
                  rank = excluded.rank,
                  top_k_bucket = excluded.top_k_bucket,
                  model_probability = excluded.model_probability,
                  actual_home_run = excluded.actual_home_run,
                  actual_home_runs = excluded.actual_home_runs,
                  actual_plate_appearances = excluded.actual_plate_appearances,
                  evaluated_at = excluded.evaluated_at,
                  raw_record = excluded.raw_record,
                  updated_at = now()
                """,
                rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync evaluated MLB HR results to Supabase.")
    parser.add_argument("--evaluated-csv", type=Path, default=DEFAULT_EVALUATED)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    rows = sync_results(args.evaluated_csv)
    print(f"Synced {rows} MLB HR result rows to Supabase")


if __name__ == "__main__":
    main()
