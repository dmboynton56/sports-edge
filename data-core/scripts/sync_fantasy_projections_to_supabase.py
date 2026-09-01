#!/usr/bin/env python3
"""Publish the generated fantasy artifact to Supabase public-read tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "data-core") not in sys.path:
    sys.path.insert(0, str(ROOT / "data-core"))

from src.utils.supabase_pg import load_supabase_credentials  # noqa: E402


DEFAULT_ARTIFACT = ROOT / "web" / "public" / "data" / "fantasy_projections.json"
DEFAULT_RETAIN_RUNS = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--retain-runs",
        type=int,
        default=DEFAULT_RETAIN_RUNS,
        help="Number of recent Supabase projection runs to retain (default: 7)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.retain_runs < 1:
        parser.error("--retain-runs must be at least 1")
    return args


def connection_string() -> str:
    credentials = load_supabase_credentials()
    host = credentials.get("db_host")
    password = credentials.get("db_password")
    if not host or not password:
        raise RuntimeError("SUPABASE_DB_HOST and SUPABASE_DB_PASSWORD are required")
    return (
        f"host={host} port={credentials.get('db_port', '5432')} "
        f"dbname={credentials.get('db_name', 'postgres')} user={credentials.get('db_user', 'postgres')} "
        f"password={password} sslmode=require"
    )


def rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    base_by_id = {str(row.get("player_id")): row for row in (payload.get("projections") or [])}
    rows = list(payload.get("projections") or [])
    for week, weekly in (payload.get("weekly") or {}).items():
        rows.extend({**base_by_id.get(str(row.get("player_id")), {}), **row} for row in (weekly or []))
    return rows


def main() -> None:
    args = parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = rows_from_payload(payload)
    run_id = f"fantasy_{payload.get('season')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    if args.dry_run:
        print(json.dumps({"run_id": run_id, "rows": len(rows), "season": payload.get("season")}, indent=2))
        return

    with psycopg.connect(connection_string()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                insert into fantasy_projection_runs
                  (run_id, season, model_version, scope, week, scoring_profile, status, metrics, gaps, source_updated_at)
                values (%s, %s, %s, 'preseason', 0, 'full_ppr', %s, %s::jsonb, %s, %s)
                """,
                (
                    run_id,
                    int(payload.get("season")),
                    payload.get("modelVersion", "fantasy-v1"),
                    payload.get("productionStatus", "candidate"),
                    json.dumps(payload.get("metrics") or {}),
                    payload.get("gaps") or [],
                    payload.get("generatedAt"),
                ),
            )
            for row in rows:
                cursor.execute(
                    """
                    insert into fantasy_player_projections
                      (run_id, player_id, player_name, position, team, season, scope, week,
                       projected_games, statline, statline_low, statline_high, points,
                       floor_points, ceiling_points, points_per_game, overall_rank,
                       position_rank, tier, adp, adp_rank, adp_tier, adp_source,
                       confidence, availability, explanation, model_version, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (run_id, player_id, season, scope, week) do update set
                      points = excluded.points,
                      floor_points = excluded.floor_points,
                      ceiling_points = excluded.ceiling_points,
                      points_per_game = excluded.points_per_game,
                      statline = excluded.statline,
                      statline_low = excluded.statline_low,
                      statline_high = excluded.statline_high,
                      updated_at = excluded.updated_at
                    """,
                    (
                        run_id,
                        row.get("player_id"),
                        row.get("player_name"),
                        row.get("position"),
                        row.get("team"),
                        int(row.get("season")),
                        row.get("scope", "preseason"),
                        int(row.get("week", 0)),
                        row.get("projected_games"),
                        json.dumps(row.get("statline") or {}),
                        json.dumps(row.get("statline_low") or {}),
                        json.dumps(row.get("statline_high") or {}),
                        row.get("points", 0),
                        row.get("floor_points", 0),
                        row.get("ceiling_points", 0),
                        row.get("points_per_game", 0),
                        row.get("overall_rank"),
                        row.get("position_rank"),
                        row.get("tier"),
                        row.get("adp"),
                        row.get("adp_rank"),
                        row.get("adp_tier"),
                        row.get("adp_source"),
                        row.get("confidence", "low"),
                        row.get("availability", "expected"),
                        row.get("explanation") or [],
                        row.get("model_version", payload.get("modelVersion", "fantasy-v1")),
                        row.get("updated_at") or payload.get("generatedAt"),
                    ),
                )
            cursor.execute(
                """
                delete from fantasy_projection_runs
                where run_id in (
                  select run_id
                  from fantasy_projection_runs
                  order by generated_at desc, run_id desc
                  offset %s
                )
                returning run_id
                """,
                (args.retain_runs,),
            )
            pruned_runs = len(cursor.fetchall())
        connection.commit()
    print(
        f"Synced {len(rows)} fantasy rows to Supabase run {run_id}; "
        f"pruned {pruned_runs} superseded runs (retaining {args.retain_runs})"
    )


if __name__ == "__main__":
    main()
