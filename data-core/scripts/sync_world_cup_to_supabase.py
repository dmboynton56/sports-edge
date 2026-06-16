#!/usr/bin/env python3
"""Sync World Cup prediction JSON into Supabase serving tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


MODEL_NAME = "sports_edge_world_cup"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rank_probs_for_team(payload: dict[str, Any], team: str, group_name: str | None) -> dict[str, Any]:
    if not group_name:
        return {}
    for row in payload.get("groupRankProbabilities", {}).get(group_name, []):
        if row.get("team") == team:
            return {
                key: row.get(key)
                for key in ("rank_1", "rank_2", "rank_3", "rank_4")
                if key in row
            }
    return {}


def sync_payload(conn, payload: dict[str, Any], *, dry_run: bool = False) -> dict[str, int]:
    model_version = str(payload["modelVersion"])
    updated_at = payload.get("updatedAt") or _now_iso()
    season = int(payload.get("season", 2026))
    matches = payload.get("matches", [])
    team_probs = payload.get("teamProbabilities", [])
    bracket_source = str(payload.get("bracketSource") or "unknown")
    simulations = int(payload.get("simulations") or 0)

    if dry_run:
        return {
            "matches": len(matches),
            "predictions": len(matches),
            "team_probabilities": len(team_probs),
            "model_runs": 1,
        }

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into world_cup_model_runs
              (model_name, model_version, started_at, simulations, bracket_source, success)
            values (%s, %s, %s, %s, %s, false)
            returning id
            """,
            (MODEL_NAME, model_version, updated_at, simulations, bracket_source),
        )
        run_id = cur.fetchone()[0]

        match_id_map: dict[str, str] = {}
        for match in matches:
            external_match_id = str(match["match_id"])
            status = str(match.get("status") or "scheduled")
            cur.execute(
                """
                insert into world_cup_matches
                  (
                    external_match_id, season, stage, group_name, kickoff_utc,
                    home_team, away_team, status, home_score, away_score,
                    source, raw_record, updated_at
                  )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
                on conflict (external_match_id) do update set
                  season = excluded.season,
                  stage = excluded.stage,
                  group_name = excluded.group_name,
                  kickoff_utc = excluded.kickoff_utc,
                  home_team = excluded.home_team,
                  away_team = excluded.away_team,
                  status = excluded.status,
                  home_score = excluded.home_score,
                  away_score = excluded.away_score,
                  source = excluded.source,
                  raw_record = excluded.raw_record,
                  updated_at = now()
                returning id
                """,
                (
                    external_match_id,
                    season,
                    match.get("stage"),
                    match.get("group"),
                    match.get("kickoff_utc"),
                    match.get("home_team"),
                    match.get("away_team"),
                    status,
                    match.get("home_score"),
                    match.get("away_score"),
                    "sports_edge_world_cup",
                    json.dumps(match, default=_json_default),
                ),
            )
            match_id_map[external_match_id] = str(cur.fetchone()[0])

        cur.execute(
            """
            delete from world_cup_match_predictions
            where model_name = %s
              and model_version = %s
            """,
            (MODEL_NAME, model_version),
        )
        for match in matches:
            match_id = match_id_map[str(match["match_id"])]
            cur.execute(
                """
                insert into world_cup_match_predictions
                  (
                    match_id, model_name, model_version, home_win_prob, draw_prob,
                    away_win_prob, home_knockout_win_prob, away_knockout_win_prob,
                    projected_home_goals, projected_away_goals, prediction_ts,
                    feature_snapshot
                  )
                values (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    match_id,
                    MODEL_NAME,
                    model_version,
                    match.get("home_win_prob"),
                    match.get("draw_prob"),
                    match.get("away_win_prob"),
                    match.get("home_knockout_win_prob"),
                    match.get("away_knockout_win_prob"),
                    match.get("projected_home_goals"),
                    match.get("projected_away_goals"),
                    match.get("prediction_ts") or updated_at,
                    json.dumps(match, default=_json_default),
                ),
            )

        cur.execute(
            """
            delete from world_cup_team_probabilities
            where model_name = %s
              and model_version = %s
            """,
            (MODEL_NAME, model_version),
        )
        for row in team_probs:
            group_name = row.get("group")
            rank_probs = _rank_probs_for_team(payload, str(row["team"]), group_name)
            cur.execute(
                """
                insert into world_cup_team_probabilities
                  (
                    team, group_name, model_name, model_version, simulation_ts,
                    simulations, bracket_source, rating, round_of_32_prob,
                    round_of_16_prob, quarterfinal_prob, semifinal_prob,
                    final_prob, champion_prob, group_rank_probs
                  )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    row.get("team"),
                    group_name,
                    MODEL_NAME,
                    model_version,
                    updated_at,
                    simulations,
                    bracket_source,
                    row.get("rating"),
                    row.get("round_of_32"),
                    row.get("round_of_16"),
                    row.get("quarterfinal"),
                    row.get("semifinal"),
                    row.get("final"),
                    row.get("champion"),
                    json.dumps(rank_probs, default=_json_default),
                ),
            )

        rows_written = len(matches) + len(matches) + len(team_probs)
        cur.execute(
            """
            update world_cup_model_runs
            set finished_at = now(),
                rows_written = %s,
                success = true
            where id = %s
            """,
            (rows_written, run_id),
        )
    conn.commit()
    return {
        "matches": len(matches),
        "predictions": len(matches),
        "team_probabilities": len(team_probs),
        "model_runs": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync World Cup prediction JSON into Supabase.")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    payload = json.loads(args.input_json.read_text())
    creds = load_supabase_credentials()
    missing = {
        "SUPABASE_URL": creds["url"],
        "SUPABASE_DB_PASSWORD or supabaseDBpass": creds["db_password"],
    }
    missing = [key for key, value in missing.items() if not value]
    if missing and not args.dry_run:
        raise RuntimeError(f"Missing Supabase credentials: {', '.join(missing)}")

    if args.dry_run:
        result = sync_payload(None, payload, dry_run=True)
        print(json.dumps(result, sort_keys=True))
        return

    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    try:
        result = sync_payload(conn, payload)
        print(json.dumps(result, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
