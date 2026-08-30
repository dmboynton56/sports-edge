#!/usr/bin/env python3
"""Create, publish, and finalize immutable MLB HR board runs.

The workflow calls this script before and after the existing prediction/odds
sync steps.  It deliberately writes a snapshot into ``mlb_home_run_board_rows``
instead of asking the website to recompute edge values from moving odds data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mlb_hr_board_contract import (  # noqa: E402
    classify_run,
    coverage_stats,
    is_priced_row,
    parse_timestamp,
    schedule_confirms_slate_over,
)
from src.data.mlb_fetcher import fetch_mlb_schedule  # noqa: E402
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402


def _now(value: str | None) -> datetime:
    parsed = parse_timestamp(value) if value else None
    return parsed or datetime.now(timezone.utc)


def _connection():
    creds = load_supabase_credentials()
    return create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )


def start_run(conn, args: argparse.Namespace) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into mlb_home_run_board_runs
              (run_key, slate_date, model_version, run_window, status, workflow_url, started_at)
            values (%s, %s, %s, %s, 'running', %s, %s)
            on conflict (run_key) do update set
              workflow_url = coalesce(excluded.workflow_url, mlb_home_run_board_runs.workflow_url)
            returning run_id, status
            """,
            (args.run_key, args.slate_date, args.model_version, args.run_window, args.workflow_url, _now(args.timestamp)),
            prepare=False,
        )
        run_id, status = cur.fetchone()
    conn.commit()
    print(json.dumps({"run_id": str(run_id), "run_key": args.run_key, "status": status}))


def _fetch_rows(conn, args: argparse.Namespace) -> list[dict[str, Any]]:
    model_like = args.model_like or "mlb-hr-v1%"
    with conn.cursor() as cur:
        cur.execute(
            """
            select
              game_id, game_date, event_time, player_id, player_name, team, opponent, venue,
              lineup_slot, lineup_status, opposing_probable_pitcher, hr_probability,
              baseline_probability, rank, model_version, prediction_ts, quality_flags,
              statcast_available, statcast_coverage,
              best_market, best_book, best_price, implied_probability, no_vig_probability,
              market_probability, edge, ev, kelly, odds_books_count,
              odds_snapshot_ts, odds_status
            from mlb_home_run_edges_latest
            where game_date = %s and model_version like %s
            order by rank nulls last, player_name
            """,
            (args.slate_date, model_like),
            prepare=False,
        )
        columns = [description[0] for description in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _previous_nonempty_board(conn, args: argparse.Namespace) -> tuple[str, list[str]] | None:
    """Return the most recent public, non-empty snapshot for this slate."""

    with conn.cursor() as cur:
        cur.execute(
            """
            select r.run_key, array_agg(distinct b.game_id)
            from mlb_home_run_board_runs r
            join mlb_home_run_board_rows b on b.run_id = r.run_id
            where r.slate_date = %s
              and r.run_key <> %s
              and r.status in ('healthy', 'partial')
            group by r.run_id, r.run_key, r.completed_at, r.started_at
            order by r.completed_at desc nulls last, r.started_at desc
            limit 1
            """,
            (args.slate_date, args.run_key),
            prepare=False,
        )
        result = cur.fetchone()
    if result is None:
        return None
    return str(result[0]), [str(game_id) for game_id in result[1] if game_id]


def _official_schedule_confirms_slate_over(slate_date: str, expected_game_ids: list[str]) -> bool:
    slate_year = datetime.fromisoformat(slate_date).year
    schedule = fetch_mlb_schedule(
        slate_year,
        game_types=("R", "F", "D", "L", "W"),
        start_date=slate_date,
        end_date=slate_date,
        include_uncompleted=True,
    )
    return schedule_confirms_slate_over(schedule.to_dict("records"), expected_game_ids)


def _refuse_unsafe_empty_overwrite(conn, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    if rows:
        return
    previous = _previous_nonempty_board(conn, args)
    if previous is None:
        return
    previous_run_key, expected_game_ids = previous
    try:
        slate_over = _official_schedule_confirms_slate_over(args.slate_date, expected_game_ids)
    except Exception as exc:  # noqa: BLE001 - schedule uncertainty must fail closed
        raise RuntimeError(
            f"Refusing empty MLB HR board publish for {args.slate_date}: the existing non-empty "
            f"board {previous_run_key} is protected because official slate completion could not be verified: {exc}"
        ) from exc
    if not slate_over:
        raise RuntimeError(
            f"Refusing empty MLB HR board publish for {args.slate_date}: the existing non-empty "
            f"board {previous_run_key} remains live because the official MLB schedule is not fully final."
        )


def publish_rows(conn, args: argparse.Namespace) -> None:
    publication_ts = _now(args.timestamp)
    rows = _fetch_rows(conn, args)
    _refuse_unsafe_empty_overwrite(conn, args, rows)
    with conn.cursor() as cur:
        cur.execute("select run_id from mlb_home_run_board_runs where run_key = %s", (args.run_key,), prepare=False)
        result = cur.fetchone()
        if result is None:
            raise RuntimeError(f"Unknown MLB HR board run: {args.run_key}")
        run_id = result[0]
        insert_rows = []
        for row in rows:
            candidate = {
                "book": row.get("best_book"),
                "american_price": row.get("best_price"),
                "market_probability": row.get("market_probability") or row.get("no_vig_probability") or row.get("implied_probability"),
                "odds_snapshot_ts": row.get("odds_snapshot_ts"),
            }
            source_status = row.get("odds_status") or "missing_odds"
            if source_status in {"ok", "raw_implied"} and not is_priced_row(candidate, publication_ts):
                source_status = "stale" if row.get("odds_snapshot_ts") else "invalid"
            if source_status not in {"ok", "raw_implied", "missing_odds", "stale", "invalid"}:
                source_status = "invalid"
            insert_rows.append(
                (
                    run_id,
                    row.get("game_date"),
                    row.get("game_id"),
                    row.get("player_id"),
                    row.get("player_name"),
                    row.get("team"),
                    row.get("opponent"),
                    row.get("venue"),
                    row.get("event_time"),
                    row.get("lineup_slot"),
                    row.get("lineup_status"),
                    row.get("opposing_probable_pitcher"),
                    row.get("model_version"),
                    row.get("hr_probability"),
                    row.get("baseline_probability"),
                    row.get("rank"),
                    row.get("best_book") if source_status in {"ok", "raw_implied"} else None,
                    row.get("best_price") if source_status in {"ok", "raw_implied"} else None,
                    row.get("implied_probability"),
                    row.get("no_vig_probability"),
                    row.get("market_probability") or row.get("no_vig_probability") or row.get("implied_probability"),
                    row.get("edge") if source_status in {"ok", "raw_implied"} else None,
                    row.get("ev") if source_status in {"ok", "raw_implied"} else None,
                    row.get("kelly") if source_status in {"ok", "raw_implied"} else None,
                    row.get("odds_snapshot_ts") if source_status in {"ok", "raw_implied"} else None,
                    source_status,
                    row.get("odds_books_count"),
                    json.dumps(row.get("quality_flags") or []),
                    row.get("statcast_available"),
                    row.get("statcast_coverage"),
                    row.get("prediction_ts"),
                    publication_ts,
                    json.dumps(row, default=str),
                )
            )
        if insert_rows:
            cur.executemany(
                """
                insert into mlb_home_run_board_rows (
                  run_id, slate_date, game_id, player_id, player_name, team, opponent, venue,
                  event_time, lineup_slot, lineup_status, opposing_probable_pitcher,
                  model_version, model_probability, baseline_probability, rank, book,
                  american_price, raw_market_probability, no_vig_market_probability,
                  market_probability, edge, ev, quarter_kelly, odds_snapshot_ts, odds_status,
                  odds_books_count, quality_flags, statcast_available, statcast_coverage,
                  prediction_ts, published_at, raw_record
                ) values (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s,
                  %s, %s, %s::jsonb
                )
                on conflict (run_id, model_version, game_id, player_id) do nothing
                """,
                insert_rows,
            )
    conn.commit()
    print(json.dumps({"run_key": args.run_key, "published_rows": len(insert_rows), "published_at": publication_ts.isoformat()}))


def finalize_run(conn, args: argparse.Namespace) -> None:
    completed_at = _now(args.timestamp)
    with conn.cursor() as cur:
        cur.execute(
            """
            select b.*
            from mlb_home_run_board_rows b
            join mlb_home_run_board_runs r on r.run_id = b.run_id
            where r.run_key = %s
            order by b.rank nulls last, b.player_name
            """,
            (args.run_key,),
            prepare=False,
        )
        columns = [description[0] for description in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        cur.execute(
            "select prediction_ts, odds_ts from mlb_home_run_board_runs where run_key = %s",
            (args.run_key,),
            prepare=False,
        )
        run = cur.fetchone()
        if run is None:
            raise RuntimeError(f"Unknown MLB HR board run: {args.run_key}")
        prediction_ts, existing_odds_ts = run
        stats = coverage_stats(rows, completed_at, now=completed_at)
        prediction_values = [parse_timestamp(row.get("prediction_ts")) for row in rows]
        prediction_values = [value for value in prediction_values if value is not None]
        odds_values = [parse_timestamp(row.get("odds_snapshot_ts")) for row in rows]
        odds_values = [value for value in odds_values if value is not None]
        latest_prediction_ts = max(prediction_values, default=prediction_ts)
        latest_odds_ts = max(odds_values, default=existing_odds_ts)
        missing = sum(1 for row in rows if not is_priced_row(row, completed_at))
        gaps = []
        if missing:
            gaps.append(f"{missing} candidates do not have a fresh valid sportsbook price.")
        if args.status:
            status = args.status
        else:
            status = classify_run(
                has_slate=bool(rows),
                source_ok=not args.source_failed,
                predictions_valid=not args.validation_failed,
                top25_coverage=stats["top25_coverage"],
            )
        cur.execute(
            """
            update mlb_home_run_board_runs
            set status = %s,
                completed_at = %s,
                gaps = %s::jsonb,
                total_candidates = %s,
                priced_candidates = %s,
                top25_denominator = %s,
                top25_priced_count = %s,
                top25_coverage = %s,
                prediction_ts = coalesce(%s, prediction_ts),
                odds_ts = coalesce(%s, odds_ts),
                validation_summary = %s::jsonb,
                updated_at = now()
            where run_key = %s
            """,
            (
                status,
                completed_at,
                json.dumps(gaps),
                stats["total_candidates"],
                stats["priced_candidates"],
                stats["top25_denominator"],
                stats["top25_priced_count"],
                stats["top25_coverage"],
                latest_prediction_ts,
                latest_odds_ts,
                json.dumps({"published_rows": len(rows), "source_failed": args.source_failed, "validation_failed": args.validation_failed}),
                args.run_key,
            ),
            prepare=False,
        )
    conn.commit()
    print(json.dumps({"run_key": args.run_key, "status": status, **stats}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "publish", "finalize"):
        command = sub.add_parser(action)
        command.add_argument("--run-key", required=True)
        command.add_argument("--slate-date", required=action in {"start", "publish"})
        command.add_argument("--model-version", default="mlb-hr-v1")
        command.add_argument("--model-like", default="mlb-hr-v1%")
        command.add_argument("--run-window", choices=("morning", "afternoon", "manual"), default="manual")
        command.add_argument("--workflow-url", default=os.getenv("GITHUB_SERVER_URL", "") + "/" + os.getenv("GITHUB_REPOSITORY", ""))
        command.add_argument("--timestamp")
        command.add_argument("--status", choices=("running", "healthy", "partial", "failed", "no_slate"))
        command.add_argument("--source-failed", action="store_true")
        command.add_argument("--validation-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    conn = _connection()
    try:
        if args.action == "start":
            start_run(conn, args)
        elif args.action == "publish":
            publish_rows(conn, args)
        else:
            finalize_run(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
