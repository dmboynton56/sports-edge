#!/usr/bin/env python3
"""Audit NBA/NFL season readiness for dashboard serving."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

LOGGER = logging.getLogger("audit_season_readiness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit league serving readiness.")
    parser.add_argument("--league", required=True, choices=["NBA", "NFL"])
    parser.add_argument("--date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=None)
    parser.add_argument("--lookahead-days", type=int, default=7)
    parser.add_argument("--week", type=int, default=None, help="NFL week filter (optional).")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args()


def _date_window(anchor: date, lookahead_days: int) -> tuple[date, date]:
    return anchor, anchor + timedelta(days=lookahead_days)


def audit_league(
    conn,
    *,
    league: str,
    start_date: date,
    end_date: date,
    week: int | None,
) -> dict[str, Any]:
    issues: list[str] = []
    report: dict[str, Any] = {
        "league": league,
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "week": week,
        "issues": issues,
        "ready": True,
    }

    with conn.cursor() as cur:
        week_clause = ""
        params: list[Any] = [league, start_date, end_date]
        if week is not None:
            week_clause = " AND g.week = %s"
            params.append(week)

        cur.execute(
            f"""
            SELECT COUNT(*) FROM games g
            WHERE g.league = %s
              AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                BETWEEN %s AND %s
            {week_clause}
            """,
            params,
        )
        report["scheduled_games"] = int(cur.fetchone()[0])

        cur.execute(
            f"""
            WITH scoped AS (
              SELECT g.id
              FROM games g
              WHERE g.league = %s
                AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                  BETWEEN %s AND %s
              {week_clause}
            ),
            latest_pred AS (
              SELECT DISTINCT ON (p.game_id)
                p.game_id,
                p.asof_ts,
                p.model_version
              FROM model_predictions p
              JOIN scoped s ON s.id = p.game_id
              ORDER BY p.game_id, p.asof_ts DESC
            )
            SELECT
              (SELECT COUNT(*) FROM scoped) AS games,
              (SELECT COUNT(*) FROM latest_pred) AS with_prediction,
              (SELECT COUNT(*) FROM latest_pred WHERE asof_ts >= NOW() - INTERVAL '48 hours') AS fresh_predictions
            """,
            params,
        )
        games, with_prediction, fresh_predictions = cur.fetchone()
        report["games_with_prediction"] = int(with_prediction)
        report["fresh_predictions"] = int(fresh_predictions)
        report["prediction_coverage_pct"] = (
            round(100.0 * with_prediction / games, 1) if games else None
        )

        cur.execute(
            f"""
            SELECT COUNT(*) FROM games g
            WHERE g.league = %s
              AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                BETWEEN %s AND %s
              AND g.book_spread IS NULL
            {week_clause}
            """,
            params,
        )
        report["missing_book_spread"] = int(cur.fetchone()[0])

        cur.execute(
            f"""
            WITH groups AS (
              SELECT
                COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date) AS gd,
                g.home_team,
                g.away_team,
                COUNT(*) AS row_count
              FROM games g
              WHERE g.league = %s
                AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                  BETWEEN %s AND %s
              {week_clause}
              GROUP BY 1, g.home_team, g.away_team
              HAVING COUNT(*) > 1
            )
            SELECT COALESCE(SUM(row_count - 1), 0) FROM groups
            """,
            params,
        )
        report["duplicate_rows"] = int(cur.fetchone()[0])

        cur.execute(
            f"""
            SELECT COUNT(*) FROM model_predictions p
            JOIN games g ON g.id = p.game_id
            WHERE g.league = %s
              AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                BETWEEN %s AND %s
              AND p.asof_ts < NOW() - INTERVAL '7 days'
            {week_clause.replace('g.week', 'g.week') if week_clause else ''}
            """,
            params,
        )
        report["stale_prediction_rows"] = int(cur.fetchone()[0])

        if league == "NFL" and week is not None:
            cur.execute(
                """
                SELECT COUNT(DISTINCT g.id)
                FROM games g
                LEFT JOIN player_impact_estimates pie ON pie.game_id = g.id
                WHERE g.league = 'NFL'
                  AND g.week = %s
                  AND pie.id IS NULL
                """,
                (week,),
            )
            report["games_missing_injury_impact"] = int(cur.fetchone()[0])

    if report["scheduled_games"] == 0:
        issues.append(f"No {league} games in window.")
    if report["scheduled_games"] and report["games_with_prediction"] < report["scheduled_games"]:
        issues.append(
            f"{report['scheduled_games'] - report['games_with_prediction']} games missing predictions."
        )
    if report["duplicate_rows"] > 0:
        issues.append(f"{report['duplicate_rows']} duplicate game rows detected.")
    if report["missing_book_spread"] > 0:
        issues.append(f"{report['missing_book_spread']} games missing book_spread.")
    if report["stale_prediction_rows"] > 0:
        issues.append(f"{report['stale_prediction_rows']} stale prediction rows in window.")

    report["ready"] = len(issues) == 0 or (
        report["scheduled_games"] > 0 and report["games_with_prediction"] > 0
    )
    report["audited_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")

    anchor = args.date or datetime.now(timezone.utc).date()
    start_date, end_date = _date_window(anchor, args.lookahead_days)

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
        report = audit_league(
            conn,
            league=args.league,
            start_date=start_date,
            end_date=end_date,
            week=args.week,
        )
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        LOGGER.info("Audit report: %s", json.dumps(report, sort_keys=True))
        print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
