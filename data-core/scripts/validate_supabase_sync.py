#!/usr/bin/env python3
"""
Validate recent Supabase sync health for games/predictions/scores.

Table/column expectations should match the checked-in inventory:
  projects/plans/docs/supabase_tables/personal_portfolio_project_tables.json
Queries use public tables `games`, `model_predictions`, etc. documented there.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

LOGGER = logging.getLogger("validate_supabase_sync")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Supabase sync health.")
    parser.add_argument("--strict", action="store_true", help="Fail on weak validation.")
    parser.add_argument(
        "--prediction-hours",
        type=int,
        default=48,
        help="Window for recent predictions (default: 48h).",
    )
    parser.add_argument(
        "--score-lookback-days",
        type=int,
        default=7,
        help="Lookback window for game/final score checks (default: 7).",
    )
    parser.add_argument(
        "--max-orphans",
        type=int,
        default=0,
        help="Maximum allowed prediction rows with no matching game (default: 0).",
    )
    parser.add_argument(
        "--book-spread-lookback-days",
        type=int,
        default=3,
        help="Lookback window for book_spread coverage checks (default: 3).",
    )
    parser.add_argument(
        "--book-spread-lookahead-days",
        type=int,
        default=14,
        help="Lookahead window for book_spread coverage checks (default: 14).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    load_dotenv(ROOT / ".env")

    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )

    report: dict[str, int] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM model_predictions
                WHERE asof_ts >= NOW() - (%s || ' hours')::interval
                """,
                (args.prediction_hours,),
            )
            report["recent_predictions"] = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*)
                FROM games
                WHERE COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                  >= (now() AT TIME ZONE 'America/Denver')::date - (%s || ' days')::interval
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                """,
                (args.score_lookback_days,),
            )
            report["recent_final_scores"] = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*)
                FROM games
                WHERE COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                  >= (now() AT TIME ZONE 'America/Denver')::date - (%s || ' days')::interval
                """,
                (args.score_lookback_days,),
            )
            report["recent_games"] = int(cur.fetchone()[0])

            cur.execute(
                """
                WITH game_groups AS (
                    SELECT
                      league,
                      COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date) AS game_date,
                      home_team,
                      away_team,
                      COUNT(*) AS row_count,
                      BOOL_OR(home_score IS NOT NULL AND away_score IS NOT NULL) AS has_score,
                      BOOL_OR(home_score IS NULL OR away_score IS NULL) AS has_missing_score
                    FROM games
                    WHERE COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                      >= (now() AT TIME ZONE 'America/Denver')::date - (%s || ' days')::interval
                      AND COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                      < (now() AT TIME ZONE 'America/Denver')::date
                    GROUP BY league, COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date), home_team, away_team
                )
                SELECT
                  COUNT(*) FILTER (WHERE NOT has_score) AS missing_score_groups,
                  COALESCE(SUM(row_count - 1) FILTER (WHERE row_count > 1), 0) AS duplicate_rows,
                  COALESCE(SUM(row_count) FILTER (WHERE has_score AND has_missing_score), 0) AS duplicate_rows_with_score
                FROM game_groups
                """,
                (args.score_lookback_days,),
            )
            missing_score_groups, duplicate_rows, duplicate_rows_with_score = cur.fetchone()
            report["past_game_groups_missing_scores"] = int(missing_score_groups)
            report["duplicate_recent_game_rows"] = int(duplicate_rows)
            report["duplicate_scored_game_rows_with_unscored_copy"] = int(duplicate_rows_with_score)

            cur.execute(
                """
                SELECT COUNT(*)
                FROM model_predictions p
                LEFT JOIN games g ON g.id = p.game_id
                WHERE p.asof_ts >= NOW() - (%s || ' hours')::interval
                  AND g.id IS NULL
                """,
                (args.prediction_hours,),
            )
            report["orphan_predictions"] = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT
                  league,
                  COUNT(*) AS games,
                  COUNT(*) FILTER (WHERE book_spread IS NULL) AS missing_book_spread
                FROM games
                WHERE COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                  >= (now() AT TIME ZONE 'America/Denver')::date - (%s || ' days')::interval
                  AND COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                  <= (now() AT TIME ZONE 'America/Denver')::date + (%s || ' days')::interval
                GROUP BY league
                """,
                (args.book_spread_lookback_days, args.book_spread_lookahead_days),
            )
            for league, games, missing_book_spread in cur.fetchall():
                league_key = str(league).lower()
                report[f"{league_key}_book_spread_window_games"] = int(games)
                report[f"{league_key}_missing_book_spread"] = int(missing_book_spread)

            cur.execute(
                """
                SELECT COUNT(*)
                FROM games
                WHERE league = 'MLB'
                  AND COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                    >= (now() AT TIME ZONE 'America/Denver')::date - 1
                  AND COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                    <= (now() AT TIME ZONE 'America/Denver')::date + 9
                """
            )
            report["mlb_window_games"] = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*)
                FROM model_predictions p
                JOIN games g ON g.id = p.game_id
                WHERE g.league = 'MLB'
                  AND p.asof_ts >= NOW() - (%s || ' hours')::interval
                """,
                (args.prediction_hours,),
            )
            report["mlb_recent_predictions"] = int(cur.fetchone()[0])
    finally:
        conn.close()

    LOGGER.info("Validation report: %s", json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))

    if args.strict:
        failures = []
        if report["recent_predictions"] <= 0:
            failures.append("No recent model_predictions rows found.")
        if report["recent_games"] > 0 and report["recent_final_scores"] <= 0:
            failures.append("Recent games exist but none have final scores.")
        if report["past_game_groups_missing_scores"] > 0:
            failures.append(
                f"Found {report['past_game_groups_missing_scores']} past game groups missing scores."
            )
        if report["orphan_predictions"] > args.max_orphans:
            failures.append(
                f"Found {report['orphan_predictions']} orphan predictions (max {args.max_orphans})."
            )
        if report.get("mlb_recent_predictions", 0) <= 0:
            failures.append("No recent MLB model_predictions rows found.")
        if failures:
            for failure in failures:
                LOGGER.error(failure)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
