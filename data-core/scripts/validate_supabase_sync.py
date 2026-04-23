#!/usr/bin/env python3
"""
Validate recent Supabase sync health for games/predictions/scores.
"""

from __future__ import annotations

import argparse
import json
import logging

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

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
                WHERE game_time_utc::date >= CURRENT_DATE - INTERVAL '3 days'
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                """
            )
            report["recent_final_scores"] = int(cur.fetchone()[0])

            cur.execute(
                """
                SELECT COUNT(*)
                FROM games
                WHERE game_time_utc::date >= CURRENT_DATE - INTERVAL '3 days'
                """
            )
            report["recent_games"] = int(cur.fetchone()[0])
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
        if failures:
            for failure in failures:
                LOGGER.error(failure)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
