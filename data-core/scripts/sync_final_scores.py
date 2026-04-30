#!/usr/bin/env python3
"""
Backfill final scores from BigQuery raw schedules into Supabase games.

Usage:
    python scripts/sync_final_scores.py --project learned-pier-478122-p7
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

from google.cloud import bigquery

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

LOGGER = logging.getLogger("sync_final_scores")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync final scores from BigQuery raw_schedules to Supabase games."
    )
    parser.add_argument("--project", required=True, help="GCP project ID.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=4,
        help="How many days back to consider final scores (default: 4).",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=1,
        help="How many days forward to include (default: 1).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def fetch_final_scores(
    client: bigquery.Client,
    project: str,
    start_date: str,
    end_date: str,
):
    query = f"""
        SELECT
            league,
            DATE(game_date) AS game_date,
            home_team,
            away_team,
            home_score,
            away_score
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league IN ('NFL', 'NBA')
          AND DATE(game_date) BETWEEN @start_date AND @end_date
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    return client.query(query, job_config=job_config).to_dataframe()


def sync_scores(conn, scores_df) -> tuple[int, int]:
    if scores_df.empty:
        return 0, 0

    updated = 0
    unmatched = 0
    update_sql = """
        UPDATE games
        SET home_score = %s,
            away_score = %s
        WHERE league = %s
          AND home_team = %s
          AND away_team = %s
          AND game_time_utc::date = %s
    """

    with conn.cursor() as cur:
        for _, row in scores_df.iterrows():
            cur.execute(
                update_sql,
                (
                    int(row["home_score"]),
                    int(row["away_score"]),
                    row["league"],
                    row["home_team"],
                    row["away_team"],
                    row["game_date"],
                ),
            )
            if cur.rowcount > 0:
                updated += int(cur.rowcount)
            else:
                unmatched += 1
    conn.commit()
    return updated, unmatched


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    now = datetime.now(timezone.utc).date()
    start_date = (now - timedelta(days=args.lookback_days)).isoformat()
    end_date = (now + timedelta(days=args.lookahead_days)).isoformat()

    client = bigquery.Client(project=args.project)
    scores = fetch_final_scores(client, args.project, start_date, end_date)
    LOGGER.info(
        "Fetched %d final-score rows from BQ for %s to %s",
        len(scores),
        start_date,
        end_date,
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
    try:
        updated, unmatched = sync_scores(conn, scores)
        LOGGER.info(
            "Final score sync complete | updated=%d unmatched=%d",
            updated,
            unmatched,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
