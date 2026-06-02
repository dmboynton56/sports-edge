#!/usr/bin/env python3
"""
Backfill final scores from BigQuery raw schedules into Supabase games.

Usage:
    python scripts/sync_final_scores.py --project learned-pier-478122-p7
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

from dotenv import load_dotenv
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials
from src.utils.team_codes import canonical_team_abbr

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
        WHERE league IN ('NFL', 'NBA', 'MLB')
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


def _score_date(value) -> str:
    if hasattr(value, "date") and not isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value).split(" ")[0]


def _game_date(value) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value).split(" ")[0]


def _match_key(league, home_team, away_team, game_date) -> tuple[str, str, str, str]:
    league_key = str(league).upper()
    home_key = canonical_team_abbr(league_key, home_team) or str(home_team).upper()
    away_key = canonical_team_abbr(league_key, away_team) or str(away_team).upper()
    return (league_key, _game_date(game_date), home_key, away_key)


def _fetch_game_id_map(conn, start_date: str, end_date: str) -> dict[tuple[str, str, str, str], list[str]]:
    game_ids_by_key: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              id,
              league,
              home_team,
              away_team,
              COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date) AS game_date
            FROM games
            WHERE league IN ('NFL', 'NBA', 'MLB')
              AND COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date) BETWEEN %s AND %s
            """,
            (start_date, end_date),
        )
        for game_id, league, home_team, away_team, game_date in cur.fetchall():
            game_ids_by_key[_match_key(league, home_team, away_team, game_date)].append(
                game_id
            )
    return dict(game_ids_by_key)


def sync_scores(conn, scores_df) -> tuple[int, int]:
    if scores_df.empty:
        return 0, 0

    start_date = min(_score_date(row["game_date"]) for _, row in scores_df.iterrows())
    end_date = max(_score_date(row["game_date"]) for _, row in scores_df.iterrows())
    game_ids_by_key = _fetch_game_id_map(conn, start_date, end_date)

    updated = 0
    unmatched = 0
    ambiguous = 0
    unmatched_samples = []
    update_sql = """
        UPDATE games
        SET home_score = %s,
            away_score = %s
        WHERE id = %s
    """

    with conn.cursor() as cur:
        for _, row in scores_df.iterrows():
            match_key = _match_key(
                row["league"],
                row["home_team"],
                row["away_team"],
                row["game_date"],
            )
            game_ids = game_ids_by_key.get(match_key, [])
            if not game_ids:
                unmatched += 1
                if len(unmatched_samples) < 10:
                    unmatched_samples.append(
                        {
                            "league": row["league"],
                            "game_date": _score_date(row["game_date"]),
                            "away_team": row["away_team"],
                            "home_team": row["home_team"],
                            "match_key": match_key,
                        }
                    )
                continue
            if len(game_ids) > 1:
                ambiguous += 1
                LOGGER.warning(
                    "Multiple Supabase games matched %s; updating %d rows",
                    match_key,
                    len(game_ids),
                )
            for game_id in game_ids:
                cur.execute(
                    update_sql,
                    (
                        int(row["home_score"]),
                        int(row["away_score"]),
                        game_id,
                    ),
                )
                updated += int(cur.rowcount)
    conn.commit()
    if unmatched_samples:
        LOGGER.warning(
            "Unmatched final-score rows (first %d): %s",
            len(unmatched_samples),
            unmatched_samples,
        )
    if ambiguous:
        LOGGER.warning("Final score sync found %d ambiguous score rows.", ambiguous)
    return updated, unmatched


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
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
