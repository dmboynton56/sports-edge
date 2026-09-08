#!/usr/bin/env python3
"""Prune historical odds snapshots after BigQuery has retained the source history.

The shared Supabase project is a serving cache, not the sports warehouse.  This
script is intentionally dry-run by default; production workflows must pass
``--apply`` explicitly.  Missing optional market tables are skipped so the
cleanup can run across seasons with different serving schemas.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


LOGGER = logging.getLogger("prune_supabase_serving_cache")


@dataclass(frozen=True)
class RetentionRule:
    table: str
    timestamp_column: str
    days: int


RETENTION_RULES = (
    RetentionRule("odds_snapshots", "snapshot_ts", 14),
    RetentionRule("pga_odds_snapshots", "snapshot_ts", 30),
    RetentionRule("mlb_home_run_odds_snapshots", "snapshot_ts", 30),
    RetentionRule("nfl_anytime_td_odds_snapshots", "snapshot_ts", 30),
    RetentionRule("cfb_odds_snapshots", "snapshot_ts", 30),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete rows past each retention window.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser.parse_args()


def _connect():
    credentials = load_supabase_credentials()
    if not credentials["db_password"]:
        raise RuntimeError("SUPABASE_DB_PASSWORD or supabaseDBpass is required")
    return create_pg_connection(
        supabase_url=credentials["url"],
        password=credentials["db_password"],
        host_override=credentials.get("db_host"),
        port=credentials["db_port"],
        database=credentials["db_name"],
        user=credentials["db_user"],
    )


def prune(*, apply: bool) -> dict[str, int]:
    conn = _connect()
    summary: dict[str, int] = {}
    try:
        for rule in RETENTION_RULES:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT COUNT(*) FROM public.{rule.table} "
                        f"WHERE {rule.timestamp_column} < NOW() - (%s * INTERVAL '1 day')",
                        (rule.days,),
                    )
                    count = int(cur.fetchone()[0])
                    summary[rule.table] = count
                    LOGGER.info(
                        "%s: %d rows past %d-day retention (%s)",
                        rule.table,
                        count,
                        rule.days,
                        "apply" if apply else "dry-run",
                    )
                    if apply and count:
                        cur.execute(
                            f"DELETE FROM public.{rule.table} "
                            f"WHERE {rule.timestamp_column} < NOW() - (%s * INTERVAL '1 day')",
                            (rule.days,),
                        )
                if apply:
                    conn.commit()
            except psycopg.errors.UndefinedTable:
                conn.rollback()
                LOGGER.info("%s: table not present; skipping", rule.table)
    finally:
        conn.close()
    return summary


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    prune(apply=args.apply)


if __name__ == "__main__":
    main()
