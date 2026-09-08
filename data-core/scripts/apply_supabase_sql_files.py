#!/usr/bin/env python3
"""Apply one or more SQL files to the configured Supabase Postgres database."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SQL files to Supabase Postgres.")
    parser.add_argument("sql_files", nargs="+", type=Path)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument(
        "--connect-retries",
        type=int,
        default=3,
        help="Number of additional connection attempts after a transient failure.",
    )
    return parser.parse_args()


def connect_with_retries(creds: dict[str, str], retries: int):
    """Connect to Supabase, retrying transient connection timeouts."""
    attempts = max(1, retries + 1)
    last_error = None
    for attempt in range(1, attempts + 1):
        conn = None
        try:
            conn = create_pg_connection(
                supabase_url=creds["url"],
                password=creds["db_password"],
                host_override=creds.get("db_host"),
                port=creds["db_port"],
                database=creds["db_name"],
                user=creds["db_user"],
            )
            # psycopg can defer the network handshake until the first query;
            # force it here so connection retries actually cover auth/timeouts.
            conn.execute("SELECT 1")
            conn.rollback()
            return conn
        except psycopg.OperationalError as exc:
            last_error = exc
            try:
                if conn is None:
                    raise RuntimeError("connection was not created")
                conn.close()
            except (RuntimeError, psycopg.Error):
                pass
            if attempt == attempts:
                raise
            delay = min(30, 5 * attempt)
            print(
                f"Supabase connection attempt {attempt}/{attempts} failed; "
                f"retrying in {delay}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise last_error  # pragma: no cover - loop always returns or raises


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    creds = load_supabase_credentials()
    missing = [
        name
        for name, value in {
            "SUPABASE_URL": creds["url"],
            "SUPABASE_DB_PASSWORD or supabaseDBpass": creds["db_password"],
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing Supabase credentials: {', '.join(missing)}")

    conn = connect_with_retries(creds, args.connect_retries)
    try:
        with conn.cursor() as cur:
            for sql_file in args.sql_files:
                sql = sql_file.read_text(encoding="utf-8")
                cur.execute(sql)
                print(f"Applied {sql_file}")
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except psycopg.Error:
            # The original connection error is more useful than a secondary
            # rollback failure after the server has already closed the socket.
            pass
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
