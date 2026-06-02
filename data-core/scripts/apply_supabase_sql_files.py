#!/usr/bin/env python3
"""Apply one or more SQL files to the configured Supabase Postgres database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SQL files to Supabase Postgres.")
    parser.add_argument("sql_files", nargs="+", type=Path)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    return parser.parse_args()


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

    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    try:
        with conn.cursor() as cur:
            for sql_file in args.sql_files:
                sql = sql_file.read_text(encoding="utf-8")
                cur.execute(sql)
                print(f"Applied {sql_file}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
