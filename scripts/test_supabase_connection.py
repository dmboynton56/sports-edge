import sys
from typing import Optional

from psycopg.rows import dict_row

from predict_WEEK_11 import create_pg_connection, load_supabase_credentials


def run_connection_check() -> Optional[int]:
    """Attempt to connect to Supabase Postgres and run a trivial query."""
    try:
        creds = load_supabase_credentials()
    except Exception as err:
        print(f"ERROR loading Supabase credentials: {err}")
        return 1

    try:
        conn = create_pg_connection(
            supabase_url=creds["url"],
            password=creds["db_password"],
            host_override=creds.get("db_host"),
            port=creds["db_port"],
            database=creds["db_name"],
            user=creds["db_user"],
        )
    except Exception as err:
        print(f"ERROR connecting to Supabase: {err}")
        return 1

    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("select current_catalog as db, current_user as user, now() at time zone 'utc' as utc_now;")
            row = cur.fetchone()
            print("Connection successful ✓")
            print(f"  database : {row['db']}")
            print(f"  user      : {row['user']}")
            print(f"  utc time  : {row['utc_now']}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(run_connection_check() or 0)
