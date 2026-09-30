#!/usr/bin/env python3
"""Read the monthly ledger and free account quota headers without buying odds."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.odds_api_client import BASE_URL, CONTINGENCY, MONTHLY_LIMIT, SOURCE_LIMITS
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


def main() -> None:
    load_dotenv(ROOT / ".env")
    now = datetime.now(timezone.utc)
    report = {
        "audited_at": now.isoformat(), "budget_month": now.date().replace(day=1).isoformat(),
        "monthly_limit": MONTHLY_LIMIT, "contingency": CONTINGENCY,
        "source_limits": SOURCE_LIMITS, "sources": [], "account": None, "gaps": [],
    }
    credentials = load_supabase_credentials()
    if credentials["db_password"]:
        try:
            with create_pg_connection(credentials["url"], credentials["db_password"], credentials.get("db_host"),
                                      credentials["db_port"], credentials["db_name"], credentials["db_user"]) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT source, COUNT(*), SUM(cost),
                          SUM(cost) FILTER (WHERE status = 'reserved')
                        FROM odds_api_request_cache WHERE budget_month = %s
                        GROUP BY source ORDER BY source
                    """, (report["budget_month"],), prepare=False)
                    report["sources"] = [dict(source=s, requests=int(n), credits=int(c), pending_credits=int(p or 0)) for s, n, c, p in cur.fetchall()]
        except Exception as exc:
            report["gaps"].append(f"Credit ledger unavailable ({type(exc).__name__}).")
    else:
        report["gaps"].append("Credit ledger credentials unavailable.")
    if os.getenv("ODDS_API_KEY"):
        try:
            response = requests.get(f"{BASE_URL}/sports/", params={"apiKey": os.environ["ODDS_API_KEY"]}, timeout=30)
            report["account"] = {
                "http_status": response.status_code,
                "used": response.headers.get("x-requests-used"),
                "remaining": response.headers.get("x-requests-remaining"),
                "last_cost": response.headers.get("x-requests-last"),
            }
        except requests.RequestException:
            report["gaps"].append("Free account quota check unavailable.")
    else:
        report["gaps"].append("ODDS_API_KEY unavailable for the free quota check.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
