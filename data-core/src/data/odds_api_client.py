"""Account-wide Starter credit protection and durable same-period response reuse.

Reservations commit before the paid HTTP request. A transaction advisory lock
serializes reservation decisions across workflows. Ambiguous network failures
keep their reservation: a retry must not spend the same credits again.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from math import ceil
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests
from psycopg.types.json import Jsonb

from src.data.odds_api_errors import OddsApiQuotaExhausted, check_odds_api_response
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.the-odds-api.com/v4"
MONTHLY_LIMIT = 500
CONTINGENCY = 50
# Covers six overlapping forecast cycles in a long month, plus recovery headroom.
SOURCE_LIMITS = {"nfl": 140, "nba": 100, "mlb": 100, "cfb": 40, "pga": 20}
LOCK_ID = 7092026500
FETCHED_AT_HEADER = "x-sports-edge-fetched-at"


class OddsApiBudgetBlocked(OddsApiQuotaExhausted):
    """Do not retry a paid request that conservation or durable state blocks."""


def request_cost(params: dict[str, Any]) -> int:
    markets = {s.strip() for s in params.get("markets", "h2h").split(",") if s.strip()}
    books = {s.strip() for s in params.get("bookmakers", "").split(",") if s.strip()}
    regions = {s.strip() for s in params.get("regions", "us").split(",") if s.strip()}
    return len(markets) * (ceil(len(books) / 10) if books else len(regions))


def check_budget(*, source: str, cost: int, used: int, remaining: int,
                 spending: dict[str, int], pending: int) -> None:
    if source not in SOURCE_LIMITS or cost < 1:
        raise OddsApiBudgetBlocked("Unsupported paid Odds API request")
    if spending.get(source, 0) + cost > SOURCE_LIMITS[source]:
        raise OddsApiBudgetBlocked(f"{source} monthly credit allocation reached ({SOURCE_LIMITS[source]})")
    reserve = CONTINGENCY
    if source != "nfl":
        reserve += max(0, SOURCE_LIMITS["nfl"] - spending.get("nfl", 0))
    available = min(remaining, MONTHLY_LIMIT - used) - pending
    if available - cost < reserve:
        raise OddsApiBudgetBlocked(f"Preserving {reserve} Odds API credits; available={available}, cost={cost}")


def _response(payload, headers, fetched_at) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode("utf-8")
    response.headers.update(headers)
    response.headers[FETCHED_AT_HEADER] = fetched_at.isoformat()
    return response


def response_timestamp(response) -> datetime:
    raw = response.headers.get(FETCHED_AT_HEADER)
    return datetime.fromisoformat(raw) if raw else datetime.now(timezone.utc)


def _reserve_or_reuse(conn, *, request_key: str, source: str, cost: int,
                     params: dict[str, Any], now: datetime, retry_empty: bool,
                     get, timeout: int) -> requests.Response | None:
    """Serialize the decision and commit a reservation before any paid HTTP."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,), prepare=False)
        cur.execute("SELECT status, payload, headers, fetched_at FROM odds_api_request_cache WHERE request_key = %s",
                    (request_key,), prepare=False)
        cached = cur.fetchone()
        if cached:
            if cached[0] != "complete":
                raise OddsApiBudgetBlocked("Odds request already reserved or failed in this period; avoiding duplicate spend")
            retry_allowed = (
                retry_empty
                and cached[2].get("x-requests-last") == "0"
                and cached[3].astimezone(ZoneInfo("America/Denver")).date()
                < now.astimezone(ZoneInfo("America/Denver")).date()
            )
            if not retry_allowed:
                conn.commit()
                LOGGER.info("Reusing %s Odds API snapshot captured at %s", source, cached[3])
                return _response(cached[1], cached[2], cached[3])
        month = now.astimezone(timezone.utc).date().replace(day=1)
        cur.execute("SELECT source, SUM(cost), SUM(cost) FILTER (WHERE status = 'reserved') FROM odds_api_request_cache WHERE budget_month = %s GROUP BY source",
                    (month,), prepare=False)
        rows = cur.fetchall()
        spending = {row[0]: int(row[1]) for row in rows}
        pending = sum(int(row[2] or 0) for row in rows)
        probe = get(f"{BASE_URL}/sports/", params={"apiKey": params["apiKey"]}, timeout=timeout)
        check_odds_api_response(probe, context="Odds API quota check")
        try:
            used = int(probe.headers["x-requests-used"])
            remaining = int(probe.headers["x-requests-remaining"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OddsApiBudgetBlocked("Quota headers unavailable; paid request skipped") from exc
        check_budget(source=source, cost=cost, used=used, remaining=remaining, spending=spending, pending=pending)
        cur.execute("""INSERT INTO odds_api_request_cache
                    (request_key, budget_month, source, cost, status)
                    VALUES (%s, %s, %s, %s, 'reserved')
                    ON CONFLICT (request_key) DO UPDATE SET
                      budget_month = EXCLUDED.budget_month, cost = EXCLUDED.cost,
                      status = 'reserved', payload = NULL, headers = '{}'::jsonb,
                      fetched_at = NULL""",
                    (request_key, month, source, cost), prepare=False)
    conn.commit()
    return None


def _record_response(conn, request_key: str, response: requests.Response,
                     estimated_cost: int, fetched_at: datetime) -> None:
    headers = {k.lower(): v for k, v in response.headers.items() if k.lower().startswith("x-requests-")}
    payload = response.json() if response.status_code == 200 else None
    try:
        charged = int(headers.get("x-requests-last", estimated_cost))
    except (TypeError, ValueError):
        charged = estimated_cost
    with conn.cursor() as cur:
        cur.execute("UPDATE odds_api_request_cache SET status = %s, cost = %s, payload = %s, headers = %s, fetched_at = %s WHERE request_key = %s",
                    ("complete" if response.status_code == 200 else "failed", charged, Jsonb(payload), Jsonb(headers), fetched_at, request_key), prepare=False)
    conn.commit()
    response.headers[FETCHED_AT_HEADER] = fetched_at.isoformat()
    LOGGER.info("Odds API cost=%s used=%s remaining=%s", charged,
                headers.get("x-requests-used"), headers.get("x-requests-remaining"))


def budgeted_get(url: str, *, params: dict[str, Any], source: str,
                 cache_period: str | None = None, timeout: int = 30,
                 retry_empty: bool = False,
                 conn=None, get=None, now: datetime | None = None) -> requests.Response:
    """Only supported current-odds endpoints can spend; missing ledger fails closed.

    No API keys or query URLs are persisted. Quota headers from the free /sports
    endpoint include account usage outside this repository. Cache timestamps
    remain the original capture time, including across retries and month reset.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "api.the-odds-api.com" or not parsed.path.startswith("/v4/sports/") or not parsed.path.rstrip("/").endswith("/odds"):
        raise OddsApiBudgetBlocked("Only current sport/event odds are supported by the Starter budget")
    if source not in SOURCE_LIMITS:
        raise OddsApiBudgetBlocked(f"Unknown Odds API source: {source}")
    supplied_now = now
    now = now or datetime.now(timezone.utc)
    period = cache_period or now.astimezone(ZoneInfo("America/Denver")).date().isoformat()
    safe_params = {key: value for key, value in params.items() if key != "apiKey"}
    identity = json.dumps([parsed.path.rstrip("/"), safe_params, source, period], sort_keys=True)
    request_key = sha256(identity.encode()).hexdigest()
    cost = request_cost(params)
    get = get or requests.get
    owns_conn = conn is None
    try:
        # Reservation failures never reach the paid request phase.
        try:
            if conn is None:
                creds = load_supabase_credentials()
                if not creds["db_password"]:
                    raise OddsApiBudgetBlocked("Missing credentials for durable Odds API credit ledger")
                conn = create_pg_connection(creds["url"], creds["db_password"], creds.get("db_host"),
                                            creds["db_port"], creds["db_name"], creds["db_user"])
            cached = _reserve_or_reuse(conn, request_key=request_key, source=source, cost=cost,
                                      params=params, now=now, retry_empty=retry_empty, get=get, timeout=timeout)
        except Exception as exc:
            if conn is not None:
                conn.rollback()
            if isinstance(exc, OddsApiQuotaExhausted):
                raise
            raise OddsApiBudgetBlocked(f"Cannot reserve Odds API credits ({type(exc).__name__}); paid request skipped") from None
        if cached is not None:
            return cached
        # Network failures retain the committed reservation; a retry cannot buy again.
        try:
            response = get(url, params=params, timeout=timeout)
            _record_response(conn, request_key, response, cost, supplied_now or datetime.now(timezone.utc))
        except requests.RequestException as exc:
            raise OddsApiBudgetBlocked(f"Odds API request interrupted ({type(exc).__name__}); reservation retained") from None
        check_odds_api_response(response, context=f"{source} odds")
        return response
    finally:
        if owns_conn and conn is not None:
            conn.close()
