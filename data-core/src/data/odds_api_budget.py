"""Shared once-per-Denver-day The Odds API budget for MLB HR + research.

HR (Player Market Refresh) and research game lines (Daily Refresh) share one
durable `odds_api_usage` row per America/Denver calendar date so they cannot
each spend Starter-plan credits on the same day.

Read path also consults existing snapshot provider stamps so a same-day Odds
call that landed before this table existed still blocks the other path.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from psycopg.errors import UndefinedTable

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

LOGGER = logging.getLogger(__name__)

SLATE_TIMEZONE = "America/Denver"
SOURCE_HR = "mlb_hr"
SOURCE_RESEARCH = "mlb_research"
VALID_SOURCES = frozenset({SOURCE_HR, SOURCE_RESEARCH})

SKIP_REASON_SHARED_BUDGET = (
    "The Odds API already used today (shared daily budget across HR + research; "
    "at most one Odds API session per Denver day)"
)
SKIP_REASON_PROPLINE_FIRST = (
    "PropLine-first: The Odds API not called (shared daily budget reserved)"
)


def denver_today() -> str:
    """America/Denver calendar date as YYYY-MM-DD."""
    return datetime.now(ZoneInfo(SLATE_TIMEZONE)).date().isoformat()


def should_skip_odds_api(already_used_today: bool, force_odds_api: bool) -> bool:
    """Keep daily conservation by default while allowing an explicit recovery run."""
    return already_used_today and not force_odds_api


def _connect() -> Any | None:
    creds = load_supabase_credentials()
    if not creds.get("url") or not creds.get("db_password"):
        return None
    return create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )


def _lookup_usage_row(conn, denver_date: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            select source
            from odds_api_usage
            where denver_date = %s
            limit 1
            """,
            (denver_date,),
            prepare=False,
        )
        row = cur.fetchone()
    return row[0] if row else None


def _lookup_hr_snapshot(conn, denver_date: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            select provider
            from mlb_home_run_odds_snapshots
            where game_date = %s
              and provider = 'the_odds_api'
            limit 1
            """,
            (denver_date,),
            prepare=False,
        )
        row = cur.fetchone()
    return SOURCE_HR if row else None


def _lookup_research_snapshot(conn, denver_date: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            select 1
            from odds_snapshots o
            join games g on g.id = o.game_id
            where g.league = 'MLB'
              and o.metadata->>'provider' = 'the_odds_api'
              and (o.snapshot_ts at time zone 'America/Denver')::date = %s
            limit 1
            """,
            (denver_date,),
            prepare=False,
        )
        row = cur.fetchone()
    return SOURCE_RESEARCH if row else None


def _safe_lookup(conn, lookup, denver_date: str) -> str | None:
    try:
        return lookup(conn, denver_date)
    except UndefinedTable:
        conn.rollback()
        return None
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        LOGGER.warning("Odds API usage lookup failed for %s: %s", denver_date, exc)
        return None


def odds_already_used_today(denver_date: str, *, conn=None) -> tuple[bool, str | None]:
    """True when The Odds API was already claimed for this Denver date.

    Returns ``(already_used, source)`` where source is ``mlb_hr``, ``mlb_research``,
    or a legacy snapshot provider string.
    """
    owns_conn = conn is None
    if conn is None:
        try:
            conn = _connect()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not check Odds API usage for %s: %s", denver_date, exc)
            return False, None
        if conn is None:
            return False, None
    try:
        source = _safe_lookup(conn, _lookup_usage_row, denver_date)
        if source:
            return True, source

        source = _safe_lookup(conn, _lookup_hr_snapshot, denver_date)
        if source:
            claim_odds_api_usage(
                denver_date,
                SOURCE_HR,
                notes="backfilled from mlb_home_run_odds_snapshots",
                conn=conn,
            )
            return True, source

        source = _safe_lookup(conn, _lookup_research_snapshot, denver_date)
        if source:
            claim_odds_api_usage(
                denver_date,
                SOURCE_RESEARCH,
                notes="backfilled from odds_snapshots",
                conn=conn,
            )
            return True, source
        return False, None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not check Odds API usage for %s: %s", denver_date, exc)
        return False, None
    finally:
        if owns_conn and conn is not None:
            conn.close()


def claim_odds_api_usage(
    denver_date: str,
    source: str,
    *,
    notes: str | None = None,
    conn=None,
) -> bool:
    """Insert the day's Odds API usage row.

    Returns True if this caller claimed the budget, False if a row already exists.
    Missing table or credentials do not block a fetch (returns True) after logging.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"invalid odds_api_usage source: {source}")

    owns_conn = conn is None
    if conn is None:
        try:
            conn = _connect()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not claim Odds API usage for %s: %s", denver_date, exc)
            return True
        if conn is None:
            LOGGER.warning("Cannot claim Odds API usage for %s: missing Supabase credentials", denver_date)
            return True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into odds_api_usage (denver_date, source, notes)
                values (%s, %s, %s)
                on conflict (denver_date) do nothing
                returning denver_date
                """,
                (denver_date, source, notes),
                prepare=False,
            )
            row = cur.fetchone()
        conn.commit()
        claimed = row is not None
        if claimed:
            LOGGER.info(
                "Claimed Odds API daily budget for %s source=%s",
                denver_date,
                source,
            )
        else:
            LOGGER.info(
                "Odds API daily budget already claimed for %s; this caller did not stamp it",
                denver_date,
            )
        return claimed
    except UndefinedTable:
        conn.rollback()
        LOGGER.warning(
            "odds_api_usage table missing; apply sql/023_odds_api_usage.sql before MLB Odds fetches"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        LOGGER.warning("Could not claim Odds API usage for %s: %s", denver_date, exc)
        return True
    finally:
        if owns_conn and conn is not None:
            conn.close()
