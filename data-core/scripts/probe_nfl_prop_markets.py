#!/usr/bin/env python3
"""Handful-of-calls NFL player-prop availability probe.

PropLine-first. The Odds API is opt-in, capped, and skipped when the shared
Starter budget cannot spare even one event-odds call. Historical Odds API
closes are estimated, not backfilled.

Does not write serving tables or live predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nfl_prop_markets import (  # noqa: E402
    NFL_PROP_MARKETS,
    NFL_SPORT_KEYS,
    aggregate_event_summaries,
    compact_closing_rows,
    credit_log_row,
    group_market_keys,
    summarize_event_odds,
)
from src.data.propline_client import PropLineClient, PropLineError  # noqa: E402

# Copied from src.data.odds_api_client so this probe does not import psycopg.
MONTHLY_LIMIT = 500
CONTINGENCY = 50
SOURCE_LIMITS = {"nfl": 140, "nba": 100, "mlb": 100, "cfb": 40, "pga": 20}

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "analysis" / "nfl-props-market-probe.json"
SPORT = "football_nfl"
ODDS_SPORT = "americanfootball_nfl"
ODDS_BASE = "https://api.the-odds-api.com/v4"
MAX_ODDS_EVENTS = 1
MAX_PROPLINE_EVENTS = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-odds-api",
        action="store_true",
        help="Spend a tiny Odds API current-odds sample (1 event × 5 markets ≤ 5 credits). Off by default.",
    )
    parser.add_argument(
        "--allow-odds-history",
        action="store_true",
        help="Spend Odds API historical event odds. Default off: one week of rec-yard closes is ~160 credits.",
    )
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(payload: Any) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "events", "sports"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def probe_propline(client: PropLineClient) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    sport_payload = client.get("/sports")
    sports = _as_list(sport_payload)
    nfl_sports = [
        row
        for row in sports
        if str(row.get("key") or "") in NFL_SPORT_KEYS or "nfl" in str(row.get("key") or "").lower()
    ]
    calls.append(
        credit_log_row(
            provider="propline",
            endpoint="/v1/sports",
            cost=1,
            remaining=client.response_meta.rate_limit_remaining,
            notes="sport catalog",
        )
    )

    events = _as_list(client.get(f"/sports/{SPORT}/events"))
    calls.append(
        credit_log_row(
            provider="propline",
            endpoint=f"/v1/sports/{SPORT}/events",
            cost=1,
            remaining=client.response_meta.rate_limit_remaining,
            notes=f"{len(events)} upcoming/in-play events",
        )
    )
    scores = _as_list(client.get(f"/sports/{SPORT}/scores"))
    calls.append(
        credit_log_row(
            provider="propline",
            endpoint=f"/v1/sports/{SPORT}/scores",
            cost=1,
            remaining=client.response_meta.rate_limit_remaining,
            notes=f"{len(scores)} score rows",
        )
    )

    market_keys: list[str] = []
    event_summaries: list[dict[str, Any]] = []
    markets_raw: Any = []
    sampled_events = events[:MAX_PROPLINE_EVENTS]
    if sampled_events:
        first_id = str(sampled_events[0].get("id"))
        markets_raw = client.get(f"/sports/{SPORT}/events/{first_id}/markets")
        calls.append(
            credit_log_row(
                provider="propline",
                endpoint=f"/v1/sports/{SPORT}/events/{{id}}/markets",
                cost=1,
                remaining=client.response_meta.rate_limit_remaining,
                notes="discovery before odds pull",
            )
        )
        if isinstance(markets_raw, list):
            market_keys = [str(row.get("key") or row) for row in markets_raw if row]
        elif isinstance(markets_raw, dict):
            market_keys = [str(row.get("key")) for row in _as_list(markets_raw)]

        requested = ",".join(NFL_PROP_MARKETS)
        for event in sampled_events:
            event_id = str(event.get("id"))
            try:
                odds = client.get(
                    f"/sports/{SPORT}/events/{event_id}/odds",
                    {"markets": requested},
                )
            except PropLineError as exc:
                calls.append(
                    credit_log_row(
                        provider="propline",
                        endpoint=f"/v1/sports/{SPORT}/events/{{id}}/odds",
                        cost=1,
                        remaining=client.response_meta.rate_limit_remaining,
                        notes=f"failed: {exc}",
                    )
                )
                continue
            calls.append(
                credit_log_row(
                    provider="propline",
                    endpoint=f"/v1/sports/{SPORT}/events/{{id}}/odds",
                    cost=1,
                    remaining=client.response_meta.rate_limit_remaining,
                    notes=f"markets={requested}",
                )
            )
            if isinstance(odds, dict):
                event_summaries.append(summarize_event_odds(odds))

    history = {"attempted": False, "redacted": None, "notes": "no completed event sampled"}
    history_payload: Any = None
    completed = [row for row in scores if row.get("completed") or str(row.get("status") or "").lower() in {"final", "completed"}]
    if completed:
        event_id = str(completed[0].get("id") or completed[0].get("event_id"))
        history["attempted"] = True
        try:
            history_payload = client.get(
                f"/sports/{SPORT}/events/{event_id}/odds/history",
                {"markets": "player_reception_yds"},
            )
            calls.append(
                credit_log_row(
                    provider="propline",
                    endpoint=f"/v1/sports/{SPORT}/events/{{id}}/odds/history",
                    cost=1,
                    remaining=client.response_meta.rate_limit_remaining,
                    notes="receiving yards only",
                )
            )
            history["redacted"] = bool(history_payload.get("redacted")) if isinstance(history_payload, dict) else False
            history["notes"] = (
                "Free/demo history is structure-only when redacted=true. Paid tiers keep 30–365 days; "
                "the CSV archive starts April 2026 and NFL grading starts 2026."
            )
            if isinstance(history_payload, dict):
                history["keys"] = sorted(history_payload.keys())
                history["bookmaker_count"] = len(history_payload.get("bookmakers") or [])
                history["has_upgrade_url"] = bool(history_payload.get("upgrade_url"))
        except PropLineError as exc:
            history["notes"] = str(exc)
            calls.append(
                credit_log_row(
                    provider="propline",
                    endpoint=f"/v1/sports/{SPORT}/events/{{id}}/odds/history",
                    cost=1,
                    remaining=client.response_meta.rate_limit_remaining,
                    notes=str(exc),
                )
            )
        try:
            closing = client.get(
                f"/sports/{SPORT}/events/{event_id}/odds/closing",
                {"markets": "player_reception_yds"},
            )
            calls.append(
                credit_log_row(
                    provider="propline",
                    endpoint=f"/v1/sports/{SPORT}/events/{{id}}/odds/closing",
                    cost=1,
                    remaining=client.response_meta.rate_limit_remaining,
                    notes="CLV helper",
                )
            )
            if isinstance(closing, dict):
                history["closing_redacted"] = bool(closing.get("redacted"))
                history["closing_keys"] = sorted(closing.keys())
                history["event_id"] = closing.get("id") or event_id
                history["home_team"] = closing.get("home_team")
                history["away_team"] = closing.get("away_team")
                history["commence_time"] = closing.get("commence_time")
                sample = compact_closing_rows(closing, market="player_reception_yds")
                source = "odds_closing" if sample else "empty"
                if not sample and isinstance(history_payload, dict):
                    history_sample = compact_closing_rows(history_payload, market="player_reception_yds")
                    if history_sample:
                        sample = history_sample
                        source = "odds_history_last_tick"
                history["closing_sample_source"] = source
                history["closing_sample"] = sample
                history["closing_sample_rows"] = len(sample)
        except PropLineError as exc:
            history["closing_error"] = str(exc)

    return {
        "nfl_sports": nfl_sports,
        "upcoming_events": len(events),
        "score_rows": len(scores),
        "market_discovery": group_market_keys(market_keys),
        "event_summaries": event_summaries,
        "coverage": aggregate_event_summaries(event_summaries),
        "history": history,
        "calls": calls,
        "request_count": client.request_count,
        "rate_limit_remaining": client.response_meta.rate_limit_remaining,
    }


def odds_account_probe(api_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    response = requests.get(f"{ODDS_BASE}/sports/", params={"apiKey": api_key}, timeout=30)
    account = {
        "http_status": response.status_code,
        "used": response.headers.get("x-requests-used"),
        "remaining": response.headers.get("x-requests-remaining"),
        "last_cost": response.headers.get("x-requests-last"),
    }
    call = credit_log_row(
        provider="the_odds_api",
        endpoint="/v4/sports/",
        cost=0,
        remaining=account["remaining"],
        used=account["used"],
        last_cost=account["last_cost"],
        notes="free catalog / quota headers",
    )
    return account, call


def should_spend_odds_api(account: Mapping[str, Any], *, nfl_spent: int, cost: int) -> tuple[bool, str]:
    try:
        remaining = int(account.get("remaining"))
        used = int(account.get("used"))
    except (TypeError, ValueError):
        return False, "Quota headers missing; Odds API paid request skipped."
    if nfl_spent + cost > SOURCE_LIMITS["nfl"]:
        return False, f"NFL monthly allocation {SOURCE_LIMITS['nfl']} would be exceeded."
    available = min(remaining, MONTHLY_LIMIT - used)
    if available - cost < CONTINGENCY:
        return False, f"Preserving {CONTINGENCY} contingency credits; available={available}, cost={cost}."
    return True, "within Starter budget"


def probe_odds_api(*, allow_current: bool, allow_history: bool) -> dict[str, Any]:
    api_key = os.getenv("ODDS_API_KEY")
    calls: list[dict[str, Any]] = []
    if not api_key:
        return {
            "skipped": True,
            "reason": "ODDS_API_KEY not set in this environment. Live current-odds sample not spent.",
            "calls": calls,
            "docs_markets": list(NFL_PROP_MARKETS),
        }
    account, catalog_call = odds_account_probe(api_key)
    calls.append(catalog_call)
    events_resp = requests.get(
        f"{ODDS_BASE}/sports/{ODDS_SPORT}/events",
        params={"apiKey": api_key},
        timeout=30,
    )
    calls.append(
        credit_log_row(
            provider="the_odds_api",
            endpoint=f"/v4/sports/{ODDS_SPORT}/events",
            cost=0,
            remaining=events_resp.headers.get("x-requests-remaining"),
            used=events_resp.headers.get("x-requests-used"),
            last_cost=events_resp.headers.get("x-requests-last"),
            notes="event list is free",
        )
    )
    events = events_resp.json() if events_resp.ok else []
    coverage = None
    spend_notes = []
    if allow_current and isinstance(events, list) and events:
        cost = len(NFL_PROP_MARKETS) * MAX_ODDS_EVENTS
        try:
            nfl_spent = int(account.get("used") or 0)
        except (TypeError, ValueError):
            nfl_spent = SOURCE_LIMITS["nfl"]
        ok, reason = should_spend_odds_api(account, nfl_spent=nfl_spent, cost=cost)
        spend_notes.append(reason)
        if ok:
            event = events[0]
            odds_resp = requests.get(
                f"{ODDS_BASE}/sports/{ODDS_SPORT}/events/{event['id']}/odds",
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": ",".join(NFL_PROP_MARKETS),
                    "oddsFormat": "american",
                },
                timeout=30,
            )
            calls.append(
                credit_log_row(
                    provider="the_odds_api",
                    endpoint=f"/v4/sports/{ODDS_SPORT}/events/{{id}}/odds",
                    cost=int(odds_resp.headers.get("x-requests-last") or cost),
                    remaining=odds_resp.headers.get("x-requests-remaining"),
                    used=odds_resp.headers.get("x-requests-used"),
                    last_cost=odds_resp.headers.get("x-requests-last"),
                    notes="current player props, 1 event, us region",
                )
            )
            if odds_resp.ok:
                coverage = aggregate_event_summaries([summarize_event_odds(odds_resp.json())])
        else:
            spend_notes.append("current event odds not fetched")
    elif not allow_current:
        spend_notes.append("current Odds API event odds skipped (--allow-odds-api not set)")

    if allow_history:
        spend_notes.append("historical event odds not fetched: 10 credits per event per market is not cheap")
    else:
        spend_notes.append("historical Odds API skipped; 2023-2025 rec-yard close backfill estimated only")

    return {
        "skipped": False,
        "account": account,
        "upcoming_events": len(events) if isinstance(events, list) else 0,
        "coverage": coverage,
        "calls": calls,
        "notes": spend_notes,
        "docs_markets": list(NFL_PROP_MARKETS),
    }


def historical_cost_estimate() -> dict[str, Any]:
    games_per_season = 17 * 16
    seasons = 3
    games = games_per_season * seasons
    credits_per_close = 10  # 1 market × 1 region × historical event odds
    discovery_per_week = 1  # historical events list
    weeks = 17 * seasons
    return {
        "market": "player_reception_yds",
        "region": "us",
        "seasons": "2023-2025 REG",
        "games": games,
        "credits_per_event_close": credits_per_close,
        "close_credits": games * credits_per_close,
        "weekly_event_discovery_credits": weeks * discovery_per_week,
        "total_credits_one_snapshot_near_close": games * credits_per_close + weeks * discovery_per_week,
        "one_week_sample_credits": 16 * credits_per_close + 1,
        "two_week_sample_credits": 32 * credits_per_close + 2,
        "starter_monthly_limit": MONTHLY_LIMIT,
        "nfl_monthly_allocation": SOURCE_LIMITS["nfl"],
        "contingency": CONTINGENCY,
        "sampled": False,
        "sample_reason": (
            "A 1-week receiving-yards close sample is ~161 credits. Daily Refresh on 2026-10-19 "
            "showed account used=63 remaining=437 and NFL ledger 63/140. 161 credits would blow "
            "the NFL allocation and eat a third of the remaining Starter month. Not cheap. Not run."
        ),
        "propline_backfill": (
            "PropLine CSV/history archive starts April 2026. NFL player-prop grading starts with "
            "the 2026 season. 2023-2025 closing lines are not in PropLine."
        ),
    }


def main() -> None:
    args = _parse_args()
    load_dotenv(ROOT / ".env")
    report: dict[str, Any] = {
        "generated_at": _now(),
        "purpose": "NFL player-prop data-availability check. Analysis only.",
        "markets": list(NFL_PROP_MARKETS),
        "propline": None,
        "odds_api": None,
        "historical_cost": historical_cost_estimate(),
    }
    propline_key = os.getenv("PROPLINE_API_KEY")
    if propline_key:
        client = PropLineClient(api_key=propline_key)
        try:
            report["propline"] = probe_propline(client)
        except PropLineError as exc:
            report["propline"] = {"error": str(exc), "request_count": client.request_count}
    else:
        report["propline"] = {
            "skipped": True,
            "reason": "PROPLINE_API_KEY not set",
        }

    report["odds_api"] = probe_odds_api(
        allow_current=args.allow_odds_api,
        allow_history=args.allow_odds_history,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "output": str(args.output),
            "propline_requests": (report.get("propline") or {}).get("request_count"),
            "odds_api_skipped": (report.get("odds_api") or {}).get("skipped"),
            "historical_credits": report["historical_cost"]["total_credits_one_snapshot_near_close"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
