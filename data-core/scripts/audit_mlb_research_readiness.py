#!/usr/bin/env python3
"""Strict operational audit for the daily MLB research-market feed.

This audit proves schedule, prediction, and sportsbook-price coverage. It does
not promote the underlying MLB team models beyond their explicit ``research``
status; model supportability is a separate evidence gate.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402

REQUIRED_MARKETS = ("moneyline", "run_line", "total")

# Pregame plus a short first-pitch grace. The previous 3-hour window kept
# already-final East/Central games "price-eligible" on evening re-runs after
# books had dropped h2h. Morning refresh still requires the full unplayed slate.
PRICE_ELIGIBLE_SQL = "game_datetime >= NOW() - INTERVAL '30 minutes'"
PRICE_ELIGIBLE_RULE = "not_started_or_first_pitch_30m"


def _matchup_label(row: dict[str, Any]) -> str:
    away = str(row.get("away_team") or "?").strip() or "?"
    home = str(row.get("home_team") or "?").strip() or "?"
    return f"{away} @ {home}"


def _missing_price_suffix(report: dict[str, Any], market: str) -> str:
    rows = (report.get("missing_prices") or {}).get(market) or []
    if not rows:
        return ""
    return ": " + ", ".join(_matchup_label(row) for row in rows)


def readiness_issues(report: dict[str, Any]) -> list[str]:
    """Return hard failures in the operational research-feed contract."""

    issues: list[str] = []
    scheduled = int(report.get("scheduled_games") or 0)
    if scheduled == 0:
        return ["No MLB games found for the requested date."]

    coverage = report.get("market_coverage") or {}
    for market in REQUIRED_MARKETS:
        row = coverage.get(market) or {}
        predicted = int(row.get("predicted_games") or 0)
        fresh_predictions = int(row.get("fresh_prediction_games") or 0)
        price_eligible = int(row.get("price_eligible_games") or 0)
        priced = int(row.get("fresh_priced_games") or 0)
        invalid = int(row.get("invalid_priced_rows") or 0)
        if predicted < scheduled:
            issues.append(f"{scheduled - predicted} games missing {market} predictions.")
        if fresh_predictions < scheduled:
            issues.append(f"{scheduled - fresh_predictions} {market} predictions are stale or missing.")
        if priced < price_eligible:
            issues.append(
                f"{price_eligible - priced} price-eligible games missing fresh paired {market} odds"
                f"{_missing_price_suffix(report, market)}."
            )
        if invalid:
            issues.append(f"{invalid} priced {market} rows violate the serving contract.")

    if int(report.get("duplicate_rows") or 0):
        issues.append(f"{report['duplicate_rows']} duplicate game-market rows detected.")
    if int(report.get("non_research_rows") or 0):
        issues.append(f"{report['non_research_rows']} rows are not labeled research.")
    return issues


def audit(conn, game_date: date) -> dict[str, Any]:
    """Audit one MLB slate without making external network requests."""

    report: dict[str, Any] = {
        "date": game_date.isoformat(),
        "research_only": True,
        "supportable_for_betting": False,
        "price_eligible_rule": PRICE_ELIGIBLE_RULE,
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM games
            WHERE league = 'MLB'
              AND COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date) = %s
            """,
            (game_date,),
            prepare=False,
        )
        report["scheduled_games"] = int(cur.fetchone()[0])

        cur.execute(
            f"""
            WITH latest AS (
              SELECT DISTINCT ON (game_pk, market) *
              FROM mlb_research_predictions
              WHERE game_date = %s
              ORDER BY game_pk, market, as_of_ts DESC
            )
            SELECT
              market,
              COUNT(DISTINCT game_pk) AS predicted_games,
              COUNT(DISTINCT game_pk) FILTER (
                WHERE as_of_ts >= NOW() - INTERVAL '24 hours'
              ) AS fresh_prediction_games,
              COUNT(DISTINCT game_pk) FILTER (
                WHERE {PRICE_ELIGIBLE_SQL}
              ) AS price_eligible_games,
              COUNT(DISTINCT game_pk) FILTER (
                WHERE {PRICE_ELIGIBLE_SQL}
                  AND odds_status = 'ok'
                  AND odds_snapshot_ts >= NOW() - INTERVAL '24 hours'
              ) AS fresh_priced_games,
              COUNT(*) FILTER (
                WHERE odds_status = 'ok' AND (
                  odds_snapshot_ts IS NULL OR best_book IS NULL
                  OR implied_probability IS NULL OR no_vig_probability IS NULL
                  OR edge IS NULL OR ev IS NULL OR kelly IS NULL
                  OR recommended_side IS NULL OR recommended_probability IS NULL
                  OR (market = 'moneyline' AND (home_price IS NULL OR away_price IS NULL))
                  OR (market = 'run_line' AND (
                    home_runline_price IS NULL OR away_runline_price IS NULL
                    OR home_runline_line IS NULL OR predicted_margin IS NULL
                  ))
                  OR (market = 'total' AND (
                    total_line IS NULL OR over_price IS NULL OR under_price IS NULL
                    OR predicted_total IS NULL
                  ))
                )
              ) AS invalid_priced_rows,
              MAX(as_of_ts) AS latest_prediction_ts,
              MAX(odds_snapshot_ts) AS latest_odds_ts
            FROM latest
            GROUP BY market
            """,
            (game_date,),
            prepare=False,
        )
        coverage = {}
        for row in cur.fetchall():
            coverage[row[0]] = {
                "predicted_games": int(row[1]),
                "fresh_prediction_games": int(row[2]),
                "price_eligible_games": int(row[3]),
                "fresh_priced_games": int(row[4]),
                "invalid_priced_rows": int(row[5]),
                "latest_prediction_ts": row[6].isoformat() if row[6] else None,
                "latest_odds_ts": row[7].isoformat() if row[7] else None,
            }
        report["market_coverage"] = coverage

        cur.execute(
            f"""
            WITH latest AS (
              SELECT DISTINCT ON (game_pk, market) *
              FROM mlb_research_predictions
              WHERE game_date = %s
              ORDER BY game_pk, market, as_of_ts DESC
            )
            SELECT market, game_pk, home_team, away_team, odds_status, game_datetime
            FROM latest
            WHERE {PRICE_ELIGIBLE_SQL}
              AND NOT (
                odds_status = 'ok'
                AND odds_snapshot_ts >= NOW() - INTERVAL '24 hours'
              )
            ORDER BY market, game_datetime, game_pk
            """,
            (game_date,),
            prepare=False,
        )
        missing_prices: dict[str, list[dict[str, Any]]] = {market: [] for market in REQUIRED_MARKETS}
        for market, game_pk, home_team, away_team, odds_status, game_datetime in cur.fetchall():
            if market not in missing_prices:
                missing_prices[market] = []
            missing_prices[market].append(
                {
                    "game_pk": int(game_pk) if game_pk is not None else None,
                    "home_team": home_team,
                    "away_team": away_team,
                    "odds_status": odds_status,
                    "game_datetime": game_datetime.isoformat() if game_datetime else None,
                }
            )
        report["missing_prices"] = missing_prices

        cur.execute(
            """
            SELECT
              COUNT(*) - COUNT(DISTINCT (game_pk, market)) AS duplicate_rows,
              COUNT(*) FILTER (WHERE model_status <> 'research') AS non_research_rows
            FROM mlb_research_predictions
            WHERE game_date = %s
            """,
            (game_date,),
            prepare=False,
        )
        duplicates, non_research = cur.fetchone()
        report["duplicate_rows"] = int(duplicates)
        report["non_research_rows"] = int(non_research)

    report["issues"] = readiness_issues(report)
    report["research_feed_ready"] = not report["issues"]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MLB research-market serving readiness.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    creds = load_supabase_credentials()
    conn = create_pg_connection(
        creds["url"], creds["db_password"], host_override=creds.get("db_host"),
        port=creds["db_port"], database=creds["db_name"], user=creds["db_user"],
    )
    try:
        report = audit(conn, args.date)
    finally:
        conn.close()
    print(json.dumps(report, indent=2 if args.json else None, sort_keys=True))
    if not report["research_feed_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
