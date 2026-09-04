#!/usr/bin/env python3
"""Audit NBA/NFL season readiness for dashboard serving."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

LOGGER = logging.getLogger("audit_season_readiness")
REQUIRED_TEAM_MARKETS = ("moneyline", "spread", "total")
FRESHNESS_HOURS = 36
NFL_AVAILABILITY_SOURCE = "nflverse_roster+sleeper"
NFL_IMPACT_MODEL_VERSION = "nfl-roster-epa-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit league serving readiness.")
    parser.add_argument("--league", required=True, choices=["NBA", "NFL"])
    parser.add_argument("--date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=None)
    parser.add_argument("--lookahead-days", type=int, default=7)
    parser.add_argument("--week", type=int, default=None, help="NFL week filter (optional).")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args()


def _date_window(anchor: date, lookahead_days: int) -> tuple[date, date]:
    return anchor, anchor + timedelta(days=lookahead_days)


def readiness_issues(report: dict[str, Any]) -> list[str]:
    """Return hard serving failures from an assembled audit report."""

    issues: list[str] = []
    scheduled = int(report.get("scheduled_games") or 0)
    if scheduled == 0:
        issues.append(f"No {report['league']} games in window.")
        return issues

    with_prediction = int(report.get("games_with_prediction") or 0)
    fresh_predictions = int(report.get("fresh_predictions") or 0)
    if with_prediction < scheduled:
        issues.append(f"{scheduled - with_prediction} games missing predictions.")
    if fresh_predictions < scheduled:
        issues.append(f"{scheduled - fresh_predictions} games missing fresh predictions.")
    if int(report.get("duplicate_rows") or 0) > 0:
        issues.append(f"{report['duplicate_rows']} duplicate game rows detected.")
    if int(report.get("missing_book_spread") or 0) > 0:
        issues.append(f"{report['missing_book_spread']} games missing book_spread.")

    for market in REQUIRED_TEAM_MARKETS:
        coverage = (report.get("market_coverage") or {}).get(market, {})
        complete = int(coverage.get("complete_games") or 0)
        fresh = int(coverage.get("fresh_games") or 0)
        if complete < scheduled:
            issues.append(f"{scheduled - complete} games missing paired {market} odds.")
        elif fresh < scheduled:
            issues.append(f"{scheduled - fresh} games have stale paired {market} odds.")

    if report.get("league") == "NFL":
        reports = int(report.get("availability_reports") or 0)
        fresh_reports = int(report.get("fresh_availability_reports") or 0)
        if reports == 0:
            issues.append("No current NFL availability reports in window.")
        elif fresh_reports < reports:
            issues.append(f"{reports - fresh_reports} latest NFL availability reports are stale.")
        missing_impacts = int(report.get("eligible_absences_missing_impact") or 0)
        if missing_impacts:
            issues.append(f"{missing_impacts} eligible NFL absences are missing impact estimates.")
        td_prediction_games = int(report.get("anytime_td_prediction_games") or 0)
        td_odds_games = int(report.get("anytime_td_odds_games") or 0)
        fresh_td_odds_games = int(report.get("fresh_anytime_td_odds_games") or 0)
        if td_prediction_games < scheduled:
            issues.append(f"{scheduled - td_prediction_games} games missing anytime-TD predictions.")
        if td_odds_games < scheduled:
            issues.append(f"{scheduled - td_odds_games} games missing anytime-TD odds.")
        elif fresh_td_odds_games < scheduled:
            issues.append(f"{scheduled - fresh_td_odds_games} games have stale anytime-TD odds.")
        if int(report.get("qualified_anytime_td_rows") or 0) == 0:
            issues.append("No anytime-TD rows passed serving guardrails.")
    return issues


def audit_league(
    conn,
    *,
    league: str,
    start_date: date,
    end_date: date,
    week: int | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "league": league,
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "week": week,
        "issues": [],
        "ready": True,
    }

    with conn.cursor() as cur:
        week_clause = ""
        params: list[Any] = [league, start_date, end_date]
        if week is not None:
            week_clause = " AND g.week = %s"
            params.append(week)

        cur.execute(
            f"""
            SELECT COUNT(*) FROM games g
            WHERE g.league = %s
              AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                BETWEEN %s AND %s
            {week_clause}
            """,
            params,
        )
        report["scheduled_games"] = int(cur.fetchone()[0])

        cur.execute(
            f"""
            WITH scoped AS (
              SELECT g.id
              FROM games g
              WHERE g.league = %s
                AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                  BETWEEN %s AND %s
              {week_clause}
            ),
            latest_pred AS (
              SELECT DISTINCT ON (p.game_id)
                p.game_id,
                p.asof_ts,
                p.model_version
              FROM model_predictions p
              JOIN scoped s ON s.id = p.game_id
              ORDER BY p.game_id, p.asof_ts DESC
            )
            SELECT
              (SELECT COUNT(*) FROM scoped) AS games,
              (SELECT COUNT(*) FROM latest_pred) AS with_prediction,
              (SELECT COUNT(*) FROM latest_pred WHERE asof_ts >= NOW() - INTERVAL '48 hours') AS fresh_predictions
            """,
            params,
        )
        games, with_prediction, fresh_predictions = cur.fetchone()
        report["games_with_prediction"] = int(with_prediction)
        report["fresh_predictions"] = int(fresh_predictions)
        report["prediction_coverage_pct"] = (
            round(100.0 * with_prediction / games, 1) if games else None
        )
        report["stale_prediction_rows"] = int(with_prediction - fresh_predictions)

        cur.execute(
            f"""
            SELECT COUNT(*) FROM games g
            WHERE g.league = %s
              AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                BETWEEN %s AND %s
              AND g.book_spread IS NULL
            {week_clause}
            """,
            params,
        )
        report["missing_book_spread"] = int(cur.fetchone()[0])

        cur.execute(
            f"""
            WITH scoped AS (
              SELECT g.id
              FROM games g
              WHERE g.league = %s
                AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                  BETWEEN %s AND %s
              {week_clause}
            ),
            latest AS (
              SELECT DISTINCT ON (o.game_id, o.market, o.selection)
                o.game_id,
                o.market,
                o.selection,
                o.snapshot_ts
              FROM odds_snapshots o
              JOIN scoped s ON s.id = o.game_id
              WHERE (
                (o.market IN ('moneyline', 'spread') AND o.selection IN ('home', 'away'))
                OR (o.market = 'total' AND o.selection IN ('over', 'under'))
              )
              ORDER BY o.game_id, o.market, o.selection, o.snapshot_ts DESC
            ),
            paired AS (
              SELECT
                game_id,
                market,
                MIN(snapshot_ts) AS pair_snapshot
              FROM latest
              GROUP BY game_id, market
              HAVING
                (market IN ('moneyline', 'spread')
                  AND BOOL_OR(selection = 'home') AND BOOL_OR(selection = 'away'))
                OR
                (market = 'total'
                  AND BOOL_OR(selection = 'over') AND BOOL_OR(selection = 'under'))
            )
            SELECT
              market,
              COUNT(*) AS complete_games,
              COUNT(*) FILTER (
                WHERE pair_snapshot >= NOW() - INTERVAL '{FRESHNESS_HOURS} hours'
              ) AS fresh_games
            FROM paired
            GROUP BY market
            """,
            params,
            prepare=False,
        )
        market_rows = {str(row[0]): (int(row[1]), int(row[2])) for row in cur.fetchall()}
        report["market_coverage"] = {}
        for market in REQUIRED_TEAM_MARKETS:
            complete, fresh = market_rows.get(market, (0, 0))
            report["market_coverage"][market] = {
                "complete_games": complete,
                "fresh_games": fresh,
                "coverage_pct": round(100.0 * complete / games, 1) if games else None,
            }

        cur.execute(
            f"""
            WITH groups AS (
              SELECT
                COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date) AS gd,
                g.home_team,
                g.away_team,
                COUNT(*) AS row_count
              FROM games g
              WHERE g.league = %s
                AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                  BETWEEN %s AND %s
              {week_clause}
              GROUP BY 1, g.home_team, g.away_team
              HAVING COUNT(*) > 1
            )
            SELECT COALESCE(SUM(row_count - 1), 0) FROM groups
            """,
            params,
        )
        report["duplicate_rows"] = int(cur.fetchone()[0])

        if league == "NFL":
            cur.execute(
                f"""
                WITH scoped AS (
                  SELECT g.id
                  FROM games g
                  WHERE g.league = %s
                    AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                      BETWEEN %s AND %s
                  {week_clause}
                ),
                latest_reports AS (
                  SELECT DISTINCT ON (
                    par.game_id,
                    COALESCE(par.player_id, par.player_name),
                    par.source
                  )
                    par.*
                  FROM player_availability_reports par
                  JOIN scoped s ON s.id = par.game_id
                  WHERE par.source = %s
                  ORDER BY
                    par.game_id,
                    COALESCE(par.player_id, par.player_name),
                    par.source,
                    par.report_ts DESC
                )
                SELECT
                  COUNT(*) AS reports,
                  COUNT(*) FILTER (
                    WHERE lr.report_ts >= NOW() - INTERVAL '{FRESHNESS_HOURS} hours'
                  ) AS fresh_reports,
                  MAX(lr.report_ts) AS latest_report_ts,
                  COUNT(*) FILTER (
                    WHERE COALESCE((lr.raw_record->>'confirmed_unavailable')::boolean, false)
                  ) AS confirmed_unavailable,
                  COUNT(*) FILTER (
                    WHERE COALESCE((lr.raw_record->>'impact_eligible')::boolean, false)
                  ) AS eligible_absences,
                  COUNT(*) FILTER (
                    WHERE COALESCE((lr.raw_record->>'impact_eligible')::boolean, false)
                      AND pie.id IS NULL
                  ) AS missing_impacts,
                  COUNT(DISTINCT lr.game_id) AS games_with_reports
                FROM latest_reports lr
                LEFT JOIN player_impact_estimates pie
                  ON pie.game_id = lr.game_id
                 AND pie.model_version = %s
                 AND (
                   (lr.player_id IS NOT NULL AND pie.player_id = lr.player_id)
                   OR (lr.player_id IS NULL AND pie.player_name = lr.player_name)
                 )
                """,
                [*params, NFL_AVAILABILITY_SOURCE, NFL_IMPACT_MODEL_VERSION],
                prepare=False,
            )
            (
                reports,
                fresh_reports,
                latest_report_ts,
                confirmed_unavailable,
                eligible_absences,
                missing_impacts,
                games_with_reports,
            ) = cur.fetchone()
            report.update(
                {
                    "availability_reports": int(reports),
                    "fresh_availability_reports": int(fresh_reports),
                    "latest_availability_report_ts": (
                        latest_report_ts.isoformat() if latest_report_ts else None
                    ),
                    "confirmed_unavailable_reports": int(confirmed_unavailable),
                    "eligible_absence_reports": int(eligible_absences),
                    "eligible_absences_missing_impact": int(missing_impacts),
                    "games_with_availability_reports": int(games_with_reports),
                }
            )

            cur.execute(
                f"""
                WITH scoped AS (
                  SELECT g.id
                  FROM games g
                  WHERE g.league = %s
                    AND COALESCE(g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
                      BETWEEN %s AND %s
                  {week_clause}
                )
                SELECT
                  (SELECT COUNT(DISTINCT p.game_id)
                   FROM nfl_anytime_td_predictions p JOIN scoped s ON s.id = p.game_id),
                  (SELECT COUNT(DISTINCT o.game_id)
                   FROM nfl_anytime_td_odds_snapshots o JOIN scoped s ON s.id = o.game_id),
                  (SELECT COUNT(DISTINCT o.game_id)
                   FROM nfl_anytime_td_odds_snapshots o JOIN scoped s ON s.id = o.game_id
                   WHERE o.snapshot_ts >= NOW() - INTERVAL '{FRESHNESS_HOURS} hours'),
                  (SELECT COUNT(*)
                   FROM nfl_anytime_td_edges_latest e JOIN scoped s ON s.id = e.game_id
                   WHERE e.odds_status = 'priced'
                     AND e.best_price <= 1000
                     AND e.sample_games >= 10
                     AND e.quality_flags = '[]'::jsonb)
                """,
                params,
                prepare=False,
            )
            td_prediction_games, td_odds_games, fresh_td_odds_games, qualified_td = cur.fetchone()
            report.update(
                {
                    "anytime_td_prediction_games": int(td_prediction_games),
                    "anytime_td_odds_games": int(td_odds_games),
                    "fresh_anytime_td_odds_games": int(fresh_td_odds_games),
                    "qualified_anytime_td_rows": int(qualified_td),
                }
            )

    report["issues"] = readiness_issues(report)
    report["ready"] = not report["issues"]
    report["audited_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")

    anchor = args.date or datetime.now(timezone.utc).date()
    start_date, end_date = _date_window(anchor, args.lookahead_days)

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
        report = audit_league(
            conn,
            league=args.league,
            start_date=start_date,
            end_date=end_date,
            week=args.week,
        )
    finally:
        conn.close()

    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        LOGGER.info("Audit report: %s", json.dumps(report, sort_keys=True))
        print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
