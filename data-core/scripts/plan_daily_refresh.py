#!/usr/bin/env python3
"""
Plan which leagues should run in the daily refresh workflow.

The plan is intentionally data-aware but not data-dependent: scheduled games in
BigQuery can activate a league outside the broad calendar window, while the
calendar window keeps in-season raw refreshes running even before today's raw
schedule has been updated.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

try:
    from google.cloud import bigquery
except ImportError:  # pragma: no cover - local unit tests do not need BigQuery
    bigquery = None


LEAGUES = ("MLB", "NBA", "NFL", "WORLD_CUP")
SLATE_TIME_ZONE = "America/Denver"


@dataclass(frozen=True)
class SeasonWindow:
    start_month: int
    start_day: int
    end_month: int
    end_day: int

    @property
    def wraps_year(self) -> bool:
        return (self.end_month, self.end_day) < (self.start_month, self.start_day)


SEASON_WINDOWS = {
    "MLB": SeasonWindow(3, 1, 11, 15),
    "NBA": SeasonWindow(10, 1, 6, 30),
    "NFL": SeasonWindow(8, 1, 2, 15),
    "WORLD_CUP": SeasonWindow(6, 1, 7, 31),
}

WORLD_CUP_TOURNAMENT_WINDOWS = {
    2026: (date(2026, 6, 11), date(2026, 7, 19)),
}


def default_anchor_date(now: datetime | None = None) -> date:
    if now is None:
        now = datetime.now(tz=ZoneInfo(SLATE_TIME_ZONE))
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(ZoneInfo(SLATE_TIME_ZONE)).date()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def season_year_for(league: str, anchor: date) -> int:
    league = league.upper()
    if league == "MLB":
        return anchor.year
    if league == "NBA":
        return anchor.year if anchor.month >= 10 else anchor.year - 1
    if league == "NFL":
        return anchor.year if anchor.month >= 8 else anchor.year - 1
    if league == "WORLD_CUP":
        return anchor.year
    raise ValueError(f"Unsupported league: {league}")


def world_cup_tournament_window_for(season: int) -> tuple[date, date]:
    return WORLD_CUP_TOURNAMENT_WINDOWS.get(season, (date(season, 6, 1), date(season, 7, 31)))


def season_bounds_for(league: str, anchor: date) -> tuple[date, date]:
    league = league.upper()
    window = SEASON_WINDOWS[league]
    if not window.wraps_year:
        return (
            date(anchor.year, window.start_month, window.start_day),
            date(anchor.year, window.end_month, window.end_day),
        )

    if (anchor.month, anchor.day) >= (window.start_month, window.start_day):
        start_year = anchor.year
        end_year = anchor.year + 1
    else:
        start_year = anchor.year - 1
        end_year = anchor.year

    return (
        date(start_year, window.start_month, window.start_day),
        date(end_year, window.end_month, window.end_day),
    )


def date_ranges_intersect(left_start: date, left_end: date, right_start: date, right_end: date) -> bool:
    return left_start <= right_end and right_start <= left_end


def calendar_active(league: str, start_date: date, end_date: date, anchor: date) -> bool:
    season_start, season_end = season_bounds_for(league, anchor)
    return date_ranges_intersect(start_date, end_date, season_start, season_end)


def count_bigquery_games(project: str, league: str, start_date: date, end_date: date) -> Optional[int]:
    if not project or bigquery is None:
        return None

    client = bigquery.Client(project=project)
    query = f"""
        SELECT COUNT(DISTINCT game_id) AS games
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league = @league
          AND game_date IS NOT NULL
          AND CAST(game_date AS DATE) BETWEEN @start_date AND @end_date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("league", "STRING", league),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    result = list(client.query(query, job_config=job_config).result())
    return int(result[0]["games"]) if result else 0


def build_plan(
    *,
    anchor: date,
    lookback_days: int,
    lookahead_days: int,
    force_full_rebuild: bool,
    project: str = "",
) -> dict:
    if lookback_days < 0 or lookahead_days < 0:
        raise ValueError("lookback_days and lookahead_days must be non-negative.")

    start_date = anchor - timedelta(days=lookback_days)
    end_date = anchor + timedelta(days=lookahead_days)
    plan: dict[str, object] = {
        "anchor_date": anchor.isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "force_full_rebuild": force_full_rebuild,
    }

    for league in LEAGUES:
        bq_games = None
        bq_error = None
        if project and league != "WORLD_CUP":
            try:
                bq_games = count_bigquery_games(project, league, start_date, end_date)
            except Exception as exc:  # noqa: BLE001
                bq_error = str(exc)

        in_calendar = calendar_active(league, start_date, end_date, anchor)
        run_league = force_full_rebuild or in_calendar or bool(bq_games)

        reasons = []
        if force_full_rebuild:
            reasons.append("force_full_rebuild")
        if bq_games:
            reasons.append(f"{bq_games} scheduled games in window")
        if in_calendar:
            reasons.append("calendar season window")
        if bq_error:
            reasons.append(f"schedule query unavailable: {bq_error}")
        if not reasons:
            reasons.append("offseason and no scheduled games in window")

        key = league.lower()
        plan[f"run_{key}"] = run_league
        plan[f"{key}_season"] = season_year_for(league, anchor)
        plan[f"{key}_scheduled_games"] = bq_games
        plan[f"{key}_reason"] = "; ".join(reasons)
        if league == "WORLD_CUP":
            wc_start, wc_end = world_cup_tournament_window_for(int(plan[f"{key}_season"]))
            plan["world_cup_start_date"] = wc_start.isoformat()
            plan["world_cup_end_date"] = wc_end.isoformat()

    plan["run_any"] = any(bool(plan[f"run_{league.lower()}"]) for league in LEAGUES)
    plan["run_market_odds"] = bool(plan["run_nba"] or plan["run_nfl"])
    return plan


def write_github_outputs(plan: dict, output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as handle:
        for key, value in plan.items():
            if isinstance(value, bool):
                value = str(value).lower()
            elif value is None:
                value = ""
            else:
                value = str(value).replace("\r", " ").replace("\n", " ")
            handle.write(f"{key}={value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan daily refresh league gates.")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID", ""), help="GCP project ID.")
    parser.add_argument(
        "--date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=default_anchor_date(),
        help=f"Anchor date in YYYY-MM-DD. Default: today in {SLATE_TIME_ZONE}.",
    )
    parser.add_argument("--lookback-days", type=int, default=1)
    parser.add_argument("--lookahead-days", type=int, default=10)
    parser.add_argument(
        "--force-full-rebuild",
        default=os.getenv("FORCE_FULL_REBUILD", "false"),
        help="Truth-like value that enables all leagues.",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Write plan keys to GITHUB_OUTPUT when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(
        anchor=args.date,
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days,
        force_full_rebuild=parse_bool(args.force_full_rebuild),
        project=args.project,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))

    output_path = os.getenv("GITHUB_OUTPUT")
    if args.github_output and output_path:
        write_github_outputs(plan, output_path)


if __name__ == "__main__":
    main()
