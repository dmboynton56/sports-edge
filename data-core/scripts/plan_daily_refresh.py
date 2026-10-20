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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.live_model_versions import dashboard_model_version
from src.utils.nfl_schedule import nfl_week_window

try:
    from google.cloud import bigquery
except ImportError:  # pragma: no cover - local unit tests do not need BigQuery
    bigquery = None


LEAGUES = ("MLB", "NBA", "NFL", "CFB", "WORLD_CUP")
SLATE_TIME_ZONE = "America/Denver"
NFL_PROVIDER_FALLBACK_MAX_SEASON = 2025
NFL_LIVE_MODEL_VERSION = dashboard_model_version("NFL")
NFL_PREDICTION_STALE_HOURS = 48
NFL_PREDICTION_LOOKAHEAD_DAYS = 7
NFL_MAX_FORECAST_WINDOW_DAYS = 14


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
    "CFB": SeasonWindow(8, 1, 1, 31),
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
    if league == "CFB":
        return anchor.year if anchor.month >= 8 else anchor.year - 1
    if league == "WORLD_CUP":
        return anchor.year
    raise ValueError(f"Unsupported league: {league}")


def nfl_provider_max_season() -> int:
    """Return the newest NFL season with in-season stats from nflreadpy.

    ``get_current_season`` is a regular-season clock, not a schedule capability
    check.  Before kickoff it can trail the schedule feed by one season, so this
    value must not be used by itself to disable an upcoming-season refresh.
    """

    try:
        import nflreadpy as nfl  # noqa: WPS433

        return int(nfl.get_current_season())
    except Exception:  # noqa: BLE001 - a capability probe must never break planning
        return NFL_PROVIDER_FALLBACK_MAX_SEASON


def nfl_schedule_available(season: int) -> bool:
    """Return whether nflverse already publishes a schedule for ``season``."""

    try:
        import nflreadpy as nfl  # noqa: WPS433

        schedule = nfl.load_schedules([season])
        if hasattr(schedule, "collect"):
            schedule = schedule.collect()
        if hasattr(schedule, "is_empty"):
            return not bool(schedule.is_empty())
        return len(schedule) > 0
    except Exception:  # noqa: BLE001 - planner should degrade to BQ/calendar evidence
        return False


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


def fetch_nfl_prediction_freshness(
    project: str,
    start_date: date,
    end_date: date,
    *,
    now: datetime | None = None,
    model_version: str = NFL_LIVE_MODEL_VERSION,
) -> Optional[dict]:
    """Return upcoming NFL games in ``[start_date, end_date]`` and prediction age.

    ``None`` means BigQuery is unavailable — callers must fail closed to the
    Tuesday freeze instead of inventing a refresh. Missing or >48h-old
    predictions are counted so Daily Refresh can re-score without spending
    Odds API credits.
    """
    if not project or bigquery is None:
        return None
    now = now or datetime.now(timezone.utc)
    client = bigquery.Client(project=project)
    query = f"""
        WITH games AS (
          SELECT DISTINCT CAST(game_id AS STRING) AS game_id, CAST(game_date AS DATE) AS game_date
          FROM `{project}.sports_edge_raw.raw_schedules`
          WHERE league = 'NFL'
            AND game_date IS NOT NULL
            AND CAST(game_date AS DATE) BETWEEN @start_date AND @end_date
            AND home_score IS NULL
            AND away_score IS NULL
        ),
        preds AS (
          SELECT CAST(game_id AS STRING) AS game_id, MAX(prediction_ts) AS latest_ts
          FROM `{project}.sports_edge_curated.model_predictions`
          WHERE league = 'NFL'
            AND model_version = @model_version
          GROUP BY game_id
        )
        SELECT
          g.game_id,
          g.game_date,
          p.latest_ts
        FROM games g
        LEFT JOIN preds p USING (game_id)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    stale_before = now - timedelta(hours=NFL_PREDICTION_STALE_HOURS)
    missing: list[str] = []
    stale: list[str] = []
    game_dates: list[date] = []
    for row in rows:
        game_dates.append(row["game_date"])
        latest = row["latest_ts"]
        if latest is None:
            missing.append(str(row["game_id"]))
            continue
        if getattr(latest, "tzinfo", None) is None:
            latest = latest.replace(tzinfo=timezone.utc)
        if latest < stale_before:
            stale.append(str(row["game_id"]))
    return {
        "upcoming_games": len(rows),
        "missing_game_ids": missing,
        "stale_game_ids": stale,
        "missing_count": len(missing),
        "stale_count": len(stale),
        "min_game_date": min(game_dates).isoformat() if game_dates else None,
        "max_game_date": max(game_dates).isoformat() if game_dates else None,
        "stale_hours": NFL_PREDICTION_STALE_HOURS,
        "model_version": model_version,
    }


def nfl_forecast_window(cycle_start: date, cycle_end: date, anchor: date) -> tuple[date, date]:
    """Cover the Tuesday-Monday cycle plus any game in the next 7 days."""
    lookahead_end = anchor + timedelta(days=NFL_PREDICTION_LOOKAHEAD_DAYS)
    start = min(cycle_start, anchor)
    end = max(cycle_end, lookahead_end)
    if (end - start).days > NFL_MAX_FORECAST_WINDOW_DAYS:
        end = start + timedelta(days=NFL_MAX_FORECAST_WINDOW_DAYS)
    return start, end


def should_run_nfl_predictions(
    *,
    run_nfl: bool,
    anchor: date,
    force_nfl_refresh: bool,
    force_full_rebuild: bool,
    freshness: Optional[dict],
) -> tuple[bool, str]:
    if not run_nfl:
        return False, "nfl inactive"
    if force_full_rebuild:
        return True, "force_full_rebuild"
    if force_nfl_refresh:
        return True, "force_nfl_refresh"
    if anchor.weekday() == 1:
        return True, "tuesday freeze"
    if freshness is None:
        return False, "prediction freshness unknown; tuesday-only"
    if freshness.get("missing_count"):
        return True, f"{freshness['missing_count']} upcoming games have no {freshness.get('model_version') or 'live'} prediction"
    if freshness.get("stale_count"):
        return True, (
            f"{freshness['stale_count']} upcoming games have predictions older than "
            f"{freshness.get('stale_hours', NFL_PREDICTION_STALE_HOURS)}h"
        )
    if freshness.get("upcoming_games"):
        return False, "upcoming games have fresh predictions"
    return False, "no upcoming unplayed games in lookahead"


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
    force_nfl_refresh: bool = False,
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
    nfl_max_season = nfl_provider_max_season()
    nfl_season = season_year_for("NFL", anchor)
    nfl_schedule_ready = nfl_schedule_available(nfl_season)
    plan["nfl_provider_max_season"] = nfl_max_season
    plan["nfl_schedule_available"] = nfl_schedule_ready

    for league in LEAGUES:
        bq_games = None
        bq_error = None
        if project and league not in {"CFB", "WORLD_CUP"}:
            try:
                bq_games = count_bigquery_games(project, league, start_date, end_date)
            except Exception as exc:  # noqa: BLE001
                bq_error = str(exc)

        in_calendar = calendar_active(league, start_date, end_date, anchor)
        season_year = season_year_for(league, anchor)
        provider_ready = (
            league != "NFL"
            or nfl_schedule_ready
            or bool(bq_games)
        )
        run_league = (force_full_rebuild or in_calendar or bool(bq_games)) and provider_ready

        reasons = []
        if force_full_rebuild:
            reasons.append("force_full_rebuild")
        if bq_games:
            reasons.append(f"{bq_games} scheduled games in window")
        if in_calendar:
            reasons.append("calendar season window")
        if bq_error:
            reasons.append(f"schedule query unavailable: {bq_error}")
        if not (force_full_rebuild or in_calendar or bool(bq_games)):
            reasons.append("offseason and no scheduled games in window")
        if league == "NFL":
            if nfl_schedule_ready:
                reasons.append(f"nflverse schedule available for season {season_year}")
            elif bq_games:
                reasons.append("schedule already present in BigQuery")
            else:
                reasons.append(f"schedule unavailable for season {season_year}")
            if season_year > nfl_max_season:
                reasons.append(
                    f"preseason mode: historical features only through {nfl_max_season}"
                )
        key = league.lower()
        plan[f"run_{key}"] = run_league
        plan[f"{key}_season"] = season_year
        plan[f"{key}_scheduled_games"] = bq_games
        plan[f"{key}_reason"] = "; ".join(reasons)
        if league == "WORLD_CUP":
            wc_start, wc_end = world_cup_tournament_window_for(int(plan[f"{key}_season"]))
            plan["world_cup_start_date"] = wc_start.isoformat()
            plan["world_cup_end_date"] = wc_end.isoformat()

    nfl_start, nfl_end = nfl_week_window(anchor)
    plan["nfl_start_date"] = nfl_start.isoformat()
    plan["nfl_end_date"] = nfl_end.isoformat()
    forecast_start, forecast_end = nfl_forecast_window(nfl_start, nfl_end, anchor)
    plan["nfl_forecast_start_date"] = forecast_start.isoformat()
    plan["nfl_forecast_end_date"] = forecast_end.isoformat()
    plan["nfl_forecast_window_days"] = (forecast_end - forecast_start).days
    freshness = None
    freshness_error = None
    if project and plan["run_nfl"]:
        try:
            freshness = fetch_nfl_prediction_freshness(
                project,
                anchor,
                anchor + timedelta(days=NFL_PREDICTION_LOOKAHEAD_DAYS),
            )
        except Exception as exc:  # noqa: BLE001
            freshness_error = str(exc)
    run_predictions, prediction_reason = should_run_nfl_predictions(
        run_nfl=bool(plan["run_nfl"]),
        anchor=anchor,
        force_nfl_refresh=force_nfl_refresh,
        force_full_rebuild=force_full_rebuild,
        freshness=freshness,
    )
    plan["run_nfl_predictions"] = run_predictions
    plan["nfl_prediction_reason"] = prediction_reason
    if freshness is not None:
        plan["nfl_upcoming_games"] = freshness["upcoming_games"]
        plan["nfl_missing_predictions"] = freshness["missing_count"]
        plan["nfl_stale_predictions"] = freshness["stale_count"]
    elif freshness_error:
        plan["nfl_prediction_reason"] = f"{prediction_reason} ({freshness_error})"
    # Month-start catches the Thursday Oct 1 reset even if Tuesday's key was empty.
    # Paid Odds API stays on this sparse cadence; ESPN DK is the free extra snapshot.
    plan["run_nfl_odds"] = bool(plan["run_nfl"] and (anchor.weekday() in {1, 6} or anchor.day == 1 or force_nfl_refresh or force_full_rebuild))
    plan["run_espn_dk_odds"] = bool(plan["run_nfl"])
    plan["run_cfb_odds"] = bool(plan["run_cfb"] and (anchor.weekday() in {3, 5} or force_full_rebuild))
    plan["run_any"] = any(bool(plan[f"run_{league.lower()}"]) for league in LEAGUES)
    plan["run_market_odds"] = bool(plan["run_nba"] or plan["run_nfl_odds"])
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
    parser.add_argument("--force-nfl-refresh", default=os.getenv("FORCE_NFL_REFRESH", "false"))
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
        force_nfl_refresh=parse_bool(args.force_nfl_refresh),
    )
    print(json.dumps(plan, indent=2, sort_keys=True))

    output_path = os.getenv("GITHUB_OUTPUT")
    if args.github_output and output_path:
        write_github_outputs(plan, output_path)


if __name__ == "__main__":
    main()
