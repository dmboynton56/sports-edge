#!/usr/bin/env python3
"""Snapshot NFL injury/starter news and emit the games that need a re-price.

Does not score the model. Daily and the injury workflow call refresh_nfl for
that, then mark the snapshot used. Zero dirty games means zero Odds API spend.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nfl_injury_news import (  # noqa: E402
    collect_live_news,
    detect_changed_games,
    window_name_for,
)
from src.data.nfl_injury_snapshots import (  # noqa: E402
    insert_snapshots,
    load_previous_used,
    mark_latest_used,
    mark_used,
)
from src.fantasy.sleeper import load_sleeper_players  # noqa: E402
from src.models.nfl_live import LIVE_MODEL_VERSION  # noqa: E402
from src.utils.nfl_schedule import DENVER  # noqa: E402
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials, upsert_games_pg  # noqa: E402
from src.utils.team_codes import canonical_nfl_abbr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh NFL injury/starter news snapshots.")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT_ID", ""))
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--lookahead-days", type=int, default=6)
    parser.add_argument("--window-name", default=None)
    parser.add_argument("--snapshot", action="store_true", help="Fetch news and write snapshots.")
    parser.add_argument("--detect", action="store_true", help="Compare to last used snapshot.")
    parser.add_argument(
        "--mark-used",
        nargs="*",
        default=None,
        help="Provider game_ids to mark as used for the latest fetch in this window.",
    )
    parser.add_argument(
        "--mark-latest-used",
        action="store_true",
        help="Mark the newest snapshot per game in the window as used for a published forecast.",
    )
    parser.add_argument("--fetched-at", default=None)
    parser.add_argument("--github-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _connect():
    creds = load_supabase_credentials()
    return create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )


def fetch_games(conn, *, start_date: date, end_date: date, project: str, season: int) -> list[dict[str, Any]]:
    from google.cloud import bigquery

    from scripts.refresh_nfl_availability import fetch_upcoming_games, fetch_upcoming_schedule_bq

    if project:
        client = bigquery.Client(project=project)
        schedule = fetch_upcoming_schedule_bq(
            client, project=project, season=season, start_date=start_date, end_date=end_date
        )
        if not schedule.empty:
            upsert_games_pg(conn, schedule)
    games = fetch_upcoming_games(conn, start_date=start_date, end_date=end_date)
    # Pair supabase uuid with BQ/nflverse id via home/away/date.
    provider: dict[tuple, str] = {}
    if project:
        query = f"""
            SELECT CAST(game_id AS STRING) AS provider_game_id, game_date, home_team, away_team
            FROM `{project}.sports_edge_raw.raw_schedules`
            WHERE league = 'NFL' AND season = @season AND game_date BETWEEN @start_date AND @end_date
            QUALIFY ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY ingested_at DESC) = 1
        """
        job = client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("season", "INT64", season),
                    bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                    bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
                ]
            ),
        )
        for row in job.result():
            home = canonical_nfl_abbr(row["home_team"]) or str(row["home_team"]).strip().upper()
            away = canonical_nfl_abbr(row["away_team"]) or str(row["away_team"]).strip().upper()
            provider[(str(row["game_date"]), home, away)] = str(row["provider_game_id"])
    rows = []
    for game in games:
        home = canonical_nfl_abbr(game.get("home_team")) or str(game.get("home_team") or "")
        away = canonical_nfl_abbr(game.get("away_team")) or str(game.get("away_team") or "")
        game_date = game.get("game_date")
        date_key = str(game_date)[:10] if game_date is not None else None
        rows.append(
            {
                **game,
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "supabase_game_id": game.get("game_id"),
                "provider_game_id": provider.get((date_key, home, away)),
            }
        )
    return rows


def scoreboard_dates(start_date: date, end_date: date) -> list[str]:
    dates = []
    cursor = start_date
    while cursor <= end_date:
        dates.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    return dates


def write_github_output(payload: dict[str, Any]) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        return
    dirty_ids = payload.get("dirty_provider_game_ids") or []
    supabase_ids = payload.get("dirty_supabase_game_ids") or []
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"dirty={'true' if dirty_ids else 'false'}\n")
        handle.write(f"reload_odds={'true' if dirty_ids else 'false'}\n")
        handle.write(f"dirty_provider_game_ids={' '.join(dirty_ids)}\n")
        handle.write(f"dirty_supabase_game_ids={' '.join(supabase_ids)}\n")
        handle.write(f"fetched_at={payload.get('fetched_at') or ''}\n")
        handle.write(f"window_name={payload.get('window_name') or ''}\n")


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    if not args.snapshot and args.mark_used is None and not args.mark_latest_used:
        args.snapshot = True
        args.detect = True
    start_date = args.start_date
    end_date = start_date + timedelta(days=args.lookahead_days)
    now = datetime.now(timezone.utc)
    window_name = args.window_name or window_name_for(now)
    payload: dict[str, Any] = {
        "window_name": window_name,
        "dirty_provider_game_ids": [],
        "dirty_supabase_game_ids": [],
        "fetched_at": None,
        "summary": {},
    }
    conn = _connect()
    try:
        if args.mark_latest_used:
            updated = 0 if args.dry_run else mark_latest_used(
                conn,
                start_date=start_date,
                end_date=end_date,
                prediction_asof_ts=now,
                model_version=LIVE_MODEL_VERSION,
                provider_game_ids=args.mark_used or None,
            )
            payload["marked_used"] = updated
            print(json.dumps(payload, sort_keys=True, default=str))
            return
        if args.mark_used is not None:
            fetched_at = args.fetched_at
            if not fetched_at:
                raise SystemExit("--fetched-at is required with --mark-used")
            updated = 0 if args.dry_run else mark_used(
                conn,
                provider_game_ids=args.mark_used,
                fetched_at=fetched_at,
                prediction_asof_ts=now,
                model_version=LIVE_MODEL_VERSION,
            )
            payload["marked_used"] = updated
            print(json.dumps(payload, sort_keys=True, default=str))
            return

        games = fetch_games(conn, start_date=start_date, end_date=end_date, project=args.project, season=args.season)
        today = datetime.now(tz=DENVER).date().isoformat()
        remaining = [
            game for game in games
            if game.get("home_team") and game.get("away_team")
            and (not game.get("game_date") or str(game.get("game_date"))[:10] >= today)
        ]
        sleeper = None
        try:
            sleeper = load_sleeper_players()
        except Exception as exc:  # noqa: BLE001
            payload["summary"]["sleeper_preload_error"] = str(exc)

        news, summary = collect_live_news(
            remaining,
            season=args.season,
            sleeper_players=sleeper,
            now=now,
            scoreboard_dates=scoreboard_dates(start_date, min(end_date, start_date + timedelta(days=7))),
        )
        payload["summary"] = summary
        payload["fetched_at"] = now.isoformat()
        payload["games"] = len(news)
        reasons_by_key: dict[str, Any] = {}
        if args.detect:
            previous = load_previous_used(conn, start_date=start_date, end_date=end_date)
            decisions = detect_changed_games(news, previous)
            reasons_by_key = {
                str(item.supabase_game_id or item.provider_game_id): item.reasons for item in decisions
            }
            payload["dirty"] = [
                {
                    "away": item.away_team,
                    "home": item.home_team,
                    "provider_game_id": item.provider_game_id,
                    "supabase_game_id": item.supabase_game_id,
                    "reasons": list(item.reasons),
                }
                for item in decisions
            ]
            payload["dirty_provider_game_ids"] = [
                str(item.provider_game_id) for item in decisions if item.provider_game_id
            ]
            payload["dirty_supabase_game_ids"] = [
                str(item.supabase_game_id) for item in decisions if item.supabase_game_id
            ]
        if args.snapshot and not args.dry_run:
            payload["inserted"] = insert_snapshots(
                conn, news, window_name=window_name, season=args.season, change_reasons=reasons_by_key
            )
        if args.github_output:
            write_github_output(payload)
        print(json.dumps(payload, sort_keys=True, default=str))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
