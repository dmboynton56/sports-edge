#!/usr/bin/env python3
"""Refresh current NFL availability and conservative absence impacts.

The nflverse season roster is the authority for team membership. Sleeper adds
same-day injury and practice context, but an impact is generated only when the
official roster marks an offensive skill player unavailable. This prevents a
questionable designation from silently changing a published game prediction.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sync_injury_reports_to_supabase import (  # noqa: E402
    build_payloads,
    resolve_game_ids,
    sync_payloads,
)
from src.fantasy.sleeper import (  # noqa: E402
    NFLVERSE_INACTIVE_STATUSES,
    NFLVERSE_OUT_STATUSES,
    load_nflverse_rosters,
    load_sleeper_players,
    sleeper_availability,
)
from src.utils.supabase_pg import (  # noqa: E402
    create_pg_connection,
    load_supabase_credentials,
    upsert_games_pg,
)
from src.utils.team_codes import canonical_nfl_abbr  # noqa: E402


MODEL_VERSION = "nfl-roster-epa-v1"
SOURCE = "nflverse_roster+sleeper"
IMPACT_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}
MIN_PLAYER_PLAYS = {"QB": 100, "RB": 40, "FB": 30, "WR": 40, "TE": 40}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh current NFL availability context.")
    parser.add_argument("--project", required=True, help="GCP project containing sports_edge_raw.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--lookahead-days", type=int, default=14)
    parser.add_argument("--history-seasons", type=int, nargs="+", default=None)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _player_name(player: Mapping[str, Any]) -> str:
    full_name = str(player.get("full_name") or "").strip()
    if full_name:
        return full_name
    return " ".join(
        str(player.get(field) or "").strip() for field in ("first_name", "last_name")
    ).strip()


def _sleeper_by_gsis(players: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(player.get("gsis_id")).strip(): player
        for player in players.values()
        if str(player.get("gsis_id") or "").strip()
    }


def _official_availability(official_status: str, sleeper: Mapping[str, Any] | None) -> str:
    if official_status in {"RES", "RSR"}:
        return "injured_reserve"
    if official_status in {"PUP", "SUS", "EXE"}:
        return "out"
    return sleeper_availability(sleeper or {})


def build_context_rows(
    games: Sequence[Mapping[str, Any]],
    official_rosters: Mapping[str, Mapping[str, Any]],
    sleeper_players: Mapping[str, Mapping[str, Any]],
    historical_impacts: Mapping[str, Mapping[str, Any]],
    *,
    season: int,
    report_ts: datetime,
    model_version: str = MODEL_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build normalized report rows for upcoming games.

    Reports are emitted only for players with a non-expected status. Impact
    fields are attached only to officially unavailable skill players with a
    sufficiently large historical play sample.
    """

    sleeper_by_gsis = _sleeper_by_gsis(sleeper_players)
    games_by_team: dict[str, list[dict[str, Any]]] = {}
    for game in games:
        home = canonical_nfl_abbr(game.get("home_team"))
        away = canonical_nfl_abbr(game.get("away_team"))
        if not home or not away:
            continue
        normalized_game = {
            **dict(game),
            "home_team": str(game.get("home_team") or "").strip().upper(),
            "away_team": str(game.get("away_team") or "").strip().upper(),
        }
        games_by_team.setdefault(home, []).append(normalized_game)
        games_by_team.setdefault(away, []).append(normalized_game)

    rows: list[dict[str, Any]] = []
    summary = {
        "official_players": len(official_rosters),
        "scoped_players": 0,
        "availability_reports": 0,
        "confirmed_unavailable": 0,
        "impact_estimates": 0,
        "inactive_skipped": 0,
        "insufficient_history": 0,
    }

    for player_id, official in official_rosters.items():
        official_status = str(official.get("status") or "").strip().upper()
        if official_status in NFLVERSE_INACTIVE_STATUSES:
            summary["inactive_skipped"] += 1
            continue

        team = canonical_nfl_abbr(official.get("team"))
        if not team or team not in games_by_team:
            continue
        summary["scoped_players"] += 1

        sleeper = sleeper_by_gsis.get(str(player_id))
        status = _official_availability(official_status, sleeper)
        if status == "expected":
            continue

        confirmed_unavailable = official_status in NFLVERSE_OUT_STATUSES
        if confirmed_unavailable:
            summary["confirmed_unavailable"] += 1

        player_name = _player_name(official) or _player_name(sleeper or {}) or str(player_id)
        position = str(official.get("position") or (sleeper or {}).get("position") or "").upper()
        historical = historical_impacts.get(str(player_id), {})
        sample_size = int(historical.get("sample_size") or 0)
        has_impact = (
            confirmed_unavailable
            and position in IMPACT_POSITIONS
            and sample_size >= MIN_PLAYER_PLAYS[position]
        )
        if confirmed_unavailable and position in IMPACT_POSITIONS and not has_impact:
            summary["insufficient_history"] += 1

        for game in games_by_team[team]:
            is_home = canonical_nfl_abbr(game["home_team"]) == team
            scheduled_team = game["home_team"] if is_home else game["away_team"]
            opponent = game["away_team"] if is_home else game["home_team"]
            raw_record = {
                "official_roster_status": official_status,
                "confirmed_unavailable": confirmed_unavailable,
                "impact_eligible": has_impact,
                "history_sample_size": sample_size,
                "minimum_player_plays": MIN_PLAYER_PLAYS.get(position),
                "sleeper_status": (sleeper or {}).get("status"),
                "sleeper_injury_status": (sleeper or {}).get("injury_status"),
                "injury_body_part": (sleeper or {}).get("injury_body_part"),
                "practice_participation": (sleeper or {}).get("practice_participation"),
                "roster_season": season,
            }
            row: dict[str, Any] = {
                "league": "NFL",
                "season": season,
                "game_id": str(game.get("game_id") or "") or None,
                "game_date": game.get("game_date"),
                # Store the schedule code because injury features match the raw
                # schedule. nflverse currently uses LAR while schedules use LA.
                "team": scheduled_team,
                "opponent": opponent,
                "player_name": player_name,
                "player_id": str(player_id),
                "position": position or None,
                "status": status,
                "report_ts": report_ts,
                "source": SOURCE,
                **raw_record,
            }
            summary["availability_reports"] += 1

            if has_impact:
                player_value = float(historical["player_value"])
                usage_share = float(historical["usage_share"])
                # A missing player can only reduce the current conservative
                # estimate. Negative historical EPA never becomes an upgrade.
                team_delta = min(0.0, -player_value * usage_share)
                row.update(
                    {
                        "metric_name": "epa_per_play",
                        "player_value": player_value,
                        "replacement_value": 0.0,
                        "usage_share": usage_share,
                        "team_delta": team_delta,
                        "sample_size": sample_size,
                        "model_version": model_version,
                        "estimated_at": report_ts,
                    }
                )
                summary["impact_estimates"] += 1
            rows.append(row)

    return rows, summary


def fetch_upcoming_games(conn, *, start_date: date, end_date: date) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              id::text AS game_id,
              COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date) AS game_date,
              home_team,
              away_team
            FROM games
            WHERE league = 'NFL'
              AND COALESCE(game_date, (game_time_utc AT TIME ZONE 'America/Denver')::date)
                BETWEEN %s AND %s
            ORDER BY game_time_utc, id
            """,
            (start_date, end_date),
            prepare=False,
        )
        columns = [description[0] for description in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def fetch_upcoming_schedule_bq(
    client: bigquery.Client,
    *,
    project: str,
    season: int,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch and normalize the schedule needed to seed Supabase game IDs."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", project):
        raise ValueError(f"Unsafe GCP project identifier: {project!r}")
    query = f"""
        SELECT
          'NFL' AS league,
          season,
          week,
          game_date,
          home_team,
          away_team,
          TIMESTAMP(
            DATETIME(
              game_date,
              COALESCE(
                SAFE.PARSE_TIME('%H:%M', JSON_VALUE(raw_record, '$.gametime')),
                TIME '00:00:00'
              )
            ),
            'America/New_York'
          ) AS game_time_utc,
          CAST(NULL AS FLOAT64) AS book_spread
        FROM `{project}.sports_edge_raw.raw_schedules`
        WHERE league = 'NFL'
          AND season = @season
          AND game_date BETWEEN @start_date AND @end_date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY ingested_at DESC) = 1
        ORDER BY game_date, game_time_utc, home_team
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("season", "INT64", season),
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        ]
    )
    frame = client.query(query, job_config=config).to_dataframe()
    if not frame.empty:
        frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
        frame["game_time_utc"] = pd.to_datetime(frame["game_time_utc"], utc=True, errors="coerce")
    return frame


def fetch_historical_impacts(
    client: bigquery.Client,
    *,
    project: str,
    seasons: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Estimate player EPA and play share from historical nflverse PBP.

    raw_record is stored as JSON-shaped text that can contain NaN, so player
    IDs are extracted with regex instead of JSON_VALUE.
    """

    if not re.fullmatch(r"[A-Za-z0-9_-]+", project):
        raise ValueError(f"Unsafe GCP project identifier: {project!r}")
    query = f"""
        WITH base AS (
          SELECT game_id, play_id, posteam, epa, raw_record
          FROM `{project}.sports_edge_raw.raw_pbp`
          WHERE league = 'NFL'
            AND season IN UNNEST(@seasons)
            AND game_id IS NOT NULL
            AND play_id IS NOT NULL
            AND posteam IS NOT NULL
            AND epa IS NOT NULL
        ),
        team_games AS (
          SELECT game_id, posteam, COUNT(DISTINCT CAST(play_id AS STRING)) AS team_plays
          FROM base
          GROUP BY game_id, posteam
        ),
        player_events AS (
          SELECT DISTINCT game_id, play_id, posteam, epa, player_id
          FROM base
          CROSS JOIN UNNEST([
            REGEXP_EXTRACT(raw_record, r'"passer_player_id":\\s*"([^"]+)"'),
            REGEXP_EXTRACT(raw_record, r'"rusher_player_id":\\s*"([^"]+)"'),
            REGEXP_EXTRACT(raw_record, r'"receiver_player_id":\\s*"([^"]+)"')
          ]) AS player_id
          WHERE player_id IS NOT NULL AND player_id != ''
        ),
        player_games AS (
          SELECT
            player_id,
            game_id,
            posteam,
            COUNT(*) AS player_plays,
            AVG(epa) AS player_epa
          FROM player_events
          GROUP BY player_id, game_id, posteam
        )
        SELECT
          pg.player_id,
          SUM(pg.player_plays) AS sample_size,
          SAFE_DIVIDE(SUM(pg.player_epa * pg.player_plays), SUM(pg.player_plays)) AS player_value,
          LEAST(1.0, SAFE_DIVIDE(SUM(pg.player_plays), SUM(tg.team_plays))) AS usage_share
        FROM player_games pg
        JOIN team_games tg USING (game_id, posteam)
        GROUP BY pg.player_id
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("seasons", "INT64", list(seasons))]
    )
    frame = client.query(query, job_config=config).to_dataframe()
    return {
        str(row["player_id"]): {
            "sample_size": int(row["sample_size"]),
            "player_value": float(row["player_value"]),
            "usage_share": float(row["usage_share"]),
        }
        for _, row in frame.iterrows()
        if row.get("player_id")
    }


def clear_generated_impacts(conn, game_ids: Sequence[str], *, model_version: str) -> int:
    """Remove stale generated impacts for exactly the games being refreshed."""

    if not game_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM player_impact_estimates
            WHERE league = 'NFL'
              AND model_version = %s
              AND game_id = ANY(%s::uuid[])
            """,
            (model_version, list(game_ids)),
            prepare=False,
        )
        return int(cur.rowcount or 0)


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    if args.lookahead_days < 0:
        raise SystemExit("--lookahead-days must be non-negative")

    history_seasons = args.history_seasons or [args.season - 2, args.season - 1]
    end_date = args.start_date + timedelta(days=args.lookahead_days)
    report_ts = datetime.now(timezone.utc)

    creds = load_supabase_credentials()
    bq_client = bigquery.Client(project=args.project)
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    try:
        schedule = fetch_upcoming_schedule_bq(
            bq_client,
            project=args.project,
            season=args.season,
            start_date=args.start_date,
            end_date=end_date,
        )
        if not args.dry_run and not schedule.empty:
            upsert_games_pg(conn, schedule)
        games = fetch_upcoming_games(conn, start_date=args.start_date, end_date=end_date)
        historical_impacts = fetch_historical_impacts(
            bq_client,
            project=args.project,
            seasons=history_seasons,
        )
        rows, summary = build_context_rows(
            games,
            load_nflverse_rosters(args.season),
            load_sleeper_players(),
            historical_impacts,
            season=args.season,
            report_ts=report_ts,
            model_version=args.model_version,
        )
        availability, impacts = build_payloads(
            rows,
            default_source=SOURCE,
            default_model_version=args.model_version,
            default_report_ts=report_ts,
        )
        availability, impacts = resolve_game_ids(conn, availability, impacts)
        unresolved = sum(payload.game_id is None for payload in availability)
        summary.update(
            {
                "games": len(games),
                "schedule_games": len(schedule),
                "history_seasons": history_seasons,
                "historical_players": len(historical_impacts),
                "unresolved_reports": unresolved,
                "dry_run": args.dry_run,
            }
        )
        if unresolved:
            raise RuntimeError(f"Could not resolve {unresolved} availability reports to games")

        if not args.dry_run:
            game_ids = sorted({str(game["game_id"]) for game in games if game.get("game_id")})
            summary["stale_impacts_deleted"] = clear_generated_impacts(
                conn, game_ids, model_version=args.model_version
            )
            inserted_availability, inserted_impacts = sync_payloads(conn, availability, impacts)
            summary["inserted_availability"] = inserted_availability
            summary["inserted_impacts"] = inserted_impacts
    finally:
        conn.close()

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
