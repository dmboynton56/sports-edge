#!/usr/bin/env python3
"""Sync normalized player availability and injury-impact rows to Supabase."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


AVAILABILITY_STATUSES = {
    "available",
    "probable",
    "questionable",
    "doubtful",
    "out",
    "inactive",
    "injured_reserve",
    "day_to_day",
}


@dataclass(frozen=True)
class AvailabilityPayload:
    league: str
    game_id: str | None
    game_date: date | None
    team: str
    opponent: str | None
    player_name: str
    player_id: str | None
    position: str | None
    status: str
    report_ts: datetime
    source: str
    raw_record: dict[str, Any]


@dataclass(frozen=True)
class ImpactPayload:
    league: str
    season: int | None
    game_id: str | None
    game_date: date | None
    team: str
    player_name: str
    player_id: str | None
    position: str | None
    metric_name: str
    player_value: float | None
    replacement_value: float | None
    usage_share: float | None
    team_delta: float
    sample_size: int | None
    model_version: str
    estimated_at: datetime
    raw_record: dict[str, Any]


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _as_text(value: Any) -> str | None:
    value = _clean(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_date(value: Any) -> date | None:
    value = _clean(value)
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def _as_datetime(value: Any, default: datetime | None = None) -> datetime:
    value = _clean(value)
    if value is None:
        if default is None:
            raise ValueError("datetime value is required")
        return default
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        if default is None:
            raise ValueError(f"Invalid datetime value: {value}")
        return default
    return ts.to_pydatetime()


def _as_float(value: Any) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    value = _clean(value)
    if value is None:
        return None
    return int(value)


def _raw_record(row: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in row.items():
        value = _clean(value)
        if isinstance(value, (datetime, date)):
            safe[key] = value.isoformat()
        elif hasattr(value, "item"):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict("records")
    if path.suffix.lower() in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict) and isinstance(loaded.get("rows"), list):
            return loaded["rows"]
    raise ValueError(f"Unsupported injury input format: {path}")


def build_payloads(
    rows: list[dict[str, Any]],
    *,
    default_source: str,
    default_model_version: str,
    default_report_ts: datetime | None = None,
) -> tuple[list[AvailabilityPayload], list[ImpactPayload]]:
    availability: list[AvailabilityPayload] = []
    impacts: list[ImpactPayload] = []
    now = default_report_ts or datetime.now(timezone.utc)

    for row in rows:
        league = (_as_text(row.get("league")) or "").upper()
        team = _as_text(row.get("team"))
        player_name = _as_text(row.get("player_name"))
        if not league or not team or not player_name:
            continue

        game_date = _as_date(row.get("game_date"))
        game_id = _as_text(row.get("game_id"))
        source = _as_text(row.get("source")) or default_source
        raw = _raw_record(row)

        status = _as_text(row.get("status"))
        if status:
            normalized_status = status.lower().replace(" ", "_").replace("-", "_")
            if normalized_status not in AVAILABILITY_STATUSES:
                normalized_status = status.lower()
            availability.append(
                AvailabilityPayload(
                    league=league,
                    game_id=game_id,
                    game_date=game_date,
                    team=team,
                    opponent=_as_text(row.get("opponent")),
                    player_name=player_name,
                    player_id=_as_text(row.get("player_id")),
                    position=_as_text(row.get("position")),
                    status=normalized_status,
                    report_ts=_as_datetime(row.get("report_ts"), default=now),
                    source=source,
                    raw_record=raw,
                )
            )

        metric_name = _as_text(row.get("metric_name"))
        if metric_name:
            player_value = _as_float(row.get("player_value"))
            replacement_value = _as_float(row.get("replacement_value"))
            usage_share = _as_float(row.get("usage_share"))
            team_delta = _as_float(row.get("team_delta"))
            if team_delta is None and player_value is not None and replacement_value is not None:
                metric_delta = replacement_value - player_value
                team_delta = metric_delta * usage_share if usage_share is not None else metric_delta
            if team_delta is None:
                continue

            impacts.append(
                ImpactPayload(
                    league=league,
                    season=_as_int(row.get("season")),
                    game_id=game_id,
                    game_date=game_date,
                    team=team,
                    player_name=player_name,
                    player_id=_as_text(row.get("player_id")),
                    position=_as_text(row.get("position")),
                    metric_name=metric_name,
                    player_value=player_value,
                    replacement_value=replacement_value,
                    usage_share=usage_share,
                    team_delta=team_delta,
                    sample_size=_as_int(row.get("sample_size")),
                    model_version=_as_text(row.get("model_version")) or default_model_version,
                    estimated_at=_as_datetime(row.get("estimated_at"), default=now),
                    raw_record=raw,
                )
            )

    return availability, impacts


def resolve_game_ids(conn, availability: list[AvailabilityPayload], impacts: list[ImpactPayload]) -> tuple[list[AvailabilityPayload], list[ImpactPayload]]:
    cache: dict[tuple[str, date | None, str, str | None], str | None] = {}

    def resolve(league: str, game_date: date | None, team: str, opponent: str | None) -> str | None:
        if game_date is None:
            return None
        key = (league, game_date, team, opponent)
        if key in cache:
            return cache[key]
        with conn.cursor() as cur:
            if opponent:
                cur.execute(
                    """
                    SELECT id
                    FROM games
                    WHERE league = %s
                      AND (
                        game_date = %s
                        OR (
                          game_date IS NULL
                          AND (game_time_utc AT TIME ZONE 'America/Denver')::date = %s
                        )
                      )
                      AND (
                        (home_team = %s AND away_team = %s)
                        OR (home_team = %s AND away_team = %s)
                      )
                    ORDER BY game_time_utc DESC, created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (league, game_date, game_date, team, opponent, opponent, team),
                    prepare=False,
                )
            else:
                cur.execute(
                    """
                    SELECT id
                    FROM games
                    WHERE league = %s
                      AND (
                        game_date = %s
                        OR (
                          game_date IS NULL
                          AND (game_time_utc AT TIME ZONE 'America/Denver')::date = %s
                        )
                      )
                      AND (home_team = %s OR away_team = %s)
                    ORDER BY game_time_utc DESC, created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (league, game_date, game_date, team, team),
                    prepare=False,
                )
            row = cur.fetchone()
        cache[key] = str(row[0]) if row else None
        return cache[key]

    resolved_availability = [
        AvailabilityPayload(
            **{
                **payload.__dict__,
                "game_id": payload.game_id or resolve(payload.league, payload.game_date, payload.team, payload.opponent),
            }
        )
        for payload in availability
    ]
    resolved_impacts = [
        ImpactPayload(
            **{
                **payload.__dict__,
                "game_id": payload.game_id or resolve(payload.league, payload.game_date, payload.team, None),
            }
        )
        for payload in impacts
    ]
    return resolved_availability, resolved_impacts


def sync_payloads(conn, availability: list[AvailabilityPayload], impacts: list[ImpactPayload]) -> tuple[int, int]:
    inserted_availability = 0
    inserted_impacts = 0
    with conn.cursor() as cur:
        for payload in availability:
            cur.execute(
                """
                DELETE FROM player_availability_reports
                WHERE league = %s
                  AND coalesce(game_id, '00000000-0000-0000-0000-000000000000'::uuid)
                    = coalesce(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                  AND coalesce(game_date, '1900-01-01'::date)
                    = coalesce(%s::date, '1900-01-01'::date)
                  AND team = %s
                  AND player_name = %s
                  AND report_ts = %s
                  AND source = %s
                """,
                (
                    payload.league,
                    payload.game_id,
                    payload.game_date,
                    payload.team,
                    payload.player_name,
                    payload.report_ts,
                    payload.source,
                ),
                prepare=False,
            )
            cur.execute(
                """
                INSERT INTO player_availability_reports (
                    league,
                    game_id,
                    game_date,
                    team,
                    opponent,
                    player_name,
                    player_id,
                    position,
                    status,
                    report_ts,
                    source,
                    raw_record
                )
                VALUES (%s, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    payload.league,
                    payload.game_id,
                    payload.game_date,
                    payload.team,
                    payload.opponent,
                    payload.player_name,
                    payload.player_id,
                    payload.position,
                    payload.status,
                    payload.report_ts,
                    payload.source,
                    json.dumps(payload.raw_record, default=_json_default),
                ),
                prepare=False,
            )
            inserted_availability += 1

        for payload in impacts:
            cur.execute(
                """
                DELETE FROM player_impact_estimates
                WHERE league = %s
                  AND coalesce(season, 0) = coalesce(%s, 0)
                  AND coalesce(game_id, '00000000-0000-0000-0000-000000000000'::uuid)
                    = coalesce(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                  AND coalesce(game_date, '1900-01-01'::date)
                    = coalesce(%s::date, '1900-01-01'::date)
                  AND team = %s
                  AND player_name = %s
                  AND metric_name = %s
                  AND model_version = %s
                """,
                (
                    payload.league,
                    payload.season,
                    payload.game_id,
                    payload.game_date,
                    payload.team,
                    payload.player_name,
                    payload.metric_name,
                    payload.model_version,
                ),
                prepare=False,
            )
            cur.execute(
                """
                INSERT INTO player_impact_estimates (
                    league,
                    season,
                    game_id,
                    game_date,
                    team,
                    player_name,
                    player_id,
                    position,
                    metric_name,
                    player_value,
                    replacement_value,
                    usage_share,
                    team_delta,
                    sample_size,
                    model_version,
                    estimated_at,
                    raw_record
                )
                VALUES (%s, %s, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    payload.league,
                    payload.season,
                    payload.game_id,
                    payload.game_date,
                    payload.team,
                    payload.player_name,
                    payload.player_id,
                    payload.position,
                    payload.metric_name,
                    payload.player_value,
                    payload.replacement_value,
                    payload.usage_share,
                    payload.team_delta,
                    payload.sample_size,
                    payload.model_version,
                    payload.estimated_at,
                    json.dumps(payload.raw_record, default=_json_default),
                ),
                prepare=False,
            )
            inserted_impacts += 1
    conn.commit()
    return inserted_availability, inserted_impacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync normalized injury reports to Supabase.")
    parser.add_argument("input_file", type=Path, help="CSV, JSON, or JSONL file of normalized injury rows.")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--source", default="manual_normalized")
    parser.add_argument("--model-version", default="injury-impact-v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-resolve-games",
        action="store_true",
        help="Do not look up missing game_id values from Supabase games.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_file)
    availability, impacts = build_payloads(
        rows,
        default_source=args.source,
        default_model_version=args.model_version,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "availability": [payload.__dict__ for payload in availability],
                    "impacts": [payload.__dict__ for payload in impacts],
                },
                indent=2,
                default=str,
            )
        )
        return

    load_dotenv(args.env_file)
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
        if not args.no_resolve_games:
            availability, impacts = resolve_game_ids(conn, availability, impacts)
        inserted_availability, inserted_impacts = sync_payloads(conn, availability, impacts)
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "inserted_availability": inserted_availability,
                "inserted_impacts": inserted_impacts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
