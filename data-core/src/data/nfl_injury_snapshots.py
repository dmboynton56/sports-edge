"""Persist and reload NFL injury/starter snapshots used for live forecasts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Mapping, Sequence
from uuid import UUID

from src.data.nfl_injury_news import GameNews, snapshot_row
from src.utils.team_codes import canonical_nfl_abbr

UPSERT_SQL = """
INSERT INTO nfl_injury_news_snapshots (
  game_id, provider_game_id, season, week, game_date, home_team, away_team,
  home_starter_qb_id, home_starter_qb_name, away_starter_qb_id, away_starter_qb_name,
  home_expected_qb_id, away_expected_qb_id, key_injuries, status_fingerprint,
  espn_home_spread, book_spread, source, fallback_used, fetched_at, window_name,
  used_for_prediction, prediction_asof_ts, model_version, change_reasons, raw_record
) VALUES (
  %s, %s, %s, %s, %s, %s, %s,
  %s, %s, %s, %s,
  %s, %s, %s::jsonb, %s,
  %s, %s, %s, %s, %s, %s,
  %s, %s, %s, %s::text[], %s::jsonb
)
"""


def _uuid_or_none(value: object) -> str | None:
    if value in (None, ""):
        return None
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError):
        return None


def insert_snapshots(
    conn,
    news_rows: Sequence[GameNews],
    *,
    window_name: str,
    season: int | None = None,
    change_reasons: Mapping[str, Sequence[str]] | None = None,
) -> int:
    if not news_rows:
        return 0
    reasons = change_reasons or {}
    with conn.cursor() as cur:
        for news in news_rows:
            key = str(news.supabase_game_id or news.provider_game_id or "")
            row = snapshot_row(news, window_name=window_name, change_reasons=reasons.get(key, ()))
            cur.execute(
                UPSERT_SQL,
                (
                    _uuid_or_none(row["game_id"]),
                    row["provider_game_id"],
                    season,
                    None,
                    row["game_date"],
                    row["home_team"],
                    row["away_team"],
                    row["home_starter_qb_id"],
                    row["home_starter_qb_name"],
                    row["away_starter_qb_id"],
                    row["away_starter_qb_name"],
                    None,
                    None,
                    json.dumps(row["key_injuries"]),
                    row["status_fingerprint"],
                    row["espn_home_spread"],
                    None,
                    row["source"],
                    row["fallback_used"],
                    row["fetched_at"],
                    row["window_name"],
                    False,
                    None,
                    None,
                    row["change_reasons"],
                    json.dumps(row["raw_record"]),
                ),
                prepare=False,
            )
    conn.commit()
    return len(news_rows)


def load_previous_used(conn, *, start_date, end_date) -> dict[str, dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (COALESCE(game_id::text, provider_game_id, home_team || away_team || COALESCE(game_date::text, '')))
              game_id, provider_game_id, home_team, away_team, game_date,
              home_starter_qb_id, home_starter_qb_name,
              away_starter_qb_id, away_starter_qb_name,
              status_fingerprint, espn_home_spread, fetched_at
            FROM nfl_injury_news_snapshots
            WHERE used_for_prediction = true
              AND COALESCE(game_date, '1900-01-01'::date) BETWEEN %s AND %s
            ORDER BY COALESCE(game_id::text, provider_game_id, home_team || away_team || COALESCE(game_date::text, '')),
                     fetched_at DESC
            """,
            (start_date, end_date),
            prepare=False,
        )
        columns = [col[0] for col in cur.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    previous: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in (row.get("game_id"), row.get("provider_game_id")):
            if key:
                previous[str(key)] = row
    return previous


def load_posted_starters(conn, provider_game_ids: Sequence[str]) -> dict[tuple[str, str], str]:
    """Latest snapshot per provider game, mapped onto v2 (game_id, team) -> gsis."""
    if not provider_game_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (provider_game_id)
              provider_game_id, home_team, away_team,
              home_starter_qb_id, away_starter_qb_id
            FROM nfl_injury_news_snapshots
            WHERE provider_game_id = ANY(%s)
            ORDER BY provider_game_id, fetched_at DESC
            """,
            (list(provider_game_ids),),
            prepare=False,
        )
        rows = cur.fetchall()
    mapping: dict[tuple[str, str], str] = {}
    for provider_game_id, home_team, away_team, home_id, away_id in rows:
        home = canonical_nfl_abbr(home_team) or str(home_team)
        away = canonical_nfl_abbr(away_team) or str(away_team)
        if home_id:
            mapping[(str(provider_game_id), home)] = str(home_id)
            mapping[(str(provider_game_id), str(home_team))] = str(home_id)
        if away_id:
            mapping[(str(provider_game_id), away)] = str(away_id)
            mapping[(str(provider_game_id), str(away_team))] = str(away_id)
    return mapping


def mark_latest_used(
    conn,
    *,
    start_date,
    end_date,
    prediction_asof_ts: datetime | None,
    model_version: str,
    provider_game_ids: Sequence[str] | None = None,
) -> int:
    """Mark the newest snapshot in the window (optionally filtered) as used for a publish."""
    asof = prediction_asof_ts or datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE nfl_injury_news_snapshots AS snap
            SET used_for_prediction = true,
                prediction_asof_ts = %s,
                model_version = %s
            FROM (
              SELECT DISTINCT ON (COALESCE(provider_game_id, game_id::text))
                id
              FROM nfl_injury_news_snapshots
              WHERE COALESCE(game_date, '1900-01-01'::date) BETWEEN %s AND %s
                AND (%s::text[] IS NULL OR provider_game_id = ANY(%s))
              ORDER BY COALESCE(provider_game_id, game_id::text), fetched_at DESC
            ) AS latest
            WHERE snap.id = latest.id
            """,
            (asof, model_version, start_date, end_date, list(provider_game_ids) if provider_game_ids else None, list(provider_game_ids) if provider_game_ids else None),
            prepare=False,
        )
        updated = int(cur.rowcount or 0)
    conn.commit()
    return updated


def mark_used(
    conn,
    *,
    provider_game_ids: Sequence[str],
    fetched_at: str,
    prediction_asof_ts: datetime | None,
    model_version: str,
    expected_qbs: Mapping[str, tuple[str | None, str | None]] | None = None,
) -> int:
    if not provider_game_ids:
        return 0
    asof = prediction_asof_ts or datetime.now(timezone.utc)
    expected_qbs = expected_qbs or {}
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE nfl_injury_news_snapshots AS snap
            SET used_for_prediction = true,
                prediction_asof_ts = %s,
                model_version = %s
            WHERE snap.provider_game_id = ANY(%s)
              AND snap.fetched_at = %s
            """,
            (asof, model_version, list(provider_game_ids), fetched_at),
            prepare=False,
        )
        updated = int(cur.rowcount or 0)
        for provider_game_id, (home_qb, away_qb) in expected_qbs.items():
            cur.execute(
                """
                UPDATE nfl_injury_news_snapshots
                SET home_expected_qb_id = COALESCE(%s, home_expected_qb_id),
                    away_expected_qb_id = COALESCE(%s, away_expected_qb_id)
                WHERE provider_game_id = %s AND fetched_at = %s
                """,
                (home_qb, away_qb, provider_game_id, fetched_at),
                prepare=False,
            )
    conn.commit()
    return updated
