"""Load player impact estimates from Supabase for injury-aware predictions."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

import pandas as pd

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


def _all_uuid_game_ids(game_ids: Optional[list[str]]) -> list[str] | None:
    """Return normalized UUIDs only when every supplied id is a UUID.

    BigQuery schedule keys (for example ``2026_01_NE_SEA``) are intentionally
    different from the UUID primary keys in Supabase. In that case callers must
    fall back to the existing date/team matching path instead of casting the
    warehouse key to ``uuid`` in Postgres.
    """

    if not game_ids:
        return None
    normalized: list[str] = []
    for game_id in game_ids:
        try:
            normalized.append(str(UUID(str(game_id))))
        except (TypeError, ValueError, AttributeError):
            return None
    return normalized


def _missing_supabase_pg_config(creds: dict[str, Any]) -> str | None:
    missing: list[str] = []
    if not creds.get("url") and not creds.get("db_host"):
        missing.append("SUPABASE_URL or SUPABASE_DB_HOST")
    if not creds.get("db_password"):
        missing.append("supabaseDBpass or SUPABASE_DB_PASSWORD")
    if missing:
        return ", ".join(missing)
    return None


def load_injury_impacts_from_supabase(
    league: str,
    *,
    game_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return player_impact_estimates rows for the given league.

    Missing Supabase credentials skip the injury join so predictions still
    publish without injury adjustments instead of crashing on a None URL.
    """
    creds = load_supabase_credentials()
    missing = _missing_supabase_pg_config(creds)
    if missing:
        print(
            "WARNING: Skipping injury impacts; missing "
            f"{missing}. Predictions will continue without injuries."
        )
        return pd.DataFrame()

    try:
        conn = create_pg_connection(
            supabase_url=creds["url"],
            password=creds["db_password"],
            host_override=creds.get("db_host"),
            port=creds["db_port"],
            database=creds["db_name"],
            user=creds["db_user"],
        )
    except ValueError as exc:
        print(
            "WARNING: Skipping injury impacts; "
            f"{exc}. Predictions will continue without injuries."
        )
        return pd.DataFrame()
    try:
        with conn.cursor() as cur:
            uuid_game_ids = _all_uuid_game_ids(game_ids)
            if uuid_game_ids:
                cur.execute(
                    """
                    SELECT
                      league,
                      game_id,
                      game_date,
                      team,
                      player_name,
                      metric_name,
                      player_value,
                      replacement_value,
                      usage_share,
                      team_delta,
                      sample_size
                    FROM player_impact_estimates
                    WHERE league = %s AND game_id = ANY(%s)
                    """,
                    (league.upper(), uuid_game_ids),
                )
            else:
                cur.execute(
                    """
                    SELECT
                      league,
                      game_id,
                      game_date,
                      team,
                      player_name,
                      metric_name,
                      player_value,
                      replacement_value,
                      usage_share,
                      team_delta,
                      sample_size
                    FROM player_impact_estimates
                    WHERE league = %s
                      AND game_date >= (now() AT TIME ZONE 'America/Denver')::date - 1
                    """,
                    (league.upper(),),
                )
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows, columns=columns)
    frame["available"] = True
    return frame


def injury_delta_columns(league: str) -> tuple[str, str]:
    if league.upper() == "NFL":
        return "home_injury_epa_delta", "away_injury_epa_delta"
    return "home_injury_net_rating_delta", "away_injury_net_rating_delta"


def extract_injury_metadata(features_df: pd.DataFrame, league: str) -> dict[str, Any]:
    """Pull injury adjustment metadata from a single-row features frame."""
    home_col, away_col = injury_delta_columns(league)
    home_delta = float(features_df[home_col].iloc[0]) if home_col in features_df.columns else 0.0
    away_delta = float(features_df[away_col].iloc[0]) if away_col in features_df.columns else 0.0
    injury_adjusted = abs(home_delta) > 0 or abs(away_delta) > 0
    return {
        "injury_adjusted": injury_adjusted,
        "home_injury_delta": home_delta,
        "away_injury_delta": away_delta,
    }
