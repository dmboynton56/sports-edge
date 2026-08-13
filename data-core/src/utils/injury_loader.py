"""Load player impact estimates from Supabase for injury-aware predictions."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


def load_injury_impacts_from_supabase(
    league: str,
    *,
    game_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return player_impact_estimates rows for the given league."""
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
        with conn.cursor() as cur:
            if game_ids:
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
                    (league.upper(), game_ids),
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
