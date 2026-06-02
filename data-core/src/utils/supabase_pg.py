import os
from datetime import date, datetime
import psycopg
from typing import Dict, Optional
import pandas as pd

def create_pg_connection(supabase_url: str, password: str, host_override: Optional[str] = None, 
                         port: str = "5432", database: str = "postgres", user: str = "postgres"):
    """Create a PostgreSQL connection to Supabase."""
    host = host_override or supabase_url.split("//")[1].split(".")[0] + ".supabase.co"
    conn_str = f"host={host} port={port} dbname={database} user={user} password={password} sslmode=require"
    return psycopg.connect(conn_str, prepare_threshold=None)

def load_supabase_credentials() -> Dict[str, str]:
    """Load Supabase credentials from environment variables."""
    return {
        "url": os.getenv("SUPABASE_URL"),
        "service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        "db_password": os.getenv("supabaseDBpass") or os.getenv("SUPABASE_DB_PASSWORD"),
        "db_host": os.getenv("SUPABASE_DB_HOST"),
        "db_port": os.getenv("SUPABASE_DB_PORT", "5432"),
        "db_name": os.getenv("SUPABASE_DB_NAME", "postgres"),
        "db_user": os.getenv("SUPABASE_DB_USER", "postgres"),
    }

def game_map_key(home_team: str, away_team: str, game_date) -> str:
    """Generate a consistent key for matching games."""
    if hasattr(game_date, "strftime"):
        date_str = game_date.strftime("%Y-%m-%d")
    else:
        date_str = str(game_date).split(" ")[0]
    return f"{date_str}_{away_team}_{home_team}"

def _date_only(val):
    if pd.isna(val):
        return None
    if hasattr(val, "to_pydatetime"):
        val = val.to_pydatetime()
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    parsed = pd.to_datetime(val, errors="coerce")
    if not pd.isna(parsed):
        return parsed.date()
    return str(val).split(" ")[0]

def fetch_injury_impacts_pg(
    conn,
    *,
    league: str,
    start_date,
    end_date,
    model_version: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch model-ready player injury impacts for a league/date window."""
    start = _date_only(start_date)
    end = _date_only(end_date)
    if start is None or end is None:
        raise ValueError("start_date and end_date are required to fetch injury impacts.")

    sql = """
        SELECT
            pie.league,
            pie.season,
            pie.game_id::text AS game_id,
            COALESCE(pie.game_date, g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date) AS game_date,
            pie.team,
            pie.player_name,
            pie.player_id,
            pie.position,
            pie.metric_name,
            pie.player_value,
            pie.replacement_value,
            pie.usage_share,
            pie.team_delta,
            pie.sample_size,
            pie.model_version,
            pie.estimated_at,
            pie.raw_record
        FROM player_impact_estimates pie
        LEFT JOIN games g ON g.id = pie.game_id
        WHERE pie.league = %s
          AND COALESCE(pie.game_date, g.game_date, (g.game_time_utc AT TIME ZONE 'America/Denver')::date)
              BETWEEN %s AND %s
          AND (%s IS NULL OR pie.model_version = %s)
        ORDER BY game_date, pie.team, pie.player_name, pie.estimated_at DESC
    """
    params = (league.upper(), start, end, model_version, model_version)
    with conn.cursor() as cur:
        cur.execute(sql, params, prepare=False)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    impacts = pd.DataFrame(rows, columns=columns)
    if impacts.empty:
        return impacts
    impacts["league"] = impacts["league"].astype(str).str.upper()
    impacts["game_date"] = pd.to_datetime(impacts["game_date"], errors="coerce")
    for col in ["player_value", "replacement_value", "usage_share", "team_delta"]:
        impacts[col] = pd.to_numeric(impacts[col], errors="coerce")
    if "sample_size" in impacts.columns:
        impacts["sample_size"] = pd.to_numeric(impacts["sample_size"], errors="coerce").astype("Int64")
    impacts["available"] = True
    return impacts

def upsert_games_pg(conn, games_df: pd.DataFrame) -> Dict[str, str]:
    """Upsert games into Supabase and return a map of keys to IDs."""
    game_id_map = {}
    
    # Helper to clean values for psycopg
    def _clean(val):
        if pd.isna(val):
            return None
        if hasattr(val, "to_pydatetime"):
            return val.to_pydatetime()
        return val

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'games'
            """
        )
        game_columns = {row[0] for row in cur.fetchall()}
        has_game_date_column = "game_date" in game_columns
        has_pitcher_columns = {
            "home_probable_pitcher",
            "away_probable_pitcher",
        }.issubset(game_columns)

        for _, row in games_df.iterrows():
            # Check if game exists
            lookup_date = _date_only(row.get("game_date", row["game_time_utc"]))
            if has_game_date_column:
                cur.execute(
                    """
                    SELECT id
                    FROM games
                    WHERE league = %s
                      AND home_team = %s
                      AND away_team = %s
                      AND (
                        game_date = %s
                        OR (
                          game_date IS NULL
                          AND (game_time_utc AT TIME ZONE 'America/Denver')::date = %s
                        )
                      )
                    ORDER BY game_time_utc DESC, created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (
                        _clean(row["league"]),
                        _clean(row["home_team"]),
                        _clean(row["away_team"]),
                        lookup_date,
                        lookup_date,
                    )
                )
            else:
                cur.execute(
                    """
                    SELECT id
                    FROM games
                    WHERE league = %s
                      AND home_team = %s
                      AND away_team = %s
                      AND (game_time_utc AT TIME ZONE 'America/Denver')::date = %s
                    ORDER BY game_time_utc DESC, created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (_clean(row["league"]), _clean(row["home_team"]), _clean(row["away_team"]), lookup_date)
                )
            res = cur.fetchone()
            
            if res:
                game_id = res[0]
                if has_game_date_column and has_pitcher_columns:
                    cur.execute(
                        """
                        UPDATE games
                        SET season = %s,
                            week = %s,
                            game_date = %s,
                            game_time_utc = %s,
                            book_spread = COALESCE(%s, book_spread),
                            home_probable_pitcher = COALESCE(%s, home_probable_pitcher),
                            away_probable_pitcher = COALESCE(%s, away_probable_pitcher)
                        WHERE id = %s
                        """,
                        (
                            _clean(row["season"]),
                            _clean(row.get("week")),
                            lookup_date,
                            _clean(row["game_time_utc"]),
                            _clean(row.get("book_spread")),
                            _clean(row.get("home_probable_pitcher")),
                            _clean(row.get("away_probable_pitcher")),
                            game_id,
                        )
                    )
                elif has_game_date_column:
                    cur.execute(
                        """
                        UPDATE games
                        SET season = %s,
                            week = %s,
                            game_date = %s,
                            game_time_utc = %s,
                            book_spread = COALESCE(%s, book_spread)
                        WHERE id = %s
                        """,
                        (
                            _clean(row["season"]),
                            _clean(row.get("week")),
                            lookup_date,
                            _clean(row["game_time_utc"]),
                            _clean(row.get("book_spread")),
                            game_id,
                        )
                    )
                elif has_pitcher_columns:
                    cur.execute(
                        """
                        UPDATE games
                        SET season = %s,
                            week = %s,
                            game_time_utc = %s,
                            book_spread = COALESCE(%s, book_spread),
                            home_probable_pitcher = COALESCE(%s, home_probable_pitcher),
                            away_probable_pitcher = COALESCE(%s, away_probable_pitcher)
                        WHERE id = %s
                        """,
                        (
                            _clean(row["season"]),
                            _clean(row.get("week")),
                            _clean(row["game_time_utc"]),
                            _clean(row.get("book_spread")),
                            _clean(row.get("home_probable_pitcher")),
                            _clean(row.get("away_probable_pitcher")),
                            game_id,
                        )
                    )
                else:
                    cur.execute(
                        "UPDATE games SET season = %s, week = %s, book_spread = COALESCE(%s, book_spread) WHERE id = %s",
                        (_clean(row["season"]), _clean(row.get("week")), _clean(row.get("book_spread")), game_id)
                    )
            else:
                if has_game_date_column and has_pitcher_columns:
                    cur.execute(
                        """
                        INSERT INTO games (
                            league,
                            season,
                            week,
                            game_date,
                            home_team,
                            away_team,
                            game_time_utc,
                            book_spread,
                            home_probable_pitcher,
                            away_probable_pitcher
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            _clean(row["league"]),
                            _clean(row["season"]),
                            _clean(row.get("week")),
                            lookup_date,
                            _clean(row["home_team"]),
                            _clean(row["away_team"]),
                            _clean(row["game_time_utc"]),
                            _clean(row.get("book_spread")),
                            _clean(row.get("home_probable_pitcher")),
                            _clean(row.get("away_probable_pitcher")),
                        )
                    )
                elif has_game_date_column:
                    cur.execute(
                        """
                        INSERT INTO games (
                            league,
                            season,
                            week,
                            game_date,
                            home_team,
                            away_team,
                            game_time_utc,
                            book_spread
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            _clean(row["league"]),
                            _clean(row["season"]),
                            _clean(row.get("week")),
                            lookup_date,
                            _clean(row["home_team"]),
                            _clean(row["away_team"]),
                            _clean(row["game_time_utc"]),
                            _clean(row.get("book_spread")),
                        )
                    )
                elif has_pitcher_columns:
                    cur.execute(
                        """
                        INSERT INTO games (
                            league,
                            season,
                            week,
                            home_team,
                            away_team,
                            game_time_utc,
                            book_spread,
                            home_probable_pitcher,
                            away_probable_pitcher
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            _clean(row["league"]),
                            _clean(row["season"]),
                            _clean(row.get("week")),
                            _clean(row["home_team"]),
                            _clean(row["away_team"]),
                            _clean(row["game_time_utc"]),
                            _clean(row.get("book_spread")),
                            _clean(row.get("home_probable_pitcher")),
                            _clean(row.get("away_probable_pitcher")),
                        )
                    )
                else:
                    cur.execute(
                        "INSERT INTO games (league, season, week, home_team, away_team, game_time_utc, book_spread) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                        (_clean(row["league"]), _clean(row["season"]), _clean(row.get("week")), _clean(row["home_team"]), _clean(row["away_team"]), _clean(row["game_time_utc"]), _clean(row.get("book_spread")))
                    )
                game_id = cur.fetchone()[0]
            
            key = game_map_key(row["home_team"], row["away_team"], lookup_date)
            game_id_map[key] = game_id
    conn.commit()
    return game_id_map
