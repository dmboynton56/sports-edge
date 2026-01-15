import os
import psycopg
from typing import Dict, Optional
import pandas as pd

def create_pg_connection(supabase_url: str, password: str, host_override: Optional[str] = None, 
                         port: str = "5432", database: str = "postgres", user: str = "postgres"):
    """Create a PostgreSQL connection to Supabase."""
    host = host_override or supabase_url.split("//")[1].split(".")[0] + ".supabase.co"
    conn_str = f"host={host} port={port} dbname={database} user={user} password={password} sslmode=require"
    return psycopg.connect(conn_str)

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

def upsert_games_pg(conn, games_df: pd.DataFrame) -> Dict[str, str]:
    """Upsert games into Supabase and return a map of keys to IDs."""
    game_id_map = {}
    with conn.cursor() as cur:
        for _, row in games_df.iterrows():
            # Check if game exists
            cur.execute(
                "SELECT id FROM games WHERE league = %s AND home_team = %s AND away_team = %s AND game_time_utc::date = %s",
                (row["league"], row["home_team"], row["away_team"], row["game_time_utc"])
            )
            res = cur.fetchone()
            
            if res:
                game_id = res[0]
                cur.execute(
                    "UPDATE games SET season = %s, week = %s WHERE id = %s",
                    (row["season"], row.get("week"), game_id)
                )
            else:
                cur.execute(
                    "INSERT INTO games (league, season, week, home_team, away_team, game_time_utc) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                    (row["league"], row["season"], row.get("week"), row["home_team"], row["away_team"], row["game_time_utc"])
                )
                game_id = cur.fetchone()[0]
            
            key = game_map_key(row["home_team"], row["away_team"], row["game_time_utc"])
            game_id_map[key] = game_id
    conn.commit()
    return game_id_map
