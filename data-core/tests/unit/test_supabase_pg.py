from datetime import datetime, timezone

import pandas as pd

from src.utils.supabase_pg import upsert_games_pg


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.pending_fetch = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        if "information_schema.columns" in sql:
            self.pending_fetch = "columns"
            return
        if "SELECT id FROM games" in sql:
            self.conn.lookup_params = params
            self.pending_fetch = "game"
            return
        if "UPDATE games SET" in sql:
            self.conn.updates.append(params)
            self.pending_fetch = None
            return
        if "INSERT INTO games" in sql:
            self.conn.inserts.append(params)
            self.pending_fetch = "insert"
            return
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchall(self):
        if self.pending_fetch == "columns":
            return [("id",), ("league",), ("game_time_utc",)]
        raise AssertionError("Unexpected fetchall")

    def fetchone(self):
        if self.pending_fetch == "game":
            return ("existing-game-id",)
        if self.pending_fetch == "insert":
            return ("inserted-game-id",)
        raise AssertionError("Unexpected fetchone")


class FakeConnection:
    def __init__(self):
        self.lookup_params = None
        self.updates = []
        self.inserts = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def test_upsert_games_uses_date_only_for_existing_game_lookup():
    conn = FakeConnection()
    game_time = datetime(2026, 6, 1, 23, 40, tzinfo=timezone.utc)
    games = pd.DataFrame(
        [
            {
                "league": "MLB",
                "season": 2026,
                "week": None,
                "home_team": "LAD",
                "away_team": "COL",
                "game_date": pd.Timestamp("2026-06-01"),
                "game_time_utc": game_time,
                "book_spread": None,
            }
        ]
    )

    game_id_map = upsert_games_pg(conn, games)

    assert conn.lookup_params == ("MLB", "LAD", "COL", pd.Timestamp("2026-06-01").date())
    assert conn.inserts == []
    assert conn.updates == [(2026, None, None, "existing-game-id")]
    assert game_id_map == {"2026-06-01_COL_LAD": "existing-game-id"}
    assert conn.commits == 1
