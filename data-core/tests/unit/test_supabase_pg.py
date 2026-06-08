from datetime import datetime, timezone

import pandas as pd

from src.utils.supabase_pg import fetch_injury_impacts_pg, upsert_games_pg


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.pending_fetch = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        if params is not None:
            assert sql.count("%s") == len(params)
        if "information_schema.columns" in sql:
            self.pending_fetch = "columns"
            return
        if "SELECT id" in sql and "FROM games" in sql:
            self.conn.lookup_params = params
            self.pending_fetch = "game"
            return
        if "UPDATE games" in sql:
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
            return self.conn.columns
        raise AssertionError("Unexpected fetchall")

    def fetchone(self):
        if self.pending_fetch == "game":
            if self.conn.existing_game_id is None:
                return None
            return ("existing-game-id",)
        if self.pending_fetch == "insert":
            return ("inserted-game-id",)
        raise AssertionError("Unexpected fetchone")


class FakeConnection:
    def __init__(self, columns=None, existing_game_id="existing-game-id"):
        self.columns = columns or [("id",), ("league",), ("game_time_utc",)]
        self.existing_game_id = existing_game_id
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


def test_upsert_games_persists_schedule_date_for_late_utc_mlb_game():
    conn = FakeConnection(
        columns=[
            ("id",),
            ("league",),
            ("season",),
            ("week",),
            ("game_date",),
            ("game_time_utc",),
            ("book_spread",),
        ]
    )
    game_time = datetime(2026, 6, 2, 1, 40, tzinfo=timezone.utc)
    game_date = pd.Timestamp("2026-06-01").date()
    games = pd.DataFrame(
        [
            {
                "league": "MLB",
                "season": 2026,
                "week": None,
                "home_team": "Arizona Diamondbacks",
                "away_team": "Los Angeles Dodgers",
                "game_date": game_date,
                "game_time_utc": game_time,
                "book_spread": None,
            }
        ]
    )

    game_id_map = upsert_games_pg(conn, games)

    assert conn.lookup_params == (
        "MLB",
        "Arizona Diamondbacks",
        "Los Angeles Dodgers",
        game_date,
        game_date,
    )
    assert conn.inserts == []
    assert conn.updates == [(2026, None, game_date, game_time, None, "existing-game-id")]
    assert game_id_map == {"2026-06-01_Los Angeles Dodgers_Arizona Diamondbacks": "existing-game-id"}


def test_upsert_games_inserts_game_date_and_probable_pitchers_when_game_is_new():
    conn = FakeConnection(
        columns=[
            ("id",),
            ("league",),
            ("season",),
            ("week",),
            ("game_date",),
            ("home_team",),
            ("away_team",),
            ("game_time_utc",),
            ("book_spread",),
            ("home_probable_pitcher",),
            ("away_probable_pitcher",),
            ("created_at",),
        ],
        existing_game_id=None,
    )
    game_time = datetime(2026, 6, 8, 0, 5, tzinfo=timezone.utc)
    game_date = pd.Timestamp("2026-06-07").date()
    games = pd.DataFrame(
        [
            {
                "league": "MLB",
                "season": 2026,
                "week": None,
                "home_team": "Chicago Cubs",
                "away_team": "Athletics",
                "game_date": game_date,
                "game_time_utc": game_time,
                "book_spread": None,
                "home_probable_pitcher": "Home Starter",
                "away_probable_pitcher": "Away Starter",
            }
        ]
    )

    game_id_map = upsert_games_pg(conn, games)

    assert conn.inserts == [
        (
            "MLB",
            2026,
            None,
            game_date,
            "Chicago Cubs",
            "Athletics",
            game_time,
            None,
            "Home Starter",
            "Away Starter",
        )
    ]
    assert conn.updates == []
    assert game_id_map == {"2026-06-07_Athletics_Chicago Cubs": "inserted-game-id"}
    assert conn.commits == 1


class FakeImpactCursor:
    description = [
        ("league",),
        ("season",),
        ("game_id",),
        ("game_date",),
        ("team",),
        ("player_name",),
        ("player_id",),
        ("position",),
        ("metric_name",),
        ("player_value",),
        ("replacement_value",),
        ("usage_share",),
        ("team_delta",),
        ("sample_size",),
        ("model_version",),
        ("estimated_at",),
        ("raw_record",),
    ]

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None, prepare=False):
        self.conn.sql = sql
        self.conn.params = params

    def fetchall(self):
        return [
            (
                "nfl",
                2026,
                "game-1",
                pd.Timestamp("2026-09-10").date(),
                "DEN",
                "B.Nix",
                "00-123",
                "QB",
                "epa_per_play",
                "0.20",
                "0.02",
                "0.65",
                "-0.117",
                100,
                "injury-impact-v1",
                datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc),
                {"source": "test"},
            )
        ]


class FakeImpactConnection:
    def __init__(self):
        self.sql = None
        self.params = None

    def cursor(self):
        return FakeImpactCursor(self)


def test_fetch_injury_impacts_pg_normalizes_rows_for_feature_join():
    conn = FakeImpactConnection()

    impacts = fetch_injury_impacts_pg(
        conn,
        league="nfl",
        start_date="2026-09-10",
        end_date="2026-09-14",
        model_version="injury-impact-v1",
    )

    assert conn.params == (
        "NFL",
        pd.Timestamp("2026-09-10").date(),
        pd.Timestamp("2026-09-14").date(),
        "injury-impact-v1",
        "injury-impact-v1",
    )
    assert "FROM player_impact_estimates" in conn.sql
    assert len(impacts) == 1
    assert impacts["league"].iloc[0] == "NFL"
    assert impacts["game_date"].iloc[0] == pd.Timestamp("2026-09-10")
    assert impacts["team_delta"].iloc[0] == -0.117
    assert bool(impacts["available"].iloc[0]) is True
