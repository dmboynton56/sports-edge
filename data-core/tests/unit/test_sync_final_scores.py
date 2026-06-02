from datetime import datetime, timezone

import pandas as pd

from scripts.sync_final_scores import sync_scores


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        if "SELECT" in sql and "FROM games" in sql:
            self.conn.select_params = params
            self.rowcount = 0
            return
        if "UPDATE games" in sql:
            self.conn.updates.append(params)
            self.rowcount = 1
            return
        raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchall(self):
        return self.conn.games


class FakeConnection:
    def __init__(self, games):
        self.games = games
        self.updates = []
        self.select_params = None
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def test_sync_scores_matches_canonical_nba_team_aliases():
    conn = FakeConnection(
        games=[
            (
                "game-1",
                "NBA",
                "SA",
                "OKC",
                datetime(2026, 5, 30, tzinfo=timezone.utc),
            )
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "league": "NBA",
                "game_date": pd.Timestamp("2026-05-30"),
                "home_team": "San Antonio Spurs",
                "away_team": "Oklahoma City Thunder",
                "home_score": 110,
                "away_score": 103,
            }
        ]
    )

    updated, unmatched = sync_scores(conn, scores)

    assert updated == 1
    assert unmatched == 0
    assert conn.updates == [(110, 103, "game-1")]
    assert conn.select_params == ("2026-05-30", "2026-05-30")
    assert conn.commits == 1


def test_sync_scores_reports_unmatched_rows_without_update():
    conn = FakeConnection(games=[])
    scores = pd.DataFrame(
        [
            {
                "league": "MLB",
                "game_date": pd.Timestamp("2026-05-30"),
                "home_team": "Los Angeles Dodgers",
                "away_team": "New York Yankees",
                "home_score": 5,
                "away_score": 2,
            }
        ]
    )

    updated, unmatched = sync_scores(conn, scores)

    assert updated == 0
    assert unmatched == 1
    assert conn.updates == []
    assert conn.commits == 1


def test_sync_scores_matches_late_utc_game_by_schedule_date():
    conn = FakeConnection(
        games=[
            (
                "game-mlb",
                "MLB",
                "Arizona Diamondbacks",
                "Los Angeles Dodgers",
                pd.Timestamp("2026-06-01").date(),
            )
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "league": "MLB",
                "game_date": pd.Timestamp("2026-06-01"),
                "home_team": "Arizona Diamondbacks",
                "away_team": "Los Angeles Dodgers",
                "home_score": 4,
                "away_score": 6,
            }
        ]
    )

    updated, unmatched = sync_scores(conn, scores)

    assert updated == 1
    assert unmatched == 0
    assert conn.updates == [(4, 6, "game-mlb")]
    assert conn.select_params == ("2026-06-01", "2026-06-01")
