from pathlib import Path

import pandas as pd

from scripts import sync_mlb_home_run_results_to_supabase as sync_module


class _Cursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None, prepare=False):
        self.connection.lookup_sql = sql
        self.connection.lookup_params.append(params)

    def fetchone(self):
        return None

    def executemany(self, sql, rows):
        self.connection.inserted_rows.extend(rows)


class _Connection:
    def __init__(self):
        self.lookup_sql = ""
        self.lookup_params = []
        self.inserted_rows = []
        self.committed = False
        self.closed = False

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.committed = False

    def close(self):
        self.closed = True


def test_board_lookup_casts_nullable_event_time_as_timestamptz(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "evaluated.csv"
    pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "player_id": 101,
                "model_version": "v1",
                "event_time": None,
                "game_date": "2026-08-15",
                "player_name": "Player One",
                "team": "DEN",
                "opponent": "SEA",
                "prediction_ts": "2026-08-15T12:00:00Z",
                "rank": 1,
                "hr_probability": 0.25,
                "actual_home_run": 1,
                "actual_home_runs": 1,
                "actual_plate_appearances": 4,
            },
            {
                "game_id": "game-2",
                "player_id": 202,
                "model_version": "v1",
                "event_time": "2026-08-16T01:30:00+00:00",
                "game_date": "2026-08-16",
                "player_name": "Player Two",
                "team": "LAD",
                "opponent": "COL",
                "prediction_ts": "2026-08-16T12:00:00Z",
                "rank": 2,
                "hr_probability": 0.20,
                "actual_home_run": 0,
                "actual_home_runs": 0,
                "actual_plate_appearances": 4,
            },
        ]
    ).to_csv(csv_path, index=False)
    connection = _Connection()
    monkeypatch.setattr(sync_module, "load_supabase_credentials", lambda: {
        "url": "https://example.supabase.co",
        "db_password": "secret",
        "db_host": "db.example",
        "db_port": 5432,
        "db_name": "postgres",
        "db_user": "postgres",
    })
    monkeypatch.setattr(sync_module, "create_pg_connection", lambda **kwargs: connection)

    assert sync_module.sync_results(csv_path) == 2
    assert "(%s::timestamptz is null or b.published_at <= %s::timestamptz)" in connection.lookup_sql
    assert connection.lookup_params[0][3] is None
    assert connection.lookup_params[1][3] == "2026-08-16T01:30:00+00:00"
    assert connection.committed is True
    assert connection.closed is True
