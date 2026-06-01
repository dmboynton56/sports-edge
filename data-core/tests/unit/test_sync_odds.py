from datetime import datetime, timezone

from scripts.sync_odds import OddsSyncResult, should_fail_zero_odds_match, sync_odds_to_supabase


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        if "SELECT id, home_team, away_team, game_time_utc" in sql:
            self.conn.select_params = params
            return
        raise AssertionError(f"Unexpected SQL: {sql}")

    def executemany(self, sql, params=None):
        self.conn.bulk_updates.append((sql, params))

    def fetchall(self):
        return self.conn.games


class FakeConnection:
    def __init__(self, games):
        self.games = games
        self.bulk_updates = []
        self.select_params = None
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def test_should_not_fail_zero_odds_match_when_dates_do_not_overlap():
    result = OddsSyncResult(
        matched_count=0,
        supabase_games=1,
        supabase_dates={"2026-05-30"},
        odds_dates={"2026-06-04"},
    )

    assert should_fail_zero_odds_match(result) is False


def test_should_fail_zero_odds_match_when_dates_overlap():
    result = OddsSyncResult(
        matched_count=0,
        supabase_games=1,
        supabase_dates={"2026-06-04"},
        odds_dates={"2026-06-04"},
    )

    assert should_fail_zero_odds_match(result) is True


def test_sync_odds_to_supabase_returns_dates_for_schedule_drift():
    conn = FakeConnection(
        games=[
            (
                "game-1",
                "SAS",
                "OKC",
                datetime(2026, 5, 30, tzinfo=timezone.utc),
            )
        ]
    )
    odds_data = [
        {
            "home_team": "San Antonio Spurs",
            "away_team": "New York Knicks",
            "commence_time": "2026-06-04T00:40:00Z",
            "bookmakers": [],
        }
    ]

    result = sync_odds_to_supabase(conn, "NBA", odds_data)

    assert result.matched_count == 0
    assert result.supabase_games == 1
    assert result.supabase_dates == {"2026-05-30"}
    assert result.odds_dates == {"2026-06-04"}
    assert conn.bulk_updates == []
    assert conn.commits == 0
