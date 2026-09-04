from datetime import datetime, timezone

from scripts.sync_odds import (
    NFL_MAPPING,
    OddsSyncResult,
    canonical_team_code,
    odds_window_days,
    pick_featured_market_outcomes,
    should_fail_zero_odds_match,
    sync_odds_to_supabase,
)


def test_nfl_odds_window_covers_the_full_weekly_slate():
    assert odds_window_days("NFL") == 14
    assert odds_window_days("NBA") == 10


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        if "SELECT id, home_team, away_team, game_time_utc, game_date" in sql:
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
                datetime(2026, 5, 30, tzinfo=timezone.utc).date(),
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

    result = sync_odds_to_supabase(
        conn,
        "NBA",
        odds_data,
        now_utc=datetime(2026, 5, 29, tzinfo=timezone.utc),
    )

    assert result.matched_count == 0
    assert result.supabase_games == 1
    assert result.supabase_dates == {"2026-05-30"}
    assert result.odds_dates == {"2026-06-03"}
    assert conn.bulk_updates == []
    assert conn.commits == 0


def _spread_event(home_team, away_team, commence_time, line=-3.5):
    return {
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence_time,
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home_team, "point": line, "price": -110},
                            {"name": away_team, "point": -line, "price": -110},
                        ],
                    }
                ],
            }
        ],
    }


def test_nfl_schedule_la_alias_matches_odds_api_lar_identity():
    assert canonical_team_code("LA", "NFL") == "LAR"
    conn = FakeConnection(
        games=[
            (
                "rams-opener",
                "LA",
                "SF",
                datetime(2026, 9, 10, tzinfo=timezone.utc),
                datetime(2026, 9, 10, tzinfo=timezone.utc).date(),
            )
        ]
    )
    odds_data = [
        _spread_event(
            "Los Angeles Rams",
            "San Francisco 49ers",
            "2026-09-11T00:35:00Z",
            line=-2.5,
        )
    ]

    result = sync_odds_to_supabase(
        conn,
        "NFL",
        odds_data,
        now_utc=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )

    assert result.matched_count == 1
    assert conn.bulk_updates[0][1] == [(-2.5, "rams-opener")]


def test_future_rematch_cannot_overwrite_an_in_window_game():
    conn = FakeConnection(
        games=[
            (
                "week-one",
                "SEA",
                "NE",
                datetime(2026, 9, 9, tzinfo=timezone.utc),
                datetime(2026, 9, 9, tzinfo=timezone.utc).date(),
            )
        ]
    )
    odds_data = [
        _spread_event(
            "Seattle Seahawks",
            "New England Patriots",
            "2026-09-17T20:00:00Z",
        )
    ]

    result = sync_odds_to_supabase(
        conn,
        "NFL",
        odds_data,
        now_utc=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )

    assert result.matched_count == 0
    assert conn.bulk_updates == []
    assert conn.commits == 0


def test_featured_market_extraction_preserves_both_sides_and_uses_market_fallback_books():
    event = {
        "home_team": "Seattle Seahawks",
        "away_team": "New England Patriots",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Seattle Seahawks", "price": -155},
                            {"name": "New England Patriots", "price": 135},
                        ],
                    }
                ],
            },
            {
                "key": "betmgm",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Seattle Seahawks", "point": -3.5, "price": -110},
                            {"name": "New England Patriots", "point": 3.5, "price": -110},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 46.5, "price": -108},
                            {"name": "Under", "point": 46.5, "price": -112},
                        ],
                    },
                ],
            },
        ],
    }

    rows = pick_featured_market_outcomes(event, "SEA", "NE", NFL_MAPPING)

    assert [(row.market, row.selection, row.book) for row in rows] == [
        ("moneyline", "home", "draftkings"),
        ("moneyline", "away", "draftkings"),
        ("spread", "home", "betmgm"),
        ("spread", "away", "betmgm"),
        ("total", "over", "betmgm"),
        ("total", "under", "betmgm"),
    ]
