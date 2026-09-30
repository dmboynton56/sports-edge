import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import requests

from scripts.sync_odds import (
    NFL_MAPPING,
    OddsSyncResult,
    canonical_team_code,
    fetch_odds_data,
    odds_window_days,
    pick_featured_market_outcomes,
    should_fail_zero_odds_match,
    sync_odds_to_supabase,
    fetch_nfl_team_totals,
    pick_team_total_outcomes,
    request_bounds,
)
from src.data.odds_api_errors import OddsApiQuotaExhausted


def test_nfl_odds_window_covers_the_full_weekly_slate():
    assert odds_window_days("NFL") == 7
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
        if "SELECT DISTINCT ON (game_id, market, selection)" in sql:
            self.conn.latest_select_params = params
            return
        raise AssertionError(f"Unexpected SQL: {sql}")

    def executemany(self, sql, params=None):
        self.conn.bulk_updates.append((sql, params))

    def fetchall(self):
        if self.conn.latest_select_params is not None:
            return self.conn.latest_snapshots
        return self.conn.games


class FakeConnection:
    def __init__(self, games):
        self.games = games
        self.bulk_updates = []
        self.select_params = None
        self.latest_select_params = None
        self.latest_snapshots = []
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
        now_utc=datetime(2026, 9, 8, 13, tzinfo=timezone.utc),
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


class _ApiResponse:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, str):
            return json.loads(self._payload)
        return self._payload


def test_fetch_odds_data_soft_fails_out_of_usage_credits():
    body = {
        "message": "Usage quota has been reached. See usage plans at https://the-odds-api.com",
        "error_code": "OUT_OF_USAGE_CREDITS",
    }
    with patch("scripts.sync_odds.budgeted_get", return_value=_ApiResponse(401, body)):
        with pytest.raises(OddsApiQuotaExhausted):
            fetch_odds_data("key", "NFL")


def test_fetch_odds_data_soft_fails_rate_limit():
    with patch("scripts.sync_odds.budgeted_get", return_value=_ApiResponse(429, "too many requests")):
        with pytest.raises(OddsApiQuotaExhausted):
            fetch_odds_data("key", "NFL")


def test_fetch_odds_data_hard_fails_invalid_api_key():
    body = {"message": "API key is invalid", "error_code": "INVALID_API_KEY"}
    with patch("scripts.sync_odds.budgeted_get", return_value=_ApiResponse(401, body)):
        with pytest.raises(RuntimeError, match="401") as caught:
            fetch_odds_data("key", "NFL")
    assert not isinstance(caught.value, OddsApiQuotaExhausted)


def test_fetch_odds_data_hard_fails_on_server_error():
    with patch("scripts.sync_odds.budgeted_get", return_value=_ApiResponse(502, "bad gateway")):
        with pytest.raises(RuntimeError, match="502") as caught:
            fetch_odds_data("key", "NFL")
    assert not isinstance(caught.value, OddsApiQuotaExhausted)


def test_fetch_odds_data_hard_fails_on_timeout():
    with patch("scripts.sync_odds.budgeted_get", side_effect=requests.Timeout("timed out")):
        with pytest.raises(requests.Timeout):
            fetch_odds_data("key", "NFL")


def test_fetch_odds_data_returns_events_on_success():
    payload = [{"id": "evt-1", "home_team": "Seattle Seahawks"}]
    with patch("scripts.sync_odds.budgeted_get", return_value=_ApiResponse(200, payload)):
        assert fetch_odds_data("key", "NFL") == payload


def test_nfl_request_bounds_stop_at_monday_in_denver():
    assert request_bounds('NFL', datetime(2026, 10, 1, 13, tzinfo=timezone.utc)) == {
        'commenceTimeFrom': '2026-09-29T06:00:00Z',
        'commenceTimeTo': '2026-10-06T05:59:59Z',
    }


def _team_total_event():
    return {
        'id': 'week-four', 'home_team': 'Seattle Seahawks', 'away_team': 'New England Patriots',
        'commence_time': '2026-10-04T20:00:00Z',
        'bookmakers': [{'key': 'fanduel', 'markets': [{'key': 'team_totals', 'outcomes': [
            {'name': pick, 'description': team, 'point': point, 'price': -110}
            for team, point in [('Seattle Seahawks', 24.5), ('New England Patriots', 19.5)]
            for pick in ['Over', 'Under']
        ]}]}],
    }


def test_team_totals_preserve_four_sides_and_fail_closed_on_incomplete_pair():
    event = _team_total_event()
    rows = pick_team_total_outcomes(event, 'SEA', 'NE', NFL_MAPPING)
    assert [(r.selection, r.line) for r in rows] == [('home_over', 24.5), ('home_under', 24.5), ('away_over', 19.5), ('away_under', 19.5)]
    event['bookmakers'][0]['markets'][0]['outcomes'][1]['point'] = 25.5
    rows = pick_team_total_outcomes(event, 'SEA', 'NE', NFL_MAPPING)
    assert [r.selection for r in rows] == ['away_over', 'away_under']


def test_team_total_fetch_skips_finished_games_and_future_cycles():
    current = _team_total_event()
    future = {**current, 'id': 'future', 'commence_time': '2026-10-11T20:00:00Z'}
    past = {**current, 'id': 'finished', 'commence_time': '2026-09-28T20:00:00Z'}
    with patch('scripts.sync_odds.budgeted_get', return_value=_ApiResponse(200, current)) as fetch:
        assert len(fetch_nfl_team_totals('key', [past, current, future], now_utc=datetime(2026, 10, 1, 13, tzinfo=timezone.utc))) == 1
        assert fetch.call_count == 1
        assert fetch.call_args.kwargs['params']['markets'] == 'team_totals'
        assert fetch.call_args.kwargs['cache_period'] == 'team-totals-2026-09-29'


def test_cached_odds_keep_original_snapshot_timestamp():
    event = _team_total_event()
    event['_snapshot_ts'] = '2026-09-29T13:05:00+00:00'
    conn = FakeConnection([('game', 'SEA', 'NE', datetime(2026, 10, 4, 20, tzinfo=timezone.utc), datetime(2026, 10, 4).date())])
    result = sync_odds_to_supabase(conn, 'NFL', [event], now_utc=datetime(2026, 10, 4, 13, tzinfo=timezone.utc))
    assert result.market_counts['team_total'] == 1
    snapshots = conn.bulk_updates[0][1]
    assert all(row[5] == datetime(2026, 9, 29, 13, 5, tzinfo=timezone.utc) for row in snapshots)


def test_cached_capture_is_not_inserted_twice_but_new_unchanged_capture_is():
    event = _team_total_event()
    captured = datetime(2026, 9, 29, 13, 5, tzinfo=timezone.utc)
    event['_snapshot_ts'] = captured.isoformat()
    conn = FakeConnection([('game', 'SEA', 'NE', datetime(2026, 10, 4, 20, tzinfo=timezone.utc), datetime(2026, 10, 4).date())])
    conn.latest_snapshots = [
        ('game', 'team_total', selection, 'fanduel', line, -110, 'week-four', captured)
        for selection, line in [('home_over', 24.5), ('home_under', 24.5), ('away_over', 19.5), ('away_under', 19.5)]
    ]
    now = datetime(2026, 10, 4, 13, tzinfo=timezone.utc)
    sync_odds_to_supabase(conn, 'NFL', [event], now_utc=now)
    assert conn.bulk_updates == []

    event['_snapshot_ts'] = now.isoformat()
    conn.latest_select_params = None
    sync_odds_to_supabase(conn, 'NFL', [event], now_utc=now)
    assert len(conn.bulk_updates[0][1]) == 4
    assert all(row[5] == now for row in conn.bulk_updates[0][1])
