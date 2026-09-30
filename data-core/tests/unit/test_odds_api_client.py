from datetime import datetime, timezone
import json

import pytest
import requests

from src.data.odds_api_client import (
    OddsApiBudgetBlocked, budgeted_get, check_budget, request_cost, response_timestamp,
)

NOW = datetime(2026, 10, 1, 13, tzinfo=timezone.utc)
URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
PARAMS = {"apiKey": "secret-test-key", "markets": "h2h,spreads,totals", "bookmakers": "draftkings,betmgm,fanduel"}


def response(payload, *, used=3, remaining=497, cost=3):
    r = requests.Response()
    r.status_code = 200
    r._content = json.dumps(payload).encode()
    r.headers.update({"x-requests-used": str(used), "x-requests-remaining": str(remaining), "x-requests-last": str(cost)})
    return r


class Ledger:
    def __init__(self):
        self.rows = {}
        self.commits = 0
        self.lock_count = 0

    def cursor(self):
        return Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class Cursor:
    def __init__(self, ledger):
        self.ledger = ledger
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params, **_):
        if "pg_advisory_xact_lock" in sql:
            self.ledger.lock_count += 1
        elif sql.startswith("SELECT status"):
            row = self.ledger.rows.get(params[0])
            self.result = (row['status'], row['payload'], row['headers'], row['fetched_at']) if row else None
        elif sql.startswith("SELECT source"):
            sums = {}
            for row in self.ledger.rows.values():
                if row['month'] != params[0]:
                    continue
                total, pending = sums.get(row['source'], (0, 0))
                sums[row['source']] = (total + row['cost'], pending + (row['cost'] if row['status'] == 'reserved' else 0))
            self.result = [(s, *v) for s, v in sums.items()]
        elif sql.startswith("INSERT"):
            key, month, source, cost = params
            assert key not in self.ledger.rows or "ON CONFLICT" in sql
            self.ledger.rows[key] = dict(month=month, source=source, cost=cost, status='reserved', payload=None, headers={}, fetched_at=None)
        elif sql.startswith("UPDATE"):
            status, cost, payload, headers, fetched_at, key = params
            self.ledger.rows[key].update(status=status, cost=cost, payload=payload.obj, headers=headers.obj, fetched_at=fetched_at)
        else:
            raise AssertionError(sql)

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result


def test_cost_is_per_market_and_book_group_not_per_game_or_team():
    assert request_cost(PARAMS) == 3
    assert request_cost({**PARAMS, 'markets': 'team_totals'}) == 1
    assert request_cost({**PARAMS, 'bookmakers': ','.join(str(n) for n in range(11))}) == 6
    assert request_cost({'markets': 'h2h,spreads,totals', 'regions': 'us,eu'}) == 6


@pytest.mark.parametrize('source,used,remaining,spending,pending', [
    ('nfl', 448, 52, {}, 0),
    ('nba', 330, 170, {}, 0),
    ('nfl', 0, 500, {'nfl': 138}, 0),
    ('nfl', 445, 55, {}, 3),
    ('nfl', 499, 500, {}, 0),
])
def test_budget_protects_account_reserve_source_cap_and_pending_reservations(source, used, remaining, spending, pending):
    with pytest.raises(OddsApiBudgetBlocked):
        check_budget(source=source, cost=3, used=used, remaining=remaining, spending=spending, pending=pending)


def test_non_nfl_calls_cannot_consume_unused_nfl_allocation():
    check_budget(source='nfl', cost=3, used=400, remaining=100, spending={}, pending=0)
    with pytest.raises(OddsApiBudgetBlocked):
        check_budget(source='nba', cost=3, used=400, remaining=100, spending={}, pending=0)


def test_reservation_commits_before_spend_and_repeat_reuses_original_capture():
    ledger = Ledger()
    calls = []
    def get(url, **_):
        calls.append(url)
        if '/odds' in url:
            assert ledger.commits == 1
            assert next(iter(ledger.rows.values()))['status'] == 'reserved'
            return response([{'id': 'game'}])
        return response([], used=0, remaining=500, cost=0)
    first = budgeted_get(URL, params=PARAMS, source='nfl', conn=ledger, get=get, now=NOW)
    second = budgeted_get(URL, params=PARAMS, source='nfl', conn=ledger, get=get, now=NOW)
    assert first.json() == second.json()
    assert response_timestamp(first) == response_timestamp(second)
    assert len(calls) == 2  # one free probe, one paid request
    assert ledger.lock_count == 2
    assert 'secret-test-key' not in json.dumps(ledger.rows, default=str)


def test_missing_credentials_make_no_http_requests(monkeypatch):
    monkeypatch.setattr('src.data.odds_api_client.load_supabase_credentials', lambda: {'db_password': None})
    with pytest.raises(OddsApiBudgetBlocked, match='credentials'):
        budgeted_get(URL, params=PARAMS, source='nfl', get=lambda *_a, **_k: pytest.fail('HTTP called'))


def test_missing_quota_headers_prevent_paid_request():
    calls = []
    def get(url, **_):
        calls.append(url)
        r = response([])
        r.headers.clear()
        return r
    with pytest.raises(OddsApiBudgetBlocked, match='headers'):
        budgeted_get(URL, params=PARAMS, source='nfl', conn=Ledger(), get=get, now=NOW)
    assert calls == ['https://api.the-odds-api.com/v4/sports/']


def test_timeout_keeps_reservation_and_blocks_duplicate_spend():
    ledger = Ledger()
    calls = []
    def get(url, **_):
        calls.append(url)
        if '/odds' in url:
            raise requests.Timeout('ambiguous transport failure')
        return response([], used=0, remaining=500, cost=0)
    for _ in range(2):
        with pytest.raises(OddsApiBudgetBlocked):
            budgeted_get(URL, params=PARAMS, source='nfl', conn=ledger, get=get, now=NOW)
    assert len(calls) == 2
    assert next(iter(ledger.rows.values()))['cost'] == 3
    assert next(iter(ledger.rows.values()))['status'] == 'reserved'


def test_month_reset_and_empty_responses_use_actual_costs():
    ledger = Ledger()
    def get(url, **_):
        return response([], used=0, remaining=500, cost=0)
    for month in (9, 10):
        budgeted_get(URL, params=PARAMS, source='nfl', conn=ledger, get=get, now=NOW.replace(month=month))
    assert len(ledger.rows) == 2
    assert {row['month'].month for row in ledger.rows.values()} == {9, 10}
    assert all(row['cost'] == 0 for row in ledger.rows.values())


def test_weekly_empty_market_can_retry_next_day_but_success_is_reused():
    ledger = Ledger()
    paid_calls = []

    def get(url, **_):
        if '/odds' not in url:
            return response([], used=0, remaining=500, cost=0)
        paid_calls.append(url)
        return response([] if len(paid_calls) == 1 else [{'id': 'game'}], cost=0 if len(paid_calls) == 1 else 1)

    options = dict(params={**PARAMS, 'markets': 'team_totals'}, source='nfl',
                   cache_period='week', retry_empty=True, conn=ledger, get=get)
    assert budgeted_get(URL, **options, now=NOW).json() == []
    assert budgeted_get(URL, **options, now=NOW).json() == []
    success = budgeted_get(URL, **options, now=NOW.replace(day=4))
    repeat = budgeted_get(URL, **options, now=NOW.replace(day=5))
    assert success.json() == repeat.json() == [{'id': 'game'}]
    assert response_timestamp(success) == response_timestamp(repeat) == NOW.replace(day=4)
    assert len(paid_calls) == 2
    assert next(iter(ledger.rows.values()))['cost'] == 1
