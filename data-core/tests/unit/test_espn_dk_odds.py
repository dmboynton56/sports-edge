from datetime import date, datetime, timezone
from pathlib import Path
import json

from src.data.espn_dk_odds import (
    ESPN_DK_BOOK,
    espn_event_to_odds_event,
    fetch_espn_dk_events,
    iter_featured_game_lines,
    parse_american_price,
    parse_espn_scoreboard_odds,
    parse_line,
    pick_draftkings_odds,
)

FIXTURE = Path(__file__).parent / "fixtures" / "espn_dk_nfl_scoreboard.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_recorded_nfl_scoreboard_maps_dk_game_lines_only():
    events = parse_espn_scoreboard_odds(_fixture())
    assert events
    ne_chi = next(event for event in events if event["id"] == "401873010")
    assert ne_chi["home_team"] == "Chicago Bears"
    assert ne_chi["away_team"] == "New England Patriots"
    assert ne_chi["commence_time"] == "2026-10-23T00:15:00Z"
    books = ne_chi["bookmakers"]
    assert len(books) == 1
    assert books[0]["key"] == ESPN_DK_BOOK
    markets = {market["key"]: market for market in books[0]["markets"]}
    assert set(markets) == {"h2h", "spreads", "totals"}
    home_spread = next(row for row in markets["spreads"]["outcomes"] if row["name"] == "Chicago Bears")
    away_spread = next(row for row in markets["spreads"]["outcomes"] if row["name"] == "New England Patriots")
    assert home_spread["point"] == -3.0
    assert away_spread["point"] == 3.0
    assert home_spread["price"] == -105
    assert away_spread["price"] == -115
    home_ml = next(row for row in markets["h2h"]["outcomes"] if row["name"] == "Chicago Bears")
    away_ml = next(row for row in markets["h2h"]["outcomes"] if row["name"] == "New England Patriots")
    assert home_ml["price"] == -155
    assert away_ml["price"] == 130
    over = next(row for row in markets["totals"]["outcomes"] if row["name"] == "Over")
    under = next(row for row in markets["totals"]["outcomes"] if row["name"] == "Under")
    assert over["point"] == 50.5
    assert under["point"] == 50.5
    assert over["price"] == -115
    assert under["price"] == -105
    rows = iter_featured_game_lines(events)
    assert {row["book"] for row in rows} == {ESPN_DK_BOOK}
    assert "team_totals" not in {row["market"] for row in rows}


def test_non_draftkings_provider_is_ignored():
    event = {
        "id": "x",
        "date": "2026-10-23T00:15Z",
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": "CHI", "displayName": "Chicago Bears"}},
                    {"homeAway": "away", "team": {"abbreviation": "NE", "displayName": "New England Patriots"}},
                ],
                "odds": [
                    {
                        "provider": {"id": "1", "name": "consensus"},
                        "spread": -3.0,
                        "overUnder": 50.5,
                    }
                ],
            }
        ],
    }
    assert pick_draftkings_odds(event["competitions"][0]) is None
    assert espn_event_to_odds_event(event) is None


def test_legacy_home_team_odds_shape_still_maps():
    event = {
        "id": "legacy",
        "date": "2026-10-23T00:15Z",
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": "CHI", "displayName": "Chicago Bears"}},
                    {"homeAway": "away", "team": {"abbreviation": "NE", "displayName": "New England Patriots"}},
                ],
                "odds": [
                    {
                        "provider": {"id": "58", "name": "DraftKings"},
                        "spread": -3.5,
                        "overUnder": 47.5,
                        "overOdds": -110,
                        "underOdds": -110,
                        "homeTeamOdds": {"moneyLine": -180, "spreadOdds": -108},
                        "awayTeamOdds": {"moneyLine": 155, "spreadOdds": -112},
                    }
                ],
            }
        ],
    }
    mapped = espn_event_to_odds_event(event)
    markets = {market["key"]: market for market in mapped["bookmakers"][0]["markets"]}
    assert markets["h2h"]["outcomes"][0]["price"] == -180
    assert markets["spreads"]["outcomes"][0]["point"] == -3.5
    assert markets["spreads"]["outcomes"][1]["point"] == 3.5
    assert markets["totals"]["outcomes"][0]["point"] == 47.5


def test_missing_price_fails_closed_without_inventing_markets():
    event = {
        "id": "incomplete",
        "date": "2026-10-23T00:15Z",
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"abbreviation": "CHI", "displayName": "Chicago Bears"}},
                    {"homeAway": "away", "team": {"abbreviation": "NE", "displayName": "New England Patriots"}},
                ],
                "odds": [{"provider": {"name": "DraftKings"}, "spread": -3.0, "overUnder": 50.5}],
            }
        ],
    }
    assert espn_event_to_odds_event(event) is None


def test_parse_helpers_reject_non_american_junk():
    assert parse_american_price("-155") == -155
    assert parse_american_price("+130") == 130
    assert parse_american_price("−110") == -110
    assert parse_american_price(0) is None
    assert parse_american_price(-90) is None
    assert parse_line("o50.5") == 50.5
    assert parse_line("+3") == 3.0
    assert parse_line("n/a") is None


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_espn_dk_events_stamps_source_and_skips_bad_days():
    payload = _fixture()
    calls = []

    def fake_get(url, headers=None, timeout=None, params=None):
        calls.append(params)
        if params.get("dates") == "20261022":
            return _Resp(payload)
        raise RuntimeError("nope")

    events = fetch_espn_dk_events(
        "NFL",
        start=date(2026, 10, 22),
        end=date(2026, 10, 23),
        now=datetime(2026, 10, 20, 2, tzinfo=timezone.utc),
        get=fake_get,
    )
    assert len(calls) == 2
    assert events[0]["_source"] == ESPN_DK_BOOK
    assert events[0]["_snapshot_ts"].startswith("2026-10-20")
