import pandas as pd

from src.data.oddspapi_odds import (
    MarketCatalog,
    decimal_to_american,
    extract_moneylines,
    extract_spread,
    flatten_historical_quotes,
    norm_team_for_sport,
    resolve_fixture_for_game,
    select_closing_quotes,
)


def _sample_payload() -> dict:
    return {
        "fixtureId": "id1100013270505056",
        "bookmakers": {
            "pinnacle": {
                "markets": {
                    "111": {
                        "outcomes": {
                            "111": {
                                "players": {
                                    "0": [
                                        {
                                            "createdAt": "2025-09-28T15:00:00.000Z",
                                            "price": 1.91,
                                            "active": True,
                                        },
                                        {
                                            "createdAt": "2025-09-28T16:00:00.000Z",
                                            "price": 1.952,
                                            "active": True,
                                        },
                                    ]
                                }
                            },
                            "112": {
                                "players": {
                                    "0": [
                                        {
                                            "createdAt": "2025-09-28T16:00:00.000Z",
                                            "price": 1.87,
                                            "active": True,
                                        }
                                    ]
                                }
                            },
                        }
                    },
                    "11356": {
                        "outcomes": {
                            "11356": {
                                "players": {
                                    "0": [
                                        {
                                            "createdAt": "2025-09-28T16:00:00.000Z",
                                            "price": 1.91,
                                            "active": True,
                                        }
                                    ]
                                }
                            },
                            "11357": {
                                "players": {
                                    "0": [
                                        {
                                            "createdAt": "2025-09-28T16:00:00.000Z",
                                            "price": 1.91,
                                            "active": True,
                                        }
                                    ]
                                }
                            },
                        }
                    },
                }
            }
        },
    }


def test_flatten_historical_quotes():
    flat = flatten_historical_quotes(_sample_payload())
    assert len(flat) == 5
    assert set(flat["outcome_id"].tolist()) == {111, 112, 11356, 11357}


def test_select_closing_quotes_uses_last_pre_start():
    flat = flatten_historical_quotes(_sample_payload())
    start_ms = int(pd.Timestamp("2025-09-28T17:00:00Z").timestamp() * 1000)
    closing = select_closing_quotes(flat, start_ms, outcome_ids={111, 112})
    home_row = closing[closing["outcome_id"] == 111].iloc[0]
    assert pd.Timestamp(home_row["created_at"]) == pd.Timestamp("2025-09-28T16:00:00.000Z")


def test_extract_moneylines():
    payload = _sample_payload()
    flat = flatten_historical_quotes(payload)
    start_ms = int(pd.Timestamp("2025-09-28T17:00:00Z").timestamp() * 1000)
    closing = select_closing_quotes(flat, start_ms, outcome_ids={111, 112})
    catalog = MarketCatalog(
        moneyline_outcome_ids={111, 112},
        outcome_side={111: "participant1", 112: "participant2"},
    )
    result = extract_moneylines(closing, catalog, payload, bookmaker="pinnacle")
    assert result["home_moneyline"] == decimal_to_american(1.952)
    assert result["away_moneyline"] == decimal_to_american(1.87)


def test_extract_spread_home_perspective():
    payload = _sample_payload()
    flat = flatten_historical_quotes(payload)
    start_ms = int(pd.Timestamp("2025-09-28T17:00:00Z").timestamp() * 1000)
    closing = select_closing_quotes(flat, start_ms, market_ids={11356})
    catalog = MarketCatalog(
        spread_market_ids={11356},
        market_meta={11356: {"handicap": -4.5, "outcomes": {11356: "1", 11357: "2"}}},
        outcome_side={11356: "participant1", 11357: "participant2"},
    )
    result = extract_spread(closing, catalog, payload, bookmaker="pinnacle")
    assert result["home_spread"] == -4.5


def test_resolve_fixture_for_game_nba_abbr():
    fixtures = [
        {
            "fixtureId": "fixture1",
            "startTime": "2025-10-22T00:00:00.000Z",
            "participant1Name": "Los Angeles Lakers",
            "participant2Name": "Golden State Warriors",
            "participant1Abbr": "LAL",
            "participant2Abbr": "GSW",
        }
    ]
    matched = resolve_fixture_for_game(
        fixtures,
        sport="NBA",
        home="LAL",
        away="GS",
        game_date="2025-10-22",
    )
    assert matched is not None
    assert matched["fixtureId"] == "fixture1"


def test_norm_team_for_sport_nba():
    assert norm_team_for_sport("GS", "NBA") == norm_team_for_sport("GSW", "NBA")
