from src.data.nfl_prop_markets import (
    NFL_PROP_MARKETS,
    aggregate_event_summaries,
    classify_book,
    compact_closing_rows,
    summarize_event_odds,
)


def test_classify_book_splits_dfs_from_sportsbooks():
    assert classify_book("draftkings") == "sportsbook"
    assert classify_book("prizepicks") == "dfs"
    assert classify_book("unknown_book") == "other"


def test_summarize_event_counts_unique_players_and_books():
    payload = {
        "id": "evt1",
        "sport_key": "football_nfl",
        "home_team": "Denver Broncos",
        "away_team": "Kansas City Chiefs",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_reception_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Courtland Sutton", "point": 49.5, "price": -110},
                            {"name": "Under", "description": "Courtland Sutton", "point": 49.5, "price": -110},
                            {"name": "Over", "description": "Troy Franklin", "point": 24.5, "price": -115},
                        ],
                    }
                ],
            },
            {
                "key": "prizepicks",
                "markets": [
                    {
                        "key": "player_reception_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Courtland Sutton", "point": 51.5, "price": -137},
                        ],
                    }
                ],
            },
        ],
    }
    summary = summarize_event_odds(payload)
    rec = next(row for row in summary["markets"] if row["market"] == "player_reception_yds")
    assert rec["present"] is True
    assert rec["player_count"] == 2
    assert rec["sportsbooks"] == ["draftkings"]
    assert rec["dfs_books"] == ["prizepicks"]
    missing = next(row for row in summary["markets"] if row["market"] == "player_pass_yds")
    assert missing["present"] is False
    assert set(row["market"] for row in summary["markets"]) == set(NFL_PROP_MARKETS)


def test_aggregate_event_summaries_averages_players_per_game():
    first = summarize_event_odds(
        {
            "id": "a",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "player_receptions",
                            "outcomes": [
                                {"name": "Over", "description": "A", "point": 4.5},
                                {"name": "Over", "description": "B", "point": 3.5},
                            ],
                        }
                    ],
                }
            ],
        }
    )
    second = summarize_event_odds(
        {
            "id": "b",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "player_receptions",
                            "outcomes": [{"name": "Over", "description": "C", "point": 5.5}],
                        }
                    ],
                }
            ],
        }
    )
    rolled = aggregate_event_summaries([first, second])
    rec = next(row for row in rolled["markets"] if row["market"] == "player_receptions")
    assert rec["events_present"] == 2
    assert rec["players_per_game_mean"] == 1.5
    assert rec["sportsbooks"] == ["fanduel"]


def test_compact_closing_rows_drops_dfs_and_pairs_over_under():
    rows = compact_closing_rows(
        {
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "player_reception_yds",
                            "outcomes": [
                                {"name": "Over", "description": "A", "point": 49.5, "price": -110},
                                {"name": "Under", "description": "A", "point": 49.5, "price": -110},
                            ],
                        }
                    ],
                },
                {
                    "key": "prizepicks",
                    "markets": [
                        {
                            "key": "player_reception_yds",
                            "outcomes": [{"name": "Over", "description": "A", "point": 51.5, "price": -137}],
                        }
                    ],
                },
            ]
        }
    )
    assert len(rows) == 1
    assert rows[0]["book"] == "draftkings"
    assert rows[0]["over_price"] == -110
    assert rows[0]["under_price"] == -110
