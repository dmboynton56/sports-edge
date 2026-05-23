import pandas as pd

from src.data.historical_odds import collapse_moneyline_consensus, flatten_odds_payload


def test_flatten_and_collapse_moneylines():
    payload = [
        {
            "id": "evt1",
            "commence_time": "2025-09-28T17:00:00Z",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2025-09-28T16:00:00Z",
                            "outcomes": [
                                {"name": "Home Team", "price": -120},
                                {"name": "Away Team", "price": 100},
                            ],
                        }
                    ],
                },
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2025-09-28T16:00:00Z",
                            "outcomes": [
                                {"name": "Home Team", "price": -110},
                                {"name": "Away Team", "price": -105},
                            ],
                        }
                    ],
                },
            ],
        }
    ]

    flat = flatten_odds_payload(payload, "MLB", "2025-09-28T16:00:00Z")
    consensus = collapse_moneyline_consensus(flat)

    assert len(flat) == 4
    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["home_moneyline"] == -115
    assert row["away_moneyline"] == -2.5
    assert row["books"] == "draftkings,fanduel"
