from datetime import date

import pandas as pd
import pytest

from src.data.nfl_historical_prop_lines import compact_graded_csv, grade_real_lines
from src.data.nfl_prop_markets import compact_closing_rows
from src.models.nfl_prop_fair_odds import (
    apply_residual_mean,
    consensus_sportsbook_line,
    display_player_name,
    injury_availability,
    no_vig_over_prob,
    player_on_event,
    probability_to_american,
    reject_dfs_rows,
    score_live_lines,
)
from src.models.nfl_receiving_yards import build_feature_frame


def test_no_vig_and_american_round_trip():
    novig = no_vig_over_prob(-110, -110)
    assert novig == pytest.approx(0.5)
    assert probability_to_american(0.75) == -300
    assert probability_to_american(0.25) == 300


def test_injury_availability_gates_out_players():
    assert injury_availability("Out") == 0.2
    assert injury_availability("Questionable") == 0.88
    assert injury_availability("Active") == 1.0


def test_residual_mean_stays_near_season_average_without_a_shock():
    rows = []
    games = []
    for week in range(1, 10):
        games.append(
            {
                "game_id": f"g{week}",
                "gameday": date(2023, 9, week),
                "home_team": "DEN",
                "away_team": "KC",
                "home_score": 24,
                "away_score": 20,
                "predicted_margin": 3,
                "closing_total": 44,
            }
        )
        rows.append(
            {
                "game_id": f"g{week}",
                "season": 2023,
                "week": week,
                "player_id": "p1",
                "player_display_name": "A",
                "position": "WR",
                "team": "DEN",
                "opponent_team": "KC",
                "targets": 8,
                "receiving_yards": 64,
                "receptions": 5,
                "attempts": 0,
                "carries": 0,
                "target_share": 0.22,
                "offense_pct": 0.9,
            }
        )
        rows.append(
            {
                "game_id": f"g{week}",
                "season": 2023,
                "week": week,
                "player_id": "qb",
                "player_display_name": "QB",
                "position": "QB",
                "team": "DEN",
                "opponent_team": "KC",
                "targets": 0,
                "receiving_yards": 0,
                "receptions": 0,
                "attempts": 35,
                "carries": 5,
                "target_share": 0.0,
                "offense_pct": 1.0,
            }
        )
    modeled = apply_residual_mean(
        build_feature_frame(pd.DataFrame(rows), pd.DataFrame(games)),
        stat="receiving_yards",
    )
    last = modeled[(modeled.player_id == "p1") & (modeled.week == 9)].iloc[0]
    assert last["baseline_mu"] == pytest.approx(64.0)
    assert abs(last["mu"] - last["baseline_mu"]) < 8.0
    assert last["mu"] > 20


def test_consensus_line_drops_dfs_and_uses_sportsbook_two_way():
    payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_reception_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Courtland Sutton", "point": 49.5, "price": -110},
                            {"name": "Under", "description": "Courtland Sutton", "point": 49.5, "price": -110},
                        ],
                    }
                ],
            },
            {
                "key": "prizepicks",
                "markets": [
                    {
                        "key": "player_reception_yds",
                        "outcomes": [{"name": "Over", "description": "Courtland Sutton", "point": 51.5, "price": -137}],
                    }
                ],
            },
        ]
    }
    assert len(compact_closing_rows(payload)) == 1
    rows = consensus_sportsbook_line(payload, market="player_reception_yds")
    assert len(rows) == 1
    assert rows[0]["line"] == 49.5
    assert "prizepicks" not in rows[0]["books"]
    live = score_live_lines(
        pd.DataFrame(
            [
                {
                    "player_display_name": "Courtland Sutton",
                    "team": "DEN",
                    "opponent_team": "KC",
                    "position": "WR",
                    "mu": 55.0,
                    "alpha": 0.3,
                    "availability": 1.0,
                }
            ]
        ),
        [{**rows[0], "home_team": "Denver Broncos", "away_team": "Kansas City Chiefs"}],
        stat="receiving_yards",
    )
    assert live[0]["opponent"] == "KC"
    assert live[0]["our_p_over"] > 0.4
    assert live[0]["live_site_published"] is False
    assert reject_dfs_rows([{**live[0], "books": ["prizepicks"]}]) == []


def test_compact_graded_csv_and_grade_label(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "event_id,commence_time,home_team,away_team,market,bookmaker,player_name,outcome_name,line,price_american,closing_price,closing_point,opening_point,actual_value,resolution,player_id\n"
        "1,2026-09-20T17:00:00+00:00,Denver Broncos,Kansas City Chiefs,player_reception_yds,draftkings,Courtland Sutton,Over,49.5,-110,-110,49.5,48.5,80,won,espn:1\n"
        "1,2026-09-20T17:00:00+00:00,Denver Broncos,Kansas City Chiefs,player_reception_yds,draftkings,Courtland Sutton,Under,49.5,-110,-110,49.5,48.5,80,lost,espn:1\n"
        "1,2026-09-20T17:00:00+00:00,Denver Broncos,Kansas City Chiefs,player_reception_yds,prizepicks,Courtland Sutton,Over,51.5,-137,-137,51.5,51.5,80,won,espn:1\n",
        encoding="utf-8",
    )
    rows = compact_graded_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["book"] == "draftkings"
    scored = pd.DataFrame(
        [
            {
                "player_display_name": "Courtland Sutton",
                "home_team": "DEN",
                "away_team": "KC",
                "game_date": "2026-09-20",
                "mu": 60.0,
                "alpha": 0.3,
                "receiving_yards": 80,
            }
        ]
    )
    report = grade_real_lines(scored, pd.DataFrame(rows), stat="receiving_yards", market="player_reception_yds")
    assert report["status"] == "small_sample"
    assert "2026" in report["label"]
    assert "not 2023-2025" in report["label"]
    assert report["n"] == 1
    assert report["win"] == 1


def test_live_join_requires_player_on_the_event():
    assert display_player_name("A.J. Brown (NE)") == "A.J. Brown"
    assert player_on_event("PHI", "Chicago Bears", "New England Patriots") is False
    assert player_on_event("NE", "Chicago Bears", "New England Patriots") is True
    projections = pd.DataFrame(
        [
            {
                "player_display_name": "A.J. Brown",
                "team": "PHI",
                "opponent_team": "DAL",
                "position": "WR",
                "mu": 70.0,
                "alpha": 0.3,
                "availability": 1.0,
            },
            {
                "player_display_name": "Christian McCaffrey",
                "team": "SF",
                "opponent_team": "ATL",
                "position": "RB",
                "mu": 30.0,
                "alpha": 0.3,
                "availability": 1.0,
            },
            {
                "player_display_name": "Christian McCaffrey",
                "team": "SF",
                "opponent_team": "WAS",
                "position": "RB",
                "mu": 31.0,
                "alpha": 0.3,
                "availability": 1.0,
            },
        ]
    )
    missed = score_live_lines(
        projections,
        [
            {
                "player": "A.J. Brown (NE)",
                "market": "player_reception_yds",
                "line": 67.5,
                "book_novig_p_over": 0.5,
                "best_over_price": -110,
                "best_under_price": -110,
                "best_over_book": "fanduel",
                "books": ["fanduel"],
                "home_team": "Chicago Bears",
                "away_team": "New England Patriots",
            }
        ],
        stat="receiving_yards",
    )
    assert missed == []
    cmc = score_live_lines(
        projections,
        [
            {
                "player": "Christian McCaffrey (SF)",
                "market": "player_reception_yds",
                "line": 40.5,
                "book_novig_p_over": 0.5,
                "best_over_price": -108,
                "best_under_price": -111,
                "best_over_book": "pinnacle",
                "books": ["pinnacle"],
                "home_team": "San Francisco 49ers",
                "away_team": "Washington Commanders",
            }
        ],
        stat="receiving_yards",
    )
    assert len(cmc) == 1
    assert cmc[0]["player"] == "Christian McCaffrey"
    assert cmc[0]["team"] == "SF"
    assert cmc[0]["opponent"] == "WAS"
    assert cmc[0]["our_mu"] == 31.0
