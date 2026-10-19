from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.models.nfl_receiving_yards import (
    apply_opportunity_model,
    build_feature_frame,
    crps_nb,
    grade_closing_sample,
    implied_team_points,
    p_over_line,
    pinball_loss,
    walk_forward_evaluate,
)


def test_implied_team_points_splits_total_by_margin():
    assert implied_team_points(10, 44, True) == pytest.approx(27.0)
    assert implied_team_points(10, 44, False) == pytest.approx(17.0)


def test_p_over_line_is_near_half_at_the_mean_for_moderate_dispersion():
    # NB mean 40, line 39.5 → P(Y >= 40) slightly under 0.5 for a right-skewed count.
    prob = p_over_line(40.0, 0.2, 39.5)
    assert 0.35 < prob < 0.55


def test_feature_frame_lags_yards_before_the_next_game():
    stats = pd.DataFrame(
        [
            {"game_id": "g1", "season": 2022, "week": 1, "player_id": "p1", "player_display_name": "A", "position": "WR", "team": "DEN", "opponent_team": "KC", "targets": 8, "receiving_yards": 80, "receptions": 5, "attempts": 0, "carries": 0, "target_share": 0.25, "offense_pct": 0.9},
            {"game_id": "g2", "season": 2022, "week": 2, "player_id": "p1", "player_display_name": "A", "position": "WR", "team": "DEN", "opponent_team": "KC", "targets": 10, "receiving_yards": 120, "receptions": 7, "attempts": 0, "carries": 0, "target_share": 0.30, "offense_pct": 0.92},
            {"game_id": "g3", "season": 2023, "week": 1, "player_id": "p1", "player_display_name": "A", "position": "WR", "team": "DEN", "opponent_team": "KC", "targets": 6, "receiving_yards": 40, "receptions": 4, "attempts": 0, "carries": 0, "target_share": 0.20, "offense_pct": 0.88},
        ]
    )
    schedule = pd.DataFrame(
        [
            {"game_id": "g1", "gameday": date(2022, 9, 11), "home_team": "DEN", "away_team": "KC", "home_score": 20, "away_score": 17, "predicted_margin": 3, "closing_total": 44},
            {"game_id": "g2", "gameday": date(2022, 9, 18), "home_team": "KC", "away_team": "DEN", "home_score": 27, "away_score": 24, "predicted_margin": 6, "closing_total": 48},
            {"game_id": "g3", "gameday": date(2023, 9, 10), "home_team": "DEN", "away_team": "KC", "home_score": 16, "away_score": 13, "predicted_margin": 2, "closing_total": 41},
        ]
    )
    features = build_feature_frame(stats, schedule)
    week1 = features[features["game_id"].eq("g1")].iloc[0]
    week3 = features[features["game_id"].eq("g3")].iloc[0]
    assert np.isnan(week1["rolling_yards_5"])
    assert week3["rolling_yards_5"] == pytest.approx(100.0)
    assert week3["games_before"] == 2
    assert week3["implied_team_points"] == pytest.approx(21.5)


def test_rolling_team_points_is_lagged_at_team_grain_not_teammate_order():
    stats = pd.DataFrame(
        [
            {"game_id": "g1", "season": 2022, "week": 1, "player_id": "p1", "player_display_name": "A", "position": "WR", "team": "DEN", "opponent_team": "KC", "targets": 8, "receiving_yards": 80, "receptions": 5, "attempts": 0, "carries": 0, "target_share": 0.25, "offense_pct": 0.9},
            {"game_id": "g1", "season": 2022, "week": 1, "player_id": "p2", "player_display_name": "B", "position": "WR", "team": "DEN", "opponent_team": "KC", "targets": 6, "receiving_yards": 40, "receptions": 4, "attempts": 0, "carries": 0, "target_share": 0.18, "offense_pct": 0.8},
            {"game_id": "g2", "season": 2022, "week": 2, "player_id": "p1", "player_display_name": "A", "position": "WR", "team": "DEN", "opponent_team": "KC", "targets": 7, "receiving_yards": 60, "receptions": 4, "attempts": 0, "carries": 0, "target_share": 0.22, "offense_pct": 0.9},
            {"game_id": "g2", "season": 2022, "week": 2, "player_id": "p2", "player_display_name": "B", "position": "WR", "team": "DEN", "opponent_team": "KC", "targets": 5, "receiving_yards": 30, "receptions": 3, "attempts": 0, "carries": 0, "target_share": 0.16, "offense_pct": 0.75},
        ]
    )
    schedule = pd.DataFrame(
        [
            {"game_id": "g1", "gameday": date(2022, 9, 11), "home_team": "DEN", "away_team": "KC", "home_score": 27, "away_score": 17, "predicted_margin": 3, "closing_total": 44},
            {"game_id": "g2", "gameday": date(2022, 9, 18), "home_team": "DEN", "away_team": "KC", "home_score": 20, "away_score": 17, "predicted_margin": 3, "closing_total": 41},
        ]
    )
    features = build_feature_frame(stats, schedule)
    week1 = features[features["game_id"].eq("g1")]
    week2 = features[features["game_id"].eq("g2")]
    assert week1["rolling_team_points"].isna().all()
    assert week2["rolling_team_points"].nunique() == 1
    assert week2["rolling_team_points"].iloc[0] == pytest.approx(27.0)


def test_opportunity_model_shrinks_extreme_ypt_toward_position_prior():
    rows = []
    for week in range(1, 12):
        rows.append(
            {
                "game_id": f"g{week}",
                "season": 2022,
                "week": week,
                "player_id": "hot",
                "player_display_name": "Hot",
                "position": "WR",
                "team": "DEN",
                "opponent_team": "KC",
                "targets": 8,
                "receiving_yards": 160,
                "receptions": 7,
                "attempts": 0,
                "carries": 0,
                "target_share": 0.22,
                "offense_pct": 0.9,
            }
        )
        rows.append(
            {
                "game_id": f"g{week}",
                "season": 2022,
                "week": week,
                "player_id": "avg",
                "player_display_name": "Avg",
                "position": "WR",
                "team": "KC",
                "opponent_team": "DEN",
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
                "season": 2022,
                "week": week,
                "player_id": "qb-den",
                "player_display_name": "QB",
                "position": "QB",
                "team": "DEN",
                "opponent_team": "KC",
                "targets": 0,
                "receiving_yards": 0,
                "receptions": 0,
                "attempts": 35,
                "carries": 8,
                "target_share": 0.0,
                "offense_pct": 1.0,
            }
        )
    stats = pd.DataFrame(rows)
    schedule = pd.DataFrame(
        [
            {
                "game_id": f"g{week}",
                "gameday": date(2022, 9, week),
                "home_team": "DEN",
                "away_team": "KC",
                "home_score": 24,
                "away_score": 20,
                "predicted_margin": 3,
                "closing_total": 44,
            }
            for week in range(1, 12)
        ]
    )
    modeled = apply_opportunity_model(build_feature_frame(stats, schedule))
    last_hot = modeled[(modeled.player_id == "hot") & (modeled.week == 11)].iloc[0]
    assert last_hot["ypt_hat"] < 18.0
    assert last_hot["ypt_hat"] > 10.0
    assert last_hot["mu"] > 20


def test_walk_forward_does_not_use_future_seasons_for_dispersion_or_mae():
    rng = np.random.default_rng(0)
    rows = []
    games = []
    for season in (2022, 2023):
        for week in range(1, 8):
            game_id = f"{season}_{week}"
            games.append(
                {
                    "game_id": game_id,
                    "gameday": date(season, 9, week),
                    "home_team": "DEN",
                    "away_team": "KC",
                    "home_score": 23,
                    "away_score": 17,
                    "predicted_margin": 4,
                    "closing_total": 45,
                }
            )
            yards = 40 + rng.integers(-10, 10)
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "player_id": "p1",
                    "player_display_name": "A",
                    "position": "WR",
                    "team": "DEN",
                    "opponent_team": "KC",
                    "targets": 7,
                    "receiving_yards": yards,
                    "receptions": 4,
                    "attempts": 0,
                    "carries": 0,
                    "target_share": 0.2,
                    "offense_pct": 0.85,
                }
            )
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "player_id": "qb",
                    "player_display_name": "QB",
                    "position": "QB",
                    "team": "DEN",
                    "opponent_team": "KC",
                    "targets": 0,
                    "receiving_yards": 0,
                    "receptions": 0,
                    "attempts": 34,
                    "carries": 3,
                    "target_share": 0.0,
                    "offense_pct": 1.0,
                }
            )
    scored, metrics = walk_forward_evaluate(
        build_feature_frame(pd.DataFrame(rows), pd.DataFrame(games)),
        eval_seasons=[2023],
    )
    assert metrics["n"] == 7
    assert scored["season"].eq(2023).all()
    assert "QB" not in set(scored["position"])
    assert metrics["mae"] < metrics["baseline_mae"] + 15
    assert metrics["roi_clv"]["n"] == 0
    assert "board_like" in metrics
    assert metrics["board_like"]["n"] <= metrics["n"]
    assert crps_nb(40, 40, 0.2) < crps_nb(40, 10, 0.2)
    assert pinball_loss([10], [10], 0.5) == 0.0


def test_grade_closing_sample_is_labeled_small_sample():
    scored = pd.DataFrame(
        [
            {
                "player_display_name": "Courtland Sutton",
                "home_team": "DEN",
                "away_team": "KC",
                "team": "DEN",
                "mu": 60.0,
                "alpha": 0.3,
                "receiving_yards": 80,
            }
        ]
    )
    report = grade_closing_sample(
        scored,
        {
            "home_team": "Denver Broncos",
            "away_team": "Kansas City Chiefs",
            "closing_sample": [
                {"player": "Courtland Sutton", "book": "draftkings", "point": 49.5, "over_price": -110, "under_price": -110}
            ],
        },
    )
    assert report["status"] == "small_sample"
    assert report["n"] == 1
    assert report["win"] == 1
    assert "small-sample" in report["label"]
