import pandas as pd
import pytest

from src.features.injury_impact import (
    add_injury_adjustment_features,
    apply_injury_adjustments_to_features,
    build_game_injury_adjustments,
    estimate_nba_player_rating_impact,
    estimate_nfl_player_epa_impact,
)


def test_estimate_nfl_player_epa_impact_uses_replacement_and_play_share():
    pbp = pd.DataFrame(
        [
            {"posteam": "DEN", "defteam": "KC", "passer_player_name": "B.Nix", "epa": 0.30},
            {"posteam": "DEN", "defteam": "KC", "passer_player_name": "B.Nix", "epa": 0.10},
            {"posteam": "DEN", "defteam": "KC", "passer_player_name": "J.Backup", "epa": -0.05},
        ]
    )

    impact = estimate_nfl_player_epa_impact(
        pbp,
        "B.Nix",
        position="QB",
        replacement_epa=0.02,
        play_share=0.65,
    )

    assert impact["available"] is True
    assert impact["team"] == "DEN"
    assert impact["player_value"] == pytest.approx(0.20)
    assert impact["replacement_value"] == pytest.approx(0.02)
    assert impact["team_delta"] == pytest.approx(-0.117)
    assert impact["sample_size"] == 2


def test_estimate_nfl_player_epa_impact_returns_empty_when_no_rows_match():
    pbp = pd.DataFrame(
        [{"posteam": "DEN", "passer_player_name": "B.Nix", "epa": 0.30}]
    )

    impact = estimate_nfl_player_epa_impact(pbp, "P.Mahomes", replacement_epa=0.01)

    assert impact["available"] is False
    assert impact["team_delta"] == 0.0
    assert impact["sample_size"] == 0


def test_estimate_nba_player_rating_impact_weights_by_minutes_share():
    player_logs = pd.DataFrame(
        [
            {
                "team": "BOS",
                "player_name": "J.Tatum",
                "net_rating": 8.0,
                "minutes_share": 0.32,
            }
        ]
    )

    impact = estimate_nba_player_rating_impact(
        player_logs,
        "J.Tatum",
        replacement_rating=-2.0,
    )

    assert impact["available"] is True
    assert impact["team"] == "BOS"
    assert impact["player_value"] == pytest.approx(8.0)
    assert impact["team_delta"] == pytest.approx(-3.2)


def test_build_game_injury_adjustments_aggregates_home_and_away_metrics():
    adjustments = build_game_injury_adjustments(
        {"home_team": "BOS", "away_team": "DEN"},
        [
            {"available": True, "team": "BOS", "metric_name": "net_rating", "team_delta": -3.2},
            {"available": True, "team": "DEN", "metric_name": "epa_per_play", "team_delta": -0.12},
            {"available": False, "team": "BOS", "metric_name": "net_rating", "team_delta": -9.0},
        ],
    )

    assert adjustments["home_injury_net_rating_delta"] == pytest.approx(-3.2)
    assert adjustments["away_injury_epa_delta"] == pytest.approx(-0.12)
    assert adjustments["home_injured_players"] == 1
    assert adjustments["away_injured_players"] == 1


def test_apply_injury_adjustments_to_features_updates_expected_columns():
    features = pd.DataFrame(
        [
            {
                "form_home_epa_off_3": 0.10,
                "form_home_epa_def_3": 0.05,
                "form_away_epa_off_3": 0.08,
            }
        ]
    )

    adjusted = apply_injury_adjustments_to_features(
        features,
        team="DEN",
        impact={"team_delta": -0.12},
        league="NFL",
        is_home=True,
    )

    assert adjusted["form_home_epa_off_3"].iloc[0] == pytest.approx(-0.02)
    assert adjusted["form_home_epa_def_3"].iloc[0] == pytest.approx(0.05)
    assert adjusted["form_away_epa_off_3"].iloc[0] == pytest.approx(0.08)
    assert adjusted["DEN_injury_delta"].iloc[0] == pytest.approx(-0.12)


def test_add_injury_adjustment_features_matches_by_game_id_and_recomputes_nfl_diff():
    games = pd.DataFrame(
        [
            {
                "game_id": "game-1",
                "game_date": "2026-09-10",
                "home_team": "DEN",
                "away_team": "KC",
                "form_home_epa_off_3": 0.10,
                "form_away_epa_off_3": 0.08,
                "form_epa_off_diff_3": 0.02,
            }
        ]
    )
    impacts = pd.DataFrame(
        [
            {
                "league": "NFL",
                "game_id": "game-1",
                "team": "DEN",
                "player_name": "B.Nix",
                "metric_name": "epa_per_play",
                "team_delta": -0.12,
            }
        ]
    )

    adjusted = add_injury_adjustment_features(games, impacts, league="NFL")

    assert adjusted["home_injury_epa_delta"].iloc[0] == pytest.approx(-0.12)
    assert adjusted["home_injured_players"].iloc[0] == 1
    assert adjusted["form_home_epa_off_3"].iloc[0] == pytest.approx(-0.02)
    assert adjusted["form_epa_off_diff_3"].iloc[0] == pytest.approx(-0.10)


def test_add_injury_adjustment_features_matches_by_date_team_for_nba():
    games = pd.DataFrame(
        [
            {
                "game_date": "2026-01-02",
                "home_team": "BOS",
                "away_team": "NYK",
                "form_home_net_rating_3": 6.0,
                "form_away_net_rating_3": 2.0,
                "form_net_rating_diff_3": 4.0,
            }
        ]
    )
    impacts = pd.DataFrame(
        [
            {
                "league": "NBA",
                "game_date": "2026-01-02",
                "team": "BOS",
                "player_name": "J.Tatum",
                "metric_name": "net_rating",
                "team_delta": -3.2,
            }
        ]
    )

    adjusted = add_injury_adjustment_features(games, impacts, league="NBA")

    assert adjusted["home_injury_net_rating_delta"].iloc[0] == pytest.approx(-3.2)
    assert adjusted["form_home_net_rating_3"].iloc[0] == pytest.approx(2.8)
    assert adjusted["form_net_rating_diff_3"].iloc[0] == pytest.approx(0.8)
