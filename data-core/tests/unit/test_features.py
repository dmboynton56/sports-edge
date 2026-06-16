import pandas as pd
import pytest
from src.features.form_metrics import (
    add_form_features_nfl,
    compute_possessions,
    compute_rolling_epa,
    compute_rolling_net_rating,
)

def test_compute_possessions():
    df = pd.DataFrame({
        'FGA': [80, 90],
        'FTA': [20, 25],
        'OREB': [10, 12],
        'TOV': [15, 14]
    })
    # Formula: FGA + 0.44 * FTA - OREB + TOV
    # Row 0: 80 + 8.8 - 10 + 15 = 93.8
    # Row 1: 90 + 11.0 - 12 + 14 = 103.0
    
    poss = compute_possessions(df)
    assert len(poss) == 2
    assert poss[0] == pytest.approx(93.8)
    assert poss[1] == pytest.approx(103.0)

def test_compute_rolling_net_rating_simple():
    # Setup mock game logs
    dates = pd.date_range(start='2024-01-01', periods=5)
    logs = pd.DataFrame({
        'team': ['BOS'] * 5,
        'game_date': dates,
        'pts': [110, 120, 115, 105, 130],
        'points_allowed': [100, 110, 105, 115, 120],
        'FGA': [90] * 5,
        'FTA': [20] * 5,
        'OREB': [10] * 5,
        'TOV': [10] * 5
        # Possessions = 90 + 8.8 - 10 + 10 = 98.8 per game
    })
    
    # Target date is after the first 3 games
    target_date = pd.Timestamp('2024-01-04') # Should look at Jan 1, 2, 3
    
    rating = compute_rolling_net_rating('BOS', target_date, logs, window=3)
    
    # Total Pts Scored (Games 1,2,3) = 110+120+115 = 345
    # Total Pts Allowed = 100+110+105 = 315
    # Total Poss = 98.8 * 3 = 296.4
    # Net Rating = (345/296.4)*100 - (315/296.4)*100 = (30/296.4)*100 ~= 10.12
    
    assert rating is not None
    assert rating == pytest.approx(10.12, rel=1e-2)

def test_compute_rolling_net_rating_insufficient_data():
    logs = pd.DataFrame({
        'team': ['BOS'],
        'game_date': [pd.Timestamp('2024-01-01')],
        'pts': [100],
        'points_allowed': [90]
    })
    
    # Window is 3, but only 1 game exists
    rating = compute_rolling_net_rating('BOS', pd.Timestamp('2024-01-05'), logs, window=3)
    assert rating is None


def test_add_form_features_nfl_vectorized_matches_scalar_and_avoids_current_game_leakage():
    play_by_play = pd.DataFrame(
        [
            {"game_id": "G1", "game_date": "2024-01-01", "posteam": "AAA", "defteam": "BBB", "epa": 0.10},
            {"game_id": "G1", "game_date": "2024-01-01", "posteam": "BBB", "defteam": "AAA", "epa": -0.20},
            {"game_id": "G2", "game_date": "2024-01-08", "posteam": "AAA", "defteam": "BBB", "epa": 0.20},
            {"game_id": "G2", "game_date": "2024-01-08", "posteam": "BBB", "defteam": "AAA", "epa": -0.10},
            {"game_id": "G3", "game_date": "2024-01-15", "posteam": "AAA", "defteam": "BBB", "epa": 0.30},
            {"game_id": "G3", "game_date": "2024-01-15", "posteam": "BBB", "defteam": "AAA", "epa": 0.00},
        ]
    )
    games = pd.DataFrame(
        [
            {"game_id": "G3", "game_date": "2024-01-15", "home_team": "AAA", "away_team": "BBB"},
            {"game_id": "G4", "game_date": "2024-01-22", "home_team": "AAA", "away_team": "BBB"},
        ]
    )

    features = add_form_features_nfl(games, play_by_play, windows=[2, 3])

    for row in features.itertuples(index=False):
        game_date = pd.Timestamp(row.game_date)
        assert row.form_home_epa_off_2 == pytest.approx(
            compute_rolling_epa("AAA", game_date, play_by_play, window=2, side="offense")
        )
        assert row.form_home_epa_def_2 == pytest.approx(
            compute_rolling_epa("AAA", game_date, play_by_play, window=2, side="defense")
        )

    jan_15 = features.iloc[0]
    assert jan_15.form_home_epa_off_2 == pytest.approx(0.15)
    assert jan_15.form_home_epa_def_2 == pytest.approx(-0.15)
    assert pd.isna(jan_15.form_home_epa_off_3)

    jan_22 = features.iloc[1]
    assert jan_22.form_home_epa_off_2 == pytest.approx(0.25)
    assert jan_22.form_home_epa_def_2 == pytest.approx(-0.05)
    assert jan_22.form_home_epa_off_3 == pytest.approx(0.20)
