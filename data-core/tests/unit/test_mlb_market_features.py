import math

import pandas as pd

from src.features.mlb_market_features import STARTER_PRIORS, build_mlb_market_features
from src.models.mlb_winner_model import default_feature_columns


def _game(game_pk, date, home_score=5, away_score=3, season=None, home_pitcher=101, away_pitcher=202, venue_id=10):
    timestamp = pd.Timestamp(date)
    return {
        "game_pk": game_pk,
        "season": season or timestamp.year,
        "game_date": timestamp.normalize(),
        "game_datetime": timestamp,
        "home_team_id": 1,
        "home_team": "Home",
        "home_probable_pitcher_id": home_pitcher,
        "home_probable_pitcher": "Home Starter",
        "away_team_id": 2,
        "away_team": "Away",
        "away_probable_pitcher_id": away_pitcher,
        "away_probable_pitcher": "Away Starter",
        "venue_id": venue_id,
        "venue_name": "Park",
        "home_score": home_score,
        "away_score": away_score,
        "home_win": int(home_score > away_score),
        "run_diff": home_score - away_score,
    }


def _boxscore(game_pk, home_pitcher=101, away_pitcher=202, home_ks=6, away_ks=4, wind_dir="Out To CF", wind_mph=8):
    return {
        "game_pk": game_pk,
        "home_actual_starter_id": home_pitcher,
        "home_actual_starter": "Home Starter",
        "home_starter_outs": 18,
        "home_starter_pitches": 90,
        "home_starter_earned_runs": 2,
        "home_starter_strikeouts": home_ks,
        "home_starter_walks": 2,
        "away_actual_starter_id": away_pitcher,
        "away_actual_starter": "Away Starter",
        "away_starter_outs": 15,
        "away_starter_pitches": 85,
        "away_starter_earned_runs": 3,
        "away_starter_strikeouts": away_ks,
        "away_starter_walks": 1,
        "home_team_strikeouts": 9,
        "home_team_walks": 4,
        "home_team_hits": 8,
        "home_team_home_runs": 1,
        "home_team_pitching_strikeouts": home_ks,
        "away_team_strikeouts": 7,
        "away_team_walks": 3,
        "away_team_hits": 6,
        "away_team_home_runs": 0,
        "away_team_pitching_strikeouts": away_ks,
        "temp_f": 72,
        "weather_condition": "Clear",
        "wind_mph": wind_mph,
        "wind_dir": wind_dir,
        "first_pitch": "1:10 PM.",
        "attendance": 20000,
        "game_duration": "2:45",
    }


def _build(games, boxscores, venue_meta=None):
    return build_mlb_market_features(
        pd.DataFrame(games),
        pd.DataFrame(boxscores),
        venue_meta or {"10": {"roofType": "Open", "elevation": 500}},
        min_prior_games=0,
    )


def test_pitcher_features_use_only_prior_actual_starts():
    features = _build(
        [_game(1, "2025-04-01T13:00:00"), _game(2, "2025-04-07T13:00:00")],
        [_boxscore(1, home_ks=6), _boxscore(2, home_ks=12)],
    )

    first, second = features.iloc[0], features.iloc[1]
    assert first["home_starter_has_history"] == 0
    assert first["home_starter_k9_last5"] == STARTER_PRIORS["k9_last5"]
    assert second["home_starter_has_history"] == 1
    assert second["home_starter_k9_last5"] == 9.0  # 6 K over 18 outs
    assert second["home_starter_pitches_last3_avg"] == 90
    assert second["home_starter_career_starts_prior"] == 1
    assert second["home_starter_days_since_last_start"] == 6


def test_pitcher_history_carries_across_seasons():
    features = _build(
        [
            _game(1, "2024-09-28T13:00:00", season=2024),
            _game(2, "2025-04-02T13:00:00", season=2025),
        ],
        [_boxscore(1, home_ks=8), _boxscore(2, home_ks=5)],
    )

    april = features.iloc[1]
    assert april["home_games_played"] == 0  # v1 state still resets by season
    assert april["home_starter_has_history"] == 1
    assert april["home_starter_k9_last5"] == 12.0
    assert april["home_starter_career_starts_prior"] == 1


def test_default_feature_columns_exclude_v2_labels_and_postgame_passthroughs():
    features = _build([_game(1, "2025-04-01T13:00:00")], [_boxscore(1)])
    selected = set(default_feature_columns(features))
    excluded = {
        "total_runs",
        "home_cover_15",
        "home_starter_ks_label",
        "away_starter_ks_label",
        "home_probable_matches_actual",
        "away_probable_matches_actual",
        "attendance",
        "game_duration",
        "first_pitch",
        "weather_condition",
        "wind_dir",
        "home_actual_starter_id",
        "home_starter_outs",
        "home_starter_strikeouts",
        "home_team_strikeouts",
        "home_team_walks",
        "home_team_hits",
        "home_team_home_runs",
        "home_team_pitching_strikeouts",
        "away_team_strikeouts",
        "away_team_walks",
        "away_team_hits",
        "away_team_home_runs",
        "away_team_pitching_strikeouts",
    }
    assert selected.isdisjoint(excluded)
    assert "home_starter_pitches_last3_avg" in selected
    assert "away_starter_outs_per_start_last5" in selected


def test_weather_derivation_for_dome_and_wind_directions():
    directions = [("Out To CF", 1, 0, 0), ("In From LF", 0, 1, 0), ("L To R", 0, 0, 1), ("None", 0, 0, 0)]
    games = [_game(index, f"2025-04-0{index}T19:00:00") for index in range(1, 5)]
    boxscores = [
        _boxscore(index, wind_dir=direction, wind_mph=0 if direction == "None" else 8)
        for index, (direction, *_expected) in enumerate(directions, start=1)
    ]
    features = _build(games, boxscores, {"10": {"roofType": "Dome", "elevation": 42}})

    assert features["is_dome_or_closed"].eq(1).all()
    assert features["is_day_game"].eq(1).all()  # parsed from local first_pitch text
    assert features["elevation"].eq(42).all()
    for row, (_direction, wind_out, wind_in, wind_cross) in zip(features.itertuples(), directions):
        assert (row.wind_out, row.wind_in, row.wind_cross) == (wind_out, wind_in, wind_cross)


def test_total_and_cover_labels_and_probable_match():
    features = _build(
        [
            _game(1, "2025-04-01T13:00:00", home_score=5, away_score=3),
            _game(2, "2025-04-02T13:00:00", home_score=4, away_score=3),
        ],
        [_boxscore(1), _boxscore(2, home_pitcher=999)],
    )

    assert features["total_runs"].tolist() == [8, 7]
    assert features["home_cover_15"].tolist() == [1, 0]
    assert bool(features.iloc[0]["home_probable_matches_actual"])
    assert not bool(features.iloc[1]["home_probable_matches_actual"])
    assert features["home_starter_ks_label"].tolist() == [6.0, 6.0]
    assert math.isclose(features.iloc[1]["combined_expected_total"], 8.0)
