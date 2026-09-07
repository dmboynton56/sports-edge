from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.nfl_experiment import (
    CalibratedProbabilityModel,
    bootstrap_improvement_probability,
    fit_calibrator,
    fit_candidate,
    market_track_status,
    run_experiment,
)
from src.models.nfl_v2 import (
    FEATURE_COLUMNS,
    LEAGUE_DEFAULTS,
    TeamState,
    audit_nfl_feature_store,
    build_nfl_feature_store,
    dataframe_fingerprint,
    mirrored_feature_row,
    no_vig_home_probability,
    normalize_team_name,
)


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": "g1", "league": "NFL", "season": 2024, "week": 1, "gameday": "2024-09-01", "game_type": "REG", "home_team": "DEN", "away_team": "KC", "home_score": 30, "away_score": 10, "location": "Home"},
            {"game_id": "g2", "league": "NFL", "season": 2024, "week": 2, "gameday": "2024-09-08", "game_type": "REG", "home_team": "KC", "away_team": "DEN", "home_score": 21, "away_score": 20, "location": "Home"},
            {"game_id": "g3", "league": "NFL", "season": 2025, "week": 1, "gameday": "2025-09-01", "game_type": "REG", "home_team": "DEN", "away_team": "KC", "home_score": 24, "away_score": 17, "location": "Neutral"},
        ]
    )


def _stats() -> pd.DataFrame:
    rows = []
    for game, home, away in (("g1", "DEN", "KC"), ("g2", "KC", "DEN"), ("g3", "DEN", "KC")):
        rows.extend(
            [
                {"game_id": game, "team": home, "off_epa": 0.2, "def_epa_allowed": -0.1, "off_success": 0.52, "def_success_allowed": 0.41, "off_explosive": 0.15, "def_explosive_allowed": 0.09, "primary_qb_id": f"{home}-QB"},
                {"game_id": game, "team": away, "off_epa": -0.1, "def_epa_allowed": 0.2, "off_success": 0.40, "def_success_allowed": 0.51, "off_explosive": 0.08, "def_explosive_allowed": 0.14, "primary_qb_id": f"{away}-QB"},
            ]
        )
    return pd.DataFrame(rows)


def test_feature_store_is_one_row_per_game_and_excludes_current_game():
    base = build_nfl_feature_store(_schedules(), _stats())
    changed = _schedules()
    changed.loc[changed["game_id"] == "g1", ["home_score", "away_score"]] = [0, 40]
    rebuilt = build_nfl_feature_store(changed, _stats())

    assert len(base) == base["game_id"].nunique() == 3
    pd.testing.assert_series_equal(
        base.loc[base.game_id == "g1", FEATURE_COLUMNS].iloc[0],
        rebuilt.loc[rebuilt.game_id == "g1", FEATURE_COLUMNS].iloc[0],
    )
    assert base.loc[base.game_id == "g2", "margin_diff_3"].iloc[0] != rebuilt.loc[rebuilt.game_id == "g2", "margin_diff_3"].iloc[0]
    assert base.loc[base.game_id == "g2", "home_history_max_date"].iloc[0] < base.loc[base.game_id == "g2", "game_date"].iloc[0]


def test_prior_season_state_is_carried_and_shrunk_not_zero_filled():
    features = build_nfl_feature_store(_schedules(), _stats())
    week_one = features.loc[features.game_id == "g3"].iloc[0]
    assert week_one["games_before_diff"] == 0
    assert abs(week_one["margin_diff_3"]) > 0
    assert abs(week_one["margin_diff_3"]) < 20


def test_rolling_windows_shift_and_exclude_unappended_current_value():
    state = TeamState(season=2025, prior=dict(LEAGUE_DEFAULTS))
    state.current["margin"] = list(range(1, 11))
    before = state.value("margin", 3)
    assert before != state.value("margin", 10)
    assert before == pytest.approx((8 + 9 + 10) / 7)
    state.current["margin"].append(100)
    assert state.value("margin", 3) != before


def test_missing_pbp_does_not_invent_epa_observations():
    features = build_nfl_feature_store(_schedules(), pd.DataFrame())
    assert features.loc[features.game_id == "g2", "adj_off_epa_diff_3"].iloc[0] == 0.0


def test_live_feature_store_emits_unplayed_without_updating_state():
    schedules = _schedules()
    future = schedules.iloc[[1]].copy()
    future["game_id"] = "g4"
    future["week"] = 4
    future["gameday"] = "2025-09-15"
    future[["home_score", "away_score"]] = None
    schedules = pd.concat([schedules, future], ignore_index=True)

    features = build_nfl_feature_store(schedules, _stats(), include_unplayed=True)
    live = features.loc[features.game_id == "g4"].iloc[0]
    assert pd.isna(live["home_win"])
    assert pd.isna(live["home_margin"])
    assert live["home_history_max_date"] < live["game_date"]
    assert len(features) == 4


def test_known_future_schedule_advances_rest_without_advancing_performance():
    schedules = _schedules().iloc[[0]].copy()
    schedules["home_score"] = np.nan
    schedules["away_score"] = np.nan
    second = schedules.copy()
    second["game_id"] = "g2-future"
    second["week"] = 2
    second["gameday"] = "2024-09-08"
    second["away_team"] = "LV"
    frame = build_nfl_feature_store(pd.concat([schedules, second]), pd.DataFrame(), include_unplayed=True)
    week_two = frame.loc[frame.game_id == "g2-future"].iloc[0]
    assert week_two["games_before_diff"] == 0
    assert week_two["rest_diff"] == pytest.approx(-7.0)


def test_duplicate_game_or_team_game_grain_fails_closed():
    duplicated_schedule = pd.concat([_schedules(), _schedules().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate season/game"):
        build_nfl_feature_store(duplicated_schedule, _stats())
    duplicated_stats = pd.concat([_stats(), _stats().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        build_nfl_feature_store(_schedules(), duplicated_stats)


def test_fingerprint_is_row_order_deterministic():
    frame = build_nfl_feature_store(_schedules(), _stats())
    assert dataframe_fingerprint(frame) == dataframe_fingerprint(frame.sample(frac=1, random_state=4))


def test_market_no_vig_and_fail_closed_status():
    assert no_vig_home_probability(-110, -110) == pytest.approx(0.5)
    assert no_vig_home_probability(None, -110) is None
    data = pd.DataFrame({"closing_market_home_prob": [0.5] * 99 + [None]})
    status = market_track_status(data, {"market_track": {"timestamp_safe": False, "minimum_coverage": 0.9}})
    assert status["coverage"] == 0.0
    assert "market_as_of_ts" in status["missing_columns"]
    assert status["enabled"] is False


def test_market_track_requires_joint_coverage_and_pre_kickoff_timestamps():
    data = pd.DataFrame(
        {
            "market_home_prob": [0.55] * 10,
            "market_spread": [-2.5] * 10,
            "market_as_of_ts": ["2026-09-10T18:00:00Z"] * 10,
            "kickoff_ts": ["2026-09-10T20:00:00Z"] * 10,
        }
    )
    config = {"market_track": {"timestamp_safe": True, "minimum_coverage": 0.9}}
    assert market_track_status(data, config)["enabled"] is True
    data.loc[0, "market_as_of_ts"] = data.loc[0, "kickoff_ts"]
    status = market_track_status(data, config)
    assert status["enabled"] is False
    assert status["timestamp_violations"] == 1


def test_historical_team_aliases_join_to_current_names():
    assert normalize_team_name("oak") == "LV"
    schedules = _schedules().iloc[[0]].copy()
    schedules.loc[:, "home_team"] = "OAK"
    stats = _stats().loc[_stats()["game_id"] == "g1"].copy()
    stats.loc[stats["team"] == "DEN", "team"] = "LV"
    features = build_nfl_feature_store(schedules, stats)
    assert features.iloc[0]["home_team"] == "LV"


def test_neutral_site_swap_symmetry_for_interpretable_model():
    rng = np.random.default_rng(5)
    train = pd.DataFrame(rng.normal(size=(120, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    train["home_field"] = rng.integers(0, 2, size=len(train))
    train["home_win"] = (train["margin_diff_3"] + 0.3 * train["home_field"] > 0).astype(int)
    model = fit_candidate({"kind": "logistic", "C": 1.0}, train, task="probability")
    neutral = train.iloc[0].copy()
    neutral["home_field"] = 0.0
    mirrored = pd.DataFrame([mirrored_feature_row(neutral)])
    p = model.predict_probability(pd.DataFrame([neutral]))[0]
    swapped = model.predict_probability(mirrored)[0]
    assert p + swapped == pytest.approx(1.0)
    calibrator = fit_calibrator("isotonic", np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    calibrated = CalibratedProbabilityModel(model, calibrator)
    assert calibrated.predict_probability(pd.DataFrame([neutral]))[0] + calibrated.predict_probability(mirrored)[0] == pytest.approx(1.0)


def test_calibration_and_bootstrap_are_deterministic():
    y = np.array([0, 0, 1, 1, 1, 0])
    raw = np.array([0.2, 0.3, 0.6, 0.7, 0.8, 0.4])
    calibrated = fit_calibrator("sigmoid", y, raw).predict(raw)
    assert np.all((calibrated > 0) & (calibrated < 1))
    first = bootstrap_improvement_probability(y, raw, np.full(len(y), 0.5), samples=100, seed=8)
    second = bootstrap_improvement_probability(y, raw, np.full(len(y), 0.5), samples=100, seed=8)
    assert first == second


def test_audit_detects_future_history():
    frame = build_nfl_feature_store(_schedules(), _stats())
    frame.loc[0, "home_history_max_date"] = frame.loc[0, "game_date"]
    audit = audit_nfl_feature_store(frame)
    assert audit["ready"] is False
    assert audit["leakage_rows"] == 1


def test_consumed_locked_test_is_closed_by_default(tmp_path):
    config = tmp_path / "consumed.yaml"
    config.write_text("experiment_state:\n  locked_test_consumed: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="locked test has already been consumed"):
        run_experiment(config, tmp_path / "missing.parquet", tmp_path / "runs")
