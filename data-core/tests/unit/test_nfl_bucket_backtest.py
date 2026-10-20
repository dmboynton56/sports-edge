from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.nfl_bucket_backtest import (
    Rule,
    annotate_predictions,
    bootstrap_roi,
    ci_excludes_zero,
    define_rules,
    evaluate_rules,
    frozen_isotonic_map_diagnostics,
    grade_ats,
    grade_moneyline,
    home_prob_bucket,
    in_toss_up_band,
    line_shopping_note,
    load_live_specs,
    model_side,
    model_side_edge,
    score_rule_rows,
    season_holds,
    strategy_verdict,
    train_walk_forward_bundle,
    walk_forward_predictions,
)
from src.models.nfl_live import DEFAULT_ARTIFACT, LIVE_MODEL_VERSION
from src.models.nfl_v2 import FEATURE_COLUMNS


def test_toss_up_and_mirror_buckets_are_partitioned():
    assert in_toss_up_band(0.40)
    assert in_toss_up_band(0.60)
    assert not in_toss_up_band(0.399)
    assert not in_toss_up_band(0.601)
    assert home_prob_bucket(0.19) == "0-20"
    assert home_prob_bucket(0.20) == "20-30"
    assert home_prob_bucket(0.35) == "30-40"
    assert home_prob_bucket(0.65) == "60-70"
    assert home_prob_bucket(0.80) == "70-80"
    assert home_prob_bucket(0.801) == "80+"
    assert model_side(0.5) is None
    assert model_side(0.51) == "home"
    assert model_side(0.49) == "away"


def test_ats_and_ml_grade_home_away_push_and_tie():
    assert grade_ats(7, -3.0, "home") == "win"
    assert grade_ats(7, -3.0, "away") == "loss"
    assert grade_ats(3, -3.0, "home") == "push"
    assert grade_ats(0, 0.0, "away") == "push"
    assert grade_moneyline(24, 17, "home") == "win"
    assert grade_moneyline(17, 24, "home") == "loss"
    assert grade_moneyline(20, 20, "away") == "push"


def test_model_side_edge_is_on_the_model_favorite():
    assert model_side_edge(0.72, 0.65) == pytest.approx(0.07)
    assert model_side_edge(0.30, 0.40) == pytest.approx(0.10)
    assert model_side_edge(0.62, 0.75) == pytest.approx(-0.13)


def _game(**overrides) -> dict:
    row = {
        "season": 2024,
        "week": 1,
        "home_team": "KC",
        "away_team": "DEN",
        "home_score": 27,
        "away_score": 20,
        "actual_margin": 7,
        "my_home_win_prob": 0.55,
        "home_spread": -3.0,
        "home_spread_odds": -110,
        "away_spread_odds": -110,
        "home_moneyline": -150,
        "away_moneyline": 130,
        "market_home_prob": 0.58,
        "model_side_edge": -0.03,
        "home_prob_bucket": "40-60",
    }
    row.update(overrides)
    return row


def test_q1_rules_compare_model_dog_and_favorite():
    frame = pd.DataFrame(
        [
            _game(my_home_win_prob=0.55, home_spread=-3.0, home_moneyline=-150, away_moneyline=130, actual_margin=7, home_score=27, away_score=20),
            _game(my_home_win_prob=0.44, home_spread=2.5, home_moneyline=120, away_moneyline=-140, actual_margin=-10, home_score=14, away_score=24),
        ]
    )
    frame = annotate_predictions(frame)
    model_ml = next(rule for rule in define_rules() if rule.rule_id == "q1_toss_up_model_ml")
    dog_ml = next(rule for rule in define_rules() if rule.rule_id == "q1_toss_up_book_dog_ml")
    fav_ats = next(rule for rule in define_rules() if rule.rule_id == "q1_toss_up_book_fav_ats_minus_110")
    model_rows = score_rule_rows(frame, model_ml)
    assert model_rows["bet_side"].tolist() == ["home", "away"]
    assert model_rows["bet_result"].tolist() == ["win", "win"]
    dog_rows = score_rule_rows(frame, dog_ml)
    assert dog_rows["bet_side"].tolist() == ["away", "home"]
    fav_rows = score_rule_rows(frame, fav_ats)
    assert fav_rows["bet_side"].tolist() == ["home", "away"]


def test_roi_bootstrap_and_multiple_comparison_flag():
    wins = pd.DataFrame({"bet_result": ["win"] * 40 + ["loss"] * 10, "bet_units": [100 / 110] * 40 + [-1.0] * 10, "season": [2022] * 25 + [2023] * 25})
    from src.models.nfl_bucket_backtest import summarize_bets

    block = summarize_bets(wins)
    assert block["wins"] == 40
    assert block["roi"] > 0
    interval = bootstrap_roi([1.0] * 30 + [-1.0] * 2, samples=200, seed=1)
    assert interval["lower_95"] > 0
    assert ci_excludes_zero({"bootstrap": interval})
    summary = {
        "n": 80,
        "roi": 0.2,
        "bootstrap": interval,
        "by_season": [
            {"season": 2022, "n": 30, "roi": 0.1},
            {"season": 2023, "n": 30, "roi": 0.12},
            {"season": 2024, "n": 30, "roi": 0.08},
        ],
    }
    oos = strategy_verdict(summary, n_tests=1, sample_kind="out_of_sample")
    assert oos["qualifies_as_strategy"] is True
    insample = strategy_verdict(summary, n_tests=1, sample_kind="in_sample")
    assert insample["qualifies_as_strategy"] is False
    many = strategy_verdict(summary, n_tests=10_000, sample_kind="out_of_sample")
    assert many["qualifies_as_strategy"] is False
    mixed = dict(summary)
    mixed["by_season"] = [
        {"season": 2022, "n": 30, "roi": 0.2},
        {"season": 2023, "n": 30, "roi": -0.2},
        {"season": 2024, "n": 30, "roi": 0.1},
    ]
    assert season_holds(mixed)["holds"] is False


def test_walk_forward_never_trains_on_the_test_season():
    rng = np.random.default_rng(7)
    rows = []
    seen_train_max = []
    for season in (2020, 2021, 2022):
        for index in range(40):
            row = {column: float(rng.normal()) for column in FEATURE_COLUMNS}
            row.update(
                {
                    "game_id": f"{season}_{index}",
                    "season": season,
                    "week": (index % 16) + 1,
                    "game_type": "REG",
                    "home_team": "KC",
                    "away_team": "DEN",
                    "home_score": 24,
                    "away_score": 17 if rng.random() > 0.4 else 28,
                    "home_win": 1,
                    "home_margin": 7.0,
                    "home_spread": -3.0,
                    "home_moneyline": -150,
                    "away_moneyline": 130,
                }
            )
            row["home_win"] = int(row["home_score"] > row["away_score"])
            row["home_margin"] = float(row["home_score"] - row["away_score"])
            rows.append(row)
    frame = pd.DataFrame(rows)

    def predict_season(train, specs, season):
        seen_train_max.append(int(train["season"].max()))
        assert int(train["season"].max()) < season
        test = frame[frame["season"] == season].copy()
        test["my_home_win_prob"] = 0.55
        test["predicted_margin"] = 2.0
        test["trained_through"] = int(train["season"].max())
        test["calibration"] = "none"
        test["calibration_mode"] = "test"
        return test

    predicted = walk_forward_predictions(
        frame,
        test_seasons=[2021, 2022],
        specs={"probability_spec": {}, "margin_spec": {}, "calibration_method": "isotonic"},
        predict_season=predict_season,
    )
    assert seen_train_max == [2020, 2021]
    assert sorted(predicted["season"].unique().tolist()) == [2021, 2022]
    assert predicted["sample_kind"].unique().tolist() == ["out_of_sample"]


def test_single_prior_season_skips_oof_isotonic():
    rng = np.random.default_rng(3)
    train = pd.DataFrame(rng.normal(size=(80, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    train["home_field"] = 1.0
    train["season"] = 2020
    train["home_win"] = (train["margin_diff_3"] > 0).astype(int)
    train["home_margin"] = train["margin_diff_3"]
    bundle = train_walk_forward_bundle(
        train,
        probability_spec={"kind": "logistic", "C": 1.0},
        margin_spec={"kind": "ridge", "alpha": 10.0},
        calibration_method="isotonic",
    )
    assert bundle["calibration"] == "none"
    assert bundle["calibration_mode"] == "single_prior_season_no_oof"
    assert bundle["trained_through"] == 2020


def test_annotate_rebuilds_actual_margin_for_ats_reload():
    frame = pd.DataFrame(
        [
            _game(actual_margin=None, home_score=27, away_score=20, home_spread=-3.0),
            _game(actual_margin=None, home_score=14, away_score=24, home_spread=2.5, my_home_win_prob=0.44),
        ]
    )
    frame = frame.drop(columns=["actual_margin"])
    frame = annotate_predictions(frame)
    rule = next(item for item in define_rules() if item.rule_id == "q1_toss_up_model_ats_minus_110")
    scored = score_rule_rows(frame, rule)
    assert scored["bet_result"].tolist() == ["win", "win"]


def test_evaluate_rules_labels_oos_and_counts_tests():
    frame = pd.DataFrame([_game() for _ in range(8)])
    frame = annotate_predictions(frame)
    rules = [
        Rule("q1_toss_up_model_ml", "q1", "toss_up_40_60", "40-60", "model", "ml", "model ml"),
        Rule("q2_edge_3_ml", "q2", "model_minus_book", "edge>=3", "model", "ml", "edge ml"),
    ]
    summaries = evaluate_rules(frame, rules, sample_kind="out_of_sample")
    assert len(summaries) == 2
    assert summaries[0]["verdict"]["bonferroni_n_tests"] == 2
    assert summaries[0]["sample_kind"] == "out_of_sample"
    note = line_shopping_note(frame)
    assert note["can_measure_1pt5_disagreement"] is False
    assert note["multi_book_rows"] == 0


def _clustered_seasons(rng, seasons=(2020, 2021, 2022), n=90):
    rows = []
    for season in seasons:
        for index in range(n):
            high = index % 2 == 0
            won = rng.random() < (0.78 if high else 0.22)
            row = {column: float(rng.normal()) for column in FEATURE_COLUMNS}
            row["margin_diff_3"] = 1.8 if high else -1.8
            row["home_field"] = 1.0
            row.update(
                {
                    "game_id": f"{season}_{index}",
                    "season": season,
                    "week": (index % 17) + 1,
                    "game_type": "REG",
                    "home_team": "KC",
                    "away_team": "DEN",
                    "home_win": int(won),
                    "home_margin": 7.0 if won else -7.0,
                    "home_score": 27 if won else 17,
                    "away_score": 17 if won else 27,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_auto_walk_forward_fills_mid_band_that_unregularized_isotonic_skips():
    rng = np.random.default_rng(9)
    train = _clustered_seasons(rng, seasons=(2020, 2021), n=100)
    test = _clustered_seasons(rng, seasons=(2022,), n=80)
    specs = {
        "kind": "logistic",
        "C": 1.0,
        "features": list(FEATURE_COLUMNS),
    }
    margin = {"kind": "ridge", "alpha": 10.0, "features": list(FEATURE_COLUMNS)}
    before = DEFAULT_ARTIFACT.stat().st_mtime_ns
    isotonic = train_walk_forward_bundle(
        train, probability_spec=specs, margin_spec=margin, calibration_method="isotonic"
    )
    auto = train_walk_forward_bundle(
        train, probability_spec=specs, margin_spec=margin, calibration_method="auto"
    )
    assert DEFAULT_ARTIFACT.stat().st_mtime_ns == before
    iso_p = np.asarray(isotonic["probability_model"].predict_probability(test), dtype=float)
    auto_p = np.asarray(auto["probability_model"].predict_probability(test), dtype=float)
    iso_band = int(((iso_p >= 0.40) & (iso_p <= 0.60)).sum())
    auto_band = int(((auto_p >= 0.40) & (auto_p <= 0.60)).sum())
    assert auto["calibration"] != "isotonic" or auto_band >= iso_band
    if iso_band == 0:
        assert auto_band > 0 or auto["calibration"] == "none"
        if auto["calibration"] == "none":
            assert len(np.unique(np.round(auto_p, 5))) > len(np.unique(np.round(iso_p, 5)))


def test_frozen_live_artifact_is_still_coarse_unregularized_isotonic():
    specs = load_live_specs()
    assert specs["calibration_method"] == "isotonic"
    payload = frozen_isotonic_map_diagnostics()
    assert payload["model_version"] == LIVE_MODEL_VERSION
    assert payload["is_unregularized_isotonic"] is True
    assert payload["n_thresholds"] >= 20
    assert payload["unique_y"] <= 20
    assert payload["grid_40_60_unique"] >= 1


def test_define_rules_covers_requested_questions():
    rules = define_rules()
    ids = {rule.rule_id for rule in rules}
    assert "q1_toss_up_model_ats_minus_110" in ids
    assert "q1_toss_up_book_dog_ml" in ids
    assert "q1_toss_up_book_fav_ats_actual" in ids
    assert "q2_under_40_book_fav_ml" in ids
    assert "q2_80+_model_ats_minus_110" in ids
    assert "q2_0-20_model_ml" in ids
    assert "q2_edge_8_ml" in ids
    assert "q2_edge_5_in_toss_up_ats_minus_110" in ids
    assert len(rules) >= 40
