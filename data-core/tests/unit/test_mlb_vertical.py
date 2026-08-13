from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluation.mlb_vertical import (
    attach_moneyline_edges,
    normalize_moneyline_sources,
    profile_feature_store,
)


def test_moneyline_sources_prioritize_fixture_prices(tmp_path):
    checkbest = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "home_moneyline": [-110, 120],
            "away_moneyline": [-110, -140],
            "source": ["checkbestodds", "checkbestodds"],
        }
    )
    oddspapi = pd.DataFrame(
        {
            "game_pk": [1],
            "home_moneyline": [-105],
            "away_moneyline": [-115],
            "book": ["pinnacle"],
            "snapshot_ts": ["2026-05-01T18:00:00Z"],
        }
    )
    checkbest_path = tmp_path / "mlb_checkbestodds.csv"
    oddspapi_path = tmp_path / "mlb_oddspapi.csv"
    checkbest.to_csv(checkbest_path, index=False)
    oddspapi.to_csv(oddspapi_path, index=False)

    selected, audit = normalize_moneyline_sources(
        [checkbest_path, oddspapi_path], target_game_pks=[1, 2, 3]
    )

    assert set(selected["game_pk"]) == {1, 2}
    assert selected.loc[selected["game_pk"].eq(1), "source"].item() == "oddspapi"
    assert audit["matched_target_games"] == 2
    assert audit["coverage"] == 2 / 3


def test_attach_moneyline_edges_removes_vig_and_calculates_ev():
    predictions = pd.DataFrame(
        {
            "game_pk": [1, 2],
            "date": ["2026-05-01", "2026-05-02"],
            "home_team": ["Home", "Home"],
            "away_team": ["Away", "Away"],
            "home_probability": [0.62, 0.48],
            "away_probability": [0.38, 0.52],
            "home_win": [1, 0],
        }
    )
    odds = pd.DataFrame(
        {
            "game_pk": [1],
            "home_moneyline": [100],
            "away_moneyline": [-130],
            "source": ["oddspapi"],
            "join_quality": ["fixture_api"],
        }
    )

    attached, summary = attach_moneyline_edges(predictions, odds)

    row = attached.loc[attached["game_pk"].eq(1)].iloc[0]
    assert np.isclose(row["market_home_probability"], 0.5 / (0.5 + 130 / 230))
    assert row["pick_side"] == "home"
    assert row["ev"] > 0
    assert row["profit_units"] == 1.0
    assert summary["matched_rows"] == 1
    assert attached.loc[attached["game_pk"].eq(2), "odds_status"].item() == "missing"


def test_feature_profile_flags_duplicate_and_future_rows():
    frame = pd.DataFrame(
        {
            "game_pk": [1, 1, 2],
            "season": [2026, 2026, 2026],
            "game_date": ["2026-08-01", "2026-08-01", "2026-08-08"],
            "game_datetime": ["2026-08-01T18:00:00Z"] * 2 + ["2026-08-08T18:00:00Z"],
            "home_win": [1, 1, 0],
            "run_diff": [1, 1, -2],
            "total_runs": [8, 8, 6],
            "home_cover_15": [0, 0, 0],
            "pregame_signal": [0.1, 0.1, 0.2],
        }
    )

    profile = profile_feature_store(frame, as_of_date="2026-08-07")

    assert profile["duplicate_game_pk_rows"] == 2
    assert profile["future_dated_rows"] == 1
    assert "duplicate_game_pk" in profile["warnings"]
    assert "future_dated_rows" in profile["warnings"]
    assert profile["status"] == "review"
