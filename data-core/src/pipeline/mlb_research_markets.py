#!/usr/bin/env python3
"""
Scoring module for MLB research markets: moneyline v3, run-line v1, and totals v1.

This module loads trained model artifacts and scores pregame predictions for a
date window. It does NOT train models or depend on gitignored feature artifacts.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.mlb_fetcher import fetch_mlb_schedule
from src.models.mlb_winner_model import build_mlb_prediction_features
from src.models.mlb_runline_model import cover_probability_from_residuals
from src.models.mlb_totals_model import over_label, _normal_over_probability


@dataclass(frozen=True)
class ResearchMarketPredictions:
    """Container for scored predictions from all research markets on a date window."""

    moneyline: pd.DataFrame  # game_pk, home_win_prob, away_win_prob
    run_line: pd.DataFrame  # game_pk, p_home_cover_15, p_away_cover_plus_15
    totals: pd.DataFrame  # game_pk, predicted_total, p_over_8_5, p_over_9_5
    score_date: date
    season: int


def load_mlb_moneyline_v3(model_path: str | Path) -> dict:
    """Load the MLB moneyline v3 artifact."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_mlb_totals_v1(model_path: str | Path) -> dict:
    """Load the MLB totals v1 artifact."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_mlb_runline_v1(model_path: str | Path) -> dict:
    """Load the MLB run-line v1 artifact."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_mlb_moneyline_v3(
    *,
    artifact: dict,
    schedule: pd.DataFrame,
    game_date: date,
    min_prior_games: int = 5,
) -> pd.DataFrame:
    """Score home-win probabilities for a single game date using the v3 artifact."""
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    games_to_score = schedule[schedule["game_date"].dt.date == game_date].copy()
    if games_to_score.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "home_win_prob", "away_win_prob"]
        )

    history = schedule[
        (schedule["game_date"].dt.date < game_date)
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()
    if history.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "home_win_prob", "away_win_prob"]
        )

    features = build_mlb_prediction_features(history, games_to_score, min_prior_games=min_prior_games)
    if features.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "home_win_prob", "away_win_prob"]
        )

    feature_cols = artifact["feature_columns"]
    probabilities = artifact["model"].predict_proba(features[feature_cols])[:, 1]
    output = features[["game_pk", "game_date", "home_team", "away_team"]].copy()
    output["home_win_prob"] = probabilities
    output["away_win_prob"] = 1.0 - probabilities
    return output


def score_mlb_totals_v1(
    *,
    artifact: dict,
    schedule: pd.DataFrame,
    game_date: date,
    min_prior_games: int = 5,
) -> pd.DataFrame:
    """Score total runs and O/U probabilities for a single game date using the v1 artifact."""
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    games_to_score = schedule[schedule["game_date"].dt.date == game_date].copy()
    if games_to_score.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "predicted_total", "p_over_8_5", "p_over_9_5"]
        )

    history = schedule[
        (schedule["game_date"].dt.date < game_date)
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()
    if history.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "predicted_total", "p_over_8_5", "p_over_9_5"]
        )

    features = build_mlb_prediction_features(history, games_to_score, min_prior_games=min_prior_games)
    if features.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "predicted_total", "p_over_8_5", "p_over_9_5"]
        )

    feature_cols = artifact["feature_columns"]
    predicted_total = artifact["model"].predict(features[feature_cols])
    sigma = artifact["probability_method"]["validation_residual_rmse_sigma"]
    p_over_8_5 = _normal_over_probability(predicted_total, 8.5, sigma)
    p_over_9_5 = _normal_over_probability(predicted_total, 9.5, sigma)

    output = features[["game_pk", "game_date", "home_team", "away_team"]].copy()
    output["predicted_total"] = predicted_total
    output["p_over_8_5"] = p_over_8_5
    output["p_over_9_5"] = p_over_9_5
    return output


def score_mlb_runline_v1(
    *,
    artifact: dict,
    schedule: pd.DataFrame,
    game_date: date,
    min_prior_games: int = 5,
) -> pd.DataFrame:
    """Score home -1.5 cover probabilities for a single game date using the v1 artifact."""
    schedule["game_date"] = pd.to_datetime(schedule["game_date"])
    games_to_score = schedule[schedule["game_date"].dt.date == game_date].copy()
    if games_to_score.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "p_home_cover_15", "p_away_cover_plus_15"]
        )

    history = schedule[
        (schedule["game_date"].dt.date < game_date)
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()
    if history.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "p_home_cover_15", "p_away_cover_plus_15"]
        )

    features = build_mlb_prediction_features(history, games_to_score, min_prior_games=min_prior_games)
    if features.empty:
        return pd.DataFrame(
            columns=["game_pk", "game_date", "home_team", "away_team", "p_home_cover_15", "p_away_cover_plus_15"]
        )

    feature_cols = artifact["feature_columns"]
    classifier = artifact["classifier"]
    p_home_cover_15 = classifier.predict_proba(features[feature_cols])[:, 1]

    output = features[["game_pk", "game_date", "home_team", "away_team"]].copy()
    output["p_home_cover_15"] = p_home_cover_15
    output["p_away_cover_plus_15"] = 1.0 - p_home_cover_15
    return output


def score_research_markets_for_date(
    *,
    game_date: date,
    season: int,
    moneyline_artifact: dict,
    totals_artifact: dict,
    runline_artifact: dict,
    season_start: Optional[date] = None,
    min_prior_games: int = 5,
) -> ResearchMarketPredictions:
    """
    Fetch the season schedule and score moneyline, run-line, and totals for a single game date.

    Returns a ResearchMarketPredictions container with three DataFrames.
    """
    if season_start is None:
        season_start = date(season, 3, 1)

    schedule = fetch_mlb_schedule(
        season,
        start_date=season_start,
        end_date=game_date,
        include_uncompleted=True,
    )
    if schedule.empty:
        raise ValueError(f"No MLB schedule rows found for season={season} through {game_date}")

    moneyline = score_mlb_moneyline_v3(
        artifact=moneyline_artifact,
        schedule=schedule,
        game_date=game_date,
        min_prior_games=min_prior_games,
    )
    totals = score_mlb_totals_v1(
        artifact=totals_artifact,
        schedule=schedule,
        game_date=game_date,
        min_prior_games=min_prior_games,
    )
    runline = score_mlb_runline_v1(
        artifact=runline_artifact,
        schedule=schedule,
        game_date=game_date,
        min_prior_games=min_prior_games,
    )

    return ResearchMarketPredictions(
        moneyline=moneyline,
        run_line=runline,
        totals=totals,
        score_date=game_date,
        season=season,
    )
