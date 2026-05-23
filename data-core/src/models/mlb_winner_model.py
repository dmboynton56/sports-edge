"""
MLB winner model training utilities.

This is a home-win model built from completed-game results, probable starters,
and venue context. Features are computed from prior games in the same season,
so the current game's result is not available to its feature row.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import pickle
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TeamState:
    games: int = 0
    wins: int = 0
    runs_for: int = 0
    runs_against: int = 0
    home_games: int = 0
    home_wins: int = 0
    away_games: int = 0
    away_wins: int = 0
    last_date: Optional[pd.Timestamp] = None
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_run_diffs: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class PitcherState:
    starts: int = 0
    team_wins: int = 0
    runs_allowed: int = 0
    run_support: int = 0
    last_date: Optional[pd.Timestamp] = None
    recent_team_wins: deque = field(default_factory=lambda: deque(maxlen=5))
    recent_run_diffs: deque = field(default_factory=lambda: deque(maxlen=5))


@dataclass
class VenueState:
    games: int = 0
    home_wins: int = 0
    total_runs: int = 0


def _rate(num: int, den: int, default: float = 0.5) -> float:
    return float(num) / float(den) if den else default


def _avg(total: int, den: int, default: float = 0.0) -> float:
    return float(total) / float(den) if den else default


def _recent_mean(values: deque, default: float) -> float:
    return float(np.mean(values)) if values else default


def _days_rest(state: TeamState, game_date: pd.Timestamp) -> float:
    if state.last_date is None:
        return 7.0
    return float(np.clip((game_date.normalize() - state.last_date.normalize()).days, 0, 14))


def _pitcher_rest(state: PitcherState, game_date: pd.Timestamp) -> float:
    if state.last_date is None:
        return 7.0
    return float(np.clip((game_date.normalize() - state.last_date.normalize()).days, 0, 30))


def _team_features(prefix: str, state: TeamState, game_date: pd.Timestamp, is_home: bool) -> Dict[str, float]:
    rest_days = _days_rest(state, game_date)
    venue_games = state.home_games if is_home else state.away_games
    venue_wins = state.home_wins if is_home else state.away_wins
    return {
        f"{prefix}_games_played": float(state.games),
        f"{prefix}_win_pct": _rate(state.wins, state.games),
        f"{prefix}_run_diff_per_game": _avg(state.runs_for - state.runs_against, state.games),
        f"{prefix}_runs_for_per_game": _avg(state.runs_for, state.games),
        f"{prefix}_runs_against_per_game": _avg(state.runs_against, state.games),
        f"{prefix}_venue_win_pct": _rate(venue_wins, venue_games),
        f"{prefix}_recent_win_pct_10": _recent_mean(state.recent_wins, 0.5),
        f"{prefix}_recent_run_diff_10": _recent_mean(state.recent_run_diffs, 0.0),
        f"{prefix}_rest_days": rest_days,
        f"{prefix}_is_b2b": float(rest_days <= 1),
    }


def _pitcher_features(prefix: str, state: PitcherState, game_date: pd.Timestamp, has_pitcher: bool) -> Dict[str, float]:
    rest_days = _pitcher_rest(state, game_date)
    return {
        f"{prefix}_starter_known": float(has_pitcher),
        f"{prefix}_starter_prior_starts": float(state.starts),
        f"{prefix}_starter_team_win_pct": _rate(state.team_wins, state.starts),
        f"{prefix}_starter_runs_allowed_per_start": _avg(state.runs_allowed, state.starts, default=4.5),
        f"{prefix}_starter_run_support_per_start": _avg(state.run_support, state.starts, default=4.5),
        f"{prefix}_starter_recent_team_win_pct_5": _recent_mean(state.recent_team_wins, 0.5),
        f"{prefix}_starter_recent_run_diff_5": _recent_mean(state.recent_run_diffs, 0.0),
        f"{prefix}_starter_rest_days": rest_days,
        f"{prefix}_starter_short_rest": float(rest_days < 5),
    }


def _venue_features(state: VenueState) -> Dict[str, float]:
    return {
        "venue_prior_games": float(state.games),
        "venue_home_win_pct": _rate(state.home_wins, state.games),
        "venue_total_runs_per_game": _avg(state.total_runs, state.games, default=8.8),
    }


def _update_state(
    state: TeamState,
    *,
    game_date: pd.Timestamp,
    runs_for: int,
    runs_against: int,
    is_home: bool,
) -> None:
    won = int(runs_for > runs_against)
    state.games += 1
    state.wins += won
    state.runs_for += int(runs_for)
    state.runs_against += int(runs_against)
    if is_home:
        state.home_games += 1
        state.home_wins += won
    else:
        state.away_games += 1
        state.away_wins += won
    state.last_date = game_date
    state.recent_wins.append(float(won))
    state.recent_run_diffs.append(float(runs_for - runs_against))


def _update_pitcher_state(
    state: PitcherState,
    *,
    game_date: pd.Timestamp,
    runs_for: int,
    runs_against: int,
) -> None:
    won = int(runs_for > runs_against)
    state.starts += 1
    state.team_wins += won
    state.runs_allowed += int(runs_against)
    state.run_support += int(runs_for)
    state.last_date = game_date
    state.recent_team_wins.append(float(won))
    state.recent_run_diffs.append(float(runs_for - runs_against))


def _update_venue_state(state: VenueState, *, home_score: int, away_score: int) -> None:
    state.games += 1
    state.home_wins += int(home_score > away_score)
    state.total_runs += int(home_score) + int(away_score)


def _nullable_int(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    return int(value)


def _empty_states() -> tuple[Dict[tuple, TeamState], Dict[tuple, PitcherState], Dict[tuple, VenueState]]:
    return defaultdict(TeamState), defaultdict(PitcherState), defaultdict(VenueState)


def _feature_row_for_game(
    game: object,
    states: Dict[tuple, TeamState],
    pitcher_states: Dict[tuple, PitcherState],
    venue_states: Dict[tuple, VenueState],
    *,
    include_target: bool,
) -> dict:
        season = int(game.season)
        game_date = pd.Timestamp(game.game_date)
        home_key = (season, int(game.home_team_id))
        away_key = (season, int(game.away_team_id))
        venue_id = _nullable_int(getattr(game, "venue_id", None))
        home_pitcher_id = _nullable_int(getattr(game, "home_probable_pitcher_id", None))
        away_pitcher_id = _nullable_int(getattr(game, "away_probable_pitcher_id", None))
        home_state = states[home_key]
        away_state = states[away_key]
        home_pitcher_state = pitcher_states[(season, home_pitcher_id)] if home_pitcher_id else PitcherState()
        away_pitcher_state = pitcher_states[(season, away_pitcher_id)] if away_pitcher_id else PitcherState()
        venue_state = venue_states[(season, venue_id)] if venue_id else VenueState()

        row = {
            "game_pk": int(game.game_pk),
            "season": season,
            "game_date": game_date,
            "game_datetime": pd.Timestamp(game.game_datetime),
            "home_team": game.home_team,
            "away_team": game.away_team,
            "home_team_id": int(game.home_team_id),
            "away_team_id": int(game.away_team_id),
            "venue_id": venue_id,
            "venue_name": getattr(game, "venue_name", None),
            "home_probable_pitcher_id": home_pitcher_id,
            "home_probable_pitcher": getattr(game, "home_probable_pitcher", None),
            "away_probable_pitcher_id": away_pitcher_id,
            "away_probable_pitcher": getattr(game, "away_probable_pitcher", None),
            "month": float(game_date.month),
            "day_of_week": float(game_date.dayofweek),
            "is_doubleheader_same_day": float(home_state.last_date is not None and home_state.last_date.normalize() == game_date.normalize()),
        }
        if include_target:
            row.update(
                {
                    "home_score": int(game.home_score),
                    "away_score": int(game.away_score),
                    "home_win": int(game.home_win),
                    "run_diff": int(game.run_diff),
                }
            )
        row.update(_team_features("home", home_state, game_date, is_home=True))
        row.update(_team_features("away", away_state, game_date, is_home=False))
        row.update(_pitcher_features("home", home_pitcher_state, game_date, has_pitcher=home_pitcher_id is not None))
        row.update(_pitcher_features("away", away_pitcher_state, game_date, has_pitcher=away_pitcher_id is not None))
        row.update(_venue_features(venue_state))

        row["games_played_diff"] = row["home_games_played"] - row["away_games_played"]
        row["win_pct_diff"] = row["home_win_pct"] - row["away_win_pct"]
        row["run_diff_per_game_diff"] = row["home_run_diff_per_game"] - row["away_run_diff_per_game"]
        row["runs_for_per_game_diff"] = row["home_runs_for_per_game"] - row["away_runs_for_per_game"]
        row["runs_against_per_game_diff"] = row["away_runs_against_per_game"] - row["home_runs_against_per_game"]
        row["recent_win_pct_10_diff"] = row["home_recent_win_pct_10"] - row["away_recent_win_pct_10"]
        row["recent_run_diff_10_diff"] = row["home_recent_run_diff_10"] - row["away_recent_run_diff_10"]
        row["rest_days_diff"] = row["home_rest_days"] - row["away_rest_days"]
        row["b2b_diff"] = row["home_is_b2b"] - row["away_is_b2b"]
        row["starter_prior_starts_diff"] = row["home_starter_prior_starts"] - row["away_starter_prior_starts"]
        row["starter_team_win_pct_diff"] = row["home_starter_team_win_pct"] - row["away_starter_team_win_pct"]
        row["starter_runs_allowed_diff"] = (
            row["away_starter_runs_allowed_per_start"] - row["home_starter_runs_allowed_per_start"]
        )
        row["starter_run_support_diff"] = (
            row["home_starter_run_support_per_start"] - row["away_starter_run_support_per_start"]
        )
        row["starter_recent_run_diff_5_diff"] = (
            row["home_starter_recent_run_diff_5"] - row["away_starter_recent_run_diff_5"]
        )
        row["starter_rest_days_diff"] = row["home_starter_rest_days"] - row["away_starter_rest_days"]
        return row


def _update_states_for_completed_game(
    game: object,
    states: Dict[tuple, TeamState],
    pitcher_states: Dict[tuple, PitcherState],
    venue_states: Dict[tuple, VenueState],
) -> None:
        season = int(game.season)
        game_date = pd.Timestamp(game.game_date)
        home_pitcher_id = _nullable_int(getattr(game, "home_probable_pitcher_id", None))
        away_pitcher_id = _nullable_int(getattr(game, "away_probable_pitcher_id", None))
        venue_id = _nullable_int(getattr(game, "venue_id", None))
        _update_state(
            states[(season, int(game.home_team_id))],
            game_date=game_date,
            runs_for=int(game.home_score),
            runs_against=int(game.away_score),
            is_home=True,
        )
        _update_state(
            states[(season, int(game.away_team_id))],
            game_date=game_date,
            runs_for=int(game.away_score),
            runs_against=int(game.home_score),
            is_home=False,
        )
        if home_pitcher_id:
            _update_pitcher_state(
                pitcher_states[(season, home_pitcher_id)],
                game_date=game_date,
                runs_for=int(game.home_score),
                runs_against=int(game.away_score),
            )
        if away_pitcher_id:
            _update_pitcher_state(
                pitcher_states[(season, away_pitcher_id)],
                game_date=game_date,
                runs_for=int(game.away_score),
                runs_against=int(game.home_score),
            )
        if venue_id:
            _update_venue_state(
                venue_states[(season, venue_id)],
                home_score=int(game.home_score),
                away_score=int(game.away_score),
            )


def _prepare_games(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["game_datetime"] = pd.to_datetime(out["game_datetime"])
    return out.sort_values(["season", "game_datetime", "game_pk"]).reset_index(drop=True)


def build_mlb_winner_features(games: pd.DataFrame, min_prior_games: int = 5) -> pd.DataFrame:
    """
    Build leakage-safe rolling team features for completed MLB games.

    Args:
        games: Completed games from `src.data.mlb_fetcher`.
        min_prior_games: Keep rows where both teams have at least this many
            same-season games before first pitch.
    """
    if games.empty:
        raise ValueError("No MLB games provided.")

    df = _prepare_games(games)
    completed = df[df["home_score"].notna() & df["away_score"].notna()].copy()

    rows = []
    states, pitcher_states, venue_states = _empty_states()

    for game in completed.itertuples(index=False):
        rows.append(
            _feature_row_for_game(
                game,
                states,
                pitcher_states,
                venue_states,
                include_target=True,
            )
        )
        _update_states_for_completed_game(game, states, pitcher_states, venue_states)

    features = pd.DataFrame(rows)
    if min_prior_games > 0:
        features = features[
            (features["home_games_played"] >= min_prior_games)
            & (features["away_games_played"] >= min_prior_games)
        ].copy()

    return features.reset_index(drop=True)


def build_mlb_prediction_features(
    history_games: pd.DataFrame,
    games_to_score: pd.DataFrame,
    min_prior_games: int = 5,
) -> pd.DataFrame:
    """
    Build pregame features for scheduled games from completed history.
    """
    if history_games.empty:
        raise ValueError("Completed MLB history is required for prediction features.")
    if games_to_score.empty:
        return pd.DataFrame()

    history = _prepare_games(history_games)
    history = history[history["home_score"].notna() & history["away_score"].notna()].copy()
    score_df = _prepare_games(games_to_score)

    states, pitcher_states, venue_states = _empty_states()
    rows = []
    for game in history.itertuples(index=False):
        _update_states_for_completed_game(game, states, pitcher_states, venue_states)

    for game in score_df.itertuples(index=False):
        row = _feature_row_for_game(
            game,
            states,
            pitcher_states,
            venue_states,
            include_target=False,
        )
        rows.append(row)

    features = pd.DataFrame(rows)
    if min_prior_games > 0 and not features.empty:
        features = features[
            (features["home_games_played"] >= min_prior_games)
            & (features["away_games_played"] >= min_prior_games)
        ].copy()
    return features.reset_index(drop=True)


def default_feature_columns(features: pd.DataFrame) -> list[str]:
    """Select numeric modeling features while excluding targets and IDs."""
    exclude = {
        "game_pk",
        "season",
        "game_date",
        "game_datetime",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
        "venue_id",
        "venue_name",
        "home_probable_pitcher_id",
        "home_probable_pitcher",
        "away_probable_pitcher_id",
        "away_probable_pitcher",
        "home_score",
        "away_score",
        "home_win",
        "run_diff",
    }
    postgame_keywords = (
        "actual_starter",
        "starter_outs",
        "starter_pitches",
        "starter_earned_runs",
        "starter_strikeouts",
        "starter_walks",
        "bullpen_",
    )
    return [
        col
        for col in features.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(features[col])
        and not any(keyword in col for keyword in postgame_keywords)
    ]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
    return float(ece)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "ece_10": expected_calibration_error(y_true, y_prob, n_bins=10),
        "avg_pred_home_win": float(np.mean(y_prob)),
        "actual_home_win_rate": float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def train_and_evaluate_mlb_winner(
    features: pd.DataFrame,
    *,
    test_season: Optional[int] = None,
    validation_season: Optional[int] = None,
    random_state: int = 42,
) -> dict:
    """
    Train candidate MLB home-win models and evaluate on a held-out season.
    """
    if features.empty:
        raise ValueError("No MLB features provided.")

    features = features.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)
    seasons = sorted(features["season"].dropna().astype(int).unique().tolist())
    if len(seasons) < 3 and (test_season is None or validation_season is None):
        raise ValueError("Need at least three seasons for default train/validation/test split.")

    test_season = int(test_season or seasons[-1])
    validation_season = int(validation_season or seasons[-2])
    if validation_season >= test_season:
        raise ValueError("validation_season must be earlier than test_season.")

    feature_cols = default_feature_columns(features)
    train_df = features[features["season"] < validation_season].copy()
    val_df = features[features["season"] == validation_season].copy()
    test_df = features[features["season"] == test_season].copy()
    final_train_df = features[features["season"] < test_season].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Split produced empty data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    candidates = {
        "logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.02,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=20,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    X_train = train_df[feature_cols]
    y_train = train_df["home_win"].astype(int).to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["home_win"].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["home_win"].astype(int).to_numpy()

    model_metrics = {}
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        val_prob = candidate.predict_proba(X_val)[:, 1]
        test_prob = candidate.predict_proba(X_test)[:, 1]
        model_metrics[name] = {
            "validation": _metrics(y_val, val_prob),
            "test": _metrics(y_test, test_prob),
        }

    selected_model_name = min(model_metrics, key=lambda name: model_metrics[name]["validation"]["brier"])
    selected_model = candidates[selected_model_name]
    selected_model.fit(final_train_df[feature_cols], final_train_df["home_win"].astype(int).to_numpy())
    selected_test_prob = selected_model.predict_proba(X_test)[:, 1]

    baseline_prob = float(final_train_df["home_win"].mean())
    baseline_test_prob = np.full(len(test_df), baseline_prob)

    return {
        "selected_model_name": selected_model_name,
        "selected_model": selected_model,
        "feature_columns": feature_cols,
        "model_metrics": model_metrics,
        "selected_refit_test": _metrics(y_test, selected_test_prob),
        "baseline": {
            "probability": baseline_prob,
            "test": _metrics(y_test, baseline_test_prob),
        },
        "splits": {
            "train_seasons": sorted(train_df["season"].astype(int).unique().tolist()),
            "validation_season": validation_season,
            "test_season": test_season,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "final_train_rows": int(len(final_train_df)),
        },
        "data_summary": {
            "rows": int(len(features)),
            "min_game_date": str(pd.to_datetime(features["game_date"]).min().date()),
            "max_game_date": str(pd.to_datetime(features["game_date"]).max().date()),
            "seasons": seasons,
            "home_win_rate": float(features["home_win"].mean()),
        },
    }


def save_mlb_winner_artifact(result: dict, output_path: str, model_version: str = "v1") -> None:
    """Persist model artifact and JSON metrics sidecar."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    artifact = {
        "model": result["selected_model"],
        "model_name": result["selected_model_name"],
        "feature_columns": result["feature_columns"],
        "model_version": model_version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "splits": result["splits"],
        "data_summary": result["data_summary"],
        "metrics": {
            "selected_refit_test": result["selected_refit_test"],
            "baseline": result["baseline"],
            "candidates": result["model_metrics"],
        },
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    metrics_path = os.path.splitext(output_path)[0] + "_metrics.json"
    metrics_payload = {key: value for key, value in artifact.items() if key != "model"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)


def summarize_metrics(result: dict) -> str:
    """Return a compact text summary for CLI output."""
    lines = [
        f"Selected model: {result['selected_model_name']}",
        f"Splits: train {result['splits']['train_seasons']}, "
        f"validation {result['splits']['validation_season']}, test {result['splits']['test_season']}",
        f"Rows: train={result['splits']['train_rows']}, validation={result['splits']['validation_rows']}, "
        f"test={result['splits']['test_rows']}",
        "Candidate metrics:",
    ]
    for name, metrics in result["model_metrics"].items():
        val = metrics["validation"]
        test = metrics["test"]
        lines.append(
            f"  {name}: val brier={val['brier']:.4f}, val log_loss={val['log_loss']:.4f}; "
            f"test acc={test['accuracy']:.4f}, brier={test['brier']:.4f}, log_loss={test['log_loss']:.4f}, "
            f"auc={test.get('roc_auc', float('nan')):.4f}, ece={test['ece_10']:.4f}"
        )
    selected = result["selected_refit_test"]
    baseline = result["baseline"]["test"]
    lines.append(
        f"Selected refit test: acc={selected['accuracy']:.4f}, brier={selected['brier']:.4f}, "
        f"log_loss={selected['log_loss']:.4f}, auc={selected.get('roc_auc', float('nan')):.4f}, "
        f"ece={selected['ece_10']:.4f}"
    )
    lines.append(
        f"Home-rate baseline: acc={baseline['accuracy']:.4f}, brier={baseline['brier']:.4f}, "
        f"log_loss={baseline['log_loss']:.4f}"
    )
    return "\n".join(lines)
