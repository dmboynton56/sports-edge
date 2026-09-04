"""Leakage-safe college-football team market model.

The model deliberately uses only information available before kickoff: a
regressed Elo rating, exponentially weighted points for/against, rest, venue,
and prior-game counts.  Coefficients are serialized as JSON so the scheduled
job is not coupled to a particular scikit-learn pickle version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score


FEATURE_COLUMNS = [
    "elo_home_edge",
    "home_expected_points",
    "away_expected_points",
    "home_offense",
    "home_defense",
    "away_offense",
    "away_defense",
    "home_games_before",
    "away_games_before",
    "rest_edge_days",
    "neutral_site",
]


@dataclass
class TeamState:
    elo: float = 1500.0
    offense: float = 28.0
    defense: float = 28.0
    games: int = 0
    last_game: datetime | None = None
    season: int | None = None

    def enter_season(self, season: int) -> None:
        if self.season is None:
            self.season = season
            return
        if season != self.season:
            self.elo = 1500.0 + 0.65 * (self.elo - 1500.0)
            self.offense = 28.0 + 0.50 * (self.offense - 28.0)
            self.defense = 28.0 + 0.50 * (self.defense - 28.0)
            self.games = 0
            self.last_game = None
            self.season = season


def _score(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_espn_scoreboard(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize ESPN scoreboard events into the model's game contract."""

    games: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        competitors = competition.get("competitors") or []
        by_side = {str(item.get("homeAway")): item for item in competitors}
        if "home" not in by_side or "away" not in by_side:
            continue
        home = by_side["home"]
        away = by_side["away"]
        status = (event.get("status") or {}).get("type") or {}
        season = event.get("season") or {}
        week = event.get("week") or {}

        def team_value(item: dict[str, Any], key: str, default: str = "") -> str:
            return str((item.get("team") or {}).get(key) or default)

        games.append(
            {
                "event_id": str(event.get("id")),
                "season": int(season.get("year") or 0),
                "week": int(week.get("number") or 0) or None,
                "game_time_utc": str(event.get("date")),
                "home_team_id": team_value(home, "id"),
                "away_team_id": team_value(away, "id"),
                "home_team": team_value(home, "displayName", team_value(home, "shortDisplayName")),
                "away_team": team_value(away, "displayName", team_value(away, "shortDisplayName")),
                "home_score": _score(home.get("score")),
                "away_score": _score(away.get("score")),
                "completed": bool(status.get("completed")),
                "status": str(status.get("name") or status.get("state") or "unknown"),
                "neutral_site": bool(competition.get("neutralSite")),
                "venue": str(((competition.get("venue") or {}).get("fullName")) or "") or None,
                "raw_record": event,
            }
        )
    return games


def _rest_days(state: TeamState, game_time: datetime) -> float:
    if state.last_game is None:
        return 14.0
    return float(min(28, max(0, (game_time.date() - state.last_game.date()).days)))


def _features(home: TeamState, away: TeamState, game: dict[str, Any], game_time: datetime) -> dict[str, float]:
    home_advantage = 0.0 if game.get("neutral_site") else 55.0
    return {
        "elo_home_edge": (home.elo - away.elo + home_advantage) / 100.0,
        "home_expected_points": 0.5 * (home.offense + away.defense),
        "away_expected_points": 0.5 * (away.offense + home.defense),
        "home_offense": home.offense,
        "home_defense": home.defense,
        "away_offense": away.offense,
        "away_defense": away.defense,
        "home_games_before": float(min(home.games, 15)),
        "away_games_before": float(min(away.games, 15)),
        "rest_edge_days": _rest_days(home, game_time) - _rest_days(away, game_time),
        "neutral_site": float(bool(game.get("neutral_site"))),
    }


def build_feature_frames(
    completed_games: Iterable[dict[str, Any]],
    future_games: Iterable[dict[str, Any]] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly pregame historical and future features."""

    states: dict[str, TeamState] = {}
    historical_rows: list[dict[str, Any]] = []

    def get_state(team_id: str, season: int) -> TeamState:
        state = states.setdefault(team_id, TeamState())
        state.enter_season(season)
        return state

    ordered = sorted(completed_games, key=lambda game: (game["game_time_utc"], game["event_id"]))
    for game in ordered:
        if not game.get("completed") or game.get("home_score") is None or game.get("away_score") is None:
            continue
        game_time = pd.Timestamp(game["game_time_utc"]).to_pydatetime()
        season = int(game["season"])
        home = get_state(str(game["home_team_id"]), season)
        away = get_state(str(game["away_team_id"]), season)
        row = {**game, **_features(home, away, game, game_time)}
        row["home_margin"] = float(game["home_score"] - game["away_score"])
        row["game_total"] = float(game["home_score"] + game["away_score"])
        row["home_win"] = float(row["home_margin"] > 0)
        historical_rows.append(row)

        expected_home = 1.0 / (1.0 + 10.0 ** (-((home.elo - away.elo) + (0 if game.get("neutral_site") else 55)) / 400.0))
        outcome = 1.0 if row["home_margin"] > 0 else 0.5 if row["home_margin"] == 0 else 0.0
        multiplier = min(2.5, math.log1p(abs(row["home_margin"])) / math.log(8.0))
        change = 32.0 * multiplier * (outcome - expected_home)
        home.elo += change
        away.elo -= change
        alpha = 0.22
        home.offense = (1 - alpha) * home.offense + alpha * float(game["home_score"])
        home.defense = (1 - alpha) * home.defense + alpha * float(game["away_score"])
        away.offense = (1 - alpha) * away.offense + alpha * float(game["away_score"])
        away.defense = (1 - alpha) * away.defense + alpha * float(game["home_score"])
        home.games += 1
        away.games += 1
        home.last_game = game_time
        away.last_game = game_time

    future_rows: list[dict[str, Any]] = []
    for game in sorted(future_games, key=lambda item: (item["game_time_utc"], item["event_id"])):
        game_time = pd.Timestamp(game["game_time_utc"]).to_pydatetime()
        season = int(game["season"])
        home = get_state(str(game["home_team_id"]), season)
        away = get_state(str(game["away_team_id"]), season)
        future_rows.append({**game, **_features(home, away, game, game_time)})

    return pd.DataFrame(historical_rows), pd.DataFrame(future_rows)


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 20.0) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    total = max(1, len(y))
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            error += mask.sum() / total * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(error)


@dataclass
class CfbMarketModel:
    model_version: str
    means: list[float]
    scales: list[float]
    margin_coefficients: list[float]
    total_coefficients: list[float]
    win_coefficients: list[float]
    margin_sigma: float
    total_sigma: float
    metrics: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _matrix(frame: pd.DataFrame) -> np.ndarray:
        return frame[FEATURE_COLUMNS].astype(float).to_numpy()

    def _scaled(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._matrix(frame)
        return (matrix - np.asarray(self.means)) / np.asarray(self.scales)

    @staticmethod
    def _linear(x: np.ndarray, coefficients: list[float] | np.ndarray) -> np.ndarray:
        coeff = np.asarray(coefficients, dtype=float)
        return coeff[0] + x @ coeff[1:]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self._scaled(frame)
        win_logit = self._linear(x, self.win_coefficients)
        result = frame.copy()
        result["predicted_margin"] = self._linear(x, self.margin_coefficients)
        result["predicted_total"] = np.maximum(20.0, self._linear(x, self.total_coefficients))
        result["home_win_probability"] = 1.0 / (1.0 + np.exp(-np.clip(win_logit, -30, 30)))
        result["predicted_home_points"] = (result["predicted_total"] + result["predicted_margin"]) / 2.0
        result["predicted_away_points"] = (result["predicted_total"] - result["predicted_margin"]) / 2.0
        return result

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        holdout_season: int,
        model_version: str = "cfb-team-v1",
    ) -> tuple["CfbMarketModel", dict[str, Any]]:
        train = frame[frame["season"] < holdout_season].copy()
        holdout = frame[frame["season"] == holdout_season].copy()
        if len(train) < 500 or len(holdout) < 100:
            raise ValueError("CFB training requires at least 500 train rows and 100 holdout rows.")

        means = cls._matrix(train).mean(axis=0)
        scales = cls._matrix(train).std(axis=0)
        scales[scales < 1e-8] = 1.0
        x_train = (cls._matrix(train) - means) / scales
        x_holdout = (cls._matrix(holdout) - means) / scales
        margin_coeff = _ridge_fit(x_train, train["home_margin"].to_numpy(dtype=float))
        total_coeff = _ridge_fit(x_train, train["game_total"].to_numpy(dtype=float))
        classifier = LogisticRegression(C=0.2, max_iter=1000).fit(x_train, train["home_win"].to_numpy(dtype=int))
        win_coeff = np.concatenate([classifier.intercept_, classifier.coef_[0]])

        margin_prediction = cls._linear(x_holdout, margin_coeff)
        total_prediction = cls._linear(x_holdout, total_coeff)
        win_probability = 1.0 / (1.0 + np.exp(-np.clip(cls._linear(x_holdout, win_coeff), -30, 30)))
        y_win = holdout["home_win"].to_numpy(dtype=int)
        metrics = {
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "holdout_season": int(holdout_season),
            "win_auc": float(roc_auc_score(y_win, win_probability)),
            "win_brier": float(brier_score_loss(y_win, win_probability)),
            "win_ece": _expected_calibration_error(y_win, win_probability),
            "margin_rmse": float(mean_squared_error(holdout["home_margin"], margin_prediction) ** 0.5),
            "margin_mae": float(mean_absolute_error(holdout["home_margin"], margin_prediction)),
            "total_rmse": float(mean_squared_error(holdout["game_total"], total_prediction) ** 0.5),
            "total_mae": float(mean_absolute_error(holdout["game_total"], total_prediction)),
        }
        metrics["supportable"] = bool(
            metrics["win_auc"] >= 0.65
            and metrics["win_brier"] <= 0.23
            and metrics["win_ece"] <= 0.06
            and metrics["margin_rmse"] <= 20.0
            and metrics["total_mae"] <= 18.0
        )

        # Refit the production coefficients on every completed season after the
        # untouched holdout has established the artifact's evidence status.
        all_means = cls._matrix(frame).mean(axis=0)
        all_scales = cls._matrix(frame).std(axis=0)
        all_scales[all_scales < 1e-8] = 1.0
        x_all = (cls._matrix(frame) - all_means) / all_scales
        all_margin = _ridge_fit(x_all, frame["home_margin"].to_numpy(dtype=float))
        all_total = _ridge_fit(x_all, frame["game_total"].to_numpy(dtype=float))
        all_classifier = LogisticRegression(C=0.2, max_iter=1000).fit(
            x_all, frame["home_win"].to_numpy(dtype=int)
        )
        all_win = np.concatenate([all_classifier.intercept_, all_classifier.coef_[0]])
        model = cls(
            model_version=model_version,
            means=all_means.tolist(),
            scales=all_scales.tolist(),
            margin_coefficients=all_margin.tolist(),
            total_coefficients=all_total.tolist(),
            win_coefficients=all_win.tolist(),
            margin_sigma=float(np.std(holdout["home_margin"].to_numpy() - margin_prediction, ddof=1)),
            total_sigma=float(np.std(holdout["game_total"].to_numpy() - total_prediction, ddof=1)),
            metrics=metrics,
        )
        return model, metrics

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CfbMarketModel":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def normal_probability_above(mean: float, threshold: float, sigma: float) -> float:
    """P(X > threshold) for a normal residual approximation."""

    if sigma <= 0:
        return float(mean > threshold)
    z = (threshold - mean) / (sigma * math.sqrt(2.0))
    return float(0.5 * math.erfc(z))
