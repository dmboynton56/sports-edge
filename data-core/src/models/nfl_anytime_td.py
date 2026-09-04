"""Out-of-time calibrated NFL anytime-touchdown model.

The target is a rushing or receiving touchdown by a player in a game. Passing
touchdowns do not count. Features are strictly lagged player/team usage and
scoring rates plus the posted game total; the held-out evaluation is therefore
representative of how Week 1 inference is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


TARGET_POSITIONS = ("QB", "RB", "WR", "TE")
BASE_STAT_COLUMNS = (
    "attempts",
    "carries",
    "targets",
    "rushing_yards",
    "receiving_yards",
    "rushing_tds",
    "receiving_tds",
    "target_share",
)
FEATURE_COLUMNS = (
    "rolling_td_rate_3",
    "rolling_td_rate_5",
    "rolling_td_rate_10",
    "rolling_tds_3",
    "rolling_tds_5",
    "rolling_opportunities_3",
    "rolling_opportunities_5",
    "rolling_yards_3",
    "rolling_yards_5",
    "rolling_targets_5",
    "rolling_carries_5",
    "rolling_target_share_5",
    "team_td_rate_5",
    "opp_td_allowed_5",
    "total_line",
    "is_home",
    "career_games_before",
    "career_td_rate",
    "career_opportunities_mean",
    "position_QB",
    "position_RB",
    "position_WR",
    "position_TE",
)


def _lagged_rolling(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby("player_id", sort=False)[column].transform(
        lambda values: values.shift(1).rolling(window, min_periods=1).mean()
    )


def build_feature_frame(player_games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe historical and future feature rows."""

    stats = player_games.copy()
    for column in BASE_STAT_COLUMNS:
        if column not in stats.columns:
            stats[column] = np.nan
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    if "player_display_name" not in stats.columns:
        stats["player_display_name"] = stats.get("player_name")
    if "team" not in stats.columns and "recent_team" in stats.columns:
        stats["team"] = stats["recent_team"]
    if "opponent_team" not in stats.columns and "opponent" in stats.columns:
        stats["opponent_team"] = stats["opponent"]

    stats["position"] = stats["position"].astype(str).str.upper()
    stats = stats[stats["position"].isin(TARGET_POSITIONS)].copy()
    if "is_future" not in stats.columns:
        stats["is_future"] = False
    else:
        stats["is_future"] = stats["is_future"].fillna(False).astype(bool)

    historical = ~stats["is_future"]
    opportunities = stats["carries"].fillna(0) + stats["targets"].fillna(0)
    participated = opportunities.gt(0) | stats["attempts"].fillna(0).gt(0)
    stats = stats[stats["is_future"] | (historical & participated)].copy()

    stats["total_tds"] = stats["rushing_tds"].fillna(0) + stats["receiving_tds"].fillna(0)
    stats["has_td"] = np.where(stats["is_future"], np.nan, stats["total_tds"].gt(0).astype(float))
    stats["opportunities"] = stats["carries"].fillna(0) + stats["targets"].fillna(0)
    stats["scrimmage_yards"] = (
        stats["rushing_yards"].fillna(0) + stats["receiving_yards"].fillna(0)
    )

    games = schedule.copy()
    if "game_date" not in games.columns:
        games["game_date"] = games.get("gameday")
    elif "gameday" in games.columns:
        games["game_date"] = games["game_date"].fillna(games["gameday"])
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    for column in ("total_line",):
        if column not in games.columns:
            games[column] = np.nan
        games[column] = pd.to_numeric(games[column], errors="coerce")
    games = games[
        ["game_id", "game_date", "home_team", "away_team", "total_line"]
    ].drop_duplicates("game_id", keep="last")
    stats = stats.merge(games, on="game_id", how="left", suffixes=("", "_schedule"))
    stats["is_home"] = (stats["team"] == stats["home_team"]).astype(float)
    stats = stats.sort_values(["game_date", "season", "week", "player_id"]).reset_index(drop=True)

    for window in (3, 5, 10):
        stats[f"rolling_td_rate_{window}"] = _lagged_rolling(stats, "has_td", window)
    for window in (3, 5):
        stats[f"rolling_tds_{window}"] = _lagged_rolling(stats, "total_tds", window)
        stats[f"rolling_opportunities_{window}"] = _lagged_rolling(
            stats, "opportunities", window
        )
        stats[f"rolling_yards_{window}"] = _lagged_rolling(stats, "scrimmage_yards", window)
    stats["rolling_targets_5"] = _lagged_rolling(stats, "targets", 5)
    stats["rolling_carries_5"] = _lagged_rolling(stats, "carries", 5)
    stats["rolling_target_share_5"] = _lagged_rolling(stats, "target_share", 5)
    stats["career_games_before"] = stats.groupby("player_id", sort=False).cumcount().astype(float)
    stats["career_td_rate"] = stats.groupby("player_id", sort=False)["has_td"].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )
    stats["career_opportunities_mean"] = stats.groupby("player_id", sort=False)[
        "opportunities"
    ].transform(lambda values: values.shift(1).expanding(min_periods=1).mean())

    team_games = (
        stats.groupby(
            ["game_id", "game_date", "team", "opponent_team"],
            dropna=False,
            as_index=False,
        )["total_tds"]
        .sum(min_count=1)
        .rename(columns={"total_tds": "team_tds"})
        .sort_values(["team", "game_date", "game_id"])
    )
    team_games["team_td_rate_5"] = team_games.groupby("team", sort=False)["team_tds"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    defense_games = team_games[
        ["game_id", "game_date", "opponent_team", "team_tds"]
    ].rename(columns={"opponent_team": "team", "team_tds": "tds_allowed"})
    defense_games = defense_games.sort_values(["team", "game_date", "game_id"])
    defense_games["opp_td_allowed_5"] = defense_games.groupby("team", sort=False)[
        "tds_allowed"
    ].transform(lambda values: values.shift(1).rolling(5, min_periods=1).mean())

    stats = stats.merge(
        team_games[["game_id", "team", "team_td_rate_5"]],
        on=["game_id", "team"],
        how="left",
    )
    stats = stats.merge(
        defense_games[["game_id", "team", "opp_td_allowed_5"]].rename(
            columns={"team": "opponent_team"}
        ),
        on=["game_id", "opponent_team"],
        how="left",
    )
    for position in TARGET_POSITIONS:
        stats[f"position_{position}"] = (stats["position"] == position).astype(float)
    return stats


def expected_calibration_error(y_true: Sequence[float], probabilities: Sequence[float], bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(y), 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if mask.any():
            error += mask.sum() / total * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


@dataclass
class AnytimeTDModel:
    booster: lgb.Booster
    medians: dict[str, float]
    calibration_coefficient: float
    calibration_intercept: float
    model_version: str = "nfl-anytime-td-v1"

    @staticmethod
    def _matrix(frame: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
        matrix = frame.reindex(columns=FEATURE_COLUMNS).apply(pd.to_numeric, errors="coerce")
        return matrix.fillna(pd.Series(medians)).fillna(0.0)

    @staticmethod
    def _calibrate(raw: np.ndarray, coefficient: float, intercept: float) -> np.ndarray:
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        return 1.0 / (1.0 + np.exp(-(coefficient * logits + intercept)))

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        *,
        model_version: str = "nfl-anytime-td-v1",
    ) -> tuple["AnytimeTDModel", dict[str, Any]]:
        train = features[(features["season"] <= 2024) & features["has_td"].notna()].copy()
        calibrate = features[
            (features["season"] == 2025)
            & (features["week"] <= 9)
            & features["has_td"].notna()
        ].copy()
        holdout = features[
            (features["season"] == 2025)
            & (features["week"] >= 10)
            & (features["week"] <= 18)
            & features["has_td"].notna()
        ].copy()
        if min(len(train), len(calibrate), len(holdout)) == 0:
            raise ValueError("Training, calibration, and 2025 holdout rows are all required")

        medians = {
            column: float(pd.to_numeric(train[column], errors="coerce").median())
            if pd.to_numeric(train[column], errors="coerce").notna().any()
            else 0.0
            for column in FEATURE_COLUMNS
        }
        x_train = cls._matrix(train, medians)
        y_train = train["has_td"].astype(int)
        fitted = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=240,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=100,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
        )
        fitted.fit(x_train, y_train)

        calibration_raw = fitted.predict_proba(cls._matrix(calibrate, medians))[:, 1]
        calibration_logit = np.log(
            np.clip(calibration_raw, 1e-6, 1 - 1e-6)
            / (1 - np.clip(calibration_raw, 1e-6, 1 - 1e-6))
        ).reshape(-1, 1)
        calibration = LogisticRegression(C=1e6, solver="lbfgs")
        calibration.fit(calibration_logit, calibrate["has_td"].astype(int))
        coefficient = float(calibration.coef_[0, 0])
        intercept = float(calibration.intercept_[0])

        model = cls(
            booster=fitted.booster_,
            medians=medians,
            calibration_coefficient=coefficient,
            calibration_intercept=intercept,
            model_version=model_version,
        )
        probabilities = model.predict_proba(holdout)
        y_holdout = holdout["has_td"].astype(int).to_numpy()
        baseline_probability = float(y_train.mean())
        baseline = np.full(len(holdout), baseline_probability)
        rolling_baseline = (
            pd.to_numeric(holdout["rolling_td_rate_5"], errors="coerce")
            .fillna(baseline_probability)
            .clip(0.01, 0.95)
            .to_numpy()
        )
        metrics = {
            "model_version": model_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_seasons": [2021, 2022, 2023, 2024],
            "calibration_period": "2025 weeks 1-9",
            "holdout_period": "2025 weeks 10-18",
            "train_rows": len(train),
            "calibration_rows": len(calibrate),
            "holdout_rows": len(holdout),
            "holdout_positive_rate": float(y_holdout.mean()),
            "brier": float(brier_score_loss(y_holdout, probabilities)),
            "log_loss": float(log_loss(y_holdout, probabilities)),
            "auc": float(roc_auc_score(y_holdout, probabilities)),
            "ece": expected_calibration_error(y_holdout, probabilities),
            "global_baseline_brier": float(brier_score_loss(y_holdout, baseline)),
            "rolling_rate_baseline_brier": float(
                brier_score_loss(y_holdout, rolling_baseline)
            ),
            "calibration_coefficient": coefficient,
            "calibration_intercept": intercept,
            "feature_columns": list(FEATURE_COLUMNS),
        }
        metrics["supportable"] = bool(
            metrics["brier"] < metrics["global_baseline_brier"] and metrics["auc"] >= 0.60
        )
        return model, metrics

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(self._matrix(frame, self.medians))
        return self._calibrate(
            np.asarray(raw, dtype=float),
            self.calibration_coefficient,
            self.calibration_intercept,
        )

    def save(self, model_path: Path, metadata_path: Path, metrics: dict[str, Any]) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(model_path))
        metadata_path.write_text(
            json.dumps(
                {
                    **metrics,
                    "medians": self.medians,
                    "model_file": model_path.name,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, model_path: Path, metadata_path: Path) -> tuple["AnytimeTDModel", dict[str, Any]]:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = cls(
            booster=lgb.Booster(model_file=str(model_path)),
            medians={key: float(value) for key, value in metadata["medians"].items()},
            calibration_coefficient=float(metadata["calibration_coefficient"]),
            calibration_intercept=float(metadata["calibration_intercept"]),
            model_version=str(metadata["model_version"]),
        )
        return model, metadata
