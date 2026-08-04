"""Original public-data fantasy projection model.

This is intentionally a transparent, component-first baseline. It learns the
relationship between recent player usage and next-season totals with a
position-aware random forest, then applies the user's scoring settings after
inference. The model is small enough to retrain in CI and produces a report
that can be compared with simple trailing-average baselines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .scoring import FULL_PPR_SCORING, projection_points


SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "K")
STAT_COLUMNS = (
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
    "two_point_conversions",
    "fg_made_0_39",
    "fg_made_40_49",
    "fg_made_50_plus",
    "fg_missed",
    "extra_points_made",
)


def _float_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)


def normalize_weekly_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Map nflverse weekly stats into the component schema."""

    if frame.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "position", "team", "season", "week", *STAT_COLUMNS])
    # Player-stat downloads can include postseason games.  Redraft season
    # projections are based on the regular season only; keeping playoffs here
    # would make games, usage, and fantasy points systematically too high.
    season_type_column = "game_type" if "game_type" in frame.columns else "season_type" if "season_type" in frame.columns else None
    if season_type_column:
        frame = frame[frame[season_type_column].astype(str).str.upper().eq("REG")]
        if frame.empty:
            return pd.DataFrame(columns=["player_id", "player_name", "position", "team", "season", "week", *STAT_COLUMNS])
    result = pd.DataFrame(index=frame.index)
    result["player_id"] = frame.get("player_id", frame.get("gsis_id", "")).astype(str)
    result["player_name"] = frame.get("player_display_name", frame.get("player_name", "")).fillna("")
    result["position"] = frame.get("position", frame.get("position_group", "")).fillna("").astype(str).str.upper()
    result["team"] = frame.get("team", "").fillna("").astype(str)
    result["season"] = pd.to_numeric(frame.get("season", 0), errors="coerce").fillna(0).astype(int)
    result["week"] = pd.to_numeric(frame.get("week", 0), errors="coerce").fillna(0).astype(int)
    result["game_id"] = frame.get("game_id", result["season"].astype(str) + "_" + result["week"].astype(str))

    result["passing_yards"] = _float_series(frame, "passing_yards")
    result["passing_tds"] = _float_series(frame, "passing_tds")
    result["interceptions"] = _float_series(frame, "passing_interceptions")
    result["rushing_yards"] = _float_series(frame, "rushing_yards")
    result["rushing_tds"] = _float_series(frame, "rushing_tds")
    result["receptions"] = _float_series(frame, "receptions")
    result["receiving_yards"] = _float_series(frame, "receiving_yards")
    result["receiving_tds"] = _float_series(frame, "receiving_tds")
    result["fumbles_lost"] = _float_series(frame, "fumbles_lost_total")
    if not result["fumbles_lost"].any():
        result["fumbles_lost"] = (
            _float_series(frame, "sack_fumbles_lost")
            + _float_series(frame, "rushing_fumbles_lost")
            + _float_series(frame, "receiving_fumbles_lost")
        )
    result["two_point_conversions"] = (
        _float_series(frame, "passing_2pt_conversions")
        + _float_series(frame, "rushing_2pt_conversions")
        + _float_series(frame, "receiving_2pt_conversions")
    )
    result["fg_made_0_39"] = _float_series(frame, "fg_made_0_19") + _float_series(frame, "fg_made_20_29") + _float_series(frame, "fg_made_30_39")
    result["fg_made_40_49"] = _float_series(frame, "fg_made_40_49")
    result["fg_made_50_plus"] = _float_series(frame, "fg_made_50_59") + _float_series(frame, "fg_made_60_")
    result["fg_missed"] = (
        _float_series(frame, "fg_missed")
        + _float_series(frame, "fg_blocked")
    )
    result["extra_points_made"] = _float_series(frame, "pat_made")
    return result


def aggregate_seasons(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame()
    group_columns = ["player_id", "player_name", "position", "team", "season"]
    result = weekly[weekly["position"].isin(SKILL_POSITIONS)].groupby(group_columns, dropna=False).agg(
        {column: "sum" for column in STAT_COLUMNS} | {"game_id": "nunique"}
    ).reset_index()
    result = result.rename(columns={"game_id": "games"})
    result["games"] = result["games"].clip(lower=1)
    for column in STAT_COLUMNS:
        result[f"{column}_per_game"] = result[column] / result["games"]
    return result


def _feature_row(history: pd.DataFrame, player_id: str, season: int, position: str) -> dict[str, float]:
    rows = history[(history["player_id"] == player_id) & (history["position"] == position) & (history["season"] < season)].sort_values("season", ascending=False)
    features: dict[str, float] = {"position_code": float(SKILL_POSITIONS.index(position))}
    for index in range(2):
        row = rows.iloc[index] if index < len(rows) else None
        features[f"prior{index + 1}_games"] = float(row["games"]) if row is not None else 0.0
        features[f"prior{index + 1}_experience"] = float(row["season"] - rows["season"].min() + 1) if row is not None and len(rows) else 0.0
        for column in STAT_COLUMNS:
            features[f"prior{index + 1}_{column}"] = float(row[column]) if row is not None else 0.0
            features[f"prior{index + 1}_{column}_per_game"] = float(row[f"{column}_per_game"]) if row is not None else 0.0
    for column in STAT_COLUMNS:
        features[f"career_{column}"] = float(rows[column].mean()) if len(rows) else 0.0
        features[f"career_{column}_per_game"] = float(rows[f"{column}_per_game"].mean()) if len(rows) else 0.0
    features["career_games"] = float(rows["games"].sum()) if len(rows) else 0.0
    return features


def build_training_rows(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    feature_columns: dict[str, list[str]] = {}
    if history.empty:
        return pd.DataFrame(), feature_columns
    for (player_id, position), player_rows in history.groupby(["player_id", "position"], dropna=False):
        for season in sorted(player_rows["season"].unique()):
            features = _feature_row(history, str(player_id), int(season), str(position))
            if not features.get("prior1_games"):
                continue
            for column in STAT_COLUMNS:
                features["target_season"] = int(season)
                features["player_id"] = player_id
                features["position"] = position
                features[f"target_{column}"] = float(player_rows.loc[player_rows["season"] == season, column].sum())
            rows.append(features)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        feature_columns = {
            column: [item for item in frame.columns if item not in {"target_season", "player_id", "position", *[f"target_{target}" for target in STAT_COLUMNS]}]
            for column in STAT_COLUMNS
        }
    return frame, feature_columns


@dataclass
class PositionModel:
    estimator: RandomForestRegressor | None
    feature_columns: list[str]
    residual_scale: dict[str, float]
    fallback: dict[str, float]
    baseline_targets: set[str]

    def predict_all(self, features: Mapping[str, float]) -> dict[str, tuple[float, float, float]]:
        """Predict every component in one forest call for fast refreshes."""

        targets = list(self.fallback)
        values: list[float] | None = None
        if self.estimator is not None:
            vector = pd.DataFrame([{column: float(features.get(column, 0.0)) for column in self.feature_columns}])
            values = [float(value) for value in self.estimator.predict(vector)[0]]
        output: dict[str, tuple[float, float, float]] = {}
        for target_index, target in enumerate(targets):
            use_prior = target in self.baseline_targets and features.get(f"prior1_{target}", 0.0) > 0
            if values is None or use_prior:
                median = max(0.0, float(features.get(f"prior1_{target}", self.fallback.get(target, 0.0))))
            else:
                median = max(0.0, values[target_index])
            spread = max(0.1, self.residual_scale.get(target, 0.1))
            output[target] = (max(0.0, median - 1.28 * spread), median, max(0.0, median + 1.28 * spread))
        return output

    def predict(self, features: Mapping[str, float], target: str) -> tuple[float, float, float]:
        """Backward-compatible single-component accessor."""

        return self.predict_all(features)[target]


@dataclass
class FantasyProjectionModel:
    models: dict[str, PositionModel]
    history: pd.DataFrame
    metrics: dict[str, Any]
    model_version: str = "fantasy-v2-rf-components-calibrated"

    @classmethod
    def fit(cls, history: pd.DataFrame, holdout_season: int | None = None) -> "FantasyProjectionModel":
        training, feature_columns = build_training_rows(history)
        models: dict[str, PositionModel] = {}
        metrics: dict[str, Any] = {"holdout_season": holdout_season, "targets": {}}
        if training.empty:
            return cls(models=models, history=history, metrics=metrics)

        for position in SKILL_POSITIONS:
            position_training = training[training["position"] == position]
            columns = feature_columns.get(STAT_COLUMNS[0], [])
            fallback = {
                target: float(position_training[f"target_{target}"].median())
                if not position_training.empty else 0.0
                for target in STAT_COLUMNS
            }
            scales = {
                target: max(0.1, float(position_training[f"target_{target}"].std() or 0.1))
                if not position_training.empty else 0.1
                for target in STAT_COLUMNS
            }
            estimator: RandomForestRegressor | None = None
            baseline_targets: set[str] = set()
            if len(position_training) >= 20 and columns:
                def make_estimator() -> RandomForestRegressor:
                    return RandomForestRegressor(
                        n_estimators=24,
                        max_depth=10,
                        max_samples=0.8,
                        min_samples_leaf=3,
                        max_features=0.75,
                        random_state=42,
                        n_jobs=-1,
                    )

                if holdout_season is not None:
                    train_rows = position_training[position_training["target_season"] != holdout_season]
                    holdout = position_training[position_training["target_season"] == holdout_season]
                    if len(train_rows) >= 20 and len(holdout):
                        validation_estimator = make_estimator()
                        validation_estimator.fit(train_rows[columns], train_rows[[f"target_{target}" for target in STAT_COLUMNS]])
                        predicted = validation_estimator.predict(holdout[columns])
                        for index, target in enumerate(STAT_COLUMNS):
                            model_mae = float(mean_absolute_error(holdout[f"target_{target}"], predicted[:, index]))
                            baseline_mae = float(mean_absolute_error(holdout[f"target_{target}"], holdout[f"prior1_{target}"]))
                            use_baseline = model_mae >= baseline_mae
                            if use_baseline:
                                baseline_targets.add(target)
                            metrics["targets"].setdefault(position, {})[target] = {
                                "mae": round(model_mae, 3),
                                "baseline_mae": round(baseline_mae, 3),
                                "beats_baseline": not use_baseline,
                                "selection": "prior_season_baseline" if use_baseline else "random_forest",
                                "sample": int(len(holdout)),
                                "validation": "out_of_time_holdout",
                            }

                estimator = make_estimator()
                estimator.fit(position_training[columns], position_training[[f"target_{target}" for target in STAT_COLUMNS]])
                residuals = position_training[[f"target_{target}" for target in STAT_COLUMNS]].to_numpy() - estimator.predict(position_training[columns])
                scales = {target: max(0.1, float(np.std(residuals[:, index]))) for index, target in enumerate(STAT_COLUMNS)}
            models[position] = PositionModel(estimator, columns, scales, fallback, baseline_targets)
        return cls(models=models, history=history, metrics=metrics)

    def project_player(
        self,
        *,
        player_id: str,
        player_name: str,
        position: str,
        team: str | None,
        season: int,
        draft_round: float | None = None,
        experience: float | None = None,
    ) -> dict[str, Any]:
        position = position.upper()
        features = _feature_row(self.history, player_id, season, position)
        has_history = bool(features.get("prior1_games"))
        if experience is not None:
            features["experience"] = float(experience)
        statline: dict[str, float] = {}
        low_statline: dict[str, float] = {}
        high_statline: dict[str, float] = {}
        predictions = self.models.get(position).predict_all(features) if self.models.get(position) else {}
        for target in STAT_COLUMNS:
            prediction = predictions.get(target)
            if prediction is None:
                continue
            low, median, high = prediction
            if not has_history:
                # Rookie/low-history priors intentionally widen uncertainty.
                median *= 0.75 if target.endswith("_tds") else 0.65
                low = max(0.0, median * 0.2)
                high = median * 2.0 + 0.1
            statline[target] = round(median, 2)
            low_statline[target] = round(low, 2)
            high_statline[target] = round(high, 2)

        prior = self.history[(self.history["player_id"] == player_id) & (self.history["position"] == position)]
        games = float(prior.sort_values("season").tail(2)["games"].mean()) if len(prior) else 17.0
        games = min(17.0, max(1.0, games))
        # Component models learn season totals.  Convert that conditional
        # total into an expected-season total when recent availability points
        # below a full 17-game schedule (for example, a returning injured
        # player), while retaining the separate projected-games field.
        season_factor = games / 17.0
        if season_factor < 1.0:
            statline = {key: round(value * season_factor, 2) for key, value in statline.items()}
            low_statline = {key: round(value * season_factor, 2) for key, value in low_statline.items()}
            high_statline = {key: round(value * season_factor, 2) for key, value in high_statline.items()}
        scored = projection_points({**statline, "position": position}, FULL_PPR_SCORING)
        low_score = projection_points({**low_statline, "position": position}, FULL_PPR_SCORING)
        high_score = projection_points({**high_statline, "position": position}, FULL_PPR_SCORING)
        explanation = self._explanation(position, prior, has_history, draft_round)
        return {
            "player_id": player_id,
            "player_name": player_name,
            "position": position,
            "team": team,
            "season": season,
            "scope": "preseason",
            "week": 0,
            "projected_games": round(games, 1),
            "statline": statline,
            "statline_low": low_statline,
            "statline_high": high_statline,
            "points": scored["median"],
            "floor_points": low_score["floor"],
            "ceiling_points": high_score["ceiling"],
            "points_per_game": round(scored["median"] / games, 2),
            "confidence": "medium" if has_history else "low",
            "availability": "expected",
            "explanation": explanation,
            "model_version": self.model_version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _explanation(position: str, prior: pd.DataFrame, has_history: bool, draft_round: float | None) -> list[str]:
        if not has_history:
            return ["Limited NFL history; rookie/low-history prior with a wider range."]
        rows = prior.sort_values("season", ascending=False)
        latest = rows.iloc[0]
        reasons = [f"Weighted recent {position} usage over {int(latest['games'])} games."]
        if len(rows) > 1:
            delta = float(latest.get("receiving_yards_per_game", 0.0) - rows.iloc[1].get("receiving_yards_per_game", 0.0))
            if abs(delta) >= 8:
                reasons.append("Recent receiving efficiency moved the forecast.")
        if draft_round is not None and draft_round <= 2:
            reasons.append("Early draft capital supports the role prior.")
        return reasons


def build_model_from_weekly(weekly: pd.DataFrame, holdout_season: int | None = None) -> FantasyProjectionModel:
    normalized = normalize_weekly_stats(weekly)
    history = aggregate_seasons(normalized)
    return FantasyProjectionModel.fit(history, holdout_season=holdout_season)


def load_nflverse_weekly(seasons: Iterable[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    loaded = nfl.load_player_stats(list(seasons))
    return loaded.to_pandas() if hasattr(loaded, "to_pandas") else pd.DataFrame(loaded)


def load_current_players(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    players = nfl.load_players()
    frame = players.to_pandas() if hasattr(players, "to_pandas") else pd.DataFrame(players)
    frame = frame.rename(columns={"gsis_id": "player_id", "display_name": "player_name", "latest_team": "team"})
    raw_position = frame.get("position", "").fillna("").astype(str).str.upper()
    position_group = frame.get("position_group", "").fillna("").astype(str).str.upper()
    frame["position"] = raw_position.where(raw_position.isin(SKILL_POSITIONS), position_group)
    frame = frame[frame["position"].isin(SKILL_POSITIONS)]
    frame = frame[frame["team"].notna() & frame["team"].astype(str).ne("")]
    frame = frame[frame.get("last_season", season) >= season - 1]
    frame = frame.drop_duplicates("player_id")
    return frame[[column for column in ["player_id", "player_name", "position", "team", "draft_round", "years_of_experience"] if column in frame.columns]]


def build_team_defense_projections(weekly: pd.DataFrame, target_season: int) -> list[dict[str, Any]]:
    """Create transparent team-defense priors from defensive player totals."""

    if weekly.empty:
        return []
    prior_season = int(pd.to_numeric(weekly["season"], errors="coerce").max())
    raw = weekly[weekly["season"] == prior_season].copy()
    teams = sorted(raw["team"].dropna().astype(str).unique())
    if not teams:
        return []
    stat_map = {
        "dst_sacks": "def_sacks",
        "dst_interceptions": "def_interceptions",
        "dst_fumble_recoveries": "fumble_recovery_opp",
        "dst_tds": "def_tds",
        "dst_safeties": "def_safeties",
    }
    rows: list[dict[str, Any]] = []
    for team in teams:
        team_raw = raw[raw["team"].astype(str) == team]
        statline = {
            output: max(0.0, float(pd.to_numeric(team_raw.get(source, 0.0), errors="coerce").fillna(0.0).sum()))
            for output, source in stat_map.items()
        }
        statline["dst_points_allowed"] = 21.0
        points = projection_points({**statline, "position": "DST"}, FULL_PPR_SCORING)["median"]
        rows.append(
            {
                "player_id": f"DST-{team}",
                "player_name": f"{team} D/ST",
                "position": "DST",
                "team": team,
                "season": target_season,
                "scope": "preseason",
                "week": 0,
                "projected_games": 17.0,
                "statline": {key: round(value * 17 / 17, 2) for key, value in statline.items()},
                "statline_low": {key: round(value * 0.65, 2) for key, value in statline.items()},
                "statline_high": {key: round(value * 1.35, 2) for key, value in statline.items()},
                "points": round(points, 2),
                "floor_points": round(points * 0.65, 2),
                "ceiling_points": round(points * 1.35, 2),
                "points_per_game": round(points / 17, 2),
                "confidence": "low",
                "availability": "expected",
                "explanation": ["Team-defense prior from recent sacks, takeaways, recoveries, and defensive touchdowns."],
                "model_version": "fantasy-v1-team-defense-prior",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows


def build_preseason_projections(
    *,
    seasons: Iterable[int],
    target_season: int,
    holdout_season: int | None = None,
) -> tuple[list[dict[str, Any]], FantasyProjectionModel]:
    weekly = load_nflverse_weekly(seasons)
    model = build_model_from_weekly(weekly, holdout_season=holdout_season)
    players = load_current_players(target_season)
    projections: list[dict[str, Any]] = []
    for row in players.itertuples(index=False):
        projection = model.project_player(
            player_id=str(row.player_id),
            player_name=str(row.player_name),
            position=str(row.position),
            team=str(row.team),
            season=target_season,
            draft_round=getattr(row, "draft_round", None),
            experience=getattr(row, "years_of_experience", None),
        )
        if projection["points"] > 0 or projection["position"] in {"QB", "RB", "WR", "TE"}:
            projections.append(projection)
    projections.extend(build_team_defense_projections(weekly, target_season))
    return projections, model
