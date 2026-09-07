"""Point-in-time NFL v2 feature and evaluation primitives.

The feature builder is deliberately sequential: it emits a game's pregame
features before updating either team's state with that game's outcome. Model
features are directional differentials plus one explicit home-field term.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from src.features.rest_schedule import TEAM_COORDINATES, haversine_distance


STATE_METRICS = (
    "adj_off_epa",
    "adj_def_epa",
    "off_success",
    "def_success",
    "off_explosive",
    "def_explosive",
    "win",
    "margin",
    "points_for",
    "points_against",
)

LEAGUE_DEFAULTS = {
    "adj_off_epa": 0.0,
    "adj_def_epa": 0.0,
    "off_success": 0.45,
    "def_success": 0.45,
    "off_explosive": 0.12,
    "def_explosive": 0.12,
    "win": 0.5,
    "margin": 0.0,
    "points_for": 22.0,
    "points_against": 22.0,
}

ROLLING_WINDOWS = (3, 5, 10)
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LAR"}

FEATURE_COLUMNS = [
    "home_field",
    "rest_diff",
    "travel_advantage_km",
    "games_before_diff",
    "qb_stability_diff",
    "qb_prior_epa_diff",
    "playoff_strength_diff",
    *[
        f"{metric}_diff_{window}"
        for window in ROLLING_WINDOWS
        for metric in STATE_METRICS
    ],
]

PBP_AGGREGATE_SQL = """
WITH plays AS (
  SELECT
    CAST(game_id AS STRING) AS game_id,
    CAST(posteam AS STRING) AS team,
    CAST(defteam AS STRING) AS opponent,
    play_type,
    SAFE_CAST(epa AS FLOAT64) AS epa,
    SAFE_CAST(yards_gained AS FLOAT64) AS yards_gained,
    COALESCE(
      SAFE_CAST(JSON_VALUE(raw_record, '$.success') AS FLOAT64),
      IF(SAFE_CAST(epa AS FLOAT64) > 0, 1.0, 0.0)
    ) AS success,
    CAST(JSON_VALUE(raw_record, '$.passer_player_id') AS STRING) AS passer_id,
    SAFE_CAST(JSON_VALUE(raw_record, '$.qb_dropback') AS INT64) AS qb_dropback,
    ingested_at
  FROM `{project}.sports_edge_raw.raw_pbp`
  WHERE league = 'NFL'
    AND season BETWEEN @start_season AND @end_season
    AND posteam IS NOT NULL
    AND defteam IS NOT NULL
    AND epa IS NOT NULL
    AND play_type IN ('pass', 'run')
),
offense AS (
  SELECT
    game_id,
    team,
    AVG(epa) AS off_epa,
    AVG(success) AS off_success,
    AVG(IF((play_type = 'pass' AND yards_gained >= 15) OR
           (play_type = 'run' AND yards_gained >= 10), 1.0, 0.0)) AS off_explosive,
    COUNT(*) AS offensive_plays,
    MAX(ingested_at) AS source_max_ingested_at
  FROM plays
  GROUP BY game_id, team
),
defense AS (
  SELECT
    game_id,
    opponent AS team,
    AVG(epa) AS def_epa_allowed,
    AVG(success) AS def_success_allowed,
    AVG(IF((play_type = 'pass' AND yards_gained >= 15) OR
           (play_type = 'run' AND yards_gained >= 10), 1.0, 0.0)) AS def_explosive_allowed
  FROM plays
  GROUP BY game_id, team
),
quarterbacks AS (
  SELECT game_id, team, passer_id AS primary_qb_id
  FROM plays
  WHERE qb_dropback = 1 AND passer_id IS NOT NULL
  GROUP BY game_id, team, passer_id
  QUALIFY ROW_NUMBER() OVER (PARTITION BY game_id, team ORDER BY COUNT(*) DESC, passer_id) = 1
)
SELECT offense.*, defense.* EXCEPT(game_id, team), quarterbacks.primary_qb_id
FROM offense
JOIN defense USING (game_id, team)
LEFT JOIN quarterbacks USING (game_id, team)
ORDER BY game_id, team
"""

META_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_date",
    "game_type",
    "home_team",
    "away_team",
    "neutral_site",
    "home_score",
    "away_score",
    "home_win",
    "home_margin",
    "home_history_max_date",
    "away_history_max_date",
    "closing_market_home_prob",
    "closing_spread",
    "closing_total",
]


def _as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _american_implied_probability(value: object) -> float | None:
    odds = _as_float(value)
    if odds is None or odds == 0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / ((-odds) + 100.0)


def no_vig_home_probability(home_moneyline: object, away_moneyline: object) -> float | None:
    home = _american_implied_probability(home_moneyline)
    away = _american_implied_probability(away_moneyline)
    if home is None or away is None or home + away <= 0:
        return None
    return home / (home + away)


def normalize_team_name(value: object) -> str:
    team = str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def _team_coordinates(team: object) -> tuple[float, float] | None:
    # Correct the legacy Raiders coordinate, which still points to Oakland.
    if str(team) == "LV":
        return (36.0908, -115.1830)
    return TEAM_COORDINATES.get(str(team))


def _truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "neutral"}
    return bool(value)


@dataclass
class TeamState:
    season: int | None = None
    prior: dict[str, float] = field(default_factory=lambda: dict(LEAGUE_DEFAULTS))
    current: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    games: int = 0
    last_game_date: pd.Timestamp | None = None
    last_location: tuple[float, float] | None = None
    last_scheduled_date: pd.Timestamp | None = None
    last_scheduled_location: tuple[float, float] | None = None
    recent_qbs: list[str] = field(default_factory=list)

    def enter_season(self, season: int, league_means: Mapping[str, float]) -> None:
        if self.season is None:
            self.season = season
            self.prior = dict(league_means)
            return
        if season == self.season:
            return

        next_prior: dict[str, float] = {}
        for metric in STATE_METRICS:
            values = self.current.get(metric, [])
            observed = float(np.mean(values)) if values else self.prior.get(metric, league_means[metric])
            sample = len(values)
            shrinkage = sample / (sample + 8.0)
            next_prior[metric] = shrinkage * observed + (1.0 - shrinkage) * league_means[metric]
        self.prior = next_prior
        self.current = defaultdict(list)
        self.games = 0
        self.recent_qbs = []
        self.season = season

    def value(self, metric: str, window: int, prior_weight: float = 4.0) -> float:
        values = self.current.get(metric, [])[-window:]
        return float((sum(values) + prior_weight * self.prior[metric]) / (len(values) + prior_weight))

    def qb_stability(self) -> float:
        recent = [qb for qb in self.recent_qbs[-3:] if qb]
        if not recent:
            return 0.5
        return float(Counter(recent).most_common(1)[0][1] / len(recent))

    def expected_qb(self) -> str | None:
        recent = [qb for qb in self.recent_qbs[-3:] if qb]
        return Counter(recent).most_common(1)[0][0] if recent else None

    def rest_days(self, game_date: pd.Timestamp) -> float:
        if self.last_scheduled_date is None:
            return 14.0
        return float(np.clip((game_date.normalize() - self.last_scheduled_date.normalize()).days, 0, 28))

    def travel_km(self, location: tuple[float, float] | None) -> float:
        if self.last_scheduled_location is None or location is None:
            return 0.0
        return float(haversine_distance(*self.last_scheduled_location, *location))

    def advance_schedule(self, game_date: pd.Timestamp, location) -> None:
        self.last_scheduled_date = game_date
        self.last_scheduled_location = location

    def update(self, values: Mapping[str, float | None], qb_id: object, game_date: pd.Timestamp, location) -> None:
        for metric, value in values.items():
            number = _as_float(value)
            if metric in STATE_METRICS and number is not None:
                self.current[metric].append(number)
        if qb_id is not None and pd.notna(qb_id) and str(qb_id).lower() != "nan":
            self.recent_qbs.append(str(qb_id))
        self.games += 1
        self.last_game_date = game_date
        self.last_location = location
        self.advance_schedule(game_date, location)


def normalize_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    frame = schedules.copy()
    if "game_date" not in frame.columns and "gameday" in frame.columns:
        frame = frame.rename(columns={"gameday": "game_date"})
    required = {"game_id", "season", "week", "game_date", "home_team", "away_team", "home_score", "away_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"NFL schedules missing required columns: {missing}")

    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.tz_localize(None)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype("Int64")
    frame["league"] = frame.get("league", "NFL")
    frame["home_team"] = frame["home_team"].map(normalize_team_name)
    frame["away_team"] = frame["away_team"].map(normalize_team_name)
    leagues = set(frame["league"].dropna().astype(str).str.upper())
    if leagues != {"NFL"}:
        raise ValueError(f"NFL dataset contains unexpected leagues: {sorted(leagues)}")
    if frame["game_id"].isna().any() or frame["game_date"].isna().any():
        raise ValueError("NFL schedules contain null game_id or game_date values.")
    duplicated = frame.duplicated(["season", "game_id"], keep=False)
    if duplicated.any():
        ids = frame.loc[duplicated, "game_id"].astype(str).unique().tolist()
        raise ValueError(f"NFL schedules contain duplicate season/game keys: {ids[:10]}")
    return frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def _pbp_lookup(team_game_stats: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if team_game_stats.empty:
        return {}
    required = {"game_id", "team"}
    missing = sorted(required - set(team_game_stats.columns))
    if missing:
        raise ValueError(f"NFL team-game PBP data missing columns: {missing}")
    duplicated = team_game_stats.duplicated(["game_id", "team"], keep=False)
    if duplicated.any():
        raise ValueError("NFL team-game PBP data is not unique on game_id/team.")
    normalized = team_game_stats.copy()
    normalized["team"] = normalized["team"].map(normalize_team_name)
    duplicated = normalized.duplicated(["game_id", "team"], keep=False)
    if duplicated.any():
        raise ValueError("NFL team-game PBP aliases collide on game_id/team.")
    return {(str(row["game_id"]), str(row["team"])): row.to_dict() for _, row in normalized.iterrows()}


def _league_means(history: Mapping[str, Sequence[float]]) -> dict[str, float]:
    means = {}
    for metric in STATE_METRICS:
        values = history.get(metric, [])
        means[metric] = float(np.mean(values)) if values else LEAGUE_DEFAULTS[metric]
    return means


def build_nfl_feature_store(
    schedules: pd.DataFrame,
    team_game_stats: pd.DataFrame,
    *,
    include_unplayed: bool = False,
) -> pd.DataFrame:
    """Build one leakage-safe pregame row per NFL game.

    Unplayed games are emitted only when ``include_unplayed`` is true and never
    update rolling team, quarterback, or league state.
    """
    games = normalize_schedules(schedules)
    if not include_unplayed:
        games = games[games["home_score"].notna() & games["away_score"].notna()].copy()
    stats = _pbp_lookup(team_game_stats)
    states: dict[str, TeamState] = defaultdict(TeamState)
    global_history: dict[str, list[float]] = defaultdict(list)
    qb_history: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for game in games.to_dict("records"):
        game_date = pd.Timestamp(game["game_date"])
        season = int(game["season"])
        home_team = str(game["home_team"])
        away_team = str(game["away_team"])
        home = states[home_team]
        away = states[away_team]
        means = _league_means(global_history)
        home.enter_season(season, means)
        away.enter_season(season, means)

        neutral = str(game.get("location", "Home")).lower() == "neutral" or _truthy(
            game.get("neutral_site", False)
        )
        location = None if neutral else _team_coordinates(home_team)
        home_qb = home.expected_qb()
        away_qb = away.expected_qb()
        home_qb_epa = float(np.mean(qb_history[home_qb][-10:])) if home_qb and qb_history[home_qb] else 0.0
        away_qb_epa = float(np.mean(qb_history[away_qb][-10:])) if away_qb and qb_history[away_qb] else 0.0
        completed = pd.notna(game.get("home_score")) and pd.notna(game.get("away_score"))
        row: dict[str, Any] = {
            "game_id": str(game["game_id"]),
            "season": season,
            "week": int(game["week"]) if pd.notna(game.get("week")) else None,
            "game_date": game_date,
            "game_type": str(game.get("game_type") or "REG"),
            "home_team": home_team,
            "away_team": away_team,
            "neutral_site": neutral,
            "home_field": 0.0 if neutral else 1.0,
            "home_score": float(game["home_score"]) if completed else None,
            "away_score": float(game["away_score"]) if completed else None,
            "home_history_max_date": home.last_game_date,
            "away_history_max_date": away.last_game_date,
            "rest_diff": home.rest_days(game_date) - away.rest_days(game_date),
            "travel_advantage_km": away.travel_km(location) - home.travel_km(location),
            "games_before_diff": float(home.games - away.games),
            "qb_stability_diff": home.qb_stability() - away.qb_stability(),
            "qb_prior_epa_diff": home_qb_epa - away_qb_epa,
            "closing_market_home_prob": no_vig_home_probability(
                game.get("home_moneyline"), game.get("away_moneyline")
            ),
            "closing_spread": _as_float(game.get("spread_line")),
            "closing_total": _as_float(game.get("total_line")),
        }
        row["home_margin"] = row["home_score"] - row["away_score"] if completed else None
        row["home_win"] = int(row["home_margin"] > 0) if completed else None
        for window in ROLLING_WINDOWS:
            for metric in STATE_METRICS:
                row[f"{metric}_diff_{window}"] = home.value(metric, window) - away.value(metric, window)
        row["playoff_strength_diff"] = (
            row["margin_diff_10"] if row["game_type"].upper() != "REG" else 0.0
        )
        rows.append(row)

        if not completed:
            home.advance_schedule(game_date, location)
            away.advance_schedule(game_date, location)
            continue

        home_stats = stats.get((str(game["game_id"]), home_team), {})
        away_stats = stats.get((str(game["game_id"]), away_team), {})
        home_pre_def = home.value("adj_def_epa", 10)
        away_pre_def = away.value("adj_def_epa", 10)
        home_pre_off = home.value("adj_off_epa", 10)
        away_pre_off = away.value("adj_off_epa", 10)

        home_off_epa = _as_float(home_stats.get("off_epa"))
        home_def_epa = _as_float(home_stats.get("def_epa_allowed"))
        away_off_epa = _as_float(away_stats.get("off_epa"))
        away_def_epa = _as_float(away_stats.get("def_epa_allowed"))
        home_values = {
            "adj_off_epa": None if home_off_epa is None else home_off_epa - away_pre_def,
            "adj_def_epa": None if home_def_epa is None else home_def_epa - away_pre_off,
            "off_success": home_stats.get("off_success"),
            "def_success": home_stats.get("def_success_allowed"),
            "off_explosive": home_stats.get("off_explosive"),
            "def_explosive": home_stats.get("def_explosive_allowed"),
            "win": float(row["home_win"]),
            "margin": float(row["home_margin"]),
            "points_for": row["home_score"],
            "points_against": row["away_score"],
        }
        away_values = {
            "adj_off_epa": None if away_off_epa is None else away_off_epa - home_pre_def,
            "adj_def_epa": None if away_def_epa is None else away_def_epa - home_pre_off,
            "off_success": away_stats.get("off_success"),
            "def_success": away_stats.get("def_success_allowed"),
            "off_explosive": away_stats.get("off_explosive"),
            "def_explosive": away_stats.get("def_explosive_allowed"),
            "win": float(1 - row["home_win"]),
            "margin": -float(row["home_margin"]),
            "points_for": row["away_score"],
            "points_against": row["home_score"],
        }
        home.update(home_values, home_stats.get("primary_qb_id"), game_date, location)
        away.update(away_values, away_stats.get("primary_qb_id"), game_date, location)
        for team_stats in (home_stats, away_stats):
            qb_id = team_stats.get("primary_qb_id")
            off_epa = _as_float(team_stats.get("off_epa"))
            if qb_id is not None and pd.notna(qb_id) and off_epa is not None:
                qb_history[str(qb_id)].append(off_epa)
        for values in (home_values, away_values):
            for metric, value in values.items():
                number = _as_float(value)
                if number is not None:
                    global_history[metric].append(number)

    result = pd.DataFrame(rows)
    audit_nfl_feature_store(result, raise_on_error=True)
    return result[[*META_COLUMNS, *FEATURE_COLUMNS]]


def audit_nfl_feature_store(frame: pd.DataFrame, *, raise_on_error: bool = False) -> dict[str, Any]:
    issues: list[str] = []
    duplicate_rows = int(frame.duplicated(["season", "game_id"]).sum()) if not frame.empty else 0
    if duplicate_rows:
        issues.append(f"{duplicate_rows} duplicate season/game feature rows")
    game_dates = pd.to_datetime(frame["game_date"], errors="coerce") if "game_date" in frame else pd.Series(dtype="datetime64[ns]")
    home_dates = pd.to_datetime(frame["home_history_max_date"], errors="coerce") if "home_history_max_date" in frame else pd.Series(pd.NaT, index=frame.index)
    away_dates = pd.to_datetime(frame["away_history_max_date"], errors="coerce") if "away_history_max_date" in frame else pd.Series(pd.NaT, index=frame.index)
    future_home = home_dates >= game_dates
    future_away = away_dates >= game_dates
    leakage_rows = int((future_home.fillna(False) | future_away.fillna(False)).sum()) if not frame.empty else 0
    if leakage_rows:
        issues.append(f"{leakage_rows} rows use same-day or future team history")
    invalid_prob = 0
    if "closing_market_home_prob" in frame:
        market = pd.to_numeric(frame["closing_market_home_prob"], errors="coerce")
        invalid_prob = int(((market <= 0) | (market >= 1)).fillna(False).sum())
        if invalid_prob:
            issues.append(f"{invalid_prob} invalid market probabilities")

    missingness = {
        column: float(frame[column].isna().mean())
        for column in FEATURE_COLUMNS
        if column in frame
    }
    audit = {
        "rows": int(len(frame)),
        "seasons": sorted(pd.to_numeric(frame.get("season"), errors="coerce").dropna().astype(int).unique().tolist()),
        "duplicate_rows": duplicate_rows,
        "leakage_rows": leakage_rows,
        "invalid_market_probability_rows": invalid_prob,
        "feature_missingness": missingness,
        "issues": issues,
        "ready": not issues,
    }
    if raise_on_error and issues:
        raise ValueError("NFL feature audit failed: " + "; ".join(issues))
    return audit


def mirrored_feature_row(row: Mapping[str, object]) -> dict[str, float]:
    """Return away/home-swapped directional features for a neutral-site game."""
    if _as_float(row.get("home_field")) != 0.0:
        raise ValueError("Swap symmetry is defined only for neutral-site rows.")
    return {
        column: 0.0 if column == "home_field" else -float(row[column])
        for column in FEATURE_COLUMNS
    }


def dataframe_fingerprint(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = list(columns or frame.columns)
    normalized = frame[selected].copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype("string")
    normalized = normalized.sort_values(selected, kind="mergesort", na_position="first").reset_index(drop=True)
    payload = pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    return sha256(payload).hexdigest()


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(y_true))
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            error += mask.sum() / total * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return float(error)


def calibration_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    bucket = pd.cut(probability, np.linspace(0, 1, bins + 1), include_lowest=True)
    table = pd.DataFrame({"actual": y_true, "probability": probability, "bucket": bucket})
    grouped = table.groupby("bucket", observed=True).agg(
        rows=("actual", "size"),
        avg_probability=("probability", "mean"),
        actual_rate=("actual", "mean"),
    )
    return [
        {
            "bucket": str(index),
            "rows": int(row.rows),
            "avg_probability": float(row.avg_probability),
            "actual_rate": float(row.actual_rate),
            "bias": float(row.avg_probability - row.actual_rate),
        }
        for index, row in grouped.iterrows()
    ]


def probability_metrics(y_true: Sequence[int], probability: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    metrics: dict[str, Any] = {
        "rows": int(len(y)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": expected_calibration_error(y, p),
        "avg_probability": float(p.mean()),
        "actual_rate": float(y.mean()),
        "home_probability_bias": float(p.mean() - y.mean()),
        "sharpness_std": float(p.std()),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "calibration": calibration_table(y, p),
    }
    if len(np.unique(y)) == 2:
        metrics["auc"] = float(roc_auc_score(y, p))
    return metrics


def margin_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return {
        "rows": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "avg_actual_margin": float(y.mean()),
        "avg_predicted_margin": float(p.mean()),
        "home_margin_bias": float(np.mean(p - y)),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
