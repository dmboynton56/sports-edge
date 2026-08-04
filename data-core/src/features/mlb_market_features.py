"""Leakage-aware MLB market features for moneyline, totals, run line, and Ks.

Weather values in this store are observed game-time boxscore values used as a
proxy for a pregame forecast. Consumers must retain that caveat in reporting.
All rolling statistics are emitted before the current game updates any state.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import json
import re
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from src.models.mlb_winner_model import (
    _empty_states,
    _feature_row_for_game,
    _prepare_games,
    _update_states_for_completed_game,
)


STARTER_PRIORS = {
    "k9_last5": 8.2,
    "bb9_last5": 3.2,
    "era_proxy_last5": 4.3,
    "pitches_last3_avg": 90.0,
    "outs_per_start_last5": 15.0,
    "days_since_last_start": 7.0,
}

OBSERVED_WEATHER_COLUMNS = [
    "temp_f",
    "weather_condition",
    "wind_mph",
    "wind_dir",
    "wind_out",
    "wind_in",
    "wind_cross",
    "is_dome_or_closed",
]

_HANDEDNESS_CACHE = Path(__file__).resolve().parents[2] / "notebooks/cache/mlb_player_handedness_cache.json"
_START_LINE_COLUMNS = (
    "starter_outs",
    "starter_strikeouts",
    "starter_walks",
    "starter_earned_runs",
    "starter_pitches",
)


def _nullable_int(value: object) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _nullable_float(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _load_handedness() -> dict[int, float]:
    """Read pitcher throw hand from the local cache; never fetch."""
    if not _HANDEDNESS_CACHE.exists():
        return {}
    with _HANDEDNESS_CACHE.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        int(player_id): float(details.get("pitch_hand") == "R")
        for player_id, details in raw.items()
        if details.get("pitch_hand") in {"L", "R"}
    }


def _boxscore_by_game(boxscores: pd.DataFrame) -> dict[int, dict]:
    if boxscores.empty:
        return {}
    if "game_pk" not in boxscores:
        raise ValueError("Boxscores must include game_pk.")
    if boxscores["game_pk"].duplicated().any():
        raise ValueError("Boxscores must contain at most one row per game_pk.")
    return {
        int(record["game_pk"]): record
        for record in boxscores.to_dict(orient="records")
    }


def _venue_details(venue_meta: Mapping, venue_id: Optional[int]) -> dict:
    if venue_id is None:
        return {}
    details = venue_meta.get(str(venue_id), venue_meta.get(venue_id, {}))
    return details if isinstance(details, dict) else {}


def _starter_features(
    prefix: str,
    pitcher_id: Optional[int],
    game_date: pd.Timestamp,
    histories: Mapping[int, deque],
    career_starts: Mapping[int, int],
    start_dates: Mapping[int, deque],
    handedness: Mapping[int, float],
) -> dict[str, float]:
    history = list(histories.get(pitcher_id, ())) if pitcher_id is not None else []
    last_five = history[-5:]
    last_three = history[-3:]
    outs = sum(line["outs"] for line in last_five)
    dates = start_dates.get(pitcher_id, ()) if pitcher_id is not None else ()
    cutoff = game_date.normalize() - pd.Timedelta(days=365)
    starts_last365 = sum(cutoff <= date.normalize() < game_date.normalize() for date in dates)

    if last_five and outs > 0:
        k9 = sum(line["strikeouts"] for line in last_five) * 27.0 / outs
        bb9 = sum(line["walks"] for line in last_five) * 27.0 / outs
        era = sum(line["earned_runs"] for line in last_five) * 27.0 / outs
        outs_per_start = outs / len(last_five)
    else:
        k9 = STARTER_PRIORS["k9_last5"]
        bb9 = STARTER_PRIORS["bb9_last5"]
        era = STARTER_PRIORS["era_proxy_last5"]
        outs_per_start = STARTER_PRIORS["outs_per_start_last5"]

    pitches = [line["pitches"] for line in last_three if np.isfinite(line["pitches"])]
    pitches_avg = float(np.mean(pitches)) if pitches else STARTER_PRIORS["pitches_last3_avg"]
    days_since = (
        float((game_date.normalize() - dates[-1].normalize()).days)
        if dates
        else STARTER_PRIORS["days_since_last_start"]
    )
    return {
        f"{prefix}_starter_k9_last5": float(k9),
        f"{prefix}_starter_bb9_last5": float(bb9),
        f"{prefix}_starter_era_proxy_last5": float(era),
        f"{prefix}_starter_pitches_last3_avg": pitches_avg,
        f"{prefix}_starter_outs_per_start_last5": float(outs_per_start),
        f"{prefix}_starter_starts_last365": float(starts_last365),
        f"{prefix}_starter_days_since_last_start": days_since,
        f"{prefix}_starter_career_starts_prior": float(career_starts.get(pitcher_id, 0)),
        f"{prefix}_starter_has_history": float(bool(history)),
        f"{prefix}_starter_throws_r": handedness.get(pitcher_id, float("nan")),
    }


def _team_market_features(prefix: str, team_id: int, histories: Mapping[int, deque]) -> dict[str, float]:
    history = list(histories.get(team_id, ()))
    if history:
        scored = float(np.mean([line["runs_scored"] for line in history]))
        allowed = float(np.mean([line["runs_allowed"] for line in history]))
        total = float(np.mean([line["total_runs"] for line in history]))
        strikeouts = [line["bat_strikeouts"] for line in history if np.isfinite(line["bat_strikeouts"])]
        bat_k = float(np.mean(strikeouts)) if strikeouts else 8.5
    else:
        scored, allowed, total, bat_k = 4.4, 4.4, 8.8, 8.5
    return {
        f"{prefix}_runs_scored_pg_15": scored,
        f"{prefix}_runs_allowed_pg_15": allowed,
        f"{prefix}_team_total_pg_15": total,
        f"{prefix}_team_bat_k_pg_15": bat_k,
    }


def _wind_buckets(wind_dir: object) -> tuple[float, float, float]:
    text = "" if wind_dir is None or pd.isna(wind_dir) else str(wind_dir).strip().lower()
    return (
        float(text.startswith("out to")),
        float(text.startswith("in from")),
        float("l to r" in text or "r to l" in text),
    )


def _first_pitch_hour(first_pitch: object, game_datetime: pd.Timestamp) -> int:
    text = "" if first_pitch is None or pd.isna(first_pitch) else str(first_pitch).strip().rstrip(".")
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*([AP]M)", text, flags=re.IGNORECASE)
    if match:
        hour = int(match.group(1)) % 12
        if match.group(3).upper() == "PM":
            hour += 12
        return hour
    return pd.Timestamp(game_datetime).hour


def _weather_features(boxscore: Mapping, venue: Mapping, game_datetime: pd.Timestamp) -> dict:
    wind_dir = boxscore.get("wind_dir")
    wind_mph = _nullable_float(boxscore.get("wind_mph"))
    wind_out, wind_in, wind_cross = _wind_buckets(wind_dir)
    roof = str(venue.get("roofType") or "").strip().lower()
    no_wind = (
        wind_dir is None
        or pd.isna(wind_dir)
        or str(wind_dir).strip().lower() in {"", "none"}
        or (np.isfinite(wind_mph) and wind_mph == 0)
    )
    fixed_roof = "dome" in roof or "closed" in roof
    retractable_closed = "retract" in roof and no_wind
    return {
        "temp_f": _nullable_float(boxscore.get("temp_f")),
        "wind_mph": wind_mph,
        "wind_dir": wind_dir,
        "wind_out": wind_out,
        "wind_in": wind_in,
        "wind_cross": wind_cross,
        "is_dome_or_closed": float(fixed_roof or retractable_closed),
        "is_day_game": float(_first_pitch_hour(boxscore.get("first_pitch"), game_datetime) < 17),
        "elevation": _nullable_float(venue.get("elevation")),
    }


def _actual_starter_line(boxscore: Mapping, prefix: str) -> tuple[Optional[int], Optional[dict]]:
    pitcher_id = _nullable_int(boxscore.get(f"{prefix}_actual_starter_id"))
    values = {name.removeprefix("starter_"): _nullable_float(boxscore.get(f"{prefix}_{name}")) for name in _START_LINE_COLUMNS}
    required = ("outs", "strikeouts", "walks", "earned_runs")
    if pitcher_id is None or not all(np.isfinite(values[name]) for name in required):
        return pitcher_id, None
    return pitcher_id, values


def _probable_matches(probable_id: Optional[int], actual_id: Optional[int]) -> object:
    if probable_id is None or actual_id is None:
        return pd.NA
    return bool(probable_id == actual_id)


def build_mlb_market_features(
    games: pd.DataFrame,
    boxscores: pd.DataFrame,
    venue_meta: Mapping,
    min_prior_games: int = 5,
) -> pd.DataFrame:
    """Build the v2 completed-game store using only prior-game rolling state.

    ``temp_f`` and wind fields are observed at game time, not archived pregame
    forecasts. They are intentionally retained as forecast proxies and marked
    as ``observed-weather`` in the build audit.
    """
    if games.empty:
        raise ValueError("No MLB games provided.")

    completed = _prepare_games(games)
    completed = completed[completed["home_score"].notna() & completed["away_score"].notna()].copy()
    boxscore_rows = _boxscore_by_game(boxscores)
    handedness = _load_handedness()

    states, pitcher_states, venue_states = _empty_states()
    pitcher_histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
    pitcher_start_dates: dict[int, deque] = defaultdict(deque)
    pitcher_career_starts: dict[int, int] = defaultdict(int)
    team_histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=15))
    rows: list[dict] = []

    for game in completed.itertuples(index=False):
        game_date = pd.Timestamp(game.game_date)
        row = _feature_row_for_game(game, states, pitcher_states, venue_states, include_target=True)
        boxscore = boxscore_rows.get(int(game.game_pk), {})
        venue = _venue_details(venue_meta, _nullable_int(getattr(game, "venue_id", None)))
        home_probable_id = _nullable_int(getattr(game, "home_probable_pitcher_id", None))
        away_probable_id = _nullable_int(getattr(game, "away_probable_pitcher_id", None))

        row.update(_starter_features("home", home_probable_id, game_date, pitcher_histories, pitcher_career_starts, pitcher_start_dates, handedness))
        row.update(_starter_features("away", away_probable_id, game_date, pitcher_histories, pitcher_career_starts, pitcher_start_dates, handedness))
        row.update(_team_market_features("home", int(game.home_team_id), team_histories))
        row.update(_team_market_features("away", int(game.away_team_id), team_histories))
        row.update(_weather_features(boxscore, venue, pd.Timestamp(game.game_datetime)))

        for key, value in boxscore.items():
            if key != "game_pk" and key not in row:
                row[key] = value

        home_actual_id, home_line = _actual_starter_line(boxscore, "home")
        away_actual_id, away_line = _actual_starter_line(boxscore, "away")
        row.update(
            {
                "total_runs": int(game.home_score) + int(game.away_score),
                "home_cover_15": int(int(game.run_diff) >= 2),
                "home_starter_ks_label": _nullable_float(boxscore.get("home_starter_strikeouts")),
                "away_starter_ks_label": _nullable_float(boxscore.get("away_starter_strikeouts")),
                "home_probable_matches_actual": _probable_matches(home_probable_id, home_actual_id),
                "away_probable_matches_actual": _probable_matches(away_probable_id, away_actual_id),
            }
        )

        for suffix in (
            "k9_last5", "bb9_last5", "era_proxy_last5", "pitches_last3_avg",
            "outs_per_start_last5", "starts_last365", "days_since_last_start",
            "career_starts_prior", "has_history", "throws_r",
        ):
            row[f"starter_{suffix}_diff"] = row[f"home_starter_{suffix}"] - row[f"away_starter_{suffix}"]
        row["runs_scored_pg_15_diff"] = row["home_runs_scored_pg_15"] - row["away_runs_scored_pg_15"]
        row["runs_allowed_pg_15_diff"] = row["home_runs_allowed_pg_15"] - row["away_runs_allowed_pg_15"]
        row["team_total_pg_15_diff"] = row["home_team_total_pg_15"] - row["away_team_total_pg_15"]
        row["team_bat_k_pg_15_diff"] = row["home_team_bat_k_pg_15"] - row["away_team_bat_k_pg_15"]
        row["combined_expected_total"] = (
            row["home_runs_scored_pg_15"] + row["away_runs_allowed_pg_15"]
            + row["away_runs_scored_pg_15"] + row["home_runs_allowed_pg_15"]
        ) / 2.0
        rows.append(row)

        _update_states_for_completed_game(game, states, pitcher_states, venue_states)
        for prefix, team_id, runs_scored, runs_allowed in (
            ("home", int(game.home_team_id), int(game.home_score), int(game.away_score)),
            ("away", int(game.away_team_id), int(game.away_score), int(game.home_score)),
        ):
            team_histories[team_id].append(
                {
                    "runs_scored": runs_scored,
                    "runs_allowed": runs_allowed,
                    "total_runs": runs_scored + runs_allowed,
                    "bat_strikeouts": _nullable_float(boxscore.get(f"{prefix}_team_strikeouts")),
                }
            )
        for pitcher_id, line in ((home_actual_id, home_line), (away_actual_id, away_line)):
            if pitcher_id is not None and line is not None:
                pitcher_histories[pitcher_id].append(line)
                pitcher_start_dates[pitcher_id].append(game_date)
                pitcher_career_starts[pitcher_id] += 1

    features = pd.DataFrame(rows)
    if min_prior_games > 0:
        features = features[
            (features["home_games_played"] >= min_prior_games)
            & (features["away_games_played"] >= min_prior_games)
        ].copy()
    return features.reset_index(drop=True)


__all__ = ["OBSERVED_WEATHER_COLUMNS", "STARTER_PRIORS", "build_mlb_market_features"]
