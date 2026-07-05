#!/usr/bin/env python3
"""Generate daily MLB home run probability candidates.

This is a probability-first v1. It uses public MLB Stats API schedule and
boxscore data, projects likely lineups from recent batting order history, and
labels every row with data-quality flags so the web app can treat the market as
candidate output until a trained/validated player-prop model replaces it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.data.mlb_fetcher import fetch_mlb_schedule  # noqa: E402
from json_utils import dumps_strict  # noqa: E402
from src.models.mlb_home_run_model import (  # noqa: E402
    FEATURE_COLUMNS,
    MODEL_VERSION as TRAINED_MODEL_VERSION,
    build_hr_feature_values,
    heuristic_hr_probability,
    load_hr_artifact,
    predict_hr_probability,
    quality_flags_for_features,
    top_feature_payload,
)
from src.models.mlb_hr_recency import games_since_last_hr, recency_quality_flags  # noqa: E402
from src.models.mlb_hr_statcast_features import (  # noqa: E402
    DEFAULT_STATCAST_MIN_BBE,
    DEFAULT_STATCAST_CACHE,
    build_torch_candidate_features,
)
from src.models.mlb_hr_torch_inference import (  # noqa: E402
    STATCAST_BLEND_MODEL_VERSION,
    apply_heuristic_blend,
    load_torch_hr_artifact,
    predict_torch_probs,
)


MLB_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
DEFAULT_CACHE = ROOT / "notebooks" / "cache" / "mlb_home_run_boxscores_2026.jsonl"
DEFAULT_MODEL_ARTIFACT = ROOT / "models" / "mlb_hr_model_v1.joblib"
DEFAULT_TORCH_ARTIFACT = ROOT / "models" / "mlb_hr_torch_statcast_model_v1.pt"
DEFAULT_CSV_OUT = ROOT / "notebooks" / "cache" / "mlb_home_run_predictions.csv"
DEFAULT_STATCAST_CSV_OUT = ROOT / "notebooks" / "cache" / "mlb_home_run_predictions_statcast_blend.csv"
DEFAULT_WEB_OUT = REPO_ROOT / "web" / "public" / "data" / "mlb_home_runs.json"
V1_MODEL_KEY = "mlb-hr-v1"
STATCAST_BLEND_KEY = STATCAST_BLEND_MODEL_VERSION
MODEL_AGREEMENT_CONSENSUS = "Consensus"
MODEL_AGREEMENT_V1_ONLY = "V1 only"
MODEL_AGREEMENT_STATCAST_BOOST = "Statcast boost"
MODEL_AGREEMENT_STATCAST_FADE = "Statcast fade"
MODEL_AGREEMENT_MISSING_STATCAST = "Missing Statcast"
MODEL_AGREEMENT_RANK_THRESHOLD = 5
STATCAST_HEALTH_COLUMNS = [
    "statcast_coverage",
    "statcast_ready_rows",
    "statcast_total_rows",
    "statcast_artifact_loaded",
]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", ".---"):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _lineup_slot(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        raw = int(str(value))
    except ValueError:
        return None
    if raw >= 100:
        return max(1, min(9, raw // 100))
    return max(1, min(9, raw))


def _hash_jitter(*parts: Any, scale: float = 0.0008) -> float:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * scale


def _read_cached_boxscores(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "game_pk" in row:
            out[int(row["game_pk"])] = row
    return out


def _write_cached_boxscores(path: Path, payloads: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(payloads[key], sort_keys=True) for key in sorted(payloads)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _fetch_boxscore(game_pk: int, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(MLB_BOXSCORE_URL.format(game_pk=int(game_pk)), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    payload["game_pk"] = int(game_pk)
    return payload


def _team_id(team_payload: dict[str, Any]) -> int | None:
    return _safe_int((team_payload.get("team") or {}).get("id"), default=0) or None


def _player_payload(team_payload: dict[str, Any], player_id: int) -> dict[str, Any]:
    return (team_payload.get("players") or {}).get(f"ID{int(player_id)}") or {}


def _extract_boxscore_rows(payload: dict[str, Any], schedule_row: pd.Series) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    game_pk = int(payload.get("game_pk"))
    game_date = pd.to_datetime(schedule_row["game_date"]).date()
    teams = payload.get("teams") or {}
    batting_rows: list[dict[str, Any]] = []
    pitching_rows: list[dict[str, Any]] = []

    for side in ("home", "away"):
        team_payload = teams.get(side) or {}
        opponent_payload = teams.get("away" if side == "home" else "home") or {}
        team_id = _team_id(team_payload)
        opponent_id = _team_id(opponent_payload)
        for player_id in team_payload.get("batters") or []:
            player = _player_payload(team_payload, int(player_id))
            stats = (player.get("stats") or {}).get("batting") or {}
            pa = _safe_int(stats.get("plateAppearances"))
            batting_rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": game_date.isoformat(),
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "home_away": side,
                    "player_id": int(player_id),
                    "player_name": (player.get("person") or {}).get("fullName"),
                    "position": (player.get("position") or {}).get("abbreviation"),
                    "lineup_slot": _lineup_slot(player.get("battingOrder")),
                    "plate_appearances": pa,
                    "at_bats": _safe_int(stats.get("atBats")),
                    "home_runs": _safe_int(stats.get("homeRuns")),
                }
            )
        for player_id in team_payload.get("pitchers") or []:
            player = _player_payload(team_payload, int(player_id))
            stats = (player.get("stats") or {}).get("pitching") or {}
            pitching_rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": game_date.isoformat(),
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "home_away": side,
                    "player_id": int(player_id),
                    "player_name": (player.get("person") or {}).get("fullName"),
                    "batters_faced": _safe_int(stats.get("battersFaced")),
                    "home_runs_allowed": _safe_int(stats.get("homeRuns")),
                    "games_started": _safe_int(stats.get("gamesStarted")),
                }
            )
    return batting_rows, pitching_rows


def _load_history(schedule: pd.DataFrame, cache_path: Path, start_date: date, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = schedule[
        (schedule["completed"] == True)  # noqa: E712
        & (schedule["game_date"].dt.date >= start_date)
        & (schedule["game_date"].dt.date < end_date)
    ].copy()
    cached = _read_cached_boxscores(cache_path)
    changed = False
    for i, game_pk in enumerate(completed["game_pk"].astype(int).tolist(), start=1):
        if game_pk not in cached:
            try:
                cached[game_pk] = _fetch_boxscore(game_pk)
                changed = True
                time.sleep(0.04)
            except Exception as exc:  # noqa: BLE001
                cached[game_pk] = {"game_pk": game_pk, "error": str(exc)}
                changed = True
        if i % 100 == 0:
            if changed:
                _write_cached_boxscores(cache_path, cached)
                changed = False
            print(f"Prepared {i} MLB boxscores...", flush=True)
    if changed:
        _write_cached_boxscores(cache_path, cached)

    batting: list[dict[str, Any]] = []
    pitching: list[dict[str, Any]] = []
    schedule_by_pk = {int(row["game_pk"]): row for _, row in completed.iterrows()}
    for game_pk in completed["game_pk"].astype(int):
        payload = cached.get(int(game_pk), {})
        if payload.get("error"):
            continue
        b_rows, p_rows = _extract_boxscore_rows(payload, schedule_by_pk[int(game_pk)])
        batting.extend(b_rows)
        pitching.extend(p_rows)
    return pd.DataFrame(batting), pd.DataFrame(pitching)


def _project_lineup(history: pd.DataFrame, team_id: int, as_of: date, max_players: int = 9) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    frame = history[(history["team_id"] == team_id) & (pd.to_datetime(history["game_date"]).dt.date < as_of)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    recent_cut = pd.Timestamp(as_of) - pd.Timedelta(days=28)
    recent = frame[frame["game_date"] >= recent_cut]
    if recent.empty:
        recent = frame.sort_values("game_date").tail(200)
    grouped = (
        recent.groupby(["player_id", "player_name"], dropna=False)
        .agg(
            recent_pa=("plate_appearances", "sum"),
            recent_hr=("home_runs", "sum"),
            starts=("game_pk", "nunique"),
            lineup_slot=("lineup_slot", lambda s: int(round(pd.to_numeric(s, errors="coerce").dropna().median())) if pd.to_numeric(s, errors="coerce").dropna().size else None),
            last_game=("game_date", "max"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["recent_pa"] > 0].copy()
    if grouped.empty:
        return grouped
    grouped["lineup_slot"] = grouped["lineup_slot"].fillna(9).astype(int)
    grouped = grouped.sort_values(["starts", "recent_pa", "last_game"], ascending=[False, False, False])
    grouped = grouped.head(max_players).copy()
    grouped = grouped.sort_values(["lineup_slot", "recent_pa"], ascending=[True, False]).reset_index(drop=True)
    grouped["lineup_slot"] = range(1, len(grouped) + 1)
    return grouped


def _player_rates(history: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    frame = history[pd.to_datetime(history["game_date"]).dt.date < as_of].copy()
    career = (
        frame.groupby(["player_id", "player_name"], dropna=False)
        .agg(pa=("plate_appearances", "sum"), hr=("home_runs", "sum"), games=("game_pk", "nunique"))
        .reset_index()
    )
    recent_cut = pd.Timestamp(as_of) - pd.Timedelta(days=28)
    recent = frame[pd.to_datetime(frame["game_date"]) >= recent_cut]
    recent_rates = (
        recent.groupby("player_id")
        .agg(recent_pa=("plate_appearances", "sum"), recent_hr=("home_runs", "sum"))
        .reset_index()
    )
    return career.merge(recent_rates, on="player_id", how="left").fillna({"recent_pa": 0, "recent_hr": 0})


def _pitcher_rates(pitching: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if pitching.empty:
        return pd.DataFrame()
    frame = pitching[pd.to_datetime(pitching["game_date"]).dt.date < as_of].copy()
    return (
        frame.groupby(["player_id", "player_name"], dropna=False)
        .agg(batters_faced=("batters_faced", "sum"), home_runs_allowed=("home_runs_allowed", "sum"), starts=("games_started", "sum"))
        .reset_index()
    )


def _venue_factors(history: pd.DataFrame, schedule: pd.DataFrame, as_of: date) -> dict[int, float]:
    if history.empty:
        return {}
    completed = schedule[schedule["game_date"].dt.date < as_of][["game_pk", "venue_id"]].copy()
    frame = history.merge(completed, on="game_pk", how="left")
    league_hr_pa = max(frame["home_runs"].sum() / max(frame["plate_appearances"].sum(), 1), 0.001)
    factors: dict[int, float] = {}
    for venue_id, group in frame.dropna(subset=["venue_id"]).groupby("venue_id"):
        if group["plate_appearances"].sum() < 250:
            continue
        rate = group["home_runs"].sum() / max(group["plate_appearances"].sum(), 1)
        factors[int(venue_id)] = float(np.clip(rate / league_hr_pa, 0.75, 1.25))
    return factors


def _candidate_features(
    *,
    batter: pd.Series,
    player_rate: dict[str, Any] | None,
    pitcher_rate: dict[str, Any] | None,
    venue_factor: float,
    league_hr_pa: float,
    is_home: bool,
) -> dict[str, float]:
    pa = _safe_float((player_rate or {}).get("pa"), 0.0) or 0.0
    hr = _safe_float((player_rate or {}).get("hr"), 0.0) or 0.0
    recent_pa = _safe_float((player_rate or {}).get("recent_pa"), 0.0) or 0.0
    recent_hr = _safe_float((player_rate or {}).get("recent_hr"), 0.0) or 0.0
    pitcher_bf = _safe_float((pitcher_rate or {}).get("batters_faced"), 0.0) or 0.0
    pitcher_hr = _safe_float((pitcher_rate or {}).get("home_runs_allowed"), 0.0) or 0.0
    slot = int(batter.get("lineup_slot") or 9)
    features = build_hr_feature_values(
        batter_pa=pa,
        batter_hr=hr,
        batter_games=(player_rate or {}).get("games", 0.0),
        batter_recent_pa=recent_pa,
        batter_recent_hr=recent_hr,
        pitcher_bf=pitcher_bf,
        pitcher_hr_allowed=pitcher_hr,
        venue_factor=venue_factor,
        league_hr_pa=league_hr_pa,
        lineup_slot=slot,
        is_home=is_home,
    )
    heuristic_probability, baseline = heuristic_hr_probability(features)
    features["baseline_probability"] = baseline
    features["heuristic_probability"] = heuristic_probability
    return features


def _score_probability(
    *,
    batter: pd.Series,
    player_rate: dict[str, Any] | None,
    pitcher_rate: dict[str, Any] | None,
    venue_factor: float,
    league_hr_pa: float,
    probable_pitcher_known: bool,
    is_home: bool,
    model_artifact: dict[str, Any] | None,
) -> tuple[float, float, list[str], list[dict[str, Any]], str]:
    features = _candidate_features(
        batter=batter,
        player_rate=player_rate,
        pitcher_rate=pitcher_rate,
        venue_factor=venue_factor,
        league_hr_pa=league_hr_pa,
        is_home=is_home,
    )
    heuristic_probability = features["heuristic_probability"]
    baseline = features["baseline_probability"]
    model_probability = predict_hr_probability(model_artifact, features)
    if model_probability is None:
        probability = heuristic_probability + _hash_jitter(batter.get("player_id"), batter.get("player_name"))
        probability = float(np.clip(probability, 0.002, 0.38))
        model_version = "mlb-hr-v1-heuristic"
    else:
        probability = model_probability + _hash_jitter(batter.get("player_id"), batter.get("player_name"), scale=0.0003)
        probability = float(np.clip(probability, 0.001, 0.45))
        model_version = str((model_artifact or {}).get("model_version") or TRAINED_MODEL_VERSION)
    flags = quality_flags_for_features(features, probable_pitcher_known=probable_pitcher_known)
    return probability, baseline, flags, top_feature_payload(features), model_version


def _build_candidates(
    schedule: pd.DataFrame,
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    games = schedule[schedule["game_date"].dt.date == as_of].copy()
    if games.empty:
        return pd.DataFrame()
    rates = _player_rates(batting, as_of)
    player_rate_by_id = {int(row["player_id"]): row.to_dict() for _, row in rates.iterrows()}
    pitcher_rates = _pitcher_rates(pitching, as_of)
    pitcher_rate_by_id = {int(row["player_id"]): row.to_dict() for _, row in pitcher_rates.iterrows()}
    league_hr_pa = max(batting["home_runs"].sum() / max(batting["plate_appearances"].sum(), 1), 0.001)
    venue_factor_by_id = _venue_factors(batting, schedule, as_of)
    recency_by_player = games_since_last_hr(batting, as_of)
    recency_map = {
        int(row["player_id"]): row.to_dict()
        for _, row in recency_by_player.iterrows()
    } if not recency_by_player.empty else {}
    rows: list[dict[str, Any]] = []

    for _, game in games.iterrows():
        game_id = f"MLB_{int(game['game_pk'])}"
        for side in ("home", "away"):
            team_id = int(game[f"{side}_team_id"])
            opponent_side = "away" if side == "home" else "home"
            opponent_team_id = int(game[f"{opponent_side}_team_id"])
            opponent_pitcher_id = game.get(f"{opponent_side}_probable_pitcher_id")
            opponent_pitcher_name = game.get(f"{opponent_side}_probable_pitcher")
            opponent_pitcher_known = pd.notna(opponent_pitcher_id)
            pitcher_rate = pitcher_rate_by_id.get(int(opponent_pitcher_id)) if opponent_pitcher_known else None
            lineup = _project_lineup(batting, team_id, as_of)
            if lineup.empty:
                continue
            for _, batter in lineup.iterrows():
                player_rate = player_rate_by_id.get(int(batter["player_id"]))
                recency = recency_map.get(int(batter["player_id"]), {})
                features = _candidate_features(
                    batter=batter,
                    player_rate=player_rate,
                    pitcher_rate=pitcher_rate,
                    venue_factor=venue_factor_by_id.get(int(game.get("venue_id") or 0), 1.0),
                    league_hr_pa=league_hr_pa,
                    is_home=side == "home",
                )
                row = {
                    "game_id": game_id,
                    "game_pk": int(game["game_pk"]),
                    "game_date": as_of.isoformat(),
                    "event_time": pd.to_datetime(game.get("game_datetime"), utc=True).isoformat(),
                    "player_id": int(batter["player_id"]),
                    "player_name": batter["player_name"],
                    "team": game[f"{side}_team_abbr"] or game[f"{side}_team"],
                    "opponent": game[f"{opponent_side}_team_abbr"] or game[f"{opponent_side}_team"],
                    "team_id": team_id,
                    "opponent_id": opponent_team_id,
                    "opposing_starter_id": int(opponent_pitcher_id) if opponent_pitcher_known else np.nan,
                    "venue": game.get("venue_name"),
                    "lineup_slot": int(batter["lineup_slot"]),
                    "lineup_status": "projected",
                    "opposing_probable_pitcher": opponent_pitcher_name if pd.notna(opponent_pitcher_name) else None,
                    "probable_pitcher_known": opponent_pitcher_known,
                    "baseline_probability": features["baseline_probability"],
                    "heuristic_probability": features["heuristic_probability"],
                    "games_since_last_hr": recency.get("games_since_last_hr"),
                    "last_hr_date": recency.get("last_hr_date"),
                }
                row.update(features)
                rows.append(row)
    return pd.DataFrame(rows)


def _finalize_predictions(
    candidates: pd.DataFrame,
    *,
    hr_probability: pd.Series,
    model_version: str,
    quality_flags: list[list[str]] | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    out["hr_probability"] = hr_probability.to_numpy()
    out["model_version"] = model_version
    out["prediction_ts"] = datetime.now(timezone.utc).isoformat()
    if quality_flags is None:
        out["quality_flags"] = [
            json.dumps(
                [
                    *quality_flags_for_features(
                        {col: row[col] for col in FEATURE_COLUMNS},
                        probable_pitcher_known=bool(row["probable_pitcher_known"]),
                    ),
                    *recency_quality_flags(row.get("games_since_last_hr")),
                ]
            )
            for _, row in out.iterrows()
        ]
    else:
        out["quality_flags"] = [
            json.dumps([*flags, *recency_quality_flags(row.get("games_since_last_hr"))])
            for flags, (_, row) in zip(quality_flags, out.iterrows(), strict=False)
        ]
    out["top_features"] = [
        json.dumps(
            top_feature_payload({col: row[col] for col in FEATURE_COLUMNS}),
        )
        for _, row in out.iterrows()
    ]
    out["confidence"] = out["quality_flags"].map(
        lambda value: float(np.clip(0.72 - 0.08 * len(json.loads(value)), 0.35, 0.78))
    )
    out = out.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _build_predictions(
    schedule: pd.DataFrame,
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    as_of: date,
    *,
    model_artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    candidates = _build_candidates(schedule, batting, pitching, as_of)
    if candidates.empty:
        return candidates
    probabilities: list[float] = []
    flags_list: list[list[str]] = []
    model_versions: list[str] = []
    for _, row in candidates.iterrows():
        features = {col: row[col] for col in FEATURE_COLUMNS}
        features["baseline_probability"] = row["baseline_probability"]
        features["heuristic_probability"] = row["heuristic_probability"]
        model_probability = predict_hr_probability(model_artifact, features)
        if model_probability is None:
            probability = row["heuristic_probability"] + _hash_jitter(row["player_id"], row["player_name"])
            probability = float(np.clip(probability, 0.002, 0.38))
            model_version = "mlb-hr-v1-heuristic"
        else:
            probability = model_probability + _hash_jitter(row["player_id"], row["player_name"], scale=0.0003)
            probability = float(np.clip(probability, 0.001, 0.45))
            model_version = str((model_artifact or {}).get("model_version") or TRAINED_MODEL_VERSION)
        probabilities.append(probability)
        flags_list.append(
            quality_flags_for_features(features, probable_pitcher_known=bool(row["probable_pitcher_known"]))
        )
        model_versions.append(model_version)
    model_version = model_versions[0] if model_versions else TRAINED_MODEL_VERSION
    if len(set(model_versions)) > 1:
        model_version = TRAINED_MODEL_VERSION
    return _finalize_predictions(
        candidates,
        hr_probability=pd.Series(probabilities),
        model_version=model_version,
        quality_flags=flags_list,
    )


def _append_quality_flag(value: Any, flag: str) -> str:
    try:
        flags = json.loads(value) if isinstance(value, str) else list(value or [])
    except (TypeError, json.JSONDecodeError):
        flags = []
    if flag not in flags:
        flags.append(flag)
    return json.dumps(flags)


def _candidate_key(row: pd.Series | dict[str, Any]) -> tuple[str, int]:
    return (str(row["game_id"]), int(row["player_id"]))


def _candidate_key_set(frame: pd.DataFrame) -> set[tuple[str, int]]:
    if frame.empty:
        return set()
    return {
        (str(game_id), int(player_id))
        for game_id, player_id in zip(frame["game_id"], frame["player_id"], strict=False)
    }


def _sort_and_rank_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    out = predictions.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _fallback_model_predictions(
    v1_predictions: pd.DataFrame,
    *,
    model_version: str,
    fallback_flag: str = "statcast_features_unavailable",
) -> pd.DataFrame:
    fallback = v1_predictions.copy()
    if fallback.empty:
        return fallback
    fallback["model_version"] = model_version
    if "quality_flags" not in fallback.columns:
        fallback["quality_flags"] = "[]"
    fallback["quality_flags"] = fallback["quality_flags"].map(
        lambda value: _append_quality_flag(value, fallback_flag)
    )
    return _sort_and_rank_predictions(fallback)


def _ensure_model_candidate_coverage(
    reference_predictions: pd.DataFrame,
    model_predictions: pd.DataFrame,
    *,
    model_version: str,
    fallback_flag: str = "statcast_features_unavailable",
) -> tuple[pd.DataFrame, list[str]]:
    gaps: list[str] = []
    if reference_predictions.empty:
        return model_predictions.copy(), gaps
    if model_predictions.empty:
        gaps.append(f"{model_version} had no rows; filled all candidates with v1 probabilities.")
        return (
            _fallback_model_predictions(
                reference_predictions,
                model_version=model_version,
                fallback_flag=fallback_flag,
            ),
            gaps,
        )

    model = model_predictions.copy()
    model["_candidate_key"] = [_candidate_key(row) for _, row in model.iterrows()]
    deduped = model.drop_duplicates("_candidate_key", keep="first")
    duplicate_count = len(model) - len(deduped)
    if duplicate_count:
        gaps.append(f"{model_version} had {duplicate_count} duplicate candidate rows; kept the first row per player.")
    model = deduped

    reference_keys = _candidate_key_set(reference_predictions)
    model_keys = set(model["_candidate_key"])
    missing_keys = reference_keys - model_keys
    if missing_keys:
        missing = reference_predictions[
            [_candidate_key(row) in missing_keys for _, row in reference_predictions.iterrows()]
        ].copy()
        fallback = _fallback_model_predictions(
            missing,
            model_version=model_version,
            fallback_flag=fallback_flag,
        )
        model = pd.concat([model.drop(columns=["_candidate_key"]), fallback], ignore_index=True)
        gaps.append(
            f"{model_version} was missing {len(missing_keys)} candidate rows; filled them with v1 probabilities."
        )
    else:
        model = model.drop(columns=["_candidate_key"])
    return _sort_and_rank_predictions(model), gaps


def _build_statcast_blend_predictions(
    candidates: pd.DataFrame,
    v1_predictions: pd.DataFrame,
    as_of: date,
    *,
    torch_artifact: dict[str, Any],
    statcast_cache: Path,
    refresh_statcast: bool = False,
    statcast_min_batter_bbe: int = DEFAULT_STATCAST_MIN_BBE,
    statcast_min_pitcher_bbe: int = DEFAULT_STATCAST_MIN_BBE,
    allow_partial_statcast: bool = False,
    statcast_deadline_seconds: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    gaps: list[str] = []
    if candidates.empty:
        return candidates, gaps

    fallback_lookup = v1_predictions.copy()
    fallback_lookup["player_id"] = fallback_lookup["player_id"].astype(int)
    fallback_probs = fallback_lookup.set_index(["game_id", "player_id"])["hr_probability"]
    probabilities: list[float] = []
    flags_list: list[list[str]] = []

    try:
        enriched = build_torch_candidate_features(
            candidates,
            as_of,
            statcast_cache=statcast_cache,
            refresh_statcast=refresh_statcast,
            statcast_min_batter_bbe=statcast_min_batter_bbe,
            statcast_min_pitcher_bbe=statcast_min_pitcher_bbe,
            allow_partial_statcast=allow_partial_statcast,
            statcast_deadline_seconds=statcast_deadline_seconds,
        ).reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        gaps.append(f"Statcast feature build failed; using v1 probabilities for blend board: {exc}")
        return (
            _fallback_model_predictions(
                v1_predictions,
                model_version=STATCAST_BLEND_KEY,
                fallback_flag="statcast_features_unavailable",
            ),
            gaps,
        )

    enriched_by_key: dict[tuple[str, int], pd.Series] = {}
    for _, row in enriched.iterrows():
        enriched_by_key.setdefault(_candidate_key(row), row)

    aligned_rows = []
    missing_enriched = 0
    for _, candidate in candidates.iterrows():
        enriched_row = enriched_by_key.get(_candidate_key(candidate))
        if enriched_row is None:
            row = candidate.copy()
            row["_statcast_enrichment_present"] = False
            missing_enriched += 1
        else:
            row = enriched_row.copy()
            row["_statcast_enrichment_present"] = True
        aligned_rows.append(row)

    aligned = pd.DataFrame(aligned_rows).reset_index(drop=True)
    if missing_enriched:
        gaps.append(
            f"Statcast enrichment returned no row for {missing_enriched} candidates; those rows use v1 probabilities."
        )

    enrichment_present = aligned.pop("_statcast_enrichment_present").fillna(False).astype(bool)
    ready_values = pd.to_numeric(
        aligned.get("statcast_feature_ready", pd.Series(0.0, index=aligned.index)),
        errors="coerce",
    ).fillna(0.0)
    ready_mask = enrichment_present & (ready_values >= 1.0)
    ready_count = int(ready_mask.sum())
    if ready_count == 0:
        gaps.append("No MLB HR candidates had Statcast-ready features; blend board mirrors v1 probabilities.")
    elif ready_count < len(aligned):
        gaps.append(
            f"Statcast features unavailable for {len(aligned) - ready_count} MLB HR candidates; those rows use v1 probabilities."
        )

    blend_probs = np.full(len(aligned), np.nan, dtype=float)
    if ready_count:
        try:
            ready = aligned.loc[ready_mask].copy()
            torch_probs = predict_torch_probs(torch_artifact, ready)
            heuristic_probs = ready["heuristic_probability"].to_numpy(dtype=float)
            blended = apply_heuristic_blend(torch_artifact, torch_probs, heuristic_probs)
            blend_probs[ready_mask.to_numpy()] = blended
        except Exception as exc:  # noqa: BLE001
            gaps.append(f"Statcast model scoring failed; using v1 probabilities for blend board: {exc}")
            ready_mask = pd.Series(False, index=aligned.index)

    for pos, row in aligned.iterrows():
        key = (row["game_id"], int(row["player_id"]))
        fallback = float(fallback_probs.get(key, row["heuristic_probability"]))
        if bool(ready_mask.iloc[pos]) and math.isfinite(float(blend_probs[pos])):
            probability = float(blend_probs[pos]) + _hash_jitter(row["player_id"], row["player_name"], scale=0.0003)
            probability = float(np.clip(probability, 0.001, 0.45))
        else:
            probability = fallback
        probabilities.append(probability)
        flags = quality_flags_for_features(
            {col: row[col] for col in FEATURE_COLUMNS},
            probable_pitcher_known=bool(row["probable_pitcher_known"]),
        )
        if not bool(ready_mask.iloc[pos]):
            flags.append("statcast_features_unavailable")
        elif row.get("statcast_feature_quality") not in (None, "full"):
            flags.append(f"statcast_{row.get('statcast_feature_quality')}_features")
        flags_list.append(flags)

    return _finalize_predictions(
        aligned.reset_index(drop=True),
        hr_probability=pd.Series(probabilities),
        model_version=STATCAST_BLEND_KEY,
        quality_flags=flags_list,
    ), gaps


def _rank_or_penalty(value: Any, penalty: int) -> int:
    if value is None:
        return penalty
    try:
        if pd.isna(value):
            return penalty
        return int(value)
    except (TypeError, ValueError):
        return penalty


def _model_agreement_label(
    *,
    v1_rank: int,
    statcast_rank: int | None,
    statcast_available: bool,
    statcast_present: bool,
) -> str:
    if not statcast_present:
        return MODEL_AGREEMENT_V1_ONLY
    if not statcast_available:
        return MODEL_AGREEMENT_MISSING_STATCAST
    rank_delta = int(statcast_rank or v1_rank) - int(v1_rank)
    if rank_delta <= -MODEL_AGREEMENT_RANK_THRESHOLD:
        return MODEL_AGREEMENT_STATCAST_BOOST
    if rank_delta >= MODEL_AGREEMENT_RANK_THRESHOLD:
        return MODEL_AGREEMENT_STATCAST_FADE
    return MODEL_AGREEMENT_CONSENSUS


def _build_probability_board(
    v1_predictions: pd.DataFrame,
    *,
    statcast_predictions: pd.DataFrame | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    if v1_predictions.empty:
        return v1_predictions.copy()

    out = v1_predictions.copy()
    out["v1_probability"] = out["hr_probability"]
    out["v1_rank"] = out["rank"].astype(int)

    statcast_present = statcast_predictions is not None and not statcast_predictions.empty
    statcast_by_candidate = (
        statcast_predictions.set_index(["game_id", "player_id"])
        if statcast_present
        else pd.DataFrame()
    )
    rank_penalty = len(out) + 25
    neutral_market_rank = len(out) + 1

    statcast_probabilities: list[float | None] = []
    statcast_ranks: list[int | None] = []
    statcast_available_values: list[bool] = []
    agreement_labels: list[str] = []
    consensus_scores: list[float] = []

    for _, row in out.iterrows():
        key = (row["game_id"], int(row["player_id"]))
        statcast_row = statcast_by_candidate.loc[key] if statcast_present and key in statcast_by_candidate.index else None
        if isinstance(statcast_row, pd.DataFrame):
            statcast_row = statcast_row.iloc[0]

        statcast_available = False
        statcast_probability: float | None = None
        statcast_rank: int | None = None
        if statcast_row is not None:
            statcast_flags = json.loads(statcast_row.get("quality_flags") or "[]")
            statcast_available = "statcast_features_unavailable" not in statcast_flags
            statcast_probability = float(statcast_row.get("hr_probability"))
            statcast_rank = int(statcast_row.get("rank"))

        v1_rank = int(row["rank"])
        v1_flags = json.loads(row.get("quality_flags") or "[]")
        statcast_component = _rank_or_penalty(statcast_rank, rank_penalty)
        flags_penalty = len(v1_flags) * 3
        if statcast_present and not statcast_available:
            flags_penalty += 6
        consensus_score = float(v1_rank + statcast_component + neutral_market_rank + flags_penalty)

        statcast_probabilities.append(statcast_probability)
        statcast_ranks.append(statcast_rank)
        statcast_available_values.append(statcast_available)
        agreement_labels.append(
            _model_agreement_label(
                v1_rank=v1_rank,
                statcast_rank=statcast_rank,
                statcast_available=statcast_available,
                statcast_present=statcast_row is not None,
            )
        )
        consensus_scores.append(consensus_score)

    out["statcast_probability"] = statcast_probabilities
    out["statcast_rank"] = statcast_ranks
    out["statcast_available"] = statcast_available_values
    out["model_agreement"] = agreement_labels
    out["consensus_score"] = consensus_scores
    out["market_signal_rank"] = neutral_market_rank
    out = out.sort_values(["consensus_score", "v1_rank"], ascending=[True, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    if top_n is not None:
        out = out.head(top_n).copy()
    return out


def _filter_to_candidate_set(predictions: pd.DataFrame, candidate_set: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or candidate_set.empty:
        return predictions.head(0).copy()
    keys = set(zip(candidate_set["game_id"], candidate_set["player_id"].astype(int), strict=False))
    mask = [
        (row["game_id"], int(row["player_id"])) in keys
        for _, row in predictions.iterrows()
    ]
    return predictions.loc[mask].copy()


def _predictions_to_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in predictions.iterrows():
        payload = {
                "id": f"{row['game_id']}-{row['player_id']}-hr",
                "sport": "MLB",
                "league": "MLB",
                "gameId": row["game_id"],
                "gameDate": row["game_date"],
                "eventTime": row["event_time"],
                "subject": f"{row['player_name']} HR",
                "playerId": str(row["player_id"]),
                "player": row["player_name"],
                "market": "home_run",
                "book": "model",
                "line": 0.5,
                "price": None,
                "modelProbability": row["hr_probability"],
                "impliedProbability": None,
                "edge": None,
                "ev": None,
                "kelly": None,
                "confidence": row["confidence"],
                "modelVersion": row["model_version"],
                "source": "MLB Stats API projected lineup",
                "updatedAt": row["prediction_ts"],
                "team": row["team"],
                "opponent": row["opponent"],
                "venue": row["venue"],
                "lineupSlot": int(row["lineup_slot"]),
                "lineupStatus": row["lineup_status"],
                "opposingProbablePitcher": row["opposing_probable_pitcher"],
                "baselineProbability": row["baseline_probability"],
                "rank": int(row["rank"]),
                "gamesSinceLastHr": None
                if row.get("games_since_last_hr") is None or (isinstance(row.get("games_since_last_hr"), float) and pd.isna(row.get("games_since_last_hr")))
                else int(row["games_since_last_hr"]),
                "lastHrDate": row.get("last_hr_date"),
                "qualityFlags": json.loads(row["quality_flags"]),
                "topFeatures": json.loads(row["top_features"]),
            }
        if "v1_probability" in row:
            payload["v1Probability"] = row.get("v1_probability")
            payload["v1Rank"] = None if pd.isna(row.get("v1_rank")) else int(row.get("v1_rank"))
        if "statcast_probability" in row:
            payload["statcastProbability"] = None if pd.isna(row.get("statcast_probability")) else row.get("statcast_probability")
            payload["statcastRank"] = None if pd.isna(row.get("statcast_rank")) else int(row.get("statcast_rank"))
            payload["statcastAvailable"] = bool(row.get("statcast_available"))
        if "model_agreement" in row:
            payload["modelAgreement"] = row.get("model_agreement")
        if "consensus_score" in row:
            payload["consensusScore"] = row.get("consensus_score")
        if "market_signal_rank" in row:
            payload["marketSignalRank"] = None if pd.isna(row.get("market_signal_rank")) else int(row.get("market_signal_rank"))
        if "statcast_feature_ready" in row:
            payload["statcastFeatureReady"] = None if pd.isna(row.get("statcast_feature_ready")) else bool(row.get("statcast_feature_ready"))
        if "statcast_feature_quality" in row:
            payload["statcastFeatureQuality"] = row.get("statcast_feature_quality")
        for col in STATCAST_HEALTH_COLUMNS:
            if col in row:
                camel = {
                    "statcast_coverage": "statcastCoverage",
                    "statcast_ready_rows": "statcastReadyRows",
                    "statcast_total_rows": "statcastTotalRows",
                    "statcast_artifact_loaded": "statcastArtifactLoaded",
                }[col]
                value = row.get(col)
                payload[camel] = None if pd.isna(value) else value
        rows.append(payload)
    return rows


def _model_payload(predictions: pd.DataFrame, gaps: list[str], model_version: str) -> dict[str, Any]:
    return {
        "modelVersion": model_version,
        "predictions": _predictions_to_rows(predictions),
        "gaps": gaps,
    }


def _has_quality_flag(value: Any, flag: str) -> bool:
    try:
        flags = json.loads(value) if isinstance(value, str) else list(value or [])
    except (TypeError, json.JSONDecodeError):
        flags = []
    return flag in flags


def _statcast_health_payload(
    statcast_predictions: pd.DataFrame,
    board_predictions: pd.DataFrame,
    *,
    enabled: bool,
    artifact_loaded: bool,
    artifact_path: Path,
    artifact_error: str | None,
    gaps: list[str],
    min_batter_bbe: int,
    min_pitcher_bbe: int,
    allow_partial: bool,
) -> dict[str, Any]:
    total_rows = int(len(statcast_predictions)) if enabled else 0
    if not enabled:
        ready_rows = 0
    elif "statcast_feature_ready" in statcast_predictions.columns:
        ready_rows = int(
            pd.to_numeric(statcast_predictions["statcast_feature_ready"], errors="coerce")
            .fillna(0)
            .ge(1.0)
            .sum()
        )
    else:
        ready_rows = int(
            sum(
                not _has_quality_flag(value, "statcast_features_unavailable")
                for value in statcast_predictions.get("quality_flags", pd.Series(dtype=object))
            )
        )
    coverage = (ready_rows / total_rows) if total_rows else 0.0
    agreement_distribution = (
        board_predictions.get("model_agreement", pd.Series(dtype=object))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {
        "enabled": enabled,
        "artifactLoaded": bool(artifact_loaded),
        "artifactPath": str(artifact_path),
        "artifactError": artifact_error,
        "coverage": float(coverage),
        "readyRows": ready_rows,
        "totalRows": total_rows,
        "unavailableRows": max(total_rows - ready_rows, 0),
        "minBatterBbe": int(min_batter_bbe),
        "minPitcherBbe": int(min_pitcher_bbe),
        "allowPartial": bool(allow_partial),
        "gaps": gaps,
        "modelAgreement": agreement_distribution,
    }


def _attach_statcast_health_columns(frame: pd.DataFrame, health: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["statcast_coverage"] = health.get("coverage")
    out["statcast_ready_rows"] = health.get("readyRows")
    out["statcast_total_rows"] = health.get("totalRows")
    out["statcast_artifact_loaded"] = health.get("artifactLoaded")
    return out


def _to_web_payload(
    v1_predictions: pd.DataFrame,
    gaps: list[str],
    *,
    board_predictions: pd.DataFrame | None = None,
    statcast_predictions: pd.DataFrame | None = None,
    statcast_gaps: list[str] | None = None,
    statcast_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_version = (
        str(v1_predictions["model_version"].dropna().iloc[0])
        if not v1_predictions.empty
        and "model_version" in v1_predictions.columns
        and not v1_predictions["model_version"].dropna().empty
        else "mlb-hr-v1-heuristic"
    )
    v1_payload = _model_payload(v1_predictions, gaps, model_version)
    default_predictions = board_predictions if board_predictions is not None else v1_predictions
    payload: dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "market": "MLB batter home runs",
        "defaultModel": V1_MODEL_KEY,
        "modelVersion": model_version,
        "productionStatus": "candidate",
        "predictions": _predictions_to_rows(default_predictions),
        "gaps": gaps,
    }
    if statcast_predictions is not None:
        statcast_payload = _model_payload(
            statcast_predictions,
            statcast_gaps or [],
            STATCAST_BLEND_KEY,
        )
        payload["models"] = {
            V1_MODEL_KEY: v1_payload,
            STATCAST_BLEND_KEY: statcast_payload,
        }
        payload["gaps"] = gaps + (statcast_gaps or [])
    if statcast_health is not None:
        payload["statcastHealth"] = statcast_health
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MLB home run probabilities for a date.")
    parser.add_argument("--date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(), default=datetime.now(timezone.utc).date())
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--history-days", type=int, default=45)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-artifact", type=Path, default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument("--torch-artifact", type=Path, default=DEFAULT_TORCH_ARTIFACT)
    parser.add_argument("--statcast-cache", type=Path, default=DEFAULT_STATCAST_CACHE)
    parser.add_argument("--refresh-statcast-cache", action="store_true")
    parser.add_argument("--skip-statcast-blend", action="store_true")
    parser.add_argument("--statcast-min-batter-bbe", type=int, default=int(os.getenv("MLB_HR_STATCAST_MIN_BATTER_BBE", str(DEFAULT_STATCAST_MIN_BBE))))
    parser.add_argument("--statcast-min-pitcher-bbe", type=int, default=int(os.getenv("MLB_HR_STATCAST_MIN_PITCHER_BBE", str(DEFAULT_STATCAST_MIN_BBE))))
    parser.add_argument("--statcast-deadline-seconds", type=float, default=float(os.getenv("MLB_HR_STATCAST_DEADLINE_SECONDS", "360")))
    parser.add_argument("--allow-partial-statcast-features", action="store_true", default=os.getenv("MLB_HR_ALLOW_PARTIAL_STATCAST", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--force-heuristic", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--statcast-out-csv", type=Path, default=DEFAULT_STATCAST_CSV_OUT)
    parser.add_argument("--web-out", type=Path, default=DEFAULT_WEB_OUT)
    parser.add_argument("--top-n", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    season = args.season or args.date.year
    history_start = args.date - timedelta(days=args.history_days)
    schedule = fetch_mlb_schedule(
        season,
        start_date=history_start,
        end_date=args.date,
        include_uncompleted=True,
    )
    if schedule.empty:
        raise SystemExit("No MLB schedule rows fetched.")
    batting, pitching = _load_history(schedule, args.cache, history_start, args.date)
    gaps: list[str] = []
    if batting.empty:
        gaps.append("No recent batting boxscores available; MLB HR probabilities not generated.")
    model_artifact = None
    if not args.force_heuristic:
        try:
            model_artifact = load_hr_artifact(args.model_artifact)
        except Exception as exc:  # noqa: BLE001
            gaps.append(f"MLB HR model artifact load failed; using heuristic fallback: {exc}")
    if model_artifact:
        print(f"Loaded MLB HR model artifact: {args.model_artifact}")
    predictions_full = _build_predictions(schedule, batting, pitching, args.date, model_artifact=model_artifact) if not batting.empty else pd.DataFrame()
    if predictions_full.empty:
        gaps.append(f"No MLB HR candidate rows generated for {args.date}.")

    statcast_predictions = pd.DataFrame()
    statcast_gaps: list[str] = []
    torch_artifact_loaded = False
    torch_artifact_error: str | None = None
    if not args.skip_statcast_blend and not predictions_full.empty:
        torch_artifact = None
        try:
            torch_artifact = load_torch_hr_artifact(args.torch_artifact)
            torch_artifact_loaded = True
            print(f"Loaded Statcast blend torch artifact: {args.torch_artifact}")
        except Exception as exc:  # noqa: BLE001
            torch_artifact_error = str(exc)
            statcast_gaps.append(f"Statcast blend torch artifact load failed: {exc}")
        if torch_artifact:
            candidates = predictions_full.drop(
                columns=["hr_probability", "model_version", "prediction_ts", "quality_flags", "top_features", "confidence", "rank"],
                errors="ignore",
            ).copy()
            statcast_predictions, statcast_gaps = _build_statcast_blend_predictions(
                candidates,
                predictions_full,
                args.date,
                torch_artifact=torch_artifact,
                statcast_cache=args.statcast_cache,
                refresh_statcast=args.refresh_statcast_cache,
                statcast_min_batter_bbe=args.statcast_min_batter_bbe,
                statcast_min_pitcher_bbe=args.statcast_min_pitcher_bbe,
                allow_partial_statcast=args.allow_partial_statcast_features,
                statcast_deadline_seconds=args.statcast_deadline_seconds,
            )
        else:
            statcast_gaps.append("Statcast blend model unavailable; statcast feed mirrors v1 probabilities.")
            statcast_predictions = _fallback_model_predictions(
                predictions_full,
                model_version=STATCAST_BLEND_KEY,
                fallback_flag="statcast_features_unavailable",
            )

    if not args.skip_statcast_blend and not predictions_full.empty:
        statcast_predictions, coverage_gaps = _ensure_model_candidate_coverage(
            predictions_full,
            statcast_predictions,
            model_version=STATCAST_BLEND_KEY,
            fallback_flag="statcast_features_unavailable",
        )
        statcast_gaps.extend(coverage_gaps)

    board_predictions = _build_probability_board(
        predictions_full,
        statcast_predictions=statcast_predictions if not statcast_predictions.empty else None,
        top_n=args.top_n,
    )
    predictions = _filter_to_candidate_set(predictions_full, board_predictions)
    if not statcast_predictions.empty:
        statcast_predictions = _filter_to_candidate_set(statcast_predictions, board_predictions)
    if not args.skip_statcast_blend and not predictions.empty:
        statcast_predictions, coverage_gaps = _ensure_model_candidate_coverage(
            predictions,
            statcast_predictions,
            model_version=STATCAST_BLEND_KEY,
            fallback_flag="statcast_features_unavailable",
        )
        statcast_gaps.extend(coverage_gaps)

    statcast_health = _statcast_health_payload(
        statcast_predictions,
        board_predictions,
        enabled=not args.skip_statcast_blend and not predictions_full.empty,
        artifact_loaded=torch_artifact_loaded,
        artifact_path=args.torch_artifact,
        artifact_error=torch_artifact_error,
        gaps=statcast_gaps,
        min_batter_bbe=args.statcast_min_batter_bbe,
        min_pitcher_bbe=args.statcast_min_pitcher_bbe,
        allow_partial=args.allow_partial_statcast_features,
    )
    predictions = _attach_statcast_health_columns(predictions, statcast_health)
    if not statcast_predictions.empty:
        statcast_predictions = _attach_statcast_health_columns(statcast_predictions, statcast_health)
    board_predictions = _attach_statcast_health_columns(board_predictions, statcast_health)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out_csv, index=False)
    if not statcast_predictions.empty:
        statcast_predictions.to_csv(args.statcast_out_csv, index=False)
    args.web_out.parent.mkdir(parents=True, exist_ok=True)
    args.web_out.write_text(
        dumps_strict(
            _to_web_payload(
                predictions,
                gaps,
                board_predictions=board_predictions,
                statcast_predictions=statcast_predictions if not statcast_predictions.empty else None,
                statcast_gaps=statcast_gaps,
                statcast_health=statcast_health,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    wrote = f"{len(predictions)} v1 predictions to {args.out_csv}"
    if not statcast_predictions.empty:
        wrote += f" and {len(statcast_predictions)} statcast blend predictions to {args.statcast_out_csv}"
    print(f"Wrote {wrote} and {args.web_out}")


if __name__ == "__main__":
    main()
