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
    MODEL_VERSION as TRAINED_MODEL_VERSION,
    build_hr_feature_values,
    heuristic_hr_probability,
    load_hr_artifact,
    predict_hr_probability,
    quality_flags_for_features,
    top_feature_payload,
)


MLB_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
DEFAULT_CACHE = ROOT / "notebooks" / "cache" / "mlb_home_run_boxscores_2026.jsonl"
DEFAULT_MODEL_ARTIFACT = ROOT / "models" / "mlb_hr_model_v1.joblib"
DEFAULT_CSV_OUT = ROOT / "notebooks" / "cache" / "mlb_home_run_predictions.csv"
DEFAULT_WEB_OUT = REPO_ROOT / "web" / "public" / "data" / "mlb_home_runs.json"


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


def _build_predictions(
    schedule: pd.DataFrame,
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    as_of: date,
    *,
    model_artifact: dict[str, Any] | None = None,
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
    rows: list[dict[str, Any]] = []

    for _, game in games.iterrows():
        game_id = f"MLB_{int(game['game_pk'])}"
        for side in ("home", "away"):
            team_id = int(game[f"{side}_team_id"])
            opponent_side = "away" if side == "home" else "home"
            opponent_pitcher_id = game.get(f"{opponent_side}_probable_pitcher_id")
            opponent_pitcher_name = game.get(f"{opponent_side}_probable_pitcher")
            opponent_pitcher_known = pd.notna(opponent_pitcher_id)
            pitcher_rate = pitcher_rate_by_id.get(int(opponent_pitcher_id)) if opponent_pitcher_known else None
            lineup = _project_lineup(batting, team_id, as_of)
            if lineup.empty:
                continue
            for _, batter in lineup.iterrows():
                player_rate = player_rate_by_id.get(int(batter["player_id"]))
                probability, baseline, flags, top_features, model_version = _score_probability(
                    batter=batter,
                    player_rate=player_rate,
                    pitcher_rate=pitcher_rate,
                    venue_factor=venue_factor_by_id.get(int(game.get("venue_id") or 0), 1.0),
                    league_hr_pa=league_hr_pa,
                    probable_pitcher_known=opponent_pitcher_known,
                    is_home=side == "home",
                    model_artifact=model_artifact,
                )
                rows.append(
                    {
                        "game_id": game_id,
                        "game_pk": int(game["game_pk"]),
                        "game_date": as_of.isoformat(),
                        "event_time": pd.to_datetime(game.get("game_datetime"), utc=True).isoformat(),
                        "player_id": int(batter["player_id"]),
                        "player_name": batter["player_name"],
                        "team": game[f"{side}_team_abbr"] or game[f"{side}_team"],
                        "opponent": game[f"{opponent_side}_team_abbr"] or game[f"{opponent_side}_team"],
                        "venue": game.get("venue_name"),
                        "lineup_slot": int(batter["lineup_slot"]),
                        "lineup_status": "projected",
                        "opposing_probable_pitcher": opponent_pitcher_name if pd.notna(opponent_pitcher_name) else None,
                        "hr_probability": probability,
                        "baseline_probability": baseline,
                        "confidence": float(np.clip(0.72 - 0.08 * len(flags), 0.35, 0.78)),
                        "model_version": model_version,
                        "prediction_ts": datetime.now(timezone.utc).isoformat(),
                        "quality_flags": json.dumps(flags),
                        "top_features": json.dumps(top_features),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("hr_probability", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _to_web_payload(predictions: pd.DataFrame, gaps: list[str]) -> dict[str, Any]:
    rows = []
    for _, row in predictions.iterrows():
        rows.append(
            {
                "id": f"{row['game_id']}-{row['player_id']}-hr",
                "sport": "MLB",
                "league": "MLB",
                "gameId": row["game_id"],
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
                "qualityFlags": json.loads(row["quality_flags"]),
                "topFeatures": json.loads(row["top_features"]),
            }
        )
    model_version = (
        str(predictions["model_version"].dropna().iloc[0])
        if not predictions.empty and "model_version" in predictions.columns and not predictions["model_version"].dropna().empty
        else "mlb-hr-v1-heuristic"
    )
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "market": "MLB batter home runs",
        "modelVersion": model_version,
        "productionStatus": "candidate",
        "predictions": rows,
        "gaps": gaps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MLB home run probabilities for a date.")
    parser.add_argument("--date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(), default=datetime.now(timezone.utc).date())
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--history-days", type=int, default=45)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-artifact", type=Path, default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument("--force-heuristic", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV_OUT)
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
    predictions = _build_predictions(schedule, batting, pitching, args.date, model_artifact=model_artifact) if not batting.empty else pd.DataFrame()
    if predictions.empty:
        gaps.append(f"No MLB HR candidate rows generated for {args.date}.")
    else:
        predictions = predictions.head(args.top_n).copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out_csv, index=False)
    args.web_out.parent.mkdir(parents=True, exist_ok=True)
    args.web_out.write_text(
        dumps_strict(_to_web_payload(predictions, gaps), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(predictions)} MLB HR predictions to {args.out_csv} and {args.web_out}")


if __name__ == "__main__":
    main()
