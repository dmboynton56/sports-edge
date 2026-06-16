#!/usr/bin/env python3
"""Train a time-split MLB batter home-run probability model.

The feature builder processes games chronologically and only uses player,
pitcher, league, and venue stats accumulated before each player-game row. That
keeps the training set aligned with the daily pregame scorer and avoids
same-game target leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from predict_mlb_home_runs import _load_history  # noqa: E402
from src.data.mlb_fetcher import fetch_mlb_schedule  # noqa: E402
from src.models.mlb_home_run_model import (  # noqa: E402
    FEATURE_COLUMNS,
    MODEL_VERSION,
    build_hr_feature_values,
    heuristic_hr_probability,
)


DEFAULT_CACHE = ROOT / "notebooks" / "cache" / "mlb_home_run_boxscores_training.jsonl"
DEFAULT_DATASET = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows.csv"
DEFAULT_MODEL = ROOT / "models" / "mlb_hr_model_v1.joblib"
DEFAULT_METRICS = ROOT / "models" / "mlb_hr_model_v1_metrics.json"
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + ["baseline_probability", "heuristic_probability"]


def _date_arg(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _fetch_schedule_range(start_date: date, end_date: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in range(start_date.year, end_date.year + 1):
        season_start = max(start_date, date(season, 1, 1))
        season_end = min(end_date, date(season, 12, 31))
        if season_start > season_end:
            continue
        print(f"Fetching MLB schedule {season_start} through {season_end}...", flush=True)
        frame = fetch_mlb_schedule(
            season,
            start_date=season_start,
            end_date=season_end,
            include_uncompleted=False,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No completed MLB games found for requested training window.")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["game_pk"]).copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["game_datetime"] = pd.to_datetime(out["game_datetime"], errors="coerce")
    return out.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)


def _starter_lookup(pitching: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    if pitching.empty:
        return {}
    frame = pitching.copy()
    frame["games_started"] = pd.to_numeric(frame["games_started"], errors="coerce").fillna(0)
    frame["batters_faced"] = pd.to_numeric(frame["batters_faced"], errors="coerce").fillna(0)
    frame = frame[frame["batters_faced"] > 0].copy()
    frame = frame.sort_values(
        ["game_pk", "team_id", "games_started", "batters_faced"],
        ascending=[True, True, False, False],
    )
    starters = frame.drop_duplicates(subset=["game_pk", "team_id"], keep="first")
    return {(int(row["game_pk"]), int(row["team_id"])): row.to_dict() for _, row in starters.iterrows() if pd.notna(row["team_id"])}


def _recent_totals(events: list[tuple[date, float, float]], as_of: date, days: int = 28) -> tuple[float, float]:
    pa = 0.0
    hr = 0.0
    cutoff = as_of - timedelta(days=days)
    for event_date, event_pa, event_hr in events:
        if cutoff <= event_date <= as_of:
            pa += event_pa
            hr += event_hr
    return pa, hr


def build_training_rows(
    schedule: pd.DataFrame,
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    *,
    min_prior_pa: int,
) -> pd.DataFrame:
    if batting.empty:
        raise ValueError("No batting boxscore rows available.")

    schedule = schedule.copy()
    schedule["game_date"] = pd.to_datetime(schedule["game_date"]).dt.date
    batting = batting.copy()
    batting["game_date"] = pd.to_datetime(batting["game_date"]).dt.date
    pitching = pitching.copy()
    pitching["game_date"] = pd.to_datetime(pitching["game_date"]).dt.date

    starters = _starter_lookup(pitching)
    schedule_by_pk = {int(row["game_pk"]): row for _, row in schedule.iterrows()}
    batting_by_game = {int(game_pk): group.copy() for game_pk, group in batting.groupby("game_pk")}
    pitching_by_game = {int(game_pk): group.copy() for game_pk, group in pitching.groupby("game_pk")}

    batter_state: dict[int, dict[str, Any]] = {}
    pitcher_state: dict[int, dict[str, float]] = {}
    venue_state: dict[int, dict[str, float]] = {}
    league_pa = 0.0
    league_hr = 0.0
    rows: list[dict[str, Any]] = []

    ordered_games = schedule.sort_values(["game_datetime", "game_pk"])["game_pk"].astype(int).tolist()
    for game_pk in ordered_games:
        if game_pk not in batting_by_game:
            continue
        game = schedule_by_pk[game_pk]
        game_date = game["game_date"]
        venue_id = int(game["venue_id"]) if pd.notna(game.get("venue_id")) else 0
        venue_stats = venue_state.get(venue_id, {"pa": 0.0, "hr": 0.0})
        league_rate = float(np.clip((league_hr / league_pa) if league_pa > 0 else 0.03, 0.001, 0.08))
        venue_rate = (venue_stats["hr"] / venue_stats["pa"]) if venue_stats["pa"] >= 250 else league_rate
        venue_factor = float(np.clip(venue_rate / league_rate, 0.7, 1.35)) if league_rate > 0 else 1.0

        game_batting = batting_by_game[game_pk]
        for _, batter in game_batting.iterrows():
            plate_appearances = _safe_float(batter.get("plate_appearances"))
            if plate_appearances <= 0 or pd.isna(batter.get("player_id")):
                continue
            player_id = int(batter["player_id"])
            team_id = int(batter["team_id"]) if pd.notna(batter.get("team_id")) else 0
            opponent_id = int(batter["opponent_id"]) if pd.notna(batter.get("opponent_id")) else 0
            starter = starters.get((game_pk, opponent_id))
            starter_id = int(starter["player_id"]) if starter and pd.notna(starter.get("player_id")) else None

            b_state = batter_state.get(player_id, {"pa": 0.0, "hr": 0.0, "games": 0.0, "events": []})
            recent_pa, recent_hr = _recent_totals(b_state["events"], game_date)
            p_state = pitcher_state.get(starter_id or -1, {"bf": 0.0, "hr": 0.0})
            features = build_hr_feature_values(
                batter_pa=b_state["pa"],
                batter_hr=b_state["hr"],
                batter_games=b_state["games"],
                batter_recent_pa=recent_pa,
                batter_recent_hr=recent_hr,
                pitcher_bf=p_state["bf"],
                pitcher_hr_allowed=p_state["hr"],
                venue_factor=venue_factor,
                league_hr_pa=league_rate,
                lineup_slot=batter.get("lineup_slot"),
                is_home=str(batter.get("home_away")) == "home",
            )
            heuristic_probability, baseline_probability = heuristic_hr_probability(features)
            if features["batter_pa_lag"] < min_prior_pa:
                continue
            rows.append(
                {
                    "game_pk": game_pk,
                    "game_date": game_date.isoformat(),
                    "player_id": player_id,
                    "player_name": batter.get("player_name"),
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "opposing_starter_id": starter_id,
                    "actual_home_run": int(_safe_float(batter.get("home_runs")) > 0),
                    "actual_home_runs": int(_safe_float(batter.get("home_runs"))),
                    "actual_plate_appearances": plate_appearances,
                    "baseline_probability": baseline_probability,
                    "heuristic_probability": heuristic_probability,
                    **features,
                }
            )

        for _, batter in game_batting.iterrows():
            plate_appearances = _safe_float(batter.get("plate_appearances"))
            if plate_appearances <= 0 or pd.isna(batter.get("player_id")):
                continue
            player_id = int(batter["player_id"])
            home_runs = _safe_float(batter.get("home_runs"))
            state = batter_state.setdefault(player_id, {"pa": 0.0, "hr": 0.0, "games": 0.0, "events": []})
            state["pa"] += plate_appearances
            state["hr"] += home_runs
            state["games"] += 1.0
            state["events"].append((game_date, plate_appearances, home_runs))
            league_pa += plate_appearances
            league_hr += home_runs
            if venue_id:
                venue = venue_state.setdefault(venue_id, {"pa": 0.0, "hr": 0.0})
                venue["pa"] += plate_appearances
                venue["hr"] += home_runs

        for _, pitcher in pitching_by_game.get(game_pk, pd.DataFrame()).iterrows():
            if pd.isna(pitcher.get("player_id")):
                continue
            pitcher_id = int(pitcher["player_id"])
            state = pitcher_state.setdefault(pitcher_id, {"bf": 0.0, "hr": 0.0, "starts": 0.0})
            state["bf"] += _safe_float(pitcher.get("batters_faced"))
            state["hr"] += _safe_float(pitcher.get("home_runs_allowed"))
            state["starts"] += _safe_float(pitcher.get("games_started"))

    if not rows:
        raise ValueError("No MLB HR training rows built after prior-PA filter.")
    return pd.DataFrame(rows)


def _top_k_hit_rate(frame: pd.DataFrame, probs: np.ndarray, k: int) -> float:
    tmp = frame[["game_date", "actual_home_run"]].copy()
    tmp["probability"] = probs
    top = tmp.sort_values(["game_date", "probability"], ascending=[True, False]).groupby("game_date", group_keys=False).head(k)
    return float(top["actual_home_run"].mean()) if len(top) else float("nan")


def _calibration(frame: pd.DataFrame, probs: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    tmp = pd.DataFrame({"actual": frame["actual_home_run"].astype(int), "probability": probs})
    try:
        tmp["bucket"] = pd.qcut(tmp["probability"], q=bins, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for bucket, group in tmp.groupby("bucket", observed=True):
        rows.append(
            {
                "bucket": str(bucket),
                "rows": int(len(group)),
                "avg_probability": float(group["probability"].mean()),
                "actual_rate": float(group["actual"].mean()),
            }
        )
    return rows


def _evaluate(frame: pd.DataFrame, probs: np.ndarray) -> dict[str, Any]:
    y = frame["actual_home_run"].astype(int).to_numpy()
    probs = np.clip(probs.astype(float), 1e-6, 1 - 1e-6)
    baseline = np.clip(frame["baseline_probability"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    metrics: dict[str, Any] = {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "baseline_brier": float(brier_score_loss(y, baseline)),
        "baseline_log_loss": float(log_loss(y, baseline, labels=[0, 1])),
        "top_10_hit_rate": _top_k_hit_rate(frame, probs, 10),
        "top_25_hit_rate": _top_k_hit_rate(frame, probs, 25),
        "calibration": _calibration(frame, probs),
    }
    if len(np.unique(y)) > 1:
        metrics["auc"] = float(roc_auc_score(y, probs))
    return metrics


def parse_args() -> argparse.Namespace:
    today = datetime.now(timezone.utc).date()
    parser = argparse.ArgumentParser(description="Train an MLB batter home-run model.")
    parser.add_argument("--start-date", type=_date_arg, default=date(2025, 3, 1))
    parser.add_argument("--end-date", type=_date_arg, default=today)
    parser.add_argument("--test-start-date", type=_date_arg, default=date(2026, 3, 1))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset-out", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--min-prior-pa", type=int, default=20)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_date >= args.end_date:
        raise SystemExit("--start-date must be before --end-date")

    if args.refresh_cache and args.cache.exists():
        args.cache.unlink()

    schedule = _fetch_schedule_range(args.start_date, args.end_date)
    batting, pitching = _load_history(schedule, args.cache, args.start_date, args.end_date)
    rows = build_training_rows(schedule, batting, pitching, min_prior_pa=args.min_prior_pa)
    rows["game_date"] = pd.to_datetime(rows["game_date"])
    rows = rows.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.dataset_out, index=False)

    train = rows[rows["game_date"].dt.date < args.test_start_date].copy()
    test = rows[rows["game_date"].dt.date >= args.test_start_date].copy()
    if len(train) < 500 or len(test) < 250:
        split_date = rows["game_date"].quantile(0.8).date()
        train = rows[rows["game_date"].dt.date < split_date].copy()
        test = rows[rows["game_date"].dt.date >= split_date].copy()
        print(f"Requested split was too small; using fallback split date {split_date}.", flush=True)
    if train["actual_home_run"].nunique() < 2 or test["actual_home_run"].nunique() < 2:
        raise SystemExit("Train/test split does not contain both target classes.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=80,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train[MODEL_FEATURE_COLUMNS], train["actual_home_run"].astype(int))

    train_probs = model.predict_proba(train[MODEL_FEATURE_COLUMNS])[:, 1]
    test_probs = model.predict_proba(test[MODEL_FEATURE_COLUMNS])[:, 1]
    metrics = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model_version": args.model_version,
        "training_window": {
            "start_date": args.start_date.isoformat(),
            "end_date_exclusive": args.end_date.isoformat(),
            "test_start_date": args.test_start_date.isoformat(),
            "min_prior_pa": args.min_prior_pa,
        },
        "leakage_controls": [
            "games are processed chronologically",
            "batter, pitcher, venue, and league stats are updated only after each game is scored",
            "actual plate appearances are retained for audit only and excluded from model features",
        ],
        "estimator": "random_forest",
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "train": _evaluate(train, train_probs),
        "test": _evaluate(test, test_probs),
    }
    artifact = {
        "model": model,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "model_version": args.model_version,
        "estimator": "random_forest",
        "trained_at": metrics["generatedAt"],
        "training_window": metrics["training_window"],
        "metrics": metrics,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_out)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    print(f"Wrote {len(rows)} training rows to {args.dataset_out}", flush=True)
    print(f"Wrote model artifact to {args.model_out}", flush=True)
    print(f"Wrote metrics to {args.metrics_out}", flush=True)
    print(
        "Test metrics: "
        f"Brier={metrics['test']['brier']:.4f}, "
        f"log_loss={metrics['test']['log_loss']:.4f}, "
        f"AUC={metrics['test'].get('auc', float('nan')):.4f}, "
        f"top10_hit_rate={metrics['test']['top_10_hit_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
