#!/usr/bin/env python3
"""Walk-forward receiving-yards prototype for WR/TE/RB, 2023-2025 OOS.

Loads free nflverse weekly stats + snap counts + schedules. Optionally scores
the frozen NFL v2 margin model (read-only) to drive team pass volume; v2 has
no totals model, so the total is nflverse closing `total_line`.

Writes metrics JSON only. Does not publish predictions or touch the live site.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.nfl_receiving_yards import (  # noqa: E402
    DEFAULT_ALPHA,
    EVAL_SEASONS,
    apply_opportunity_model,
    attach_distribution,
    build_feature_frame,
    grade_closing_sample,
    walk_forward_evaluate,
)
from src.models.nfl_v2 import FEATURE_COLUMNS, build_nfl_feature_store, normalize_team_name  # noqa: E402

CACHE_DIR = ROOT / "notebooks" / "cache"
DEFAULT_METRICS = REPO_ROOT / "docs" / "analysis" / "nfl-receiving-yards-prototype-metrics.json"
DEFAULT_PROBE = REPO_ROOT / "docs" / "analysis" / "nfl-props-market-probe.json"
HISTORY_SEASONS = list(range(2020, 2027))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--history-seasons", type=int, nargs="+", default=HISTORY_SEASONS)
    parser.add_argument("--eval-seasons", type=int, nargs="+", default=list(EVAL_SEASONS))
    parser.add_argument(
        "--skip-v2",
        action="store_true",
        help="Use nflverse closing spread instead of frozen v2 predicted margin.",
    )
    return parser.parse_args()


def _to_pandas(value) -> pd.DataFrame:
    if hasattr(value, "collect"):
        value = value.collect()
    return value.to_pandas() if hasattr(value, "to_pandas") else pd.DataFrame(value)


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    stats = _to_pandas(nfl.load_player_stats(seasons, summary_level="week"))
    if "season_type" in stats.columns:
        stats = stats[stats["season_type"].astype(str).str.upper().eq("REG")].copy()
    return stats


def load_snaps(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    snaps = _to_pandas(nfl.load_snap_counts(seasons))
    if snaps.empty:
        return snaps
    if "game_type" in snaps.columns:
        snaps = snaps[snaps["game_type"].astype(str).str.upper().eq("REG")].copy()
    snaps["team"] = snaps["team"].map(normalize_team_name)
    return snaps


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    games = _to_pandas(nfl.load_schedules(seasons))
    if "game_type" in games.columns:
        games = games[games["game_type"].astype(str).str.upper().eq("REG")].copy()
    games["home_team"] = games["home_team"].map(normalize_team_name)
    games["away_team"] = games["away_team"].map(normalize_team_name)
    games["closing_total"] = pd.to_numeric(games.get("total_line"), errors="coerce")
    # nflverse spread_line is the away spread (negative = away favorite).
    # Home margin ≈ spread_line; Sports Edge home spread = -spread_line.
    games["nflverse_home_margin"] = pd.to_numeric(games.get("spread_line"), errors="coerce")
    games["total_source"] = "nflverse_total_line"
    return games


def merge_snaps(stats: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    if snaps.empty:
        stats["offense_pct"] = np.nan
        return stats
    snap_rows = snaps.rename(columns={"player": "snap_player"})[
        ["game_id", "pfr_player_id", "team", "week", "season", "offense_pct", "offense_snaps", "snap_player"]
    ].copy()
    if "pfr_id" in stats.columns:
        merged = stats.merge(
            snap_rows,
            left_on=["game_id", "pfr_id"],
            right_on=["game_id", "pfr_player_id"],
            how="left",
            suffixes=("", "_snap"),
        )
        if merged["offense_pct"].notna().mean() > 0.4:
            return merged
    name = stats.get("player_display_name", stats.get("player_name")).astype(str).str.lower()
    snap_name = snap_rows["snap_player"].astype(str).str.lower()
    snap_lookup = snap_rows.assign(_name=snap_name)
    stats = stats.assign(_name=name)
    merged = stats.merge(
        snap_lookup,
        on=["game_id", "team", "_name"],
        how="left",
        suffixes=("", "_snap"),
    )
    return merged.drop(columns=["_name"], errors="ignore")


def attach_v2_margin(schedules: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    from scripts.eval_nfl_2026_week7 import load_team_games
    from src.models.nfl_live import load_live_artifact

    games = schedules.copy()
    team_games = load_team_games(seasons)
    features = build_nfl_feature_store(games, team_games, include_unplayed=False)
    artifact = load_live_artifact()
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"v2 features missing {missing}")
    margin = np.asarray(artifact["margin_model"].predict_margin(features), dtype=float)
    scored = features[["game_id"]].copy()
    scored["predicted_margin"] = margin
    scored["spread_source"] = "nfl_v2_live_margin_readonly"
    out = games.merge(scored, on="game_id", how="left")
    missing_margin = out["predicted_margin"].isna()
    out.loc[missing_margin, "predicted_margin"] = out.loc[missing_margin, "nflverse_home_margin"]
    out.loc[missing_margin, "spread_source"] = "nflverse_spread_line_fallback"
    out["spread_source"] = out["spread_source"].fillna("nfl_v2_live_margin_readonly")
    return out


def attach_nflverse_margin(schedules: pd.DataFrame) -> pd.DataFrame:
    games = schedules.copy()
    games["predicted_margin"] = games["nflverse_home_margin"]
    games["spread_source"] = "nflverse_spread_line"
    return games


def main() -> None:
    args = _parse_args()
    print("Loading nflverse weekly stats...", flush=True)
    stats = load_weekly(args.history_seasons)
    print(f"weekly={len(stats)}", flush=True)
    print("Loading snap counts...", flush=True)
    snaps = load_snaps(args.history_seasons)
    print(f"snaps={len(snaps)}", flush=True)
    stats = merge_snaps(stats, snaps)
    print("Loading schedules...", flush=True)
    schedules = load_schedules(args.history_seasons)
    spread_note = "nflverse closing spread/total"
    if args.skip_v2:
        schedules = attach_nflverse_margin(schedules)
    else:
        try:
            print("Scoring frozen v2 margin model (read-only)...", flush=True)
            schedules = attach_v2_margin(schedules, args.history_seasons)
            spread_note = "v2 predicted margin + nflverse closing total"
        except Exception as exc:  # noqa: BLE001
            print(f"v2 scoring unavailable ({type(exc).__name__}: {exc}); using nflverse close", flush=True)
            schedules = attach_nflverse_margin(schedules)
            spread_note = f"nflverse closing spread/total (v2 failed: {type(exc).__name__})"

    features = build_feature_frame(stats, schedules)
    scored, metrics = walk_forward_evaluate(features, eval_seasons=args.eval_seasons)
    probe_path = DEFAULT_PROBE
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        history = (probe.get("propline") or {}).get("history") or {}
        if history.get("closing_sample"):
            latest_alpha = DEFAULT_ALPHA
            if metrics.get("alpha_by_season"):
                latest_alpha = float(list(metrics["alpha_by_season"].values())[-1])
            modeled = attach_distribution(
                apply_opportunity_model(features),
                alpha_by_season={int(season): latest_alpha for season in features["season"].dropna().unique()},
            )
            metrics["roi_clv"] = grade_closing_sample(modeled, history)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "nfl-receiving-yards-prototype-v0",
        "live_site_touched": False,
        "predictions_published": False,
        "spread_total_source": spread_note,
        "history_seasons": args.history_seasons,
        "eval_seasons": args.eval_seasons,
        "rows_scored": int(len(scored)),
        "snap_coverage": float(pd.to_numeric(features.get("offense_pct"), errors="coerce").notna().mean())
        if not features.empty
        else 0.0,
        "metrics": metrics,
        "notes": [
            "Walk-forward by season for NB dispersion; player features are lagged week-to-week.",
            "Baseline is the player's season-to-date receiving yards, then career, then 5-game rolling.",
            "Routes are not in public nflverse weekly/PFR rec/FTN charting; snap share is the route proxy.",
            "v2 has no totals model. Totals always come from nflverse total_line.",
            "ROI/CLV is labeled small-sample/not-run when historical lines were not purchased.",
        ],
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.metrics_output), "n": metrics.get("n"), "mae": metrics.get("mae"), "baseline_mae": metrics.get("baseline_mae")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
