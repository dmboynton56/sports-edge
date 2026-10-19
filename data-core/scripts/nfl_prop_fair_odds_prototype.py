#!/usr/bin/env python3
"""Stats-only NFL receiving fair odds + free-line sample grade.

Walk-forward residual mean for receiving yards and receptions, 2023-2025.
Grades the PropLine CC-BY 14-day 2026 sample when the cache is present.
Pulls this week's live PropLine sportsbook lines (no DFS) for a comparison
table written to docs/analysis only.

Does not publish to the site or change live predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nfl_receiving_yards_prototype import (  # noqa: E402
    HISTORY_SEASONS,
    attach_nflverse_margin,
    attach_v2_margin,
    load_schedules,
    load_snaps,
    load_weekly,
    merge_snaps,
)
from src.data.nfl_historical_prop_lines import cache_frame, grade_real_lines  # noqa: E402
from src.data.nfl_injury_news import ESPN_INJURIES_URL, USER_AGENT, parse_espn_injuries  # noqa: E402
from src.data.propline_client import PropLineClient, PropLineError  # noqa: E402
from src.models.nfl_prop_fair_odds import (  # noqa: E402
    RECEPTION_LINES,
    YARD_LINES,
    _norm_name,
    apply_name_availability,
    apply_residual_mean,
    attach_distribution,
    attach_injury_availability,
    consensus_sportsbook_line,
    project_upcoming_rows,
    reject_dfs_rows,
    score_live_lines,
    walk_forward_stat,
)
from src.models.nfl_receiving_yards import DEFAULT_ALPHA, EVAL_SEASONS, build_feature_frame  # noqa: E402
from src.models.nfl_v2 import normalize_team_name  # noqa: E402

DEFAULT_METRICS = REPO_ROOT / "docs" / "analysis" / "nfl-props-fair-odds-metrics.json"
DEFAULT_LIVE = REPO_ROOT / "docs" / "analysis" / "nfl-props-week-live-comparison.json"
LIVE_MARKETS = ("player_reception_yds", "player_receptions")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--live-output", type=Path, default=DEFAULT_LIVE)
    parser.add_argument("--history-seasons", type=int, nargs="+", default=HISTORY_SEASONS)
    parser.add_argument("--eval-seasons", type=int, nargs="+", default=list(EVAL_SEASONS))
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument(
        "--rescore-live",
        type=Path,
        help="Reuse lines already stored in a live comparison JSON. No new PropLine calls.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip 2023-2025 walk-forward.")
    return parser.parse_args()


def load_injuries(seasons: list[int]) -> pd.DataFrame | None:
    try:
        import nflreadpy as nfl
    except ImportError:
        return None
    try:
        raw = nfl.load_injuries(seasons)
    except Exception:  # noqa: BLE001
        return None
    if hasattr(raw, "collect"):
        raw = raw.collect()
    frame = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
    if frame.empty:
        return None
    return frame


def espn_status_by_name() -> dict[str, str]:
    import requests

    try:
        response = requests.get(ESPN_INJURIES_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        rows = parse_espn_injuries(response.json())
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for row in rows:
        name = _norm_name(row.name)
        if name:
            out[name] = row.status
    return out


def _as_list(payload: Any) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "events", "sports"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def pull_live_propline(markets: tuple[str, ...] = LIVE_MARKETS) -> dict[str, Any]:
    api_key = os.getenv("PROPLINE_API_KEY")
    if not api_key:
        return {"skipped": True, "reason": "PROPLINE_API_KEY not set", "events": [], "lines": []}
    client = PropLineClient(api_key=api_key)
    events = _as_list(client.get("/sports/football_nfl/events"))
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=8)
    upcoming = []
    for event in events:
        try:
            start = datetime.fromisoformat(str(event.get("commence_time") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        if now - timedelta(hours=6) <= start <= horizon:
            upcoming.append(event)
    lines: list[dict[str, Any]] = []
    event_summaries = []
    for event in upcoming[:16]:
        event_id = str(event.get("id"))
        try:
            odds = client.get(
                f"/sports/football_nfl/events/{event_id}/odds",
                {"markets": ",".join(markets)},
            )
        except PropLineError as exc:
            event_summaries.append({"event_id": event_id, "error": str(exc)})
            continue
        if not isinstance(odds, dict):
            continue
        event_summaries.append(
            {
                "event_id": event_id,
                "home_team": odds.get("home_team") or event.get("home_team"),
                "away_team": odds.get("away_team") or event.get("away_team"),
                "commence_time": odds.get("commence_time") or event.get("commence_time"),
            }
        )
        for market in markets:
            for row in consensus_sportsbook_line(odds, market=market):
                row["home_team"] = event_summaries[-1]["home_team"]
                row["away_team"] = event_summaries[-1]["away_team"]
                row["commence_time"] = event_summaries[-1]["commence_time"]
                lines.append(row)
    return {
        "skipped": False,
        "request_count": client.request_count,
        "rate_limit_remaining": client.response_meta.rate_limit_remaining,
        "events": event_summaries,
        "lines": reject_dfs_rows(lines),
    }


def main() -> None:
    args = _parse_args()
    print("Loading nflverse weekly stats...", flush=True)
    stats = load_weekly(args.history_seasons)
    snaps = load_snaps(args.history_seasons)
    stats = merge_snaps(stats, snaps)
    schedules = load_schedules(args.history_seasons)
    spread_note = "nflverse closing spread/total"
    if args.skip_v2:
        schedules = attach_nflverse_margin(schedules)
    else:
        try:
            print("Scoring frozen v2 margin (read-only)...", flush=True)
            schedules = attach_v2_margin(schedules, args.history_seasons)
            spread_note = "v2 predicted margin + nflverse closing total"
        except Exception as exc:  # noqa: BLE001
            print(f"v2 unavailable ({type(exc).__name__}: {exc})", flush=True)
            schedules = attach_nflverse_margin(schedules)
            spread_note = f"nflverse close (v2 failed: {type(exc).__name__})"

    features = build_feature_frame(stats, schedules)
    injuries = load_injuries(args.history_seasons)
    features = attach_injury_availability(features, injuries)
    print(f"features={len(features)} injury_join={'yes' if injuries is not None else 'no'}", flush=True)

    if args.skip_eval:
        yards_metrics = json.loads(args.metrics_output.read_text(encoding="utf-8")).get("receiving_yards", {})
        rec_metrics = json.loads(args.metrics_output.read_text(encoding="utf-8")).get("receptions", {})
    else:
        yards_scored, yards_metrics = walk_forward_stat(
            features, stat="receiving_yards", eval_seasons=args.eval_seasons, synthetic_lines=YARD_LINES
        )
        rec_scored, rec_metrics = walk_forward_stat(
            features, stat="receptions", eval_seasons=args.eval_seasons, synthetic_lines=RECEPTION_LINES
        )
        lines = cache_frame()
        yards_2026 = attach_distribution(apply_residual_mean(features, stat="receiving_yards"))
        yards_2026 = yards_2026[pd.to_numeric(yards_2026["season"], errors="coerce").eq(2026)]
        rec_2026 = attach_distribution(apply_residual_mean(features, stat="receptions"))
        rec_2026 = rec_2026[pd.to_numeric(rec_2026["season"], errors="coerce").eq(2026)]
        yards_metrics["roi_clv"] = grade_real_lines(yards_2026, lines, stat="receiving_yards", market="player_reception_yds")
        rec_metrics["roi_clv"] = grade_real_lines(rec_2026, lines, stat="receptions", market="player_receptions")

    live_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live_site_published": False,
        "predictions_published": False,
        "skipped": True,
    }
    if not args.skip_live:
        if args.rescore_live and args.rescore_live.exists():
            print(f"Rescoring cached PropLine lines from {args.rescore_live}...", flush=True)
            cached = json.loads(args.rescore_live.read_text(encoding="utf-8"))
            live = {
                "skipped": False,
                "rescored": True,
                "request_count": cached.get("request_count"),
                "rate_limit_remaining": cached.get("rate_limit_remaining"),
                "events": cached.get("events") or [],
                "lines": cached.get("lines") or [],
            }
        else:
            print("Pulling live PropLine sportsbook lines...", flush=True)
            live = pull_live_propline()
        live_payload.update(live)
        if not live.get("skipped"):
            upcoming = schedules.copy()
            if "game_date" not in upcoming.columns:
                upcoming["game_date"] = upcoming.get("gameday")
            upcoming["game_date"] = pd.to_datetime(upcoming["game_date"], errors="coerce")
            now = pd.Timestamp.now("UTC").tz_localize(None)
            upcoming = upcoming[
                upcoming["game_date"].notna()
                & (upcoming["game_date"] >= now.normalize())
                & (upcoming["game_date"] <= now + pd.Timedelta(days=8))
            ]
            statuses = espn_status_by_name()
            modeled = attach_distribution(apply_residual_mean(features, stat="receiving_yards"))
            rec_modeled = attach_distribution(apply_residual_mean(features, stat="receptions"))
            if upcoming.empty:
                teams = {
                    normalize_team_name(event.get("home_team"))
                    for event in live.get("events") or []
                } | {
                    normalize_team_name(event.get("away_team"))
                    for event in live.get("events") or []
                }
                projections = apply_name_availability(
                    modeled[modeled["team"].isin(teams)].sort_values("game_date").groupby("player_id", sort=False).tail(1),
                    statuses,
                )
                rec_proj = apply_name_availability(
                    rec_modeled[rec_modeled["team"].isin(teams)].sort_values("game_date").groupby("player_id", sort=False).tail(1),
                    statuses,
                )
            else:
                projections = apply_name_availability(project_upcoming_rows(modeled, upcoming), statuses)
                rec_proj = apply_name_availability(project_upcoming_rows(rec_modeled, upcoming), statuses)
            yard_lines = [row for row in live.get("lines") or [] if row.get("market") == "player_reception_yds"]
            rec_lines = [row for row in live.get("lines") or [] if row.get("market") == "player_receptions"]
            comparison = reject_dfs_rows(score_live_lines(projections, yard_lines, stat="receiving_yards"))
            comparison += reject_dfs_rows(score_live_lines(rec_proj, rec_lines, stat="receptions"))
            live_payload["comparison"] = comparison
            live_payload["example_rows"] = comparison[:10]
            live_payload["n"] = len(comparison)
            live_payload["skipped"] = False
            live_payload["join_rule"] = (
                "Name match after stripping book team suffixes; player team must be "
                "one of the event's two teams. Duplicate Player (TEAM) rows collapsed."
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "nfl-prop-fair-odds-residual-v0",
        "live_site_touched": False,
        "predictions_published": False,
        "spread_total_source": spread_note,
        "history_seasons": args.history_seasons,
        "eval_seasons": args.eval_seasons,
        "receiving_yards": yards_metrics,
        "receptions": rec_metrics,
        "historical_lines": {
            "source": "docs/analysis/nfl-propline-graded-14d-compact.json",
            "coverage": "2026-09-18 to 2026-09-29 only. Not 2023-2025.",
            "yards_roi": yards_metrics.get("roi_clv"),
            "receptions_roi": rec_metrics.get("roi_clv"),
        },
        "notes": [
            "Mean is season-to-date blended with trailing-8. Opportunity is a residual scale.",
            "Baseline remains plain season-average (career / trail fallback) for the 17.63 comparison.",
            "Live comparison is analysis-only and is not a board, edge, or published prediction.",
        ],
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    args.live_output.write_text(json.dumps(live_payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": str(args.metrics_output),
                "live": str(args.live_output),
                "yards_mae": yards_metrics.get("mae"),
                "yards_baseline": yards_metrics.get("baseline_mae"),
                "rec_mae": rec_metrics.get("mae"),
                "live_n": live_payload.get("n"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
