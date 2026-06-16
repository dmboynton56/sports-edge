#!/usr/bin/env python3
"""Export a strict JSON PGA tournament dashboard artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pga_odds_fetcher import fetch_and_summarize  # noqa: E402


ARCHIVE = ROOT / "src" / "data" / "archive"
MAIN_TSV = ARCHIVE / "pga_results_2001-2025.tsv"
SUPP = ARCHIVE / "pga_results_espn_supplement.tsv"
DEFAULT_PRED = ROOT / "notebooks" / "cache" / "us_open_2026_predictions.csv"
DEFAULT_OUT = REPO_ROOT / "web" / "public" / "data" / "pga_tournaments" / "us_open_2026.json"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsEdge/1.0)"}


def _rel_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _json_clean(value.item())
    return value


def _load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not path.exists():
        return [], meta
    df = pd.read_csv(path)
    return df.to_dict(orient="records"), meta


def _load_merged_results(main_path: Path, supp_path: Path) -> pd.DataFrame:
    base_cols = [
        "season",
        "start",
        "tournament",
        "name",
        "position",
        "score",
        "round1",
        "round2",
        "round3",
        "round4",
        "total",
    ]
    frames: list[pd.DataFrame] = []
    for path in (main_path, supp_path):
        if path.exists():
            raw = pd.read_csv(path, sep="\t")
            frames.append(raw[[col for col in base_cols if col in raw.columns]])
    if not frames:
        return pd.DataFrame(columns=base_cols)
    merged = pd.concat(frames, ignore_index=True)
    merged["start"] = pd.to_datetime(merged["start"], errors="coerce")
    return merged.drop_duplicates(subset=["season", "start", "tournament", "name"], keep="last")


def _form_summaries(df: pd.DataFrame, year: int, max_recent_per_player: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if df.empty:
        return [], {}
    df = df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    events = []
    for (tournament, start), group in df[df["start"].dt.year == year].groupby(["tournament", "start"], sort=False):
        events.append(
            {
                "tournament": str(tournament),
                "start": str(start.date()) if pd.notna(start) else "",
                "players": int(len(group)),
            }
        )
    events.sort(key=lambda row: row["start"])

    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in df.sort_values("start", ascending=False).iterrows():
        by_player[str(row["name"])].append(
            {
                "tournament": str(row["tournament"]),
                "start": str(row["start"])[:10] if pd.notna(row["start"]) else "",
                "position": str(row["position"]),
                "scoreToPar": str(row["score"]) if pd.notna(row.get("score")) else "",
                "r1": row["round1"] if pd.notna(row.get("round1")) else None,
                "r2": row["round2"] if pd.notna(row.get("round2")) else None,
                "r3": row["round3"] if pd.notna(row.get("round3")) else None,
                "r4": row["round4"] if pd.notna(row.get("round4")) else None,
                "total": int(row["total"]) if pd.notna(row.get("total")) and str(row["total"]) else None,
            }
        )
    return events, {name: rows[:max_recent_per_player] for name, rows in by_player.items()}


def _score_to_par_str(raw: Any) -> str:
    text = str(raw or "").strip()
    if text.upper() in {"E", "0"}:
        return "E"
    return text


def _sort_key_to_par(to_par: str) -> float:
    text = to_par.strip().upper()
    if text == "E":
        return 0.0
    if text in {"WD", "DQ", "MDF", "DNS", "CUT", ""}:
        return 999.0
    try:
        return float(text.replace("+", ""))
    except ValueError:
        return 999.0


def _fetch_live_leaderboard() -> dict[str, Any] | None:
    try:
        response = requests.get(ESPN_SCOREBOARD, headers=ESPN_HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None
    events = data.get("events") or []
    if not events:
        return None
    event = events[0]
    comp = (event.get("competitions") or [{}])[0]
    competitors = comp.get("competitors") or []
    if not competitors:
        return None
    status_type = (comp.get("status") or {}).get("type", {})
    rows = []
    for competitor in competitors:
        athlete = competitor.get("athlete") or {}
        rounds: dict[int, int] = {}
        for line in competitor.get("linescores") or []:
            period = line.get("period")
            value = line.get("value")
            if period and value is not None:
                rounds[int(period)] = int(value)
        rows.append(
            {
                "player": athlete.get("displayName") or athlete.get("fullName") or "?",
                "toPar": _score_to_par_str(competitor.get("score")),
                "thru": str((competitor.get("status") or {}).get("displayThru") or ""),
                "totalStrokes": sum(rounds.values()) if rounds else None,
                "rounds": rounds,
                "status": ((competitor.get("status") or {}).get("type") or {}).get("description", ""),
            }
        )
    rows.sort(key=lambda row: (_sort_key_to_par(row["toPar"]), row["totalStrokes"] or 999, row["player"]))
    for index, row in enumerate(rows, start=1):
        row["position"] = index
        row["positionDisplay"] = str(index)
    return {
        "event": event.get("name", ""),
        "eventDate": event.get("date", ""),
        "currentRound": (comp.get("status") or {}).get("period", 1),
        "status": status_type.get("description", ""),
        "isCompleted": bool(status_type.get("completed")),
        "fetchedAt": datetime.now(timezone.utc).isoformat(),
        "players": rows,
    }


def _prob_from_prediction(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _build_normalized_markets(predictions: list[dict[str, Any]], event_time: str | None, model_version: str) -> list[dict[str, Any]]:
    rows = []
    market_keys = [
        ("win", "best_calibrated_target_win_prob"),
        ("top10", "best_calibrated_target_top10_prob"),
        ("top20", "best_calibrated_target_top20_prob"),
        ("make_cut", "best_calibrated_target_made_cut_prob"),
    ]
    for pred in predictions:
        for market, key in market_keys:
            prob = _prob_from_prediction(pred, key)
            if prob is None:
                continue
            rows.append(
                {
                    "id": f"PGA-{pred.get('player')}-{market}",
                    "sport": "PGA",
                    "league": "PGA",
                    "gameId": "us_open_2026",
                    "eventTime": event_time,
                    "subject": pred.get("player"),
                    "player": pred.get("player"),
                    "market": market,
                    "book": "model",
                    "line": None,
                    "price": None,
                    "modelProbability": prob,
                    "impliedProbability": None,
                    "edge": None,
                    "ev": None,
                    "kelly": None,
                    "confidence": pred.get("confidence"),
                    "modelVersion": model_version,
                    "source": pred.get("source"),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PGA tournament dashboard JSON.")
    parser.add_argument("--pred-csv", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tournament-key", default="us_open_2026")
    parser.add_argument("--event-name", default="U.S. Open Championship")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--course-name", default="Shinnecock Hills Golf Club")
    parser.add_argument("--course-par", type=int, default=70)
    parser.add_argument("--course-yardage", type=int, default=7440)
    parser.add_argument("--start-date", default="2026-06-18")
    parser.add_argument("--end-date", default="2026-06-21")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--live-odds", action="store_true")
    parser.add_argument("--skip-leaderboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, pred_meta = _load_predictions(args.pred_csv)
    merged = _load_merged_results(MAIN_TSV, SUPP)
    events, recent_by_player = _form_summaries(merged, year=args.season, max_recent_per_player=12)
    model_version = str(pred_meta.get("model_version") or pred_meta.get("modelVersion") or "pga-baseline-v0")

    market_odds = None
    if args.live_odds and not args.skip_odds:
        names = [str(row.get("player")) for row in predictions if row.get("player")]
        odds_key = "us_open" if args.tournament_key.startswith("us_open") else args.tournament_key
        try:
            market_odds = fetch_and_summarize(odds_key, prediction_names=names)
        except Exception as exc:
            market_odds = {"error": str(exc), "playerOdds": [], "books": []}

    live_leaderboard = None if args.skip_leaderboard else _fetch_live_leaderboard()
    normalized_markets = _build_normalized_markets(predictions, args.start_date, model_version)
    payload: dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "event": {
            "eventKey": args.tournament_key,
            "name": args.event_name,
            "season": args.season,
            "course": args.course_name,
            "par": args.course_par,
            "yardage": args.course_yardage,
            "startDate": args.start_date,
            "endDate": args.end_date,
            "status": "pre_tournament",
        },
        "predictions": predictions,
        "normalizedMarkets": normalized_markets,
        "predictionMeta": pred_meta,
        "espnSupplement": {"path": _rel_to_repo(SUPP), "rows": int(len(pd.read_csv(SUPP, sep="\t"))) if SUPP.exists() else 0},
        "mergedResults": {"mainPath": _rel_to_repo(MAIN_TSV), "supplementPath": _rel_to_repo(SUPP), "mergedRows": int(len(merged))},
        "tournaments2026": events,
        "recentByPlayer": recent_by_player,
        "gaps": pred_meta.get("quality_notes", []),
    }
    if market_odds:
        payload["marketOdds"] = market_odds
    if live_leaderboard:
        payload["liveLeaderboard"] = live_leaderboard

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _json_clean(payload)
    args.out.write_text(json.dumps(cleaned, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.out} ({len(predictions)} predictions, {len(normalized_markets)} market rows)")


if __name__ == "__main__":
    main()
