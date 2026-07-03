#!/usr/bin/env python3
"""Generate the PGA tournament registry from ESPN's season calendar."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ESPN_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsEdge/1.0)"}
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
CORE_EVENT_URL = "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga/events/{event_id}?lang=en&region=us"
DEFAULT_OUT = ROOT / "config" / "pga_tournaments.yaml"
MAJOR_PATTERNS = ("masters", "pga championship", "u s open", "the open")


def _fetch_json(url: str, *, timeout: int) -> dict[str, Any]:
    response = requests.get(url, headers=ESPN_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _slug(value: str, season: int) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    text = re.sub(r"_+", "_", text)
    if text == "u_s_open":
        text = "us_open"
    return f"{text}_{season}"


def _date(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return str(value or "")[:10]


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _course_meta(event: dict[str, Any]) -> dict[str, Any]:
    courses = event.get("courses") or []
    course = next((row for row in courses if row.get("host")), None) or (courses[0] if courses else {})
    return {
        "course": course.get("name") or "",
        "par": int(course.get("shotsToPar") or 72),
        "yardage": int(course["totalYards"]) if course.get("totalYards") is not None else None,
    }


def _priority(name: str) -> int:
    normalized = _normalize_name(name)
    if any(pattern in normalized for pattern in MAJOR_PATTERNS):
        return 100
    if "presidents cup" in normalized:
        return 90
    return 10


def build_registry(season: int, *, timeout: int, include_completed: bool) -> dict[str, Any]:
    scoreboard = _fetch_json(SCOREBOARD_URL, timeout=timeout)
    calendar = next(iter(scoreboard.get("leagues") or []), {}).get("calendar") or []
    tournaments = []
    today = datetime.now(timezone.utc).date()
    for item in calendar:
        event_id = str(item.get("id") or "")
        if not event_id:
            continue
        start = _date(item.get("startDate"))
        end = _date(item.get("endDate"))
        if start[:4] != str(season):
            continue
        if not include_completed and end and datetime.fromisoformat(end).date() < today:
            continue
        event = _fetch_json(CORE_EVENT_URL.format(event_id=event_id), timeout=timeout)
        name = str(event.get("name") or item.get("label") or event_id)
        course = _course_meta(event)
        row: dict[str, Any] = {
            "key": _slug(name, season),
            "espn_event_id": event_id,
            "odds_key": re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"),
            "espn_match": [name, str(event.get("shortName") or name)],
            "name": name,
            "course": course["course"],
            "par": course["par"],
            "start_date": start,
            "end_date": end,
            "total_rounds": 4,
            "cut_after_round": 2,
            "cut_size": 60 if _slug(name, season) == f"us_open_{season}" else 65,
            "cut_rule": "top_n_and_ties",
            "field_source": "espn",
            "prediction_window_days": 4,
            "post_window_days": 2,
            "priority": _priority(name),
        }
        if course["yardage"] is not None:
            row["yardage"] = course["yardage"]
        tournaments.append(row)
    return {"season": season, "tournaments": tournaments}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate config/pga_tournaments.yaml from ESPN.")
    parser.add_argument("--season", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--remaining-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_registry(args.season, timeout=args.timeout, include_completed=not args.remaining_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(registry['tournaments'])} PGA registry entries to {args.out}")


if __name__ == "__main__":
    main()
