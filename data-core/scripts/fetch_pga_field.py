#!/usr/bin/env python3
"""Fetch a PGA tournament field from ESPN event payloads."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsEdge/1.0)"}
CORE_EVENT_URL = "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga/events/{event_id}?lang=en&region=us"
SITE_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"


def _fetch_json(url: str, *, timeout: int) -> dict[str, Any]:
    response = requests.get(url.replace("http://", "https://"), headers=ESPN_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _event_date_token(event: dict[str, Any]) -> str | None:
    raw = event.get("date")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.strftime("%Y%m%d")


def _competitors_from_event(event: dict[str, Any], *, timeout: int) -> list[dict[str, Any]]:
    competitors: list[dict[str, Any]] = []
    for competition in event.get("competitions") or []:
        competitors.extend(competition.get("competitors") or [])
        ref = (competition.get("competitors") or {}).get("$ref") if isinstance(competition.get("competitors"), dict) else None
        if ref:
            competitors.extend(_fetch_json(ref, timeout=timeout).get("items") or [])
        comp_ref = competition.get("$ref")
        if comp_ref and not competition.get("competitors"):
            comp_payload = _fetch_json(comp_ref, timeout=timeout)
            competitors.extend(comp_payload.get("competitors") or [])
    return competitors


def _scoreboard_event(event_id: str, event: dict[str, Any], *, timeout: int) -> dict[str, Any] | None:
    urls = [SITE_SCOREBOARD_URL]
    date_token = _event_date_token(event)
    if date_token:
        urls.append(f"{SITE_SCOREBOARD_URL}?dates={date_token}")
    for url in urls:
        payload = _fetch_json(url, timeout=timeout)
        for candidate in payload.get("events") or []:
            if str(candidate.get("id")) == str(event_id):
                return candidate
    return None


def _athlete_payload(competitor: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    athlete = competitor.get("athlete")
    if isinstance(athlete, dict) and athlete.get("displayName"):
        return athlete
    ref = athlete.get("$ref") if isinstance(athlete, dict) else None
    if ref:
        return _fetch_json(ref, timeout=timeout)
    return {}


def _clean_player(competitor: dict[str, Any], *, timeout: int) -> dict[str, Any] | None:
    athlete = _athlete_payload(competitor, timeout=timeout)
    name = str(
        athlete.get("displayName")
        or athlete.get("fullName")
        or competitor.get("displayName")
        or ""
    ).strip()
    if not name:
        return None
    parts = name.split()
    country = athlete.get("citizenship") or ((athlete.get("citizenshipCountry") or {}).get("name"))
    flag = athlete.get("flag") or {}
    return {
        "player_id": str(athlete.get("id") or competitor.get("id") or "").strip() or None,
        "player": name,
        "first_name": athlete.get("firstName") or (parts[0] if parts else ""),
        "last_name": athlete.get("lastName") or (parts[-1] if parts else ""),
        "country": country,
        "country_code": (flag.get("alt") or "").lower() or None,
        "amateur": bool(athlete.get("amateur", False)),
        "past_champion": False,
        "content_url": next(
            (
                link.get("href")
                for link in athlete.get("links", [])
                if "playercard" in (link.get("rel") or [])
            ),
            None,
        ),
    }


def _event_meta(event: dict[str, Any]) -> dict[str, Any]:
    course = next((row for row in event.get("courses") or [] if row.get("host")), None)
    course = course or next(iter(event.get("courses") or []), {})
    return {
        "event_name": event.get("name") or event.get("shortName"),
        "season": int(str((event.get("season") or {}).get("year") or event.get("date", "")[:4] or datetime.now().year)),
        "course": course.get("name"),
        "par": course.get("shotsToPar"),
        "yardage": course.get("totalYards"),
        "start_date": str(event.get("date") or "")[:10],
        "end_date": str(event.get("endDate") or "")[:10],
    }


def fetch_field(event_id: str, *, timeout: int = 30) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event = _fetch_json(CORE_EVENT_URL.format(event_id=event_id), timeout=timeout)
    competitors = _competitors_from_event(event, timeout=timeout)
    if not competitors:
        site_event = _scoreboard_event(event_id, event, timeout=timeout)
        if site_event:
            competitors = _competitors_from_event(site_event, timeout=timeout)
    players = [_clean_player(row, timeout=timeout) for row in competitors]
    players = [row for row in players if row and row["player"]]
    players = sorted(
        {row["player"]: row for row in players}.values(),
        key=lambda row: (row.get("last_name") or "", row.get("first_name") or "", row.get("player_id") or ""),
    )
    return _event_meta(event), players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a PGA tournament field from ESPN.")
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--event-key", default="")
    parser.add_argument("--event-name", default="")
    parser.add_argument("--season", type=int)
    parser.add_argument("--course")
    parser.add_argument("--par", type=int)
    parser.add_argument("--yardage", type=int)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--text-out", type=Path, required=True)
    parser.add_argument("--min-players", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta, players = fetch_field(args.event_id, timeout=args.timeout)
    if len(players) < args.min_players:
        raise SystemExit(f"Expected at least {args.min_players} ESPN field players; got {len(players)}.")

    payload = {
        "event_key": args.event_key or f"espn_{args.event_id}",
        "event_name": args.event_name or meta.get("event_name") or args.event_id,
        "season": args.season or meta.get("season"),
        "course": args.course or meta.get("course"),
        "start_date": args.start_date or meta.get("start_date"),
        "end_date": args.end_date or meta.get("end_date"),
        "par": args.par or meta.get("par"),
        "yardage": args.yardage or meta.get("yardage"),
        "field_size": len(players),
        "source": "ESPN golf event competitors",
        "source_url": CORE_EVENT_URL.format(event_id=args.event_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "players": players,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.text_out.parent.mkdir(parents=True, exist_ok=True)
    args.text_out.write_text("\n".join(row["player"] for row in players) + "\n", encoding="utf-8")
    print(f"Wrote {len(players)} ESPN players to {args.json_out} and {args.text_out}")


if __name__ == "__main__":
    main()
