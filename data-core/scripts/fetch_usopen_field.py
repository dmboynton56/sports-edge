#!/usr/bin/env python3
"""Fetch the current U.S. Open player field from the USGA public endpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_ENDPOINT = (
    "https://www.usopen.com/content/api/players.resource=@@content@@usopen@@2026@@players@@"
    "_jcr_content@@root@@all_player.year=2026.json"
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT = ROOT / "src" / "data" / "fields" / "us_open_2026_field.json"
DEFAULT_TEXT_OUT = ROOT / "src" / "data" / "fields" / "us_open_2026_field.txt"


def _extract_cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for block in payload.get("data", []):
        cards = block.get("resultset", {}).get("cards")
        if isinstance(cards, list):
            return cards
    return []


def _clean_player(card: dict[str, Any]) -> dict[str, Any]:
    first = str(card.get("firstName") or "").strip()
    last = str(card.get("lastName") or "").strip()
    country = card.get("country") or {}
    name = " ".join(part for part in (first, last) if part).strip()
    return {
        "player_id": str(card.get("identifier") or "").strip() or None,
        "player": name,
        "first_name": first,
        "last_name": last,
        "country": country.get("name"),
        "country_code": country.get("code"),
        "amateur": bool(card.get("amateur")),
        "past_champion": bool(card.get("champion")),
        "content_url": card.get("contentUrl"),
    }


def fetch_field(endpoint: str) -> list[dict[str, Any]]:
    response = requests.get(endpoint, timeout=30)
    response.raise_for_status()
    players = [_clean_player(card) for card in _extract_cards(response.json())]
    players = [player for player in players if player["player"]]
    players.sort(key=lambda row: (row["last_name"], row["first_name"], row["player_id"] or ""))
    return players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the 2026 U.S. Open field from USGA.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--text-out", type=Path, default=DEFAULT_TEXT_OUT)
    parser.add_argument("--min-players", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    players = fetch_field(args.endpoint)
    if len(players) < args.min_players:
        raise SystemExit(f"Expected at least {args.min_players} players; got {len(players)}.")

    payload = {
        "event_key": "us_open_2026",
        "event_name": "U.S. Open Championship",
        "season": 2026,
        "course": "Shinnecock Hills Golf Club",
        "start_date": "2026-06-18",
        "end_date": "2026-06-21",
        "par": 70,
        "yardage": 7440,
        "field_size": len(players),
        "source": "USGA public players endpoint",
        "source_url": args.endpoint,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "players": players,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.text_out.write_text(
        "\n".join(player["player"] for player in players) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(players)} players to {args.json_out} and {args.text_out}")


if __name__ == "__main__":
    main()
