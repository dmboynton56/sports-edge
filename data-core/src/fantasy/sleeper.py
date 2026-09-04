"""Daily Sleeper roster and injury context for fantasy projections.

Sleeper exposes a public, read-only NFL player directory.  Its documentation
asks consumers to fetch the full directory no more than once per day, so the
fantasy refresh job makes one request and reuses the in-memory result for every
projection in the artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import requests


SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl?active=true"
UNAVAILABLE_INJURY_STATUSES = {"ir", "na", "out", "pup", "sus", "susp", "dnr"}
UNAVAILABLE_ROSTER_STATUSES = {"inactive", "injured reserve", "suspended"}
NFLVERSE_OUT_STATUSES = {"RES", "RSR", "PUP", "SUS", "EXE"}
NFLVERSE_INACTIVE_STATUSES = {"CUT", "DEV", "RLS"}


def load_sleeper_players(*, timeout: float = 30.0) -> dict[str, dict[str, Any]]:
    """Fetch the public NFL player directory once for a refresh run."""

    response = requests.get(
        SLEEPER_PLAYERS_URL,
        headers={"Accept": "application/json", "User-Agent": "sports-edge-fantasy/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Sleeper returned a non-object player directory")
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, Mapping)}


def load_nflverse_rosters(season: int) -> dict[str, dict[str, Any]]:
    """Return the published season roster keyed by stable GSIS player ID."""
    import nflreadpy as nfl  # Imported lazily so context-only tests stay lightweight.

    roster = nfl.load_rosters([int(season)])
    if hasattr(roster, "collect"):
        roster = roster.collect()
    if hasattr(roster, "to_pandas"):
        roster = roster.to_pandas()

    result: dict[str, dict[str, Any]] = {}
    for record in roster.to_dict("records"):
        gsis_id = str(record.get("gsis_id") or "").strip()
        if gsis_id:
            result[gsis_id] = dict(record)
    return result


def merge_sleeper_context(
    projections: Iterable[dict[str, Any]],
    players: Mapping[str, Mapping[str, Any]],
    official_rosters: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Merge current team, depth-chart, roster, and injury fields in place."""

    by_gsis: dict[str, Mapping[str, Any]] = {}
    by_name_position: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for player in players.values():
        gsis_id = str(player.get("gsis_id") or "").strip()
        if gsis_id:
            by_gsis[gsis_id] = player
        name = _normalize_name(player.get("full_name") or _full_name(player))
        position = str(player.get("position") or "").upper()
        if name and position:
            by_name_position.setdefault((name, position), []).append(player)

    summary = {"matched": 0, "unmatched": 0, "questionable": 0, "unavailable": 0}
    if official_rosters is not None:
        summary.update({"official_matched": 0, "official_active": 0})
    for projection in projections:
        player = by_gsis.get(str(projection.get("player_id") or ""))
        if player is None:
            candidates = by_name_position.get(
                (_normalize_name(projection.get("player_name")), str(projection.get("position") or "").upper()),
                [],
            )
            player = _best_match(candidates, str(projection.get("team") or ""))
        if player is None:
            summary["unmatched"] += 1
            continue

        summary["matched"] += 1
        official = (official_rosters or {}).get(str(projection.get("player_id") or "").strip())
        if official is None:
            official = (official_rosters or {}).get(str(player.get("gsis_id") or "").strip())
        if official is not None:
            summary["official_matched"] += 1
            official_status = str(official.get("status") or "").strip().upper()
            projection["official_roster_status"] = official_status or None
            if official_status == "ACT":
                summary["official_active"] += 1

        team = str((official or {}).get("team") or player.get("team") or "").strip().upper()
        if team:
            projection["team"] = team

        availability = sleeper_availability(player)
        official_status = str((official or {}).get("status") or "").strip().upper()
        if official_status in NFLVERSE_OUT_STATUSES:
            availability = "out"
        elif official_status in NFLVERSE_INACTIVE_STATUSES:
            availability = "inactive"
        projection["availability"] = availability
        projection["roster_status"] = player.get("status")
        projection["injury_status"] = player.get("injury_status")
        projection["injury_body_part"] = player.get("injury_body_part")
        projection["practice_participation"] = player.get("practice_participation")
        projection["depth_chart_order"] = _integer(player.get("depth_chart_order"))
        projection["availability_updated_at"] = _timestamp(player.get("news_updated"))

        if availability == "questionable" or availability == "doubtful":
            summary["questionable"] += 1
        elif availability in {"out", "inactive"}:
            summary["unavailable"] += 1

        if availability != "expected":
            detail = str(player.get("injury_body_part") or "").strip()
            status = str(player.get("injury_status") or player.get("status") or availability).strip()
            note = f"Sleeper roster feed: {status}{f' ({detail})' if detail else ''}."
            explanations = [str(item) for item in (projection.get("explanation") or [])]
            if note not in explanations:
                explanations.append(note)
            projection["explanation"] = explanations
    return summary


def sleeper_availability(player: Mapping[str, Any]) -> str:
    """Normalize Sleeper injury and roster fields into serving statuses."""
    injury = str(player.get("injury_status") or "").strip().lower()
    roster = str(player.get("status") or "").strip().lower()
    if injury in UNAVAILABLE_INJURY_STATUSES:
        return "out"
    if roster in UNAVAILABLE_ROSTER_STATUSES:
        return "inactive"
    if injury == "doubtful":
        return "doubtful"
    if injury == "questionable":
        return "questionable"
    return "expected"


# Backwards-compatible private name for callers written before this helper was public.
_availability = sleeper_availability


def _best_match(candidates: list[Mapping[str, Any]], team: str) -> Mapping[str, Any] | None:
    if not candidates:
        return None
    normalized_team = team.strip().upper()
    for candidate in candidates:
        if str(candidate.get("team") or "").strip().upper() == normalized_team:
            return candidate
    active = [candidate for candidate in candidates if str(candidate.get("status") or "").lower() == "active"]
    return active[0] if len(active) == 1 else candidates[0] if len(candidates) == 1 else None


def _full_name(player: Mapping[str, Any]) -> str:
    return " ".join(str(player.get(key) or "").strip() for key in ("first_name", "last_name")).strip()


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").lower().replace(".", "").replace("'", "").split())


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _timestamp(value: Any) -> str | None:
    try:
        milliseconds = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()
