"""ESPN PGA leaderboard parsing and tournament cut helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable
import math
import re
import unicodedata

import requests


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
ESPN_CORE_EVENT = "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga/events/{event_id}?lang=en&region=us"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsEdge/1.0)"}
INACTIVE_TO_PAR = {"WD", "DQ", "DNS", "CUT", "MDF", ""}
INACTIVE_STATUS = {"Withdrawn", "Disqualified"}


class EspnScoreboardError(RuntimeError):
    """Raised when the ESPN PGA scoreboard cannot be fetched after retries."""


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.strip().lower().split())


def _is_player_active(player: dict[str, Any]) -> bool:
    """Return whether a player is active (not withdrawn, DQ'd, etc)."""
    toPar = str(player.get("toPar") or "").upper()
    status = str(player.get("status") or "")
    return toPar not in INACTIVE_TO_PAR and status not in INACTIVE_STATUS


def normalize_event_name(name: str) -> str:
    text = normalize_name(name).replace(".", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def score_to_par_str(raw: Any) -> str:
    text = str(raw or "").strip()
    if text.upper() in {"E", "0"}:
        return "E"
    return text


def sort_key_to_par(to_par: Any) -> float:
    text = str(to_par or "").strip().upper()
    if text == "E":
        return 0.0
    if text in INACTIVE_TO_PAR:
        return 999.0
    try:
        return float(text.replace("+", ""))
    except ValueError:
        return 999.0


def parse_to_par_value(to_par: Any) -> float | None:
    value = sort_key_to_par(to_par)
    return None if value >= 999 else value


def event_matches(event: dict[str, Any], patterns: Iterable[str]) -> int:
    """Return a fuzzy-match score for an ESPN event against registry patterns."""

    names = [
        event.get("name"),
        event.get("shortName"),
        ((event.get("competitions") or [{}])[0]).get("note"),
    ]
    event_names = [normalize_event_name(str(name)) for name in names if name]
    pattern_names = [normalize_event_name(pattern) for pattern in patterns if pattern]
    score = 0
    for event_name in event_names:
        for pattern in pattern_names:
            if not event_name or not pattern:
                continue
            if event_name == pattern:
                score = max(score, 100)
            elif pattern in event_name:
                score = max(score, 60)
            elif event_name in pattern:
                score = max(score, 40)
    return score


def _safe_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _assign_positions(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        row["position"] = index
        row["positionDisplay"] = str(index)

    index = 0
    while index < len(rows):
        end = index
        while end + 1 < len(rows) and rows[end + 1]["toPar"] == rows[index]["toPar"]:
            end += 1
        rank = index + 1
        tied = end > index
        for pos in range(index, end + 1):
            rows[pos]["positionDisplay"] = f"T{rank}" if tied else str(rank)
        index = end + 1


def parse_leaderboard_event(event: dict[str, Any]) -> dict[str, Any] | None:
    comp = (event.get("competitions") or [{}])[0]
    competitors = comp.get("competitors") or []
    if not competitors:
        return None

    status_type = (comp.get("status") or {}).get("type") or {}
    rows: list[dict[str, Any]] = []
    for competitor in competitors:
        athlete = competitor.get("athlete") or {}
        rounds: dict[int, int] = {}
        round_holes: dict[int, int] = {}
        for line in competitor.get("linescores") or []:
            period = _safe_int(line.get("period"))
            value = _safe_int(line.get("value"))
            if period is not None and value is not None:
                rounds[period] = value
                round_holes[period] = len(line.get("linescores") or [])
        to_par = score_to_par_str(competitor.get("score"))
        total_strokes = sum(rounds.values()) if rounds else None
        rows.append(
            {
                "player": athlete.get("displayName") or athlete.get("fullName") or "?",
                "toPar": to_par,
                "thru": str((competitor.get("status") or {}).get("displayThru") or ""),
                "totalStrokes": total_strokes,
                "rounds": rounds,
                "roundHoles": round_holes,
                "status": ((competitor.get("status") or {}).get("type") or {}).get("description", ""),
            }
        )

    rows.sort(key=lambda row: (sort_key_to_par(row["toPar"]), row["totalStrokes"] or 999, row["player"]))
    _assign_positions(rows)
    return {
        "event": event.get("name", ""),
        "eventDate": event.get("date", ""),
        "currentRound": (comp.get("status") or {}).get("period", 1),
        "status": status_type.get("description", ""),
        "statusState": status_type.get("state", ""),
        "isCompleted": bool(status_type.get("completed")),
        "fetchedAt": datetime.now(timezone.utc).isoformat(),
        "players": rows,
    }


def _fetch_json(url: str, *, timeout: int) -> dict[str, Any]:
    """Fetch one ESPN Core resource, normalizing the legacy http refs."""

    response = requests.get(url.replace("http://", "https://"), headers=ESPN_HEADERS, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"ESPN Core returned a non-object payload for {url}")
    return payload


def _resolve_core_ref(value: Any, *, timeout: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    ref = value.get("$ref")
    if ref:
        return _fetch_json(str(ref), timeout=timeout)
    return value


def _core_linescores(value: Any, *, timeout: int) -> list[dict[str, Any]]:
    payload = _resolve_core_ref(value, timeout=timeout)
    items = payload.get("items") if isinstance(payload, dict) else None
    if items is None and isinstance(payload, dict) and payload.get("period") is not None:
        items = [payload]
    return [item for item in (items or []) if isinstance(item, dict)]


def _normalize_core_competitor(competitor: dict[str, Any], *, timeout: int) -> dict[str, Any] | None:
    """Hydrate one Core competitor into the site-scoreboard competitor shape."""

    try:
        base = _resolve_core_ref(competitor, timeout=timeout) if competitor.get("$ref") else competitor
        athlete = _resolve_core_ref(base.get("athlete"), timeout=timeout)
        status = _resolve_core_ref(base.get("status"), timeout=timeout)
        score = _resolve_core_ref(base.get("score"), timeout=timeout)
        linescores = _core_linescores(base.get("linescores"), timeout=timeout)
    except (requests.RequestException, ValueError, TypeError) as exc:
        # A single withdrawn/retired competitor should not discard the full
        # board. The minimum-player guard in fetch_core_event protects against
        # accepting a mostly-empty Core response.
        return None

    name = str(athlete.get("displayName") or athlete.get("fullName") or "").strip()
    if not name:
        return None
    status_type = status.get("type") or {}
    position = status.get("position") or {}
    return {
        "id": str(base.get("id") or ""),
        "athlete": {
            "displayName": name,
            "fullName": athlete.get("fullName") or name,
        },
        "score": score.get("displayValue") or score.get("completedRoundsDisplayValue") or "",
        "position": position.get("displayName") or position.get("id"),
        "linescores": linescores,
        "status": {
            "displayThru": str(status.get("thru") or status.get("hole") or ""),
            "position": position,
            "type": {
                "id": status_type.get("id"),
                "name": status_type.get("name"),
                "state": status_type.get("state", ""),
                "completed": bool(status_type.get("completed")),
                "description": status_type.get("description") or status.get("displayValue") or "",
                "detail": status_type.get("detail"),
                "shortDetail": status_type.get("shortDetail"),
            },
        },
    }


def fetch_core_event(
    event_id: str,
    *,
    timeout: int = 30,
    max_workers: int = 8,
    min_players: int = 25,
) -> dict[str, Any]:
    """Fetch and normalize an ESPN Core event into the site event shape.

    ESPN Core exposes the event and each competitor's athlete, status, score,
    and linescores as separate resources. Requests are bounded so a refresh
    cannot create an unbounded fan-out.
    """

    if not str(event_id).strip():
        raise EspnScoreboardError("ESPN Core fallback requires an event id")
    root = _fetch_json(ESPN_CORE_EVENT.format(event_id=str(event_id).strip()), timeout=timeout)
    competitions = [row for row in (root.get("competitions") or []) if isinstance(row, dict)]
    competition = next((row for row in competitions if row.get("competitors")), None)
    if competition is None and competitions:
        competition = _resolve_core_ref(competitions[0], timeout=timeout)
    if not competition:
        raise EspnScoreboardError(f"ESPN Core event {event_id} has no competition payload")

    raw_competitors = [row for row in (competition.get("competitors") or []) if isinstance(row, dict)]
    if not raw_competitors:
        raise EspnScoreboardError(f"ESPN Core event {event_id} has no competitors")
    workers = max(1, min(int(max_workers), 32))
    normalized: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(raw_competitors))) as executor:
        futures = [executor.submit(_normalize_core_competitor, row, timeout=timeout) for row in raw_competitors]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception:
                row = None
            if row:
                normalized.append(row)

    normalized.sort(key=lambda row: (_safe_int(row.get("position")) or 9999, row["athlete"]["displayName"]))
    if len(normalized) < max(int(min_players), 1):
        raise EspnScoreboardError(
            f"ESPN Core event {event_id} returned only {len(normalized)} usable competitors "
            f"(minimum {max(int(min_players), 1)})"
        )

    status = _resolve_core_ref(competition.get("status"), timeout=timeout)
    season = root.get("season")
    year = (season or {}).get("year") if isinstance(season, dict) and not season.get("$ref") else None
    if year is None:
        year = str(root.get("date") or "")[:4]
    return {
        "id": str(root.get("id") or event_id),
        "name": root.get("name") or root.get("shortName") or str(event_id),
        "shortName": root.get("shortName") or root.get("name") or str(event_id),
        "date": root.get("date") or competition.get("date"),
        "endDate": root.get("endDate") or competition.get("endDate"),
        "season": {"year": int(year) if str(year).isdigit() else year},
        "competitions": [
            {
                "id": str(competition.get("id") or event_id),
                "date": competition.get("date") or root.get("date"),
                "endDate": competition.get("endDate") or root.get("endDate"),
                "status": status,
                "competitors": normalized,
            }
        ],
    }


def fetch_core_leaderboard(
    event_id: str,
    *,
    timeout: int = 30,
    max_workers: int = 8,
    min_players: int = 25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(normalized_event, parsed_leaderboard)`` from ESPN Core."""

    event = fetch_core_event(
        event_id,
        timeout=timeout,
        max_workers=max_workers,
        min_players=min_players,
    )
    leaderboard = parse_leaderboard_event(event)
    if leaderboard is None:
        raise EspnScoreboardError(f"ESPN Core event {event_id} did not contain a usable leaderboard")
    return event, leaderboard


def fetch_leaderboard_event(
    *,
    espn_match: Iterable[str] = (),
    espn_event_id: str | None = None,
    scoreboard: dict[str, Any] | None = None,
    timeout: int = 30,
    min_players: int = 25,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Fetch Site first, then fall back to the Core event endpoint."""

    site_error: Exception | None = None
    if scoreboard is None:
        try:
            scoreboard = fetch_scoreboard(timeout=timeout)
        except EspnScoreboardError as exc:
            site_error = exc
            scoreboard = {}
    events = scoreboard.get("events") or {}
    if isinstance(events, list) and events:
        patterns = tuple(espn_match or ())
        scored = sorted(
            ((event_matches(event, patterns), event) for event in events),
            key=lambda row: row[0],
            reverse=True,
        )
        event = scored[0][1] if scored and (not patterns or scored[0][0] > 0) else None
        if event is not None:
            leaderboard = parse_leaderboard_event(event)
            if leaderboard is not None:
                return event, leaderboard

    if espn_event_id:
        try:
            return fetch_core_leaderboard(
                espn_event_id,
                timeout=timeout,
                min_players=min_players,
            )
        except Exception as exc:
            if site_error:
                raise EspnScoreboardError(
                    f"ESPN Site API failed ({site_error}); Core API fallback failed: {exc}"
                ) from exc
            raise EspnScoreboardError(f"ESPN Core API fallback failed: {exc}") from exc
    if site_error:
        raise EspnScoreboardError(str(site_error)) from site_error
    return None, None


def fetch_live_leaderboard(
    *,
    espn_match: Iterable[str] = (),
    scoreboard: dict[str, Any] | None = None,
    espn_event_id: str | None = None,
    timeout: int = 30,
) -> dict[str, Any] | None:
    """Fetch ESPN Site scoreboard, falling back to Core when an event id is given."""

    if espn_event_id:
        _event, leaderboard = fetch_leaderboard_event(
            espn_match=espn_match,
            espn_event_id=espn_event_id,
            scoreboard=scoreboard,
            timeout=timeout,
        )
        return leaderboard

    if scoreboard is None:
        scoreboard = fetch_scoreboard(timeout=timeout)
        if scoreboard is None:
            return None

    events = scoreboard.get("events") or []
    if not events:
        return None

    patterns = tuple(espn_match or ())
    if patterns:
        scored = sorted(((event_matches(event, patterns), event) for event in events), key=lambda row: row[0], reverse=True)
        if not scored or scored[0][0] <= 0:
            return None
        event = scored[0][1]
    else:
        event = events[0]
    return parse_leaderboard_event(event)


def fetch_scoreboard(*, timeout: int = 30, max_attempts: int = 3, backoff_seconds: float = 2.0) -> dict[str, Any]:
    last_error: Exception | None = None
    attempts = max(int(max_attempts), 1)
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(ESPN_SCOREBOARD, headers=ESPN_HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                break
            if attempt >= attempts:
                break
            import time

            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise EspnScoreboardError(f"Failed to fetch ESPN PGA scoreboard after {attempts} attempts: {last_error}")


def rounds_completed_from_leaderboard(leaderboard: dict[str, Any], *, total_rounds: int = 4) -> int:
    current_round = int(leaderboard.get("currentRound") or 1)
    status = str(leaderboard.get("status") or "").lower()
    state = str(leaderboard.get("statusState") or "").lower()
    if leaderboard.get("isCompleted") or "complete" in status or state == "post":
        candidate = min(current_round, total_rounds)
    else:
        candidate = max(0, min(current_round - 1, total_rounds))

    players = leaderboard.get("players") or []
    for round_no in range(candidate, 0, -1):
        if _round_complete_for_field(players, round_no):
            return round_no
    return 0


def _round_complete_for_field(players: list[dict[str, Any]], round_no: int) -> bool:
    relevant = [player for player in players if _is_player_active(player)]
    if not relevant:
        return False
    for player in relevant:
        rounds_dict = player.get("rounds") or {}
        holes_dict = player.get("roundHoles") or {}
        # Try both int and string keys for compatibility
        round_value = rounds_dict.get(round_no) or rounds_dict.get(str(round_no))
        holes_played = holes_dict.get(round_no) or holes_dict.get(str(round_no))
        if round_value is None:
            return False
        if holes_played is not None and holes_played < 18:
            return False
    return True


def determine_cut(
    players: list[dict[str, Any]],
    *,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None]:
    """Apply top-N-and-ties cut logic."""

    valid = [player for player in players if str(player.get("toPar") or "").upper() not in INACTIVE_TO_PAR]
    valid.sort(key=lambda player: (sort_key_to_par(player["toPar"]), player.get("totalStrokes") or 999, player["player"]))
    if not valid:
        return [], players[:], None
    if len(valid) <= top_n:
        inactive = [player for player in players if str(player.get("toPar") or "").upper() in INACTIVE_TO_PAR]
        return valid, inactive, sort_key_to_par(valid[-1]["toPar"])

    cut_line = sort_key_to_par(valid[top_n - 1]["toPar"])
    made = [player for player in valid if sort_key_to_par(player["toPar"]) <= cut_line]
    missed = [player for player in valid if sort_key_to_par(player["toPar"]) > cut_line]
    missed.extend(player for player in players if str(player.get("toPar") or "").upper() in INACTIVE_TO_PAR)
    return made, missed, cut_line


def active_players_for_round_state(
    players: list[dict[str, Any]],
    *,
    rounds_completed: int,
    cut_after_round: int,
    cut_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None, bool]:
    """Return players to simulate, players out, cut line, and whether cut was applied."""

    if rounds_completed >= cut_after_round:
        made, missed, cut_line = determine_cut(players, top_n=cut_size)
        return made, missed, cut_line, True
    active = [player for player in players if str(player.get("toPar") or "").upper() not in {"WD", "DQ", "DNS", ""}]
    inactive = [player for player in players if player not in active]
    return active, inactive, None, False


def format_cut_line(cut_line: float | None) -> str | None:
    if cut_line is None:
        return None
    if cut_line == 0:
        return "E"
    if float(cut_line).is_integer():
        return f"+{int(cut_line)}" if cut_line > 0 else str(int(cut_line))
    return f"+{cut_line:.1f}" if cut_line > 0 else f"{cut_line:.1f}"
