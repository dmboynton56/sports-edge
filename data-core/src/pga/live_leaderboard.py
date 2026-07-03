"""ESPN PGA leaderboard parsing and tournament cut helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable
import math
import re
import unicodedata

import requests


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsEdge/1.0)"}
INACTIVE_TO_PAR = {"WD", "DQ", "DNS", "CUT", "MDF", ""}


class EspnScoreboardError(RuntimeError):
    """Raised when the ESPN PGA scoreboard cannot be fetched after retries."""


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.strip().lower().split())


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


def fetch_live_leaderboard(
    *,
    espn_match: Iterable[str] = (),
    scoreboard: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any] | None:
    """Fetch ESPN scoreboard and return the matched tournament leaderboard."""

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
    for attempt in range(1, max(int(max_attempts), 1) + 1):
        try:
            response = requests.get(ESPN_SCOREBOARD, headers=ESPN_HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            import time

            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise EspnScoreboardError(f"Failed to fetch ESPN PGA scoreboard after {max_attempts} attempts: {last_error}")


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
    relevant = [
        player
        for player in players
        if str(player.get("toPar") or "").upper() not in {"WD", "DQ", "DNS", ""}
    ]
    if not relevant:
        return False
    for player in relevant:
        round_value = (player.get("rounds") or {}).get(round_no)
        holes_played = (player.get("roundHoles") or {}).get(round_no)
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
