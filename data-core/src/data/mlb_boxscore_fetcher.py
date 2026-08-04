"""
MLB boxscore fetcher for actual starters and bullpen usage.

This is optional enrichment for the MLB feature store. It stores one row per
game with actual starting pitcher IDs/names plus coarse bullpen usage.
"""

from __future__ import annotations

import re
import time
from typing import Iterable, Optional

import pandas as pd
import requests


MLB_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"


def _info_values(info: list) -> dict:
    """Return boxscore info values keyed by normalized label."""
    if not isinstance(info, list):
        return {}
    return {
        str(item.get("label", "")).strip().lower(): item.get("value")
        for item in info
        if isinstance(item, dict) and item.get("label")
    }


def _parse_weather(info: list) -> dict:
    """Parse weather and park-relative wind fields without raising."""
    values = _info_values(info)
    weather = values.get("weather")
    wind = values.get("wind")
    result = {
        "temp_f": None,
        "weather_condition": None,
        "wind_mph": None,
        "wind_dir": None,
    }

    if isinstance(weather, str):
        match = re.fullmatch(r"\s*(-?\d+)\s+degrees\s*,\s*(.+?)\s*", weather)
        if match:
            result["temp_f"] = int(match.group(1))
            result["weather_condition"] = match.group(2)

    if isinstance(wind, str):
        match = re.fullmatch(r"\s*(\d+)\s+mph\s*,\s*(.+?)\s*", wind)
        if match:
            result["wind_mph"] = int(match.group(1))
            result["wind_dir"] = match.group(2).removesuffix(".")

    return result


def _parse_game_info(info: list) -> dict:
    """Parse weather and general game metadata from a boxscore info list."""
    values = _info_values(info)
    result = _parse_weather(info)
    result.update(
        {
            "first_pitch": values.get("first pitch"),
            "attendance": None,
            "game_duration": values.get("game duration", values.get("t")),
        }
    )

    attendance = values.get("attendance", values.get("att"))
    if attendance is not None:
        try:
            result["attendance"] = int(str(attendance).replace(",", "").strip().removesuffix("."))
        except (TypeError, ValueError):
            pass
    return result


def _pitching_stats(team_payload: dict, pitcher_id: int) -> dict:
    return (
        team_payload.get("players", {})
        .get(f"ID{pitcher_id}", {})
        .get("stats", {})
        .get("pitching", {})
    )


def _pitcher_name(team_payload: dict, pitcher_id: Optional[int]) -> Optional[str]:
    if pitcher_id is None:
        return None
    return (
        team_payload.get("players", {})
        .get(f"ID{pitcher_id}", {})
        .get("person", {})
        .get("fullName")
    )


def _starter_id(team_payload: dict) -> Optional[int]:
    for pitcher_id in team_payload.get("pitchers", []):
        stats = _pitching_stats(team_payload, pitcher_id)
        if int(stats.get("gamesStarted") or 0) == 1:
            return int(pitcher_id)
    pitchers = team_payload.get("pitchers", [])
    return int(pitchers[0]) if pitchers else None


def _int_stat(stats: dict, key: str) -> int:
    value = stats.get(key)
    if value in (None, "", ".---"):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _team_stats_summary(team_payload: dict, prefix: str) -> dict:
    team_stats = team_payload.get("teamStats") or {}
    batting = team_stats.get("batting") or {}
    pitching = team_stats.get("pitching") or {}
    return {
        f"{prefix}_team_strikeouts": _int_stat(batting, "strikeOuts"),
        f"{prefix}_team_walks": _int_stat(batting, "baseOnBalls"),
        f"{prefix}_team_hits": _int_stat(batting, "hits"),
        f"{prefix}_team_home_runs": _int_stat(batting, "homeRuns"),
        f"{prefix}_team_pitching_strikeouts": _int_stat(pitching, "strikeOuts"),
    }


def _team_pitching_summary(team_payload: dict, prefix: str) -> dict:
    starter = _starter_id(team_payload)
    starter_stats = _pitching_stats(team_payload, starter) if starter else {}
    bullpen_pitchers = 0
    bullpen_outs = 0
    bullpen_earned_runs = 0
    bullpen_pitches = 0

    for pitcher_id in team_payload.get("pitchers", []):
        stats = _pitching_stats(team_payload, pitcher_id)
        if starter is not None and int(pitcher_id) == starter:
            continue
        bullpen_pitchers += 1
        bullpen_outs += _int_stat(stats, "outs")
        bullpen_earned_runs += _int_stat(stats, "earnedRuns")
        bullpen_pitches += _int_stat(stats, "pitchesThrown")

    return {
        f"{prefix}_actual_starter_id": starter,
        f"{prefix}_actual_starter": _pitcher_name(team_payload, starter),
        f"{prefix}_starter_outs": _int_stat(starter_stats, "outs"),
        f"{prefix}_starter_pitches": _int_stat(starter_stats, "pitchesThrown"),
        f"{prefix}_starter_earned_runs": _int_stat(starter_stats, "earnedRuns"),
        f"{prefix}_starter_strikeouts": _int_stat(starter_stats, "strikeOuts"),
        f"{prefix}_starter_walks": _int_stat(starter_stats, "baseOnBalls"),
        f"{prefix}_bullpen_pitchers": bullpen_pitchers,
        f"{prefix}_bullpen_outs": bullpen_outs,
        f"{prefix}_bullpen_earned_runs": bullpen_earned_runs,
        f"{prefix}_bullpen_pitches": bullpen_pitches,
    }


def fetch_mlb_boxscore(game_pk: int, timeout: int = 30) -> dict:
    """Fetch and summarize one MLB boxscore."""
    response = requests.get(MLB_BOXSCORE_URL.format(game_pk=int(game_pk)), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    teams = payload.get("teams", {})
    row = {"game_pk": int(game_pk)}
    row.update(_team_pitching_summary(teams.get("home", {}), "home"))
    row.update(_team_pitching_summary(teams.get("away", {}), "away"))
    row.update(_team_stats_summary(teams.get("home", {}), "home"))
    row.update(_team_stats_summary(teams.get("away", {}), "away"))
    row.update(_parse_game_info(payload.get("info", [])))
    return row


def fetch_mlb_boxscores(
    game_pks: Iterable[int],
    *,
    timeout: int = 30,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """Fetch summarized boxscores for many game IDs."""
    rows = []
    for i, game_pk in enumerate(game_pks, start=1):
        try:
            rows.append(fetch_mlb_boxscore(int(game_pk), timeout=timeout))
        except Exception as exc:  # noqa: BLE001
            rows.append({"game_pk": int(game_pk), "boxscore_error": str(exc)})
        if i % 100 == 0:
            print(f"Fetched {i} MLB boxscores...")
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return pd.DataFrame(rows)
