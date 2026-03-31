"""
Fetch completed PGA Tour leaderboards from ESPN's public scoreboard API.

Maps responses into the same column schema as pga_results_*.tsv (tab-separated).
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsEdge/1.0; +https://github.com)"}


def _get_json(url: str, params: Optional[dict] = None) -> dict:
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=45)
    r.raise_for_status()
    return r.json()


def fetch_season_calendar(season_year: int) -> List[dict]:
    """Return calendar entries for a PGA season (from live scoreboard payload)."""
    data = _get_json(SCOREBOARD_URL)
    leagues = data.get("leagues") or []
    if not leagues:
        return []
    cal = leagues[0].get("calendar") or []
    out = []
    for c in cal:
        sd = c.get("startDate") or ""
        try:
            y = int(sd[:4]) if len(sd) >= 4 else 0
        except ValueError:
            y = 0
        if y == season_year:
            out.append(
                {
                    "id": c.get("id"),
                    "label": c.get("label"),
                    "startDate": c.get("startDate"),
                    "endDate": c.get("endDate"),
                }
            )
    return out


def _parse_espn_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def fetch_scoreboard_date(ymd: str) -> dict:
    """ymd = YYYYMMDD"""
    return _get_json(SCOREBOARD_URL, params={"dates": ymd})


def _round_strokes(linescores: List[dict]) -> Tuple[Optional[float], ...]:
    by_period = {}
    for block in linescores or []:
        p = block.get("period")
        if p in (1, 2, 3, 4):
            v = block.get("value")
            if v is not None:
                by_period[p] = float(v)
    return tuple(by_period.get(i) for i in (1, 2, 3, 4))


def _golf_positions(competitors: List[dict]) -> List[str]:
    """Assign 1, T2, ... from total strokes (lower better). Same index order as competitors."""
    n = len(competitors)
    meta: List[dict] = []
    for i, c in enumerate(competitors):
        name = (c.get("athlete") or {}).get("fullName") or ""
        ls = c.get("linescores") or []
        r1, r2, r3, r4 = _round_strokes(ls)
        rounds = [x for x in (r1, r2, r3, r4) if x is not None and x > 0]
        total = sum(rounds) if len(rounds) >= 4 else (sum(rounds) if len(rounds) >= 2 else None)
        score_s = (c.get("score") or "").strip()
        meta.append(
            {
                "i": i,
                "name": name,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "r4": r4,
                "total4": sum(rounds) if len(rounds) >= 4 else None,
                "total2": sum(rounds[:2]) if len(rounds) >= 2 else None,
                "score_str": score_s,
                "n_rounds": len(rounds),
            }
        )

    pos_out = [""] * n
    finishers = [m for m in meta if m["total4"] is not None]
    finishers.sort(key=lambda m: (m["total4"], m["name"]))
    i = 0
    while i < len(finishers):
        j = i
        while j + 1 < len(finishers) and finishers[j + 1]["total4"] == finishers[i]["total4"]:
            j += 1
        rank = i + 1
        lbl = str(rank) if i == j else f"T{rank}"
        for k in range(i, j + 1):
            pos_out[finishers[k]["i"]] = lbl
        i = j + 1

    for m in meta:
        if pos_out[m["i"]]:
            continue
        ss = m["score_str"].upper()
        if ss in ("WD", "DQ", "MDF", "DNS"):
            pos_out[m["i"]] = ss
        elif m["n_rounds"] <= 2 and m["n_rounds"] > 0:
            pos_out[m["i"]] = "CUT"
        else:
            pos_out[m["i"]] = ss or "WD"

    return pos_out


def event_to_tsv_rows(event: dict, competition: dict) -> List[dict]:
    ev_name = event.get("name") or "Unknown Event"
    season = (event.get("season") or {}).get("year")
    if season is None:
        season = datetime.now(timezone.utc).year

    start_dt = _parse_espn_dt(competition.get("date") or event.get("date"))
    end_dt = _parse_espn_dt(competition.get("endDate") or event.get("endDate"))
    start_s = start_dt.strftime("%Y-%m-%d") if start_dt else ""
    end_s = end_dt.strftime("%Y-%m-%d") if end_dt else start_s
    location = f"{ev_name} — ESPN"

    competitors = competition.get("competitors") or []
    positions = _golf_positions(competitors)

    rows = []
    for idx, c in enumerate(competitors):
        athlete = c.get("athlete") or {}
        name = athlete.get("fullName") or ""
        r1, r2, r3, r4 = _round_strokes(c.get("linescores") or [])
        position = positions[idx] if idx < len(positions) else ""
        score_disp = (c.get("score") or "").strip()

        total_val: Optional[int] = None
        if r1 and r2 and r3 and r4:
            total_val = int(r1 + r2 + r3 + r4)
        elif r1 and r2:
            total_val = int(r1 + r2) if position == "CUT" else None

        row = {
            "season": int(season),
            "start": start_s,
            "end": end_s,
            "tournament": ev_name,
            "location": location,
            "position": position,
            "name": name,
            "score": score_disp if score_disp else (str(position) if position in ("CUT", "WD", "DQ") else ""),
            "round1": r1 if r1 is not None else "",
            "round2": r2 if r2 is not None else "",
            "round3": r3 if r3 is not None else "",
            "round4": r4 if r4 is not None else "",
            "total": total_val if total_val is not None else "",
            "earnings": "",
            "fedex_points": "",
        }
        rows.append(row)
    return rows


def find_populated_scoreboard_date(
    start: datetime, end: datetime, min_players: int = 30
) -> Optional[Tuple[str, dict]]:
    """Scan day-by-day between start and end (inclusive) for a scoreboard with enough players."""
    cur = start.date()
    last = end.date()
    while cur <= last + timedelta(days=2):
        ymd = cur.strftime("%Y%m%d")
        try:
            data = fetch_scoreboard_date(ymd)
        except requests.RequestException:
            cur += timedelta(days=1)
            time.sleep(0.15)
            continue
        events = data.get("events") or []
        if not events:
            cur += timedelta(days=1)
            time.sleep(0.1)
            continue
        ev = events[0]
        comp = (ev.get("competitions") or [{}])[0]
        n = len(comp.get("competitors") or [])
        if n >= min_players:
            return ymd, data
        cur += timedelta(days=1)
        time.sleep(0.1)
    return None


def fetch_season_results(
    season_year: int,
    *,
    only_completed_before: Optional[datetime] = None,
    sleep_s: float = 0.2,
    min_players: int = 30,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch all tournaments in season_year from ESPN calendar that are complete.
    Returns (dataframe, log lines).
    """
    if only_completed_before is None:
        only_completed_before = datetime.now(timezone.utc)

    log: List[str] = []
    cal = fetch_season_calendar(season_year)
    log.append(f"Calendar entries for {season_year}: {len(cal)}")

    all_rows: List[dict] = []
    seen_ids: set = set()

    for entry in cal:
        eid = entry.get("id")
        label = entry.get("label")
        start_dt = _parse_espn_dt(entry.get("startDate"))
        end_dt = _parse_espn_dt(entry.get("endDate"))
        if not start_dt or not eid:
            continue
        if end_dt and end_dt > only_completed_before:
            log.append(f"skip future: {label}")
            continue

        if eid in seen_ids:
            continue

        found = find_populated_scoreboard_date(start_dt, end_dt or start_dt, min_players=min_players)
        if not found:
            log.append(f"no board: {label} ({eid})")
            continue
        ymd, data = found
        events = data.get("events") or []
        if not events:
            continue
        ev = events[0]
        if ev.get("id") != eid:
            log.append(f"date {ymd} returned different event ({ev.get('name')}), skip {label}")
            continue
        comp = (ev.get("competitions") or [{}])[0]
        st = ((comp.get("status") or {}).get("type") or {}).get("completed")
        if st is not True:
            log.append(f"not final: {label} (date {ymd})")
            continue

        rows = event_to_tsv_rows(ev, comp)
        all_rows.extend(rows)
        seen_ids.add(eid)
        log.append(f"ok {label}: {len(rows)} rows (date={ymd})")
        time.sleep(sleep_s)

    if not all_rows:
        return pd.DataFrame(), log

    df = pd.DataFrame(all_rows)
    return df, log
