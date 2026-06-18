"""MLB home run player-prop odds from The Odds API.

This module fetches current event-level player props and normalizes
`batter_home_runs` outcomes into one row per book/player/side/line.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Iterable, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.odds_math import american_to_decimal, american_to_implied

load_dotenv()

LOG = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
HR_MARKET = "batter_home_runs"
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_LINE = 0.5


class MlbHrOddsError(RuntimeError):
    """Raised when The Odds API returns a non-recoverable response."""


@dataclass
class OddsApiResponseMeta:
    requests_remaining: Optional[str] = None
    requests_used: Optional[str] = None
    requests_last: Optional[str] = None


@dataclass
class OddsApiClient:
    api_key: str
    timeout: int = 30
    min_request_interval_sec: float = 0.25
    request_count: int = 0
    response_meta: OddsApiResponseMeta = field(default_factory=OddsApiResponseMeta)
    _last_request_at: float = 0.0

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)

        query = {"apiKey": self.api_key, **(params or {})}
        response = requests.get(f"{BASE_URL}{path}", params=query, timeout=self.timeout)
        self._last_request_at = time.time()
        self.request_count += 1
        self.response_meta = OddsApiResponseMeta(
            requests_remaining=response.headers.get("x-requests-remaining"),
            requests_used=response.headers.get("x-requests-used"),
            requests_last=response.headers.get("x-requests-last"),
        )

        if response.status_code >= 400:
            try:
                payload = response.json()
                message = payload.get("message") or payload.get("error") or response.text
            except ValueError:
                message = response.text
            raise MlbHrOddsError(f"The Odds API {response.status_code}: {message}")

        if not response.text:
            return {}
        return response.json()


def get_api_key() -> str:
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise ValueError("ODDS_API_KEY not found in environment")
    return key


def normalize_name(name: object) -> str:
    raw = str(name or "")
    nfkd = unicodedata.normalize("NFKD", raw)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.lower().strip()
    ascii_name = re.sub(r"\b(jr\.?|sr\.?|iii|ii|iv|v)\b", "", ascii_name)
    ascii_name = re.sub(r"[^a-z0-9\s]", "", ascii_name)
    return re.sub(r"\s+", " ", ascii_name).strip()


def normalize_team(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def utc_day_bounds(day: date) -> tuple[str, str]:
    start = datetime.combine(day, dt_time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _time_distance_seconds(left: object, right: pd.Timestamp) -> float:
    parsed = pd.to_datetime(left, utc=True, errors="coerce")
    if pd.isna(parsed):
        return float("inf")
    return abs((parsed - right).total_seconds())


def fetch_mlb_events(
    client: OddsApiClient,
    *,
    game_date: date,
    sport_key: str = SPORT_KEY,
) -> list[dict[str, Any]]:
    start, end = utc_day_bounds(game_date)
    payload = client.get(
        f"/sports/{sport_key}/events",
        {
            "dateFormat": "iso",
            "commenceTimeFrom": start,
            "commenceTimeTo": end,
        },
    )
    return payload if isinstance(payload, list) else []


def fetch_event_hr_odds(
    client: OddsApiClient,
    *,
    event_id: str,
    sport_key: str = SPORT_KEY,
    regions: str = DEFAULT_REGIONS,
    market: str = HR_MARKET,
    odds_format: str = DEFAULT_ODDS_FORMAT,
) -> dict[str, Any]:
    payload = client.get(
        f"/sports/{sport_key}/events/{event_id}/odds",
        {
            "regions": regions,
            "markets": market,
            "dateFormat": "iso",
            "oddsFormat": odds_format,
        },
    )
    return payload if isinstance(payload, dict) else {}


def match_events_to_schedule(events: Iterable[dict[str, Any]], schedule: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return provider event id -> MLB schedule metadata."""

    if schedule.empty:
        return {}

    schedule_rows = []
    for _, row in schedule.iterrows():
        schedule_rows.append(
            {
                "game_id": f"MLB_{int(row['game_pk'])}",
                "game_pk": int(row["game_pk"]),
                "game_date": pd.to_datetime(row["game_date"]).date().isoformat(),
                "event_time": pd.to_datetime(row.get("game_datetime"), utc=True).isoformat()
                if pd.notna(row.get("game_datetime"))
                else None,
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "home_team_abbr": row.get("home_team_abbr"),
                "away_team_abbr": row.get("away_team_abbr"),
                "home_keys": {
                    normalize_team(row.get("home_team")),
                    normalize_team(row.get("home_team_abbr")),
                },
                "away_keys": {
                    normalize_team(row.get("away_team")),
                    normalize_team(row.get("away_team_abbr")),
                },
            }
        )

    matched: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = str(event.get("id") or "")
        home_key = normalize_team(event.get("home_team"))
        away_key = normalize_team(event.get("away_team"))
        if not event_id or not home_key or not away_key:
            continue

        candidates = [
            row
            for row in schedule_rows
            if home_key in row["home_keys"] and away_key in row["away_keys"]
        ]
        if not candidates:
            candidates = [
                row
                for row in schedule_rows
                if home_key in row["away_keys"] and away_key in row["home_keys"]
            ]
        if len(candidates) > 1:
            event_time = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
            if pd.notna(event_time):
                candidates = sorted(
                    candidates,
                    key=lambda row: _time_distance_seconds(row.get("event_time"), event_time),
                )
        if candidates:
            matched[event_id] = {
                **candidates[0],
                "provider_event_id": event_id,
                "provider_home_team": event.get("home_team"),
                "provider_away_team": event.get("away_team"),
                "provider_commence_time": event.get("commence_time"),
            }
    return matched


def flatten_event_hr_odds(
    payload: dict[str, Any],
    *,
    game_meta: Optional[dict[str, Any]] = None,
    snapshot_ts: Optional[str] = None,
    target_market: str = HR_MARKET,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    game_meta = game_meta or {}
    event_id = str(payload.get("id") or game_meta.get("provider_event_id") or "")
    snapshot = snapshot_ts or datetime.now(timezone.utc).isoformat()

    for bookmaker in payload.get("bookmakers") or []:
        book = bookmaker.get("key")
        book_title = bookmaker.get("title")
        for market in bookmaker.get("markets") or []:
            market_key = market.get("key")
            if market_key != target_market:
                continue
            last_update = market.get("last_update") or bookmaker.get("last_update")
            for outcome in market.get("outcomes") or []:
                player_name = outcome.get("description") or outcome.get("participant") or outcome.get("player")
                side = outcome.get("name")
                price = outcome.get("price")
                line = outcome.get("point")
                if not player_name or not side or price is None:
                    continue
                try:
                    american_price = int(price)
                except (TypeError, ValueError):
                    continue
                try:
                    line_value = float(line) if line is not None else None
                except (TypeError, ValueError):
                    line_value = None
                rows.append(
                    {
                        "game_id": game_meta.get("game_id"),
                        "game_pk": game_meta.get("game_pk"),
                        "game_date": game_meta.get("game_date"),
                        "event_time": game_meta.get("event_time") or payload.get("commence_time"),
                        "provider": "the_odds_api",
                        "provider_event_id": event_id,
                        "market": market_key,
                        "player_name": player_name,
                        "normalized_player_name": normalize_name(player_name),
                        "line": line_value,
                        "side": str(side).strip().title(),
                        "book": book,
                        "book_title": book_title,
                        "price": american_price,
                        "decimal_odds": american_to_decimal(american_price),
                        "implied_probability": american_to_implied(american_price),
                        "last_update": last_update,
                        "snapshot_ts": snapshot,
                        "raw_record": outcome,
                    }
                )
    return pd.DataFrame(rows)


def fetch_day_hr_odds(
    client: OddsApiClient,
    *,
    game_date: date,
    schedule: pd.DataFrame,
    regions: str = DEFAULT_REGIONS,
    market: str = HR_MARKET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    events = fetch_mlb_events(client, game_date=game_date)
    event_map = match_events_to_schedule(events, schedule)
    frames: list[pd.DataFrame] = []
    gaps: list[str] = []
    snapshot_ts = datetime.now(timezone.utc).isoformat()

    for event in events:
        event_id = str(event.get("id") or "")
        meta = event_map.get(event_id)
        if not meta:
            gaps.append(f"Unmatched Odds API MLB event {event_id}: {event.get('away_team')} at {event.get('home_team')}")
            continue
        try:
            payload = fetch_event_hr_odds(client, event_id=event_id, regions=regions, market=market)
        except MlbHrOddsError as exc:
            gaps.append(f"Odds fetch failed for {meta['game_id']}: {exc}")
            continue
        frame = flatten_event_hr_odds(payload, game_meta=meta, snapshot_ts=snapshot_ts, target_market=market)
        if frame.empty:
            gaps.append(f"No {market} odds returned for {meta['game_id']}")
            continue
        frames.append(frame)

    odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    audit = {
        "sportKey": SPORT_KEY,
        "market": market,
        "regions": regions,
        "gameDate": game_date.isoformat(),
        "eventsReturned": len(events),
        "eventsMatched": len(event_map),
        "oddsRows": int(len(odds)),
        "apiRequests": client.request_count,
        "apiCreditsRemaining": client.response_meta.requests_remaining,
        "apiCreditsUsed": client.response_meta.requests_used,
        "lastRequestCost": client.response_meta.requests_last,
        "gaps": gaps,
    }
    return odds, audit
