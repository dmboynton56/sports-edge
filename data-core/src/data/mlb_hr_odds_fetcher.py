"""MLB home run player-prop odds from The Odds API with PropLine fallback.

This module fetches current event-level player props and normalizes standard
and alternate home-run outcomes into one row per book/player/side/line.

When The Odds API key is missing, quota-exhausted, or returns 401/429, the
fetcher falls back to PropLine if PROPLINE_API_KEY is set. PropLine event IDs
are separate from The Odds API, so we join events by team names and commence_time.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.odds_math import american_to_decimal, american_to_implied
from src.data.propline_client import (
    PropLineClient,
    PropLineError,
    fetch_propline_event_odds,
    fetch_propline_mlb_events,
    get_propline_api_key,
)

load_dotenv()

LOG = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
HR_MARKET = "batter_home_runs"
HR_ALTERNATE_MARKET = "batter_home_runs_alternate"
HR_MARKETS = (HR_MARKET, HR_ALTERNATE_MARKET)
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_LINE = 0.5
SLATE_TIMEZONE = "America/Denver"


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


def _normalize_markets(markets: Iterable[str] | str | None) -> tuple[str, ...]:
    """Return a stable, de-duplicated list of provider market keys."""

    if markets is None:
        values: Iterable[str] = HR_MARKETS
    elif isinstance(markets, str):
        values = markets.split(",")
    else:
        values = markets
    normalized = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    return normalized or HR_MARKETS


def slate_day_bounds(day: date, *, timezone_name: str = SLATE_TIMEZONE) -> tuple[str, str]:
    """Return UTC bounds for one local slate date.

    The Odds API accepts UTC timestamps, while the product defines a slate in
    America/Denver. Construct both local midnights independently so DST days
    correctly span 23 or 25 hours instead of assuming a fixed 24-hour UTC day.
    """

    local_tz = ZoneInfo(timezone_name)
    start_local = datetime.combine(day, dt_time.min, tzinfo=local_tz)
    end_local = datetime.combine(day + timedelta(days=1), dt_time.min, tzinfo=local_tz)
    start = start_local.astimezone(timezone.utc)
    end = end_local.astimezone(timezone.utc)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def utc_day_bounds(day: date) -> tuple[str, str]:
    """Backward-compatible alias for the local Denver slate bounds."""

    return slate_day_bounds(day)


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
    start, end = slate_day_bounds(game_date)
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
    market: str | None = None,
    markets: Iterable[str] | str | None = None,
    odds_format: str = DEFAULT_ODDS_FORMAT,
) -> dict[str, Any]:
    requested_markets = _normalize_markets(markets if markets is not None else market)
    payload = client.get(
        f"/sports/{sport_key}/events/{event_id}/odds",
        {
            "regions": regions,
            "markets": ",".join(requested_markets),
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
    target_market: str | None = None,
    target_markets: Iterable[str] | str | None = None,
    provider: str = "the_odds_api",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    game_meta = game_meta or {}
    event_id = str(payload.get("id") or game_meta.get("provider_event_id") or "")
    snapshot = snapshot_ts or datetime.now(timezone.utc).isoformat()
    requested_markets = _normalize_markets(target_markets if target_markets is not None else target_market)

    for bookmaker in payload.get("bookmakers") or []:
        book = bookmaker.get("key")
        book_title = bookmaker.get("title")
        for market in bookmaker.get("markets") or []:
            market_key = market.get("key")
            if market_key not in requested_markets:
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
                        "provider": provider,
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
    market: str | None = None,
    markets: Iterable[str] | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested_markets = _normalize_markets(markets if markets is not None else market)
    events = fetch_mlb_events(client, game_date=game_date)
    event_map = match_events_to_schedule(events, schedule)
    frames: list[pd.DataFrame] = []
    gaps: list[str] = []
    snapshot_ts = datetime.now(timezone.utc).isoformat()

    event_market_rows: dict[str, set[str]] = {}
    for event in events:
        event_id = str(event.get("id") or "")
        meta = event_map.get(event_id)
        if not meta:
            gaps.append(f"Unmatched Odds API MLB event {event_id}: {event.get('away_team')} at {event.get('home_team')}")
            continue
        try:
            payload = fetch_event_hr_odds(client, event_id=event_id, regions=regions, markets=requested_markets)
        except MlbHrOddsError as exc:
            gaps.append(f"Odds fetch failed for {meta['game_id']}: {exc}")
            continue
        frame = flatten_event_hr_odds(
            payload, game_meta=meta, snapshot_ts=snapshot_ts, target_markets=requested_markets, provider="the_odds_api"
        )
        if frame.empty:
            gaps.append(f"No requested HR odds returned for {meta['game_id']}")
            continue
        event_market_rows[meta["game_id"]] = set(frame["market"].dropna().astype(str))
        frames.append(frame)

    odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    market_rows = (
        odds.groupby("market", dropna=False).size().astype(int).to_dict()
        if not odds.empty
        else {}
    )
    events_with_market = {
        market_key: sum(market_key in markets_for_event for markets_for_event in event_market_rows.values())
        for market_key in requested_markets
    }
    events_missing_market = {
        market_key: sorted(
            meta["game_id"]
            for event_id, meta in event_map.items()
            if market_key not in event_market_rows.get(meta["game_id"], set())
        )
        for market_key in requested_markets
    }
    audit = {
        "sportKey": SPORT_KEY,
        "market": requested_markets[0] if len(requested_markets) == 1 else HR_MARKET,
        "markets": list(requested_markets),
        "regions": regions,
        "gameDate": game_date.isoformat(),
        "eventsReturned": len(events),
        "eventsMatched": len(event_map),
        "oddsRows": int(len(odds)),
        "oddsRowsByMarket": market_rows,
        "eventsWithMarket": events_with_market,
        "eventsMissingMarket": events_missing_market,
        "apiRequests": client.request_count,
        "apiCreditsRemaining": client.response_meta.requests_remaining,
        "apiCreditsUsed": client.response_meta.requests_used,
        "lastRequestCost": client.response_meta.requests_last,
        "gaps": gaps,
        "provider": "the_odds_api",
    }
    return odds, audit


def fetch_day_hr_odds_propline(
    client: PropLineClient,
    *,
    game_date: date,
    schedule: pd.DataFrame,
    market: str | None = None,
    markets: Iterable[str] | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch MLB HR odds from PropLine as a fallback to The Odds API.

    PropLine event IDs are separate from The Odds API. We fetch all PropLine
    events, join them to the schedule by team names + commence_time, then
    fetch odds for matched events.

    Args:
        client: PropLine API client
        game_date: Game date for slate (used to filter schedule)
        schedule: MLB schedule DataFrame with game_pk, teams, game_datetime
        market: Single market key (backwards compat)
        markets: List or comma-separated market keys

    Returns:
        (odds_df, audit_dict) with provider="propline"
    """
    requested_markets = _normalize_markets(markets if markets is not None else market)
    events = fetch_propline_mlb_events(client)
    event_map = match_events_to_schedule(events, schedule)
    frames: list[pd.DataFrame] = []
    gaps: list[str] = []
    snapshot_ts = datetime.now(timezone.utc).isoformat()

    event_market_rows: dict[str, set[str]] = {}
    for event in events:
        event_id = str(event.get("id") or "")
        meta = event_map.get(event_id)
        if not meta:
            gaps.append(f"Unmatched PropLine MLB event {event_id}: {event.get('away_team')} at {event.get('home_team')}")
            continue
        try:
            payload = fetch_propline_event_odds(client, event_id=event_id, markets=requested_markets)
        except PropLineError as exc:
            gaps.append(f"PropLine odds fetch failed for {meta['game_id']}: {exc}")
            continue
        frame = flatten_event_hr_odds(
            payload, game_meta=meta, snapshot_ts=snapshot_ts, target_markets=requested_markets, provider="propline"
        )
        if frame.empty:
            gaps.append(f"No requested HR odds returned from PropLine for {meta['game_id']}")
            continue
        event_market_rows[meta["game_id"]] = set(frame["market"].dropna().astype(str))
        frames.append(frame)

    odds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    market_rows = (
        odds.groupby("market", dropna=False).size().astype(int).to_dict()
        if not odds.empty
        else {}
    )
    events_with_market = {
        market_key: sum(market_key in markets_for_event for markets_for_event in event_market_rows.values())
        for market_key in requested_markets
    }
    events_missing_market = {
        market_key: sorted(
            meta["game_id"]
            for event_id, meta in event_map.items()
            if market_key not in event_market_rows.get(meta["game_id"], set())
        )
        for market_key in requested_markets
    }
    audit = {
        "sportKey": SPORT_KEY,
        "market": requested_markets[0] if len(requested_markets) == 1 else HR_MARKET,
        "markets": list(requested_markets),
        "gameDate": game_date.isoformat(),
        "eventsReturned": len(events),
        "eventsMatched": len(event_map),
        "oddsRows": int(len(odds)),
        "oddsRowsByMarket": market_rows,
        "eventsWithMarket": events_with_market,
        "eventsMissingMarket": events_missing_market,
        "apiRequests": client.request_count,
        "gaps": gaps,
        "provider": "propline",
    }
    return odds, audit
