"""
Historical sportsbook odds helpers.

Currently supports The Odds API v4 historical snapshot endpoint. The endpoint
requires a paid Odds API plan; callers should handle `HistoricalOddsUnavailable`
as an expected data-access blocker rather than a code failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import requests


THE_ODDS_API_HISTORICAL_URL = "https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds"

SPORT_KEYS = {
    "MLB": "baseball_mlb",
    "NBA": "basketball_nba",
    "NFL": "americanfootball_nfl",
    "NHL": "icehockey_nhl",
}


class HistoricalOddsUnavailable(RuntimeError):
    """Raised when historical odds are unavailable for the configured key."""


@dataclass
class HistoricalOddsSnapshot:
    requested_ts: str
    returned_ts: Optional[str]
    previous_ts: Optional[str]
    next_ts: Optional[str]
    rows: pd.DataFrame


def fetch_historical_odds_snapshot(
    *,
    api_key: str,
    sport: str,
    snapshot_ts: str,
    markets: str = "h2h",
    regions: str = "us",
    bookmakers: Optional[str] = "draftkings,fanduel,betmgm",
    timeout: int = 30,
) -> HistoricalOddsSnapshot:
    """Fetch one historical odds snapshot from The Odds API."""
    sport_key = SPORT_KEYS.get(sport.upper(), sport)
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": snapshot_ts,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    response = requests.get(
        THE_ODDS_API_HISTORICAL_URL.format(sport_key=sport_key),
        params=params,
        timeout=timeout,
    )
    if response.status_code == 401 and "HISTORICAL_UNAVAILABLE" in response.text:
        raise HistoricalOddsUnavailable(response.text)
    if response.status_code == 402 and "HISTORICAL_UNAVAILABLE" in response.text:
        raise HistoricalOddsUnavailable(response.text)
    response.raise_for_status()
    payload = response.json()

    return HistoricalOddsSnapshot(
        requested_ts=snapshot_ts,
        returned_ts=payload.get("timestamp"),
        previous_ts=payload.get("previous_timestamp"),
        next_ts=payload.get("next_timestamp"),
        rows=flatten_odds_payload(payload.get("data", []), sport.upper(), snapshot_ts),
    )


def flatten_odds_payload(events: Iterable[dict], sport: str, snapshot_ts: str) -> pd.DataFrame:
    """Flatten The Odds API event payload into one row per outcome."""
    rows = []
    for event in events:
        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "sport": sport.upper(),
                            "event_id": event.get("id"),
                            "snapshot_ts": snapshot_ts,
                            "commence_time": event.get("commence_time"),
                            "home_team": event.get("home_team"),
                            "away_team": event.get("away_team"),
                            "book": book_key,
                            "market": market_key,
                            "outcome_name": outcome.get("name"),
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                            "last_update": market.get("last_update"),
                        }
                    )
    return pd.DataFrame(rows)


def collapse_moneyline_consensus(odds_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse flattened h2h rows to one consensus moneyline row per event.

    Prices are averaged across books. This keeps a simple stable format for
    backtests; a sharper later version can prefer closing Pinnacle or best line.
    """
    if odds_rows.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "snapshot_ts",
                "commence_time",
                "home_team",
                "away_team",
                "home_moneyline",
                "away_moneyline",
                "books",
            ]
        )
    h2h = odds_rows[odds_rows["market"].isin(["h2h", "moneyline"])].copy()
    if h2h.empty:
        return pd.DataFrame()

    event_cols = ["event_id", "snapshot_ts", "commence_time", "home_team", "away_team"]
    rows = []
    for keys, group in h2h.groupby(event_cols, dropna=False):
        row = dict(zip(event_cols, keys))
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_prices = pd.to_numeric(group.loc[group["outcome_name"] == home_team, "price"], errors="coerce")
        away_prices = pd.to_numeric(group.loc[group["outcome_name"] == away_team, "price"], errors="coerce")
        row["home_moneyline"] = float(home_prices.mean()) if home_prices.notna().any() else pd.NA
        row["away_moneyline"] = float(away_prices.mean()) if away_prices.notna().any() else pd.NA
        row["books"] = ",".join(sorted(group["book"].dropna().unique().tolist()))
        rows.append(row)
    return pd.DataFrame(rows)


def snapshot_ts_for_date(value: object, hour_utc: int = 16) -> str:
    """Build an ISO timestamp for a daily historical odds snapshot."""
    dt = pd.to_datetime(value)
    return datetime(dt.year, dt.month, dt.day, int(hour_utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
