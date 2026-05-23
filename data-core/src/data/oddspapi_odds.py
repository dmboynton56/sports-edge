"""
OddsPapi historical odds helpers.

Uses the v4 REST API (api.oddspapi.io/v4). Keys on the free tier currently
authenticate against v4; v5 returns invalid apiKey for the same key.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from src.utils.team_codes import canonical_nba_abbr

ODDSPAPI_BASE_URL = "https://api.oddspapi.io/v4"
DEFAULT_BOOKMAKER = "pinnacle"

SPORT_CONFIG: dict[str, dict[str, int]] = {
    "NBA": {"sport_id": 11, "tournament_id": 132},
    "NFL": {"sport_id": 14, "tournament_id": 31},
    "MLB": {"sport_id": 13, "tournament_id": 109},
}


class OddsPapiError(RuntimeError):
    """Raised when OddsPapi returns a non-recoverable API error."""


class OddsPapiRateLimited(OddsPapiError):
    """Raised when OddsPapi rate limits the caller."""


@dataclass
class OddsPapiClient:
    api_key: str
    timeout: int = 30
    request_count: int = 0
    quota_remaining: Optional[str] = None
    min_request_interval_sec: float = 2.0
    _last_request_at: float = 0.0

    def get(self, path: str, params: Optional[dict[str, Any]] = None, *, _attempt: int = 0) -> Any:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.min_request_interval_sec:
            time.sleep(self.min_request_interval_sec - elapsed)

        query = {"apiKey": self.api_key, **(params or {})}
        response = requests.get(
            f"{ODDSPAPI_BASE_URL}{path}",
            params=query,
            timeout=self.timeout,
        )
        self._last_request_at = time.time()
        self.request_count += 1
        self.quota_remaining = response.headers.get("X-RateLimit-Remaining")

        if response.status_code == 429:
            if _attempt >= 5:
                raise OddsPapiRateLimited("OddsPapi rate limit retries exhausted")
            retry_after = 2.0
            try:
                payload = response.json()
                details = payload.get("error", {}).get("retryMs")
                if details:
                    retry_after = max(float(details) / 1000.0, 1.0)
            except ValueError:
                retry_after = float(response.headers.get("Retry-After", "2"))
            time.sleep(retry_after)
            return self.get(path, params, _attempt=_attempt + 1)

        if response.status_code >= 400:
            try:
                payload = response.json()
                message = payload.get("error", {}).get("message") or payload.get("message") or response.text
            except ValueError:
                message = response.text
            raise OddsPapiError(f"OddsPapi {response.status_code}: {message}")

        if not response.text:
            return {}
        return response.json()


@dataclass
class MarketCatalog:
    moneyline_outcome_ids: set[int] = field(default_factory=set)
    spread_market_ids: set[int] = field(default_factory=set)
    outcome_side: dict[int, str] = field(default_factory=dict)
    market_meta: dict[int, dict[str, Any]] = field(default_factory=dict)


def norm_team(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def norm_team_for_sport(value: object, sport: str) -> str:
    if sport.upper() == "NBA":
        abbr = canonical_nba_abbr(str(value or ""))
        if abbr:
            return norm_team(abbr)
    return norm_team(value)


def decimal_to_american(price: object) -> Optional[int]:
    if price is None or pd.isna(price):
        return None
    decimal_price = float(price)
    if decimal_price <= 1:
        return None
    if decimal_price >= 2:
        return int(round((decimal_price - 1) * 100))
    return int(round(-100 / (decimal_price - 1)))


def load_fixture_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_fixture_cache(path: str | Path, cache: dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def fixture_cache_key(sport: str, game_date: object, home: object, away: object) -> str:
    day = pd.to_datetime(game_date).date().isoformat()
    return f"{sport.upper()}:{day}:{norm_team_for_sport(home, sport)}:{norm_team_for_sport(away, sport)}"


def discover_sport_config(client: OddsPapiClient, sport: str) -> dict[str, int]:
    sport = sport.upper()
    if sport in SPORT_CONFIG:
        return SPORT_CONFIG[sport]

    sports = client.get("/sports")
    sport_slug = {
        "MLB": "baseball",
        "NBA": "basketball",
        "NFL": "american-football",
    }.get(sport, sport.lower())
    sport_id = None
    for row in sports:
        slug = str(row.get("slug", "")).lower()
        name = str(row.get("sportName", "")).lower()
        if sport_slug in slug or sport.lower() in name:
            sport_id = int(row["sportId"])
            break
    if sport_id is None:
        raise OddsPapiError(f"Could not resolve OddsPapi sportId for {sport}")

    tournaments = client.get("/tournaments", {"sportId": sport_id})
    tournament_id = None
    for row in tournaments:
        name = str(row.get("tournamentName", "")).upper()
        if sport in name or (sport == "MLB" and "MLB" in name):
            tournament_id = int(row["tournamentId"])
            break
    if tournament_id is None and tournaments:
        tournament_id = int(tournaments[0]["tournamentId"])
    if tournament_id is None:
        raise OddsPapiError(f"Could not resolve OddsPapi tournamentId for {sport}")

    config = {"sport_id": sport_id, "tournament_id": tournament_id}
    SPORT_CONFIG[sport] = config
    return config


def fetch_market_catalog(client: OddsPapiClient, sport_id: int) -> MarketCatalog:
    markets = client.get("/markets", {"sportId": sport_id})
    catalog = MarketCatalog()
    for market in markets:
        if market.get("sportId") != sport_id:
            continue
        market_id = int(market["marketId"])
        market_type = str(market.get("marketType") or "").lower()
        period = str(market.get("period") or "")
        outcomes = market.get("outcomes") or []
        outcome_map = {
            int(outcome["outcomeId"]): str(outcome.get("outcomeName") or "")
            for outcome in outcomes
            if outcome.get("outcomeId") is not None
        }
        side_map = {}
        for outcome_id, outcome_name in outcome_map.items():
            if outcome_name == "1":
                side_map[outcome_id] = "participant1"
            elif outcome_name == "2":
                side_map[outcome_id] = "participant2"

        catalog.market_meta[market_id] = {
            "market_type": market_type,
            "handicap": market.get("handicap"),
            "period": period,
            "outcomes": outcome_map,
        }

        if market_type in {"moneyline", "winner"}:
            catalog.moneyline_outcome_ids.update(outcome_map)
            catalog.outcome_side.update(side_map)
        if market_type in {"spreads", "handicap"} and period in {"result", "fulltime"}:
            catalog.spread_market_ids.add(market_id)
            catalog.outcome_side.update(side_map)
    return catalog


def fetch_fixtures_for_day(
    client: OddsPapiClient,
    *,
    sport_id: int,
    tournament_id: int,
    game_date: object,
) -> list[dict[str, Any]]:
    day = pd.to_datetime(game_date).strftime("%Y-%m-%d")
    try:
        fixtures = client.get(
            "/fixtures",
            {
                "sportId": sport_id,
                "tournamentId": tournament_id,
                "from": f"{day}T00:00:00Z",
                "to": f"{day}T23:59:59Z",
            },
        )
    except OddsPapiError as exc:
        if "No fixtures found" in str(exc):
            return []
        raise
    return fixtures if isinstance(fixtures, list) else []


def fetch_fixture_historical_odds(
    client: OddsPapiClient,
    *,
    fixture_id: str,
    bookmaker: str = DEFAULT_BOOKMAKER,
) -> dict[str, Any]:
    return client.get(
        "/historical-odds",
        {"fixtureId": fixture_id, "bookmakers": bookmaker},
    )


def _fixture_team_keys(fixture: dict[str, Any], sport: str) -> tuple[str, str]:
    keys = []
    for prefix in ("participant1", "participant2"):
        name = fixture.get(f"{prefix}Name") or ""
        abbr = fixture.get(f"{prefix}Abbr") or ""
        candidates = [norm_team_for_sport(name, sport), norm_team_for_sport(abbr, sport)]
        keys.append(next((key for key in candidates if key), ""))
    return keys[0], keys[1]


def resolve_fixture_for_game(
    fixtures: list[dict[str, Any]],
    *,
    sport: str,
    home: object,
    away: object,
    game_date: object,
) -> Optional[dict[str, Any]]:
    home_key = norm_team_for_sport(home, sport)
    away_key = norm_team_for_sport(away, sport)
    target_day = pd.to_datetime(game_date).date()

    matches = []
    for fixture in fixtures:
        start_ts = fixture.get("trueStartTime") or fixture.get("startTime")
        fixture_day = pd.to_datetime(start_ts).date()
        if fixture_day != target_day:
            continue
        p1_key, p2_key = _fixture_team_keys(fixture, sport)
        if {p1_key, p2_key} == {home_key, away_key}:
            matches.append(fixture)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None

    for fixture in matches:
        p1_key, p2_key = _fixture_team_keys(fixture, sport)
        if p1_key == home_key and p2_key == away_key:
            return fixture
    return matches[0]


def flatten_historical_quotes(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixture_id = payload.get("fixtureId")
    bookmakers = payload.get("bookmakers") or {}

    for bookmaker, book_data in bookmakers.items():
        markets = (book_data or {}).get("markets") or {}
        for market_id, market_data in markets.items():
            outcomes = (market_data or {}).get("outcomes") or {}
            for outcome_id, outcome_data in outcomes.items():
                players = (outcome_data or {}).get("players") or {}
                for _player_id, quotes in players.items():
                    if not isinstance(quotes, list):
                        continue
                    for quote in quotes:
                        rows.append(
                            {
                                "fixture_id": fixture_id,
                                "bookmaker": bookmaker,
                                "market_id": int(market_id),
                                "outcome_id": int(outcome_id),
                                "price": quote.get("price"),
                                "created_at": quote.get("createdAt"),
                                "changed_at_ms": int(pd.Timestamp(quote.get("createdAt")).timestamp() * 1000)
                                if quote.get("createdAt")
                                else None,
                                "active": quote.get("active", True),
                                "limit": quote.get("limit"),
                            }
                        )
    return pd.DataFrame(rows)


def fixture_start_ms(fixture: dict[str, Any]) -> int:
    start_ts = fixture.get("trueStartTime") or fixture.get("startTime")
    return int(pd.Timestamp(start_ts).timestamp() * 1000)


def select_closing_quotes(
    flat: pd.DataFrame,
    start_time_ms: int,
    *,
    market_ids: Optional[set[int]] = None,
    outcome_ids: Optional[set[int]] = None,
) -> pd.DataFrame:
    if flat.empty:
        return flat.copy()

    filtered = flat.copy()
    if "changed_at_ms" not in filtered.columns or filtered["changed_at_ms"].isna().all():
        filtered["changed_at_ms"] = pd.to_datetime(filtered["created_at"], utc=True).astype("int64") // 10**6
    filtered["changed_at_ms"] = pd.to_numeric(filtered["changed_at_ms"], errors="coerce")
    filtered = filtered[filtered["changed_at_ms"].notna()]
    filtered = filtered[filtered["changed_at_ms"] < int(start_time_ms)]
    filtered = filtered[filtered["active"].fillna(True)]
    if market_ids:
        filtered = filtered[filtered["market_id"].isin(market_ids)]
    if outcome_ids:
        filtered = filtered[filtered["outcome_id"].isin(outcome_ids)]
    if filtered.empty:
        return filtered

    filtered = filtered.sort_values("changed_at_ms")
    return (
        filtered.groupby(["fixture_id", "bookmaker", "market_id", "outcome_id"], dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _participants_rotated(_payload: dict[str, Any], _bookmaker: str) -> bool:
    return False


def _map_side_to_home_away(side: str, *, participants_rotated: bool) -> str:
    if participants_rotated:
        return "away" if side == "participant1" else "home"
    return "home" if side == "participant1" else "away"


def extract_moneylines(
    closing: pd.DataFrame,
    catalog: MarketCatalog,
    payload: dict[str, Any],
    *,
    bookmaker: str = DEFAULT_BOOKMAKER,
) -> dict[str, Optional[float]]:
    ml = closing[closing["outcome_id"].isin(catalog.moneyline_outcome_ids)].copy()
    if ml.empty:
        ml = closing.copy()
    rotated = _participants_rotated(payload, bookmaker)
    home_price = None
    away_price = None
    snapshot_ts = None

    for _, row in ml.iterrows():
        outcome_id = int(row["outcome_id"])
        side = catalog.outcome_side.get(outcome_id)
        if side is None:
            continue
        home_away = _map_side_to_home_away(side, participants_rotated=rotated)
        american = decimal_to_american(row.get("price"))
        if home_away == "home":
            home_price = float(american) if american is not None else home_price
        else:
            away_price = float(american) if american is not None else away_price
        snapshot_ts = row.get("created_at") or snapshot_ts

    return {
        "home_moneyline": home_price,
        "away_moneyline": away_price,
        "snapshot_ts": snapshot_ts,
    }


def extract_spread(
    closing: pd.DataFrame,
    catalog: MarketCatalog,
    payload: dict[str, Any],
    *,
    bookmaker: str = DEFAULT_BOOKMAKER,
) -> dict[str, Optional[float]]:
    spread_rows = closing[closing["market_id"].isin(catalog.spread_market_ids)].copy()
    if spread_rows.empty:
        return {"home_spread": None, "home_price": None, "snapshot_ts": None}

    best_market_id = None
    best_score = None
    best_snapshot = None
    best_price = None

    for market_id, group in spread_rows.groupby("market_id"):
        meta = catalog.market_meta.get(int(market_id), {})
        outcomes = meta.get("outcomes", {})
        outcome_ids = list(outcomes.keys())
        if len(outcome_ids) < 2:
            continue
        side_rows = []
        for outcome_id in outcome_ids[:2]:
            rows = group[group["outcome_id"] == outcome_id]
            if rows.empty:
                side_rows = []
                break
            side_rows.append(rows.iloc[-1])
        if len(side_rows) != 2:
            continue
        price_gap = abs(float(side_rows[0]["price"]) - float(side_rows[1]["price"]))
        if best_score is None or price_gap < best_score:
            best_score = price_gap
            best_market_id = int(market_id)
            best_price = float(side_rows[0]["price"])
            best_snapshot = side_rows[0].get("created_at")

    if best_market_id is None:
        return {"home_spread": None, "home_price": None, "snapshot_ts": None}

    handicap = catalog.market_meta.get(best_market_id, {}).get("handicap")
    return {
        "home_spread": float(handicap) if handicap is not None else None,
        "home_price": best_price,
        "snapshot_ts": best_snapshot,
    }


def _cache_entry(cache: dict[str, Any], cache_key: str) -> Optional[dict[str, Any]]:
    value = cache.get(cache_key)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"fixture_id": value}
    return None


def resolve_and_fetch_closing_odds(
    client: OddsPapiClient,
    *,
    sport: str,
    home: object,
    away: object,
    game_date: object,
    bookmaker: str = DEFAULT_BOOKMAKER,
    fixture_cache: Optional[dict[str, Any]] = None,
    market_catalog: Optional[MarketCatalog] = None,
    config: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    sport = sport.upper()
    config = config or discover_sport_config(client, sport)
    catalog = market_catalog or fetch_market_catalog(client, config["sport_id"])
    cache = fixture_cache if fixture_cache is not None else {}
    cache_key = fixture_cache_key(sport, game_date, home, away)

    cached = _cache_entry(cache, cache_key)
    fixture_payload: Optional[dict[str, Any]] = None
    fixture_id = cached.get("fixture_id") if cached else None

    if cached and cached.get("startTime"):
        fixture_payload = cached
    elif fixture_id:
        fixture_payload = {"fixtureId": fixture_id, "startTime": cached.get("startTime")}
    else:
        fixtures = fetch_fixtures_for_day(
            client,
            sport_id=config["sport_id"],
            tournament_id=config["tournament_id"],
            game_date=game_date,
        )
        matched = resolve_fixture_for_game(
            fixtures,
            sport=sport,
            home=home,
            away=away,
            game_date=game_date,
        )
        if matched is None:
            return {
                "matched": False,
                "fixture_id": None,
                "historical_payload": None,
                "moneylines": {},
                "spread": {},
            }
        fixture_id = matched["fixtureId"]
        fixture_payload = matched
        cache[cache_key] = {
            "fixture_id": fixture_id,
            "startTime": matched.get("trueStartTime") or matched.get("startTime"),
            "participant1Name": matched.get("participant1Name"),
            "participant2Name": matched.get("participant2Name"),
        }

    try:
        historical = fetch_fixture_historical_odds(client, fixture_id=fixture_id, bookmaker=bookmaker)
    except OddsPapiError as exc:
        if "No historical odds found" in str(exc):
            return {
                "matched": False,
                "fixture_id": fixture_id,
                "historical_payload": None,
                "moneylines": {},
                "spread": {},
                "error": str(exc),
            }
        raise

    flat = flatten_historical_quotes(historical)
    start_time_ms = fixture_start_ms(fixture_payload)
    closing_ml = select_closing_quotes(
        flat,
        start_time_ms,
        outcome_ids=catalog.moneyline_outcome_ids,
    )
    closing_spread = select_closing_quotes(
        flat,
        start_time_ms,
        market_ids=catalog.spread_market_ids,
    )

    return {
        "matched": True,
        "fixture_id": fixture_id,
        "historical_payload": historical,
        "moneylines": extract_moneylines(closing_ml, catalog, historical, bookmaker=bookmaker),
        "spread": extract_spread(closing_spread, catalog, historical, bookmaker=bookmaker),
    }
