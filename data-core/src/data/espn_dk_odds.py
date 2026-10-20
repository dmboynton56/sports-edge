"""Free ESPN scoreboard DraftKings game lines (no API key).

Maps ``competitions[].odds`` into The Odds API event shape used by
``sync_odds`` / CFB snapshots. Game lines only: spread, moneyline, total.
Source / book tag is ``espn_dk``. Missing prices fail closed — no invented
American odds or lines.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import requests

GetFn = Callable[..., requests.Response]

ESPN_DK_BOOK = "espn_dk"
ESPN_DK_TITLE = "ESPN DraftKings"
USER_AGENT = "sports-edge-espn-dk/1.0 (+https://github.com/dmboynton56/sports-edge)"
SERVING_TIMEZONE = ZoneInfo("America/Denver")

SCOREBOARD_URLS = {
    "NFL": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "NBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "CFB": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
}

# ESPN provider id 100 is current DK; 58 appears in older dumps.
DRAFTKINGS_PROVIDER_IDS = frozenset({"58", "100"})


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def parse_american_price(value: object) -> int | None:
    if value is None or value == "":
        return None
    text = str(value).strip().replace("−", "-").replace("–", "-").replace(" ", "")
    try:
        price = int(float(text))
    except (TypeError, ValueError):
        return None
    if price == 0 or abs(price) < 100:
        return None
    return price


def parse_line(value: object) -> float | None:
    if value is None or value == "":
        return None
    text = (
        str(value)
        .strip()
        .lower()
        .replace("−", "-")
        .replace("–", "-")
        .replace("+", "")
        .replace("o", "")
        .replace("u", "")
        .replace(" ", "")
    )
    try:
        line = float(text)
    except (TypeError, ValueError):
        return None
    if line != line or line in {float("inf"), float("-inf")}:
        return None
    return line


def is_draftkings_odds(odds: Mapping[str, Any]) -> bool:
    provider = odds.get("provider") if isinstance(odds.get("provider"), Mapping) else {}
    name = str(provider.get("name") or provider.get("displayName") or "").strip().lower()
    provider_id = str(provider.get("id") or "").strip()
    if "draftkings" in name or "draft kings" in name:
        return True
    return provider_id in DRAFTKINGS_PROVIDER_IDS


def pick_draftkings_odds(competition: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = [row for row in (competition.get("odds") or []) if isinstance(row, Mapping)]
    for row in rows:
        if is_draftkings_odds(row):
            return row
    return None


def _close_quote(side: object) -> Mapping[str, Any]:
    if not isinstance(side, Mapping):
        return {}
    close = side.get("close")
    if isinstance(close, Mapping):
        return close
    return side


def _team_block(competition: Mapping[str, Any], home_away: str) -> Mapping[str, Any]:
    for competitor in competition.get("competitors") or []:
        if not isinstance(competitor, Mapping):
            continue
        if str(competitor.get("homeAway") or "").lower() == home_away:
            team = competitor.get("team") if isinstance(competitor.get("team"), Mapping) else {}
            return {"competitor": competitor, "team": team}
    return {"competitor": {}, "team": {}}


def _team_name(block: Mapping[str, Any]) -> str:
    team = block.get("team") if isinstance(block.get("team"), Mapping) else {}
    return str(team.get("displayName") or team.get("name") or "").strip()


def _team_abbr(block: Mapping[str, Any]) -> str:
    team = block.get("team") if isinstance(block.get("team"), Mapping) else {}
    return str(team.get("abbreviation") or "").strip()


def _event_commence(event: Mapping[str, Any], competition: Mapping[str, Any]) -> str | None:
    raw = competition.get("date") or competition.get("startDate") or event.get("date")
    if not raw:
        return None
    text = str(raw).replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def espn_odds_to_bookmaker(odds: Mapping[str, Any], home_name: str, away_name: str) -> dict[str, Any] | None:
    """Build one Odds-API bookmaker with complete game-line pairs only."""
    home_odds = odds.get("homeTeamOdds") if isinstance(odds.get("homeTeamOdds"), Mapping) else {}
    away_odds = odds.get("awayTeamOdds") if isinstance(odds.get("awayTeamOdds"), Mapping) else {}
    moneyline = odds.get("moneyline") if isinstance(odds.get("moneyline"), Mapping) else {}
    point_spread = odds.get("pointSpread") if isinstance(odds.get("pointSpread"), Mapping) else {}
    total = odds.get("total") if isinstance(odds.get("total"), Mapping) else {}

    home_ml = parse_american_price(
        _close_quote(moneyline.get("home")).get("odds")
        or home_odds.get("moneyLine")
        or home_odds.get("moneyline")
    )
    away_ml = parse_american_price(
        _close_quote(moneyline.get("away")).get("odds")
        or away_odds.get("moneyLine")
        or away_odds.get("moneyline")
    )

    home_spread_quote = _close_quote(point_spread.get("home"))
    away_spread_quote = _close_quote(point_spread.get("away"))
    home_spread = parse_line(home_spread_quote.get("line") or odds.get("spread"))
    away_spread = parse_line(away_spread_quote.get("line"))
    if home_spread is not None and away_spread is None:
        away_spread = -home_spread
    home_spread_price = parse_american_price(home_spread_quote.get("odds") or home_odds.get("spreadOdds"))
    away_spread_price = parse_american_price(away_spread_quote.get("odds") or away_odds.get("spreadOdds"))

    over_quote = _close_quote(total.get("over"))
    under_quote = _close_quote(total.get("under"))
    total_line = parse_line(over_quote.get("line") or under_quote.get("line") or odds.get("overUnder"))
    over_price = parse_american_price(over_quote.get("odds") or odds.get("overOdds"))
    under_price = parse_american_price(under_quote.get("odds") or odds.get("underOdds"))

    markets: list[dict[str, Any]] = []
    if home_ml is not None and away_ml is not None and home_name and away_name:
        markets.append(
            {
                "key": "h2h",
                "outcomes": [
                    {"name": home_name, "price": home_ml},
                    {"name": away_name, "price": away_ml},
                ],
            }
        )
    if (
        home_spread is not None
        and away_spread is not None
        and home_spread_price is not None
        and away_spread_price is not None
        and home_name
        and away_name
    ):
        markets.append(
            {
                "key": "spreads",
                "outcomes": [
                    {"name": home_name, "point": home_spread, "price": home_spread_price},
                    {"name": away_name, "point": away_spread, "price": away_spread_price},
                ],
            }
        )
    if total_line is not None and over_price is not None and under_price is not None:
        markets.append(
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": total_line, "price": over_price},
                    {"name": "Under", "point": total_line, "price": under_price},
                ],
            }
        )
    if not markets:
        return None
    return {
        "key": ESPN_DK_BOOK,
        "title": ESPN_DK_TITLE,
        "markets": markets,
    }


def espn_event_to_odds_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    competition = (event.get("competitions") or [{}])[0]
    if not isinstance(competition, Mapping):
        return None
    odds = pick_draftkings_odds(competition)
    if odds is None:
        return None
    home = _team_block(competition, "home")
    away = _team_block(competition, "away")
    home_name = _team_name(home)
    away_name = _team_name(away)
    commence = _event_commence(event, competition)
    if not home_name or not away_name or not commence:
        return None
    bookmaker = espn_odds_to_bookmaker(odds, home_name, away_name)
    if bookmaker is None:
        return None
    return {
        "id": str(event.get("id") or competition.get("id") or ""),
        "sport_key": None,
        "commence_time": commence,
        "home_team": home_name,
        "away_team": away_name,
        "home_abbr": _team_abbr(home) or None,
        "away_abbr": _team_abbr(away) or None,
        "bookmakers": [bookmaker],
        "_source": ESPN_DK_BOOK,
    }


def parse_espn_scoreboard_odds(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in payload.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        mapped = espn_event_to_odds_event(event)
        if not mapped or not mapped.get("id") or mapped["id"] in seen:
            continue
        seen.add(mapped["id"])
        events.append(mapped)
    return events


def scoreboard_date_tokens(start: date, end: date) -> list[str]:
    tokens: list[str] = []
    cursor = start
    while cursor <= end:
        tokens.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    return tokens


def scoreboard_params(league: str, *, dates: str | None = None, week: int | None = None, season_type: int | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": 200 if league == "CFB" else 100}
    if dates:
        params["dates"] = dates
    if week is not None:
        params["week"] = int(week)
    if season_type is not None:
        params["seasontype"] = int(season_type)
    if league == "CFB":
        params["groups"] = 80
    return params


def fetch_espn_scoreboard(
    league: str,
    *,
    dates: str | None = None,
    week: int | None = None,
    season_type: int | None = None,
    get: GetFn | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = SCOREBOARD_URLS[league.upper()]
    fetch = get or requests.get
    response = fetch(url, headers=_headers(), timeout=timeout, params=scoreboard_params(
        league.upper(), dates=dates, week=week, season_type=season_type
    ))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"ESPN {league} scoreboard returned a non-object payload")
    return payload


def fetch_espn_dk_events(
    league: str,
    *,
    start: date | None = None,
    end: date | None = None,
    week: int | None = None,
    season_type: int | None = None,
    now: datetime | None = None,
    get: GetFn | None = None,
) -> list[dict[str, Any]]:
    """Fetch ESPN scoreboards and return Odds-API-shaped DK game-line events."""
    league_key = league.upper()
    if league_key not in SCOREBOARD_URLS:
        raise ValueError(f"ESPN DK scoreboard is not configured for {league}")
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(SERVING_TIMEZONE).date()
    start = start or local
    end = end or (start + timedelta(days=7))

    payloads: list[Mapping[str, Any]] = []
    if week is not None:
        payloads.append(fetch_espn_scoreboard(league_key, week=week, season_type=season_type or 2, get=get))
    else:
        for token in scoreboard_date_tokens(start, end):
            try:
                payloads.append(fetch_espn_scoreboard(league_key, dates=token, get=get))
            except Exception:
                continue

    merged: dict[str, dict[str, Any]] = {}
    snapshot_ts = now.astimezone(timezone.utc).isoformat()
    for payload in payloads:
        for event in parse_espn_scoreboard_odds(payload):
            event["_snapshot_ts"] = snapshot_ts
            merged[event["id"]] = event
    return list(merged.values())


def iter_featured_game_lines(events: Iterable[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Helper for tests: flatten book/market/selection rows, never team totals."""
    rows = []
    for event in events:
        for book in event.get("bookmakers") or []:
            for market in book.get("markets") or []:
                if market.get("key") not in {"h2h", "spreads", "totals"}:
                    continue
                for outcome in market.get("outcomes") or []:
                    rows.append(
                        {
                            "event_id": event.get("id"),
                            "book": book.get("key"),
                            "market": market.get("key"),
                            "name": outcome.get("name"),
                            "line": outcome.get("point"),
                            "price": outcome.get("price"),
                        }
                    )
    return rows
