"""NFL player-prop market keys and Odds-API-shaped payload summaries.

Analysis-only. Does not write predictions or spend credits by itself.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

NFL_SPORT_KEYS = ("football_nfl", "americanfootball_nfl")

# Keys the user asked about, in Odds API / PropLine vocabulary.
NFL_PROP_MARKETS = (
    "player_reception_yds",
    "player_receptions",
    "player_rush_yds",
    "player_pass_yds",
    "player_anytime_td",
)

DFS_BOOK_KEYS = frozenset(
    {
        "prizepicks",
        "underdog",
        "sleeper",
        "dabble",
        "parlayplay",
        "draftkings_pick6",
        "pick6",
    }
)

SPORTSBOOK_BOOK_KEYS = frozenset(
    {
        "draftkings",
        "fanduel",
        "betmgm",
        "williamhill_us",
        "caesars",
        "bovada",
        "pinnacle",
        "betrivers",
        "fanatics",
        "espnbet",
        "hardrockbet",
        "fliff",
        "betonlineag",
        "lowvig",
        "betus",
        "mybookieag",
        "ballybet",
        "hardrock",
        "novig",
        "prophetx",
        "kalshi",
        "polymarket",
        "smarkets",
        "matchbook",
    }
)


def classify_book(book_key: str) -> str:
    key = str(book_key or "").strip().lower()
    if key in DFS_BOOK_KEYS:
        return "dfs"
    if key in SPORTSBOOK_BOOK_KEYS:
        return "sportsbook"
    return "other"


def _player_name(outcome: Mapping[str, Any]) -> str:
    description = str(outcome.get("description") or "").strip()
    if description:
        return description
    name = str(outcome.get("name") or "").strip()
    if name.lower() not in {"over", "under", "yes", "no"}:
        return name
    return ""


def summarize_event_odds(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Collapse one event odds payload into per-market book/player counts."""

    event = payload or {}
    markets: dict[str, dict[str, Any]] = {}
    for book in event.get("bookmakers") or []:
        book_key = str(book.get("key") or "")
        book_class = classify_book(book_key)
        for market in book.get("markets") or []:
            market_key = str(market.get("key") or "")
            row = markets.setdefault(
                market_key,
                {
                    "market": market_key,
                    "books": set(),
                    "sportsbooks": set(),
                    "dfs_books": set(),
                    "other_books": set(),
                    "players": set(),
                    "outcome_count": 0,
                },
            )
            row["books"].add(book_key)
            if book_class == "dfs":
                row["dfs_books"].add(book_key)
            elif book_class == "sportsbook":
                row["sportsbooks"].add(book_key)
            else:
                row["other_books"].add(book_key)
            for outcome in market.get("outcomes") or []:
                row["outcome_count"] += 1
                player = _player_name(outcome)
                if player:
                    row["players"].add(player)

    summarized = []
    for market_key in NFL_PROP_MARKETS:
        row = markets.get(market_key, {"market": market_key, "books": set(), "sportsbooks": set(), "dfs_books": set(), "other_books": set(), "players": set(), "outcome_count": 0})
        summarized.append(
            {
                "market": market_key,
                "present": market_key in markets,
                "book_count": len(row["books"]),
                "sportsbook_count": len(row["sportsbooks"]),
                "dfs_book_count": len(row["dfs_books"]),
                "player_count": len(row["players"]),
                "outcome_count": int(row["outcome_count"]),
                "sportsbooks": sorted(row["sportsbooks"]),
                "dfs_books": sorted(row["dfs_books"]),
                "other_books": sorted(row["other_books"]),
            }
        )
    return {
        "event_id": event.get("id"),
        "sport_key": event.get("sport_key"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "commence_time": event.get("commence_time"),
        "markets": summarized,
    }


def aggregate_event_summaries(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean/median players-per-game and union of books across sampled events."""

    by_market: dict[str, dict[str, Any]] = {
        market: {
            "market": market,
            "events_present": 0,
            "player_counts": [],
            "sportsbooks": set(),
            "dfs_books": set(),
            "other_books": set(),
        }
        for market in NFL_PROP_MARKETS
    }
    rows = list(summaries)
    for summary in rows:
        for market in summary.get("markets") or []:
            bucket = by_market[str(market["market"])]
            if market.get("present"):
                bucket["events_present"] += 1
                bucket["player_counts"].append(int(market.get("player_count") or 0))
            bucket["sportsbooks"].update(market.get("sportsbooks") or [])
            bucket["dfs_books"].update(market.get("dfs_books") or [])
            bucket["other_books"].update(market.get("other_books") or [])

    markets = []
    for market, bucket in by_market.items():
        counts = bucket["player_counts"]
        markets.append(
            {
                "market": market,
                "events_sampled": len(rows),
                "events_present": bucket["events_present"],
                "players_per_game_mean": (sum(counts) / len(counts)) if counts else 0.0,
                "players_per_game_median": (sorted(counts)[len(counts) // 2] if counts else 0),
                "sportsbooks": sorted(bucket["sportsbooks"]),
                "dfs_books": sorted(bucket["dfs_books"]),
                "other_books": sorted(bucket["other_books"]),
            }
        )
    return {"events_sampled": len(rows), "markets": markets}


def credit_log_row(
    *,
    provider: str,
    endpoint: str,
    cost: int | None,
    remaining: str | None = None,
    used: str | None = None,
    last_cost: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "endpoint": endpoint,
        "estimated_cost": cost,
        "x_requests_remaining": remaining,
        "x_requests_used": used,
        "x_requests_last": last_cost,
        "notes": notes,
    }


def compact_closing_rows(payload: Mapping[str, Any], *, market: str = "player_reception_yds") -> list[dict[str, Any]]:
    """One main sportsbook line per player/book. Alternates dropped. No DFS mix."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for book in payload.get("bookmakers") or []:
        book_key = str(book.get("key") or "")
        if classify_book(book_key) != "sportsbook":
            continue
        for market_row in book.get("markets") or []:
            if str(market_row.get("key") or "") != market:
                continue
            by_player_point: dict[tuple[str, object], dict[str, Any]] = {}
            for outcome in market_row.get("outcomes") or []:
                player = _player_name(outcome)
                point = outcome.get("point")
                price = outcome.get("price")
                snapshots = outcome.get("snapshots") or []
                if snapshots:
                    last = snapshots[-1] if isinstance(snapshots[-1], dict) else {}
                    price = last.get("price", price)
                    point = last.get("point", point)
                if not player or point is None:
                    continue
                item = by_player_point.setdefault(
                    (player, point),
                    {"player": player, "book": book_key, "point": point},
                )
                name = str(outcome.get("name") or "").lower()
                if name == "over":
                    item["over_price"] = price
                elif name == "under":
                    item["under_price"] = price
            for item in by_player_point.values():
                grouped.setdefault((item["player"], book_key), []).append(item)

    rows = []
    for options in grouped.values():
        two_way = [row for row in options if row.get("over_price") is not None and row.get("under_price") is not None]
        pool = two_way or options
        def juice_gap(row: dict[str, Any]) -> float:
            try:
                return abs(float(row.get("over_price")) + 110.0)
            except (TypeError, ValueError):
                return 10_000.0
        rows.append(min(pool, key=juice_gap))
    preferred = {"draftkings", "fanduel", "pinnacle", "betmgm", "bovada"}
    trimmed = [row for row in rows if row["book"] in preferred]
    return trimmed or rows[:80]


def group_market_keys(keys: Iterable[str]) -> dict[str, list[str]]:
    wanted = set(NFL_PROP_MARKETS)
    found = []
    extras = []
    for key in keys:
        item = str(key)
        if item in wanted:
            found.append(item)
        else:
            extras.append(item)
    return {"requested_present": sorted(found), "other": sorted(extras)}
