"""Load compact free historical NFL prop lines (read-only cache).

The only freely redistributable file we could actually download that joins to
nflverse names is PropLine's CC-BY-4.0 14-day Kaggle/Hugging Face sample
(2026-09-18 to 2026-09-29). That is not a 2023–2025 archive.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.data.nfl_prop_markets import classify_book
from src.models.nfl_prop_fair_odds import no_vig_over_prob
from src.models.nfl_receiving_yards import DEFAULT_ALPHA, _norm_player, p_over_line
from src.models.nfl_v2 import normalize_team_name

PREFERRED_BOOKS = ("draftkings", "fanduel", "pinnacle", "betmgm", "bovada")
MARKETS = ("player_reception_yds", "player_receptions")
DEFAULT_CACHE = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "analysis"
    / "nfl-propline-graded-14d-compact.json"
)
SOURCE = {
    "name": "PropLine graded player props sample",
    "url": "https://huggingface.co/datasets/propline/graded-player-props-sample",
    "kaggle": "https://www.kaggle.com/datasets/andyt338/graded-player-props-mlb-nfl-23-sportsbooks",
    "license": "CC BY 4.0",
    "years": "2026-09-18 to 2026-09-29 (last 14 days when published)",
    "not_years": "2023-2025",
}


def compact_graded_csv(path: Path, *, markets: Iterable[str] = MARKETS) -> list[dict[str, Any]]:
    """One preferred-book two-way close per player/event/market. DFS dropped."""

    wanted = set(markets)
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            market = row.get("market")
            book = str(row.get("bookmaker") or "").lower()
            if market not in wanted or classify_book(book) != "sportsbook":
                continue
            side = str(row.get("outcome_name") or "").lower()
            if side not in {"over", "under"}:
                continue
            point = row.get("closing_point") or row.get("line")
            key = (row.get("event_id"), row.get("player_name"), market, book, str(point))
            item = grouped.setdefault(
                key,
                {
                    "event_id": row.get("event_id"),
                    "commence_time": row.get("commence_time"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "market": market,
                    "book": book,
                    "player": row.get("player_name"),
                    "player_id": row.get("player_id") or None,
                    "line": point,
                    "opening_point": row.get("opening_point"),
                    "actual": row.get("actual_value"),
                },
            )
            try:
                price = float(row.get("closing_price") or row.get("price_american"))
            except (TypeError, ValueError):
                price = None
            if side == "over":
                item["over_price"] = price
                item["resolution"] = row.get("resolution")
            else:
                item["under_price"] = price

    two_way = [
        row
        for row in grouped.values()
        if row.get("over_price") is not None and row.get("under_price") is not None
    ]
    by_player: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in two_way:
        by_player[(row["event_id"], row["player"], row["market"])].append(row)

    compact = []
    preferred = set(PREFERRED_BOOKS)
    for options in by_player.values():
        pool = [row for row in options if row["book"] in preferred] or options

        def juice(row: dict[str, Any]) -> float:
            try:
                return abs(float(row.get("over_price")) + 110.0)
            except (TypeError, ValueError):
                return 10_000.0

        compact.append(min(pool, key=juice))
    for row in compact:
        try:
            row["line"] = float(row["line"])
        except (TypeError, ValueError):
            continue
        try:
            row["opening_point"] = float(row["opening_point"]) if row.get("opening_point") not in (None, "") else None
        except (TypeError, ValueError):
            row["opening_point"] = None
        try:
            row["actual"] = float(row["actual"]) if row.get("actual") not in (None, "") else None
        except (TypeError, ValueError):
            row["actual"] = None
        row["novig_p_over"] = no_vig_over_prob(row.get("over_price"), row.get("under_price"))
    return compact


def write_cache(rows: list[dict[str, Any]], path: Path = DEFAULT_CACHE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": SOURCE,
        "n": len(rows),
        "markets": sorted({str(row.get("market")) for row in rows}),
        "books": sorted({str(row.get("book")) for row in rows}),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def load_cache(path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    if not path.exists():
        return {"source": SOURCE, "n": 0, "rows": []}
    return json.loads(path.read_text(encoding="utf-8"))


def cache_frame(path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    payload = load_cache(path)
    frame = pd.DataFrame(payload.get("rows") or [])
    if frame.empty:
        return frame
    frame["game_date"] = pd.to_datetime(frame.get("commence_time"), errors="coerce").dt.tz_localize(None).dt.normalize()
    frame["home_team"] = frame["home_team"].map(normalize_team_name)
    frame["away_team"] = frame["away_team"].map(normalize_team_name)
    frame["_name"] = frame["player"].map(_norm_player)
    return frame


def grade_real_lines(
    scored: pd.DataFrame,
    lines: pd.DataFrame,
    *,
    stat: str,
    market: str,
) -> dict[str, Any]:
    """Small-sample ROI at close plus open-to-close line CLV. Not a strategy."""

    empty = {
        "status": "not_run",
        "label": "small-sample",
        "n": 0,
        "market": market,
        "stat": stat,
    }
    if scored.empty or lines.empty:
        empty["reason"] = "No scored rows or no cached lines."
        return empty
    sample = lines[lines["market"].eq(market)].copy()
    if sample.empty:
        empty["reason"] = f"Cache has no {market} rows."
        return empty
    if "_name" not in sample.columns:
        sample["_name"] = sample.get("player", sample.get("player_name")).map(_norm_player)
    if "home_team" in sample.columns:
        sample["home_team"] = sample["home_team"].map(normalize_team_name)
    if "away_team" in sample.columns:
        sample["away_team"] = sample["away_team"].map(normalize_team_name)
    if "game_date" not in sample.columns and "commence_time" in sample.columns:
        sample["game_date"] = pd.to_datetime(sample["commence_time"], errors="coerce").dt.tz_localize(None).dt.normalize()
    games = scored.copy()
    games["_name"] = games.get("player_display_name", games.get("player_name")).map(_norm_player)
    games["game_date"] = pd.to_datetime(games.get("game_date"), errors="coerce").dt.tz_localize(None).dt.normalize()
    merged = sample.merge(games, on="_name", how="inner", suffixes=("_line", ""))
    if "home_team" in merged.columns and "home_team_line" in merged.columns:
        matched = merged[
            ((merged["home_team"] == merged["home_team_line"]) & (merged["away_team"] == merged["away_team_line"]))
            | ((merged["home_team"] == merged["away_team_line"]) & (merged["away_team"] == merged["home_team_line"]))
        ]
        if not matched.empty:
            merged = matched
    if "game_date" in merged.columns and "game_date_line" in merged.columns:
        dated = merged[merged["game_date"].eq(merged["game_date_line"])]
        if dated.empty:
            delta = (merged["game_date"] - merged["game_date_line"]).abs()
            dated = merged[delta <= pd.Timedelta(days=1)]
        if not dated.empty:
            merged = dated
    bets = []
    for row in merged.to_dict("records"):
        try:
            point = float(row.get("line"))
            mu = float(row["mu"])
            alpha = float(row.get("alpha") or DEFAULT_ALPHA)
            actual = float(row.get("actual") if row.get("actual") is not None else row.get(stat))
            over_price = float(row.get("over_price"))
            under_price = float(row.get("under_price") or -110)
        except (TypeError, ValueError):
            continue
        p_over = p_over_line(mu, alpha, point)
        side = "over" if p_over >= 0.5 else "under"
        pushed = actual == point
        won = actual > point if side == "over" else actual < point
        price = over_price if side == "over" else under_price
        if pushed:
            profit = 0.0
        elif won:
            profit = price / 100.0 if price > 0 else 100.0 / abs(price)
        else:
            profit = -1.0
        open_line = row.get("opening_point")
        try:
            open_line = float(open_line) if open_line is not None else None
        except (TypeError, ValueError):
            open_line = None
        if open_line is None:
            clv = None
        else:
            move = point - open_line
            clv = move if side == "over" else -move
        bets.append(
            {
                "player": row.get("player"),
                "book": row.get("book"),
                "line": point,
                "open": open_line,
                "side": side,
                "p_over": p_over,
                "actual": actual,
                "profit": profit,
                "clv_line_move": clv,
            }
        )
    if not bets:
        return {
            "status": "sampled_unjoined",
            "label": "small-sample — 2026 PropLine 14d, not 2023-2025",
            "n": 0,
            "market": market,
            "cache_rows": int(len(sample)),
            "reason": "Name/date/team join missed every cached close.",
        }
    profits = [bet["profit"] for bet in bets]
    clvs = [
        float(bet["clv_line_move"])
        for bet in bets
        if bet["clv_line_move"] is not None and pd.notna(bet["clv_line_move"])
    ]
    return {
        "status": "small_sample",
        "label": "small-sample — 2026-09-18..29 PropLine CC-BY sample, not 2023-2025, not a strategy",
        "n": len(bets),
        "market": market,
        "unique_players": len({bet["player"] for bet in bets}),
        "books": sorted({str(bet["book"]) for bet in bets}),
        "flat_roi": float(sum(profits) / len(profits)),
        "units": float(sum(profits)),
        "win": int(sum(1 for bet in bets if bet["profit"] > 0)),
        "loss": int(sum(1 for bet in bets if bet["profit"] < 0)),
        "push": int(sum(1 for bet in bets if bet["profit"] == 0)),
        "clv_mean_line_move_toward_side": float(sum(clvs) / len(clvs)) if clvs else None,
        "clv_n": len(clvs),
    }
