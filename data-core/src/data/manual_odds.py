"""
Manual odds storage and management for PGA markets.

Supports multi-market odds (win, top5, top10, top20, made_cut, matchups)
from any source — screenshots, manual entry, or API. Stores as a single
JSON file that the export pipeline picks up.

Schema: notebooks/cache/pga_manual_odds.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from src.data.odds_math import (
    american_to_decimal,
    american_to_implied,
    best_available,
    consensus_no_vig,
    edge,
    edge_signal,
    ev,
    kelly_fraction,
    overround,
)

LOG = logging.getLogger(__name__)

MarketType = Literal["win", "top5", "top10", "top20", "made_cut", "frl", "matchup"]

VALID_MARKETS: list[MarketType] = ["win", "top5", "top10", "top20", "made_cut", "frl", "matchup"]

MODEL_PROB_KEYS: dict[MarketType, list[str]] = {
    "win": [
        "best_calibrated_target_win_prob",
        "lr_target_win_prob",
    ],
    "top5": [],  # MC only: sim_top5_pct / 100
    "top10": [
        "best_calibrated_target_top10_prob",
        "rf_target_top10_prob",
        "lr_target_top10_prob",
    ],
    "top20": [
        "best_calibrated_target_top20_prob",
        "meta_ensemble_target_top20_prob",
        "lr_target_top20_prob",
    ],
    "made_cut": [
        "best_calibrated_target_made_cut_prob",
        "meta_ensemble_target_made_cut_prob",
        "lr_target_made_cut_prob",
    ],
    "frl": [],
    "matchup": [],
}

MC_PROB_KEYS: dict[MarketType, str] = {
    "win": "sim_win_pct",
    "top5": "sim_top5_pct",
    "top10": "sim_top10_pct",
    "top20": "sim_top20_pct",
}

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "notebooks" / "cache" / "pga_manual_odds.json"


def empty_store(tournament: str = "Masters Tournament") -> Dict[str, Any]:
    """Create an empty manual odds store."""
    return {
        "tournament": tournament,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "entries": [],
    }


def load_store(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the manual odds store from disk, or return empty."""
    path = path or DEFAULT_PATH
    if path.exists():
        return json.loads(path.read_text())
    return empty_store()


def save_store(store: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Save the manual odds store to disk."""
    path = path or DEFAULT_PATH
    store["updatedAt"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2))
    return path


def add_entry(
    store: Dict[str, Any],
    market: MarketType,
    book: str,
    player_odds: Dict[str, int],
    source: str = "",
    captured_at: Optional[str] = None,
    round_num: Optional[int] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Add a market odds entry to the store.

    Args:
        store: the odds store dict
        market: market type (win, top5, top10, top20, made_cut, frl, matchup)
        book: bookmaker name (e.g., "draftkings", "fanduel")
        player_odds: {player_name: american_odds} dict
        source: where the odds came from (e.g., "screenshot", "manual", "api")
        captured_at: ISO timestamp when odds were captured (default: now)
        round_num: which round the odds are for (e.g., 3 = after R2 cut)
        notes: any additional context

    Returns:
        The new entry dict (also appended to store["entries"])
    """
    if market not in VALID_MARKETS:
        raise ValueError(f"Invalid market '{market}'. Valid: {VALID_MARKETS}")

    entry = {
        "id": f"{market}_{book}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "market": market,
        "book": book,
        "capturedAt": captured_at or datetime.now(timezone.utc).isoformat(),
        "source": source,
        "roundNum": round_num,
        "notes": notes,
        "playerOdds": {
            name: {
                "price": price,
                "implied": round(american_to_implied(price), 5),
                "decimal": round(american_to_decimal(price), 3),
            }
            for name, price in player_odds.items()
        },
    }
    store["entries"].append(entry)
    return entry


def add_matchup_entry(
    store: Dict[str, Any],
    book: str,
    player_a: str,
    player_b: str,
    odds_a: int,
    odds_b: int,
    source: str = "",
    captured_at: Optional[str] = None,
    round_num: Optional[int] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Add a head-to-head matchup odds entry."""
    entry = {
        "id": f"matchup_{book}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "market": "matchup",
        "book": book,
        "capturedAt": captured_at or datetime.now(timezone.utc).isoformat(),
        "source": source,
        "roundNum": round_num,
        "notes": notes,
        "matchup": {
            "playerA": player_a,
            "playerB": player_b,
            "oddsA": {"price": odds_a, "implied": round(american_to_implied(odds_a), 5)},
            "oddsB": {"price": odds_b, "implied": round(american_to_implied(odds_b), 5)},
        },
    }
    store["entries"].append(entry)
    return entry


def get_latest_by_market(store: Dict[str, Any]) -> Dict[MarketType, List[Dict[str, Any]]]:
    """
    Group entries by market type. Returns latest entries per market+book combo.
    """
    by_market: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entry in store.get("entries", []):
        mkt = entry["market"]
        book = entry["book"]
        if mkt not in by_market:
            by_market[mkt] = {}
        existing = by_market[mkt].get(book)
        if not existing or entry["capturedAt"] > existing["capturedAt"]:
            by_market[mkt][book] = entry

    result: Dict[str, List[Dict[str, Any]]] = {}
    for mkt, books in by_market.items():
        result[mkt] = list(books.values())
    return result  # type: ignore


def build_market_summary(
    entries: List[Dict[str, Any]],
    market: MarketType,
) -> Dict[str, Any]:
    """
    Build a summary for a single market from multiple book entries.
    Returns: {books, overrounds, playerOdds: [{player, bestPrice, bestBook, consensusImplied, bookOdds}]}
    """
    books = [e["book"] for e in entries]
    book_implied_dicts: List[Dict[str, float]] = []
    book_overrounds: Dict[str, float] = {}

    all_player_book_odds: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for entry in entries:
        bk = entry["book"]
        implied_dict: Dict[str, float] = {}
        for player, info in entry.get("playerOdds", {}).items():
            implied_dict[player] = info["implied"]
            if player not in all_player_book_odds:
                all_player_book_odds[player] = {}
            all_player_book_odds[player][bk] = info

        book_implied_dicts.append(implied_dict)
        book_overrounds[bk] = round(overround(implied_dict), 4)

    consensus = consensus_no_vig(book_implied_dicts) if book_implied_dicts else {}

    player_summaries = []
    for player in sorted(consensus, key=lambda p: consensus.get(p, 0), reverse=True):
        player_books = all_player_book_odds.get(player, {})
        bp, bb = best_available({b: info["price"] for b, info in player_books.items()})
        player_summaries.append({
            "player": player,
            "bestPrice": bp,
            "bestBook": bb,
            "consensusImplied": round(consensus.get(player, 0), 5),
            "bookOdds": player_books,
        })

    latest_capture = max((e["capturedAt"] for e in entries), default="")

    return {
        "market": market,
        "books": books,
        "overrounds": book_overrounds,
        "capturedAt": latest_capture,
        "playerOdds": player_summaries,
    }


def get_model_prob(
    pred: Dict[str, Any],
    market: MarketType,
) -> Optional[float]:
    """Extract best model probability for a given market from a prediction record."""
    for key in MODEL_PROB_KEYS.get(market, []):
        val = pred.get(key)
        if val is not None and val == val:
            return float(val)

    mc_key = MC_PROB_KEYS.get(market)
    if mc_key:
        val = pred.get(mc_key)
        if val is not None and val == val:
            return float(val) / 100.0

    return None


def compute_edges_for_market(
    predictions: List[Dict[str, Any]],
    market_summary: Dict[str, Any],
    market: MarketType,
) -> List[Dict[str, Any]]:
    """
    Compute edges for all players in a market.
    Returns list of edge entries sorted by edge descending.
    """
    odds_by_player = {po["player"]: po for po in market_summary.get("playerOdds", [])}
    pred_by_name = {p["player"]: p for p in predictions}

    edges = []
    for player, po in odds_by_player.items():
        pred = pred_by_name.get(player)
        if not pred:
            continue

        model_prob = get_model_prob(pred, market)
        if model_prob is None:
            continue

        consensus = po["consensusImplied"]
        if consensus <= 0:
            continue

        bp = po["bestPrice"]
        bb = po["bestBook"]
        e = edge(model_prob, consensus)
        e_val = ev(model_prob, bp)
        kf = kelly_fraction(model_prob, bp)

        edges.append({
            "player": player,
            "market": market,
            "modelProb": round(model_prob, 5),
            "marketImplied": round(consensus, 5),
            "edge": round(e, 5),
            "ev": round(e_val, 5),
            "kellyFraction": round(kf, 5),
            "bestPrice": bp,
            "bestBook": bb,
            "signal": edge_signal(e),
        })

    edges.sort(key=lambda x: x["edge"], reverse=True)
    return edges
