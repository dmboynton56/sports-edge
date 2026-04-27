#!/usr/bin/env python3
"""
Bundle Masters predictions + ESPN 2026 supplement + market odds into
web/public/data/pga_masters_dashboard.json.

  cd data-core && .venv/bin/python scripts/export_pga_dashboard.py
  .venv/bin/python scripts/export_pga_dashboard.py --skip-odds
  .venv/bin/python scripts/export_pga_dashboard.py --odds-cache notebooks/cache/pga_odds_masters_20260409.json
"""
from __future__ import annotations

import glob as globmod
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
repo_root = Path(project_root).parent

import requests

from src.data.odds_math import edge, ev, kelly_fraction, edge_signal
from src.data.pga_odds_fetcher import fetch_and_summarize, match_player_name, build_name_index
from src.data.manual_odds import (
    load_store as load_manual_store,
    get_latest_by_market,
    build_market_summary,
    compute_edges_for_market,
    get_model_prob,
    VALID_MARKETS,
)

LOG = logging.getLogger("export_pga_dashboard")

ARCHIVE = Path(project_root) / "src" / "data" / "archive"
MAIN_TSV = ARCHIVE / "pga_results_2001-2025.tsv"
SUPP = ARCHIVE / "pga_results_espn_supplement.tsv"
PRED = Path(project_root) / "notebooks" / "cache" / "masters_2026_predictions.csv"
PRED_META = PRED.with_suffix(".meta.json")
ODDS_CACHE_DIR = Path(project_root) / "notebooks" / "cache"
MANUAL_ODDS_PATH = ODDS_CACHE_DIR / "pga_manual_odds.json"
OUT = repo_root / "web" / "public" / "data" / "pga_masters_dashboard.json"


def load_predictions() -> tuple[list[dict], dict]:
    meta = {}
    if PRED_META.exists():
        meta = json.loads(PRED_META.read_text())
    if not PRED.exists():
        return [], meta
    df = pd.read_csv(PRED)
    return df.to_dict(orient="records"), meta


def load_merged_results(main_path: Path, supp_path: Path) -> pd.DataFrame:
    """Same merge/dedupe as feature builder: main archive + ESPN supplement."""
    base_cols = [
        "season",
        "start",
        "tournament",
        "name",
        "position",
        "score",
        "round1",
        "round2",
        "round3",
        "round4",
        "total",
    ]
    frames: list[pd.DataFrame] = []
    for path in (main_path, supp_path):
        if not path.exists():
            continue
        raw = pd.read_csv(path, sep="\t")
        take = [c for c in base_cols if c in raw.columns]
        frames.append(raw[take])
    if not frames:
        return pd.DataFrame(columns=base_cols)
    comb = pd.concat(frames, ignore_index=True)
    comb["start"] = pd.to_datetime(comb["start"], errors="coerce")
    comb = comb.drop_duplicates(subset=["season", "start", "tournament", "name"], keep="last")
    return comb


def form_summaries(
    df: pd.DataFrame, year_filter: int | None, max_recent_per_player: int
) -> tuple[list[dict], dict]:
    """Per-tournament rollup (optional year) + recent starts per player (merged history)."""
    if df.empty:
        return [], {}
    df = df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    ev_df = df[df["start"].dt.year == year_filter] if year_filter is not None else df
    events = []
    for (tourn, start), g in ev_df.groupby(["tournament", "start"], sort=False):
        events.append(
            {
                "tournament": str(tourn),
                "start": str(start.date()) if pd.notna(start) else "",
                "players": int(len(g)),
            }
        )
    events.sort(key=lambda x: x["start"])

    by_player: dict[str, list] = defaultdict(list)
    for _, r in df.sort_values("start", ascending=False).iterrows():
        by_player[str(r["name"])].append(
            {
                "tournament": str(r["tournament"]),
                "start": str(r["start"])[:10] if pd.notna(r["start"]) else "",
                "position": str(r["position"]),
                "scoreToPar": str(r["score"]) if pd.notna(r.get("score")) else "",
                "r1": r["round1"] if pd.notna(r.get("round1")) else None,
                "r2": r["round2"] if pd.notna(r.get("round2")) else None,
                "r3": r["round3"] if pd.notna(r.get("round3")) else None,
                "r4": r["round4"] if pd.notna(r.get("round4")) else None,
                "total": int(r["total"]) if pd.notna(r.get("total")) and str(r["total"]) else None,
            }
        )

    recent_by_player = {k: v[:max_recent_per_player] for k, v in by_player.items()}
    return events, recent_by_player


def _rel_to_repo(p: str) -> str:
    try:
        return str(Path(p).relative_to(repo_root))
    except ValueError:
        return p


def _find_latest_odds_cache(tournament: str = "masters") -> Optional[Path]:
    """Find the most recently cached odds JSON for the given tournament."""
    pattern = str(ODDS_CACHE_DIR / f"pga_odds_{tournament}_*.json")
    files = sorted(globmod.glob(pattern), reverse=True)
    return Path(files[0]) if files else None


def _load_odds(
    cache_path: Optional[Path],
    prediction_names: List[str],
    skip_odds: bool,
    live_fetch: bool,
) -> Optional[Dict[str, Any]]:
    """
    Load odds data from cache or live API.
    Returns the summary dict or None if unavailable.
    """
    if skip_odds:
        return None

    if cache_path and cache_path.exists():
        LOG.info("Loading cached odds from %s", cache_path)
        data = json.loads(cache_path.read_text())
        if data.get("playerOdds"):
            return data
        LOG.warning("Cached odds file has no playerOdds, will try live fetch")

    if live_fetch:
        try:
            LOG.info("Fetching live odds from The Odds API...")
            return fetch_and_summarize("masters", prediction_names=prediction_names)
        except Exception as exc:
            LOG.warning("Could not fetch live odds: %s", exc)

    auto = _find_latest_odds_cache()
    if auto and (not cache_path or auto != cache_path):
        LOG.info("Falling back to latest cached odds: %s", auto)
        return json.loads(auto.read_text())

    return None


def _build_odds_index(odds_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build player -> odds info lookup from the odds summary."""
    idx: Dict[str, Dict[str, Any]] = {}
    for po in odds_data.get("playerOdds", []):
        idx[po["player"]] = po
    return idx


def _get_model_win_prob(pred: Dict[str, Any]) -> Optional[float]:
    """
    Extract the best available model win probability from a prediction record.
    Tries calibrated first, then LR, then MC simulation.
    """
    for key in ("best_calibrated_target_win_prob", "lr_target_win_prob", "meta_ensemble_target_win_prob"):
        val = pred.get(key)
        if val is not None and val == val:  # not NaN
            return float(val)
    sim = pred.get("sim_win_pct")
    if sim is not None and sim == sim:
        return float(sim) / 100.0
    return None


def _enrich_predictions_with_odds(
    preds: List[Dict[str, Any]],
    odds_data: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Add market odds fields to each prediction dict.
    Also build the edges list (all players, sorted by edge desc).
    """
    odds_idx = _build_odds_index(odds_data)
    name_index = build_name_index(list(odds_idx.keys()))
    edges: List[Dict[str, Any]] = []

    for pred in preds:
        player = pred.get("player", "")
        matched = match_player_name(player, name_index)
        po = odds_idx.get(matched) if matched else None

        if not po:
            pred["market_implied_win"] = None
            pred["best_price_win"] = None
            pred["best_book_win"] = None
            pred["edge_win"] = None
            pred["ev_win"] = None
            continue

        consensus = po["consensusImplied"]
        bp = po["bestPrice"]
        bb = po["bestBook"]

        pred["market_implied_win"] = round(consensus, 5)
        pred["best_price_win"] = bp
        pred["best_book_win"] = bb
        pred["book_odds_win"] = po.get("bookOdds", {})

        model_prob = _get_model_win_prob(pred)
        if model_prob is not None and consensus > 0:
            e = edge(model_prob, consensus)
            e_val = ev(model_prob, bp)
            kf = kelly_fraction(model_prob, bp)
            pred["edge_win"] = round(e, 5)
            pred["ev_win"] = round(e_val, 5)
            pred["kelly_win"] = round(kf, 5)

            edges.append({
                "player": player,
                "market": "win",
                "modelProb": round(model_prob, 5),
                "marketImplied": round(consensus, 5),
                "edge": round(e, 5),
                "ev": round(e_val, 5),
                "kellyFraction": round(kf, 5),
                "bestPrice": bp,
                "bestBook": bb,
                "signal": edge_signal(e),
            })
        else:
            pred["edge_win"] = None
            pred["ev_win"] = None
            pred["kelly_win"] = None

    edges.sort(key=lambda x: x["edge"], reverse=True)
    return preds, edges


ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SportsEdge/1.0)"}

PAR = 72  # Augusta National


def _score_to_par_str(score_raw: str) -> str:
    """Normalise ESPN 'score' field to a consistent to-par string."""
    s = str(score_raw).strip()
    if s.upper() in ("E", "0"):
        return "E"
    return s


def fetch_live_leaderboard() -> Optional[Dict[str, Any]]:
    """Fetch the current ESPN PGA scoreboard and return a leaderboard dict if an event is live/in-progress."""
    try:
        r = requests.get(ESPN_SCOREBOARD, headers=ESPN_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        LOG.warning("ESPN scoreboard fetch failed: %s", exc)
        return None

    events = data.get("events", [])
    if not events:
        return None

    ev = events[0]
    comp = (ev.get("competitions") or [{}])[0]
    status_type = (comp.get("status") or {}).get("type", {})
    status_desc = status_type.get("description", "")
    is_completed = status_type.get("completed", False)
    competitors = comp.get("competitors", [])
    if not competitors:
        return None

    ev_name = ev.get("name", "")
    ev_date = ev.get("date", "")
    current_round = (comp.get("status") or {}).get("period", 1)

    rows = []
    for c in competitors:
        athlete = c.get("athlete") or {}
        name = athlete.get("displayName") or athlete.get("fullName", "?")
        score_raw = c.get("score", "")
        linescores = c.get("linescores") or []
        thru = (c.get("status") or {}).get("displayThru") or (c.get("status") or {}).get("thru") or ""
        player_status = ((c.get("status") or {}).get("type") or {}).get("description", "")

        round_scores = {}
        for ls in linescores:
            period = ls.get("period")
            val = ls.get("value")
            if period and val is not None:
                round_scores[int(period)] = int(val)

        total_strokes = sum(round_scores.values()) if round_scores else None
        to_par = _score_to_par_str(score_raw)

        rows.append({
            "player": name,
            "toPar": to_par,
            "thru": str(thru),
            "totalStrokes": total_strokes,
            "rounds": round_scores,
            "status": player_status,
        })

    rows.sort(key=lambda x: (
        _sort_key_to_par(x["toPar"]),
        x["totalStrokes"] or 999,
        x["player"],
    ))

    for i, row in enumerate(rows):
        row["position"] = i + 1

    # Assign tied positions
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1]["toPar"] == rows[i]["toPar"]:
            j += 1
        rank = i + 1
        tied = j > i
        for k in range(i, j + 1):
            rows[k]["positionDisplay"] = f"T{rank}" if tied else str(rank)
        i = j + 1

    return {
        "event": ev_name,
        "eventDate": ev_date,
        "currentRound": current_round,
        "status": status_desc,
        "isCompleted": is_completed,
        "fetchedAt": datetime.now(timezone.utc).isoformat(),
        "players": rows,
    }


def _sort_key_to_par(to_par: str) -> float:
    s = to_par.strip().upper()
    if s == "E":
        return 0.0
    if s in ("WD", "DQ", "MDF", "DNS", "CUT", ""):
        return 999.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return 999.0


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    parser = argparse.ArgumentParser(description="Export PGA dashboard JSON with predictions + odds.")
    parser.add_argument("--skip-odds", action="store_true", help="Skip odds integration entirely")
    parser.add_argument("--odds-cache", type=Path, default=None, help="Path to cached odds JSON file")
    parser.add_argument("--live-odds", action="store_true", help="Fetch live odds from API (uses 1 credit)")
    args = parser.parse_args()

    preds, pred_meta = load_predictions()
    if isinstance(pred_meta.get("feature_store"), str):
        pred_meta = {**pred_meta, "feature_store": _rel_to_repo(pred_meta["feature_store"])}
    supplement_meta: dict = {"path": _rel_to_repo(str(SUPP)), "rows": 0}
    merged = load_merged_results(MAIN_TSV, SUPP)
    merged_meta = {
        "mainPath": _rel_to_repo(str(MAIN_TSV)),
        "supplementPath": _rel_to_repo(str(SUPP)),
        "mergedRows": int(len(merged)),
    }
    events: list = []
    recent_by_player: dict = {}

    if SUPP.exists():
        sdf = pd.read_csv(SUPP, sep="\t")
        supplement_meta["rows"] = len(sdf)
        supplement_meta["seasons"] = sorted(sdf["season"].dropna().unique().tolist()) if "season" in sdf.columns else []

    if not merged.empty:
        events, recent_by_player = form_summaries(merged, year_filter=2026, max_recent_per_player=12)

    prediction_names = [p["player"] for p in preds if "player" in p]
    odds_data = _load_odds(
        cache_path=args.odds_cache,
        prediction_names=prediction_names,
        skip_odds=args.skip_odds,
        live_fetch=args.live_odds,
    )

    market_odds_payload: Optional[Dict[str, Any]] = None
    edges_payload: List[Dict[str, Any]] = []

    if odds_data and odds_data.get("playerOdds"):
        preds, edges_payload = _enrich_predictions_with_odds(preds, odds_data)
        market_odds_payload = {
            "tournament": odds_data.get("tournament", ""),
            "fetchedAt": odds_data.get("fetchedAt", ""),
            "commenceTime": odds_data.get("commenceTime", ""),
            "books": odds_data.get("books", []),
            "overrounds": odds_data.get("overrounds", {}),
            "playerOdds": odds_data.get("playerOdds", []),
        }
        n_pos = sum(1 for e in edges_payload if e["signal"] == "positive")
        LOG.info("API odds merged: %d players, %d positive edges", len(market_odds_payload["playerOdds"]), n_pos)
    else:
        LOG.info("No API odds data available")

    # If mid-tournament predictions exist, API odds are stale (pre-tournament);
    # drop those edges so only fresh manual odds edges are used.
    midtourn_csv_check = ODDS_CACHE_DIR / "masters_2026_midtournament.csv"
    if midtourn_csv_check.exists() and edges_payload:
        LOG.info("Mid-tournament data found — discarding %d stale pre-tournament API edges", len(edges_payload))
        edges_payload = []

    # --- Live ESPN leaderboard ---
    live_leaderboard: Optional[Dict[str, Any]] = None
    if not args.skip_odds:
        live_leaderboard = fetch_live_leaderboard()
        if live_leaderboard:
            LOG.info(
                "Live leaderboard: %s — %s — %d players (round %d)",
                live_leaderboard["event"],
                live_leaderboard["status"],
                len(live_leaderboard["players"]),
                live_leaderboard["currentRound"],
            )

    # --- Mid-tournament predictions (if available) ---
    midtournament_payload: Optional[Dict[str, Any]] = None
    mt_preds_for_edges: Optional[List[Dict[str, Any]]] = None
    midtourn_csv = ODDS_CACHE_DIR / "masters_2026_midtournament.csv"
    midtourn_meta = midtourn_csv.with_suffix(".meta.json")
    if midtourn_csv.exists():
        mt_df = pd.read_csv(midtourn_csv)
        mt_meta = json.loads(midtourn_meta.read_text()) if midtourn_meta.exists() else {}
        mt_records = mt_df.to_dict(orient="records")
        midtournament_payload = {
            "meta": mt_meta,
            "predictions": mt_records,
        }
        mt_preds_for_edges = mt_records
        LOG.info(
            "Mid-tournament predictions loaded: %d players (round %s)",
            len(mt_df),
            mt_meta.get("rounds_completed", "?"),
        )

    # --- Manual odds (multi-market: top5, top10, top20, made_cut, matchups) ---
    # When mid-tournament predictions exist, use them for edge computation
    # since post-cut odds reflect remaining rounds, not the full tournament.
    placement_markets: Dict[str, Any] = {}
    all_market_edges: List[Dict[str, Any]] = list(edges_payload)

    if not args.skip_odds and MANUAL_ODDS_PATH.exists():
        manual_store = load_manual_store(MANUAL_ODDS_PATH)
        by_market = get_latest_by_market(manual_store)
        LOG.info("Manual odds store: %d markets (%s)", len(by_market), ", ".join(by_market.keys()))

        edge_preds = mt_preds_for_edges if mt_preds_for_edges else preds
        if mt_preds_for_edges:
            LOG.info("Using mid-tournament predictions for edge computation")

        for mkt, entries in by_market.items():
            if mkt == "matchup":
                placement_markets["matchups"] = entries
                continue
            summary = build_market_summary(entries, mkt)  # type: ignore
            placement_markets[mkt] = summary
            mkt_edges = compute_edges_for_market(edge_preds, summary, mkt)  # type: ignore
            all_market_edges.extend(mkt_edges)

            n_pos = sum(1 for e in mkt_edges if e["signal"] == "positive")
            LOG.info("  %s: %d players, %d positive edges", mkt, len(summary["playerOdds"]), n_pos)

            for pred in preds:
                player = pred.get("player", "")
                po = next((p for p in summary["playerOdds"] if p["player"] == player), None)
                if po:
                    pred[f"market_implied_{mkt}"] = po["consensusImplied"]
                    pred[f"best_price_{mkt}"] = po["bestPrice"]
                    pred[f"best_book_{mkt}"] = po["bestBook"]
                    pred[f"book_odds_{mkt}"] = po.get("bookOdds", {})
                    me = next((e for e in mkt_edges if e["player"] == player), None)
                    if me:
                        pred[f"edge_{mkt}"] = me["edge"]
                        pred[f"ev_{mkt}"] = me["ev"]
                        pred[f"kelly_{mkt}"] = me["kellyFraction"]
    elif not args.skip_odds:
        LOG.info("No manual odds file at %s", MANUAL_ODDS_PATH)

    all_market_edges.sort(key=lambda x: x["edge"], reverse=True)

    payload: Dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "predictions": preds,
        "predictionMeta": pred_meta,
        "espnSupplement": supplement_meta,
        "mergedResults": merged_meta,
        "tournaments2026": events,
        "recentByPlayer": recent_by_player,
    }
    if market_odds_payload:
        payload["marketOdds"] = market_odds_payload
    if all_market_edges:
        payload["edges"] = all_market_edges
    if placement_markets:
        payload["placementMarkets"] = placement_markets
    if live_leaderboard:
        payload["liveLeaderboard"] = live_leaderboard
    if midtournament_payload:
        payload["midtournament"] = midtournament_payload

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    parts = [
        f"{len(preds)} predictions",
        f"{len(events)} 2026 events",
        f"{len(recent_by_player)} players w/ recent history",
    ]
    if all_market_edges:
        n_total_pos = sum(1 for e in all_market_edges if e["signal"] == "positive")
        markets_with_edges = set(e["market"] for e in all_market_edges)
        parts.append(f"{len(all_market_edges)} edges across {len(markets_with_edges)} markets ({n_total_pos} positive)")
    if live_leaderboard:
        parts.append(f"live leaderboard: {len(live_leaderboard['players'])} players (R{live_leaderboard['currentRound']})")
    if midtournament_payload:
        parts.append(f"mid-tournament: {len(midtournament_payload['predictions'])} players after R{midtournament_payload['meta'].get('rounds_completed', '?')}")
    print(f"Wrote {OUT} ({', '.join(parts)})")


if __name__ == "__main__":
    main()
