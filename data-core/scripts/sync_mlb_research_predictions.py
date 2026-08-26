#!/usr/bin/env python3
"""
Sync MLB research market predictions to Supabase.

This script scores moneyline v3, run-line v1, and totals v1 for a given date
and writes them to the mlb_research_predictions table.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from supabase import Client, create_client
import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pipeline.mlb_research_markets import (
    load_mlb_moneyline_v3,
    load_mlb_runline_v1,
    load_mlb_totals_v1,
    score_research_markets_for_date,
)
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONEYLINE_MODEL = ROOT / "models" / "mlb_winner_model_v3.pkl"
DEFAULT_TOTALS_MODEL = ROOT / "models" / "mlb_totals_model_v1.pkl"
DEFAULT_RUNLINE_MODEL = ROOT / "models" / "mlb_runline_model_v1.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync MLB research predictions to Supabase.")
    parser.add_argument("--date", required=True, help="Game date in YYYY-MM-DD.")
    parser.add_argument("--season", type=int, help="MLB season year (defaults to --date year).")
    parser.add_argument("--moneyline-model", default=str(DEFAULT_MONEYLINE_MODEL))
    parser.add_argument("--totals-model", default=str(DEFAULT_TOTALS_MODEL))
    parser.add_argument("--runline-model", default=str(DEFAULT_RUNLINE_MODEL))
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Score but do not write to Supabase.")
    return parser.parse_args()


def _game_id(game_pk: int) -> str:
    return f"MLB_{game_pk}"


def _prediction_id(game_pk: int, market: str, model_version: str) -> str:
    """Stable deterministic prediction ID."""
    payload = f"{game_pk}:{market}:{model_version}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _fetch_latest_odds(conn) -> dict[tuple[int, str], dict]:
    """Fetch latest odds snapshot for each game_pk and market.
    
    Returns a dict keyed by (game_pk, market) with odds data.
    Market names: 'moneyline', 'run_line', 'total'
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH latest_odds AS (
                SELECT DISTINCT ON (game_id, market)
                    game_id,
                    book,
                    market,
                    line,
                    price,
                    metadata,
                    snapshot_ts
                FROM odds_snapshots
                WHERE game_id LIKE 'MLB_%'
                  AND market IN ('moneyline', 'run_line', 'total')
                  AND snapshot_ts >= NOW() - INTERVAL '24 hours'
                ORDER BY game_id, market, snapshot_ts DESC
            )
            SELECT 
                game_id,
                market,
                book,
                line,
                price,
                metadata,
                snapshot_ts
            FROM latest_odds
            """
        )
        rows = cur.fetchall()
    
    odds_map = {}
    for row in rows:
        game_id, market, book, line, price, metadata, snapshot_ts = row
        game_pk = int(game_id.replace("MLB_", ""))
        
        if (game_pk, market) not in odds_map:
            odds_map[(game_pk, market)] = {
                "book": book,
                "line": float(line) if line is not None else None,
                "price": int(price) if price is not None else None,
                "metadata": metadata or {},
                "snapshot_ts": snapshot_ts,
            }
    
    return odds_map


def _build_moneyline_rows(df: pd.DataFrame, as_of_ts: datetime, odds_map: dict) -> list[dict]:
    """Transform moneyline DataFrame into Supabase row format."""
    rows = []
    for _, row in df.iterrows():
        game_pk = int(row["game_pk"])
        odds = odds_map.get((game_pk, "moneyline"))
        
        # Determine odds status
        if odds and odds["price"] is not None:
            odds_status = "ok"
            odds_snapshot_ts = odds["snapshot_ts"].isoformat() if odds["snapshot_ts"] else None
            best_book = odds["book"]
            home_price = float(odds["price"])
            away_price = float(odds["metadata"].get("away_price")) if odds["metadata"].get("away_price") else None
        else:
            odds_status = "missing_odds"
            odds_snapshot_ts = None
            best_book = None
            home_price = None
            away_price = None
        
        rows.append(
            {
                "prediction_id": _prediction_id(game_pk, "moneyline", "v3"),
                "league": "MLB",
                "market": "moneyline",
                "model_version": "v3",
                "model_status": "research",
                "game_id": _game_id(game_pk),
                "game_pk": game_pk,
                "season": pd.to_datetime(row["game_date"]).year,
                "game_date": str(pd.to_datetime(row["game_date"]).date()),
                "game_datetime": str(pd.to_datetime(row["game_datetime"])) if pd.notna(row.get("game_datetime")) else None,
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "venue": None,
                "as_of_ts": as_of_ts.isoformat(),
                "home_win_prob": float(row["home_win_prob"]),
                "away_win_prob": float(row["away_win_prob"]),
                "odds_status": odds_status,
                "odds_snapshot_ts": odds_snapshot_ts,
                "best_book": best_book,
                "home_price": home_price,
                "away_price": away_price,
            }
        )
    return rows


def _build_runline_rows(df: pd.DataFrame, as_of_ts: datetime, odds_map: dict) -> list[dict]:
    """Transform run-line DataFrame into Supabase row format."""
    rows = []
    for _, row in df.iterrows():
        game_pk = int(row["game_pk"])
        odds = odds_map.get((game_pk, "run_line"))
        
        # Determine odds status
        if odds and odds["price"] is not None and odds["line"] is not None:
            odds_status = "ok"
            odds_snapshot_ts = odds["snapshot_ts"].isoformat() if odds["snapshot_ts"] else None
            best_book = odds["book"]
            # Assume home is -1.5; price is for home
            home_runline_price = float(odds["price"])
            away_runline_price = float(odds["metadata"].get("away_price")) if odds["metadata"].get("away_price") else None
        else:
            odds_status = "missing_odds"
            odds_snapshot_ts = None
            best_book = None
            home_runline_price = None
            away_runline_price = None
        
        rows.append(
            {
                "prediction_id": _prediction_id(game_pk, "run_line", "v1"),
                "league": "MLB",
                "market": "run_line",
                "model_version": "v1",
                "model_status": "research",
                "game_id": _game_id(game_pk),
                "game_pk": game_pk,
                "season": pd.to_datetime(row["game_date"]).year,
                "game_date": str(pd.to_datetime(row["game_date"]).date()),
                "game_datetime": str(pd.to_datetime(row["game_datetime"])) if pd.notna(row.get("game_datetime")) else None,
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "venue": None,
                "as_of_ts": as_of_ts.isoformat(),
                "p_home_cover_15": float(row["p_home_cover_15"]),
                "p_away_cover_plus_15": float(row["p_away_cover_plus_15"]),
                "odds_status": odds_status,
                "odds_snapshot_ts": odds_snapshot_ts,
                "best_book": best_book,
                "home_runline_price": home_runline_price,
                "away_runline_price": away_runline_price,
            }
        )
    return rows


def _build_totals_rows(df: pd.DataFrame, as_of_ts: datetime, odds_map: dict) -> list[dict]:
    """Transform totals DataFrame into Supabase row format."""
    rows = []
    for _, row in df.iterrows():
        game_pk = int(row["game_pk"])
        odds = odds_map.get((game_pk, "total"))
        
        # Determine odds status
        if odds and odds["price"] is not None and odds["line"] is not None:
            odds_status = "ok"
            odds_snapshot_ts = odds["snapshot_ts"].isoformat() if odds["snapshot_ts"] else None
            best_book = odds["book"]
            total_line = float(odds["line"])
            over_price = float(odds["price"])  # price is for over
            under_price = float(odds["metadata"].get("under_price")) if odds["metadata"].get("under_price") else None
        else:
            odds_status = "missing_odds"
            odds_snapshot_ts = None
            best_book = None
            total_line = None
            over_price = None
            under_price = None
        
        rows.append(
            {
                "prediction_id": _prediction_id(game_pk, "total", "v1"),
                "league": "MLB",
                "market": "total",
                "model_version": "v1",
                "model_status": "research",
                "game_id": _game_id(game_pk),
                "game_pk": game_pk,
                "season": pd.to_datetime(row["game_date"]).year,
                "game_date": str(pd.to_datetime(row["game_date"]).date()),
                "game_datetime": str(pd.to_datetime(row["game_datetime"])) if pd.notna(row.get("game_datetime")) else None,
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "venue": None,
                "as_of_ts": as_of_ts.isoformat(),
                "predicted_total": float(row["predicted_total"]),
                "p_over_8_5": float(row["p_over_8_5"]),
                "p_over_9_5": float(row["p_over_9_5"]),
                "odds_status": odds_status,
                "odds_snapshot_ts": odds_snapshot_ts,
                "best_book": best_book,
                "total_line": total_line,
                "over_price": over_price,
                "under_price": under_price,
            }
        )
    return rows


def _upsert_rows(client: Client, rows: list[dict], table: str = "mlb_research_predictions") -> None:
    """Upsert rows to Supabase in batches."""
    if not rows:
        return
    
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        client.table(table).upsert(batch, on_conflict="prediction_id").execute()


def main() -> None:
    args = parse_args()
    game_date = pd.to_datetime(args.date).date()
    season = args.season or game_date.year
    as_of_ts = datetime.now(timezone.utc)

    print(f"Loading model artifacts...")
    moneyline_artifact = load_mlb_moneyline_v3(args.moneyline_model)
    totals_artifact = load_mlb_totals_v1(args.totals_model)
    runline_artifact = load_mlb_runline_v1(args.runline_model)

    print(f"Scoring research markets for {game_date} (season={season})...")
    result = score_research_markets_for_date(
        game_date=game_date,
        season=season,
        moneyline_artifact=moneyline_artifact,
        totals_artifact=totals_artifact,
        runline_artifact=runline_artifact,
        min_prior_games=args.min_prior_games,
    )

    # Fetch latest odds from Supabase
    creds = load_supabase_credentials()
    pg_conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    
    try:
        print("Fetching latest odds from Supabase...")
        odds_map = _fetch_latest_odds(pg_conn)
        print(f"Fetched odds for {len(odds_map)} (game_pk, market) pairs")
    finally:
        pg_conn.close()

    moneyline_rows = _build_moneyline_rows(result.moneyline, as_of_ts, odds_map)
    runline_rows = _build_runline_rows(result.run_line, as_of_ts, odds_map)
    totals_rows = _build_totals_rows(result.totals, as_of_ts, odds_map)

    all_rows = moneyline_rows + runline_rows + totals_rows
    
    # Count rows with odds
    ml_with_odds = sum(1 for r in moneyline_rows if r["odds_status"] == "ok")
    rl_with_odds = sum(1 for r in runline_rows if r["odds_status"] == "ok")
    tot_with_odds = sum(1 for r in totals_rows if r["odds_status"] == "ok")
    
    print(f"\nPrepared {len(all_rows)} research prediction rows:")
    print(f"  Moneyline: {len(moneyline_rows)} ({ml_with_odds} with odds)")
    print(f"  Run-line: {len(runline_rows)} ({rl_with_odds} with odds)")
    print(f"  Totals: {len(totals_rows)} ({tot_with_odds} with odds)")

    if args.dry_run:
        print("Dry-run mode: skipping Supabase write.")
        return

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")

    client = create_client(supabase_url, supabase_key)
    _upsert_rows(client, all_rows)
    print(f"Upserted {len(all_rows)} rows to mlb_research_predictions.")


if __name__ == "__main__":
    main()
