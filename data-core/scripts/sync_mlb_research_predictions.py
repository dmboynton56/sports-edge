#!/usr/bin/env python3
"""
Sync MLB research market predictions to Supabase.

This script scores moneyline v3, run-line v1, and totals v1 for a given date
and writes them to the mlb_research_predictions table.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import psycopg
from psycopg import sql
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pipeline.mlb_research_markets import (
    load_mlb_moneyline_v3,
    load_mlb_runline_v1,
    load_mlb_totals_v1,
    score_research_markets_for_date,
)
from src.models.mlb_runline_model import cover_probability_from_residuals
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


def _fetch_latest_odds(conn) -> dict[tuple[int, str], dict[str, dict]]:
    """Fetch paired, fresh odds keyed by ``(game_pk, research_market)``."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH latest_odds AS (
                SELECT DISTINCT ON (metadata->>'game_pk', market, selection)
                    (metadata->>'game_pk')::bigint AS game_pk,
                    book,
                    market,
                    selection,
                    line,
                    price,
                    metadata,
                    snapshot_ts
                FROM odds_snapshots
                WHERE metadata ? 'game_pk'
                  AND market IN ('moneyline', 'spread', 'total')
                  AND selection IS NOT NULL
                  AND snapshot_ts >= NOW() - INTERVAL '24 hours'
                ORDER BY metadata->>'game_pk', market, selection, snapshot_ts DESC
            )
            SELECT game_pk, market, selection, book, line, price, metadata, snapshot_ts
            FROM latest_odds
            """,
            prepare=False,
        )
        rows = cur.fetchall()

    odds_map: dict[tuple[int, str], dict[str, dict]] = {}
    for game_pk, market, selection, book, line, price, metadata, snapshot_ts in rows:
        research_market = "run_line" if market == "spread" else market
        odds_map.setdefault((int(game_pk), research_market), {})[str(selection)] = {
            "book": book,
            "line": float(line) if line is not None else None,
            "price": float(price),
            "metadata": metadata or {},
            "snapshot_ts": snapshot_ts,
        }
    return odds_map


def _implied_probability(price: float) -> float:
    return 100.0 / (price + 100.0) if price > 0 else abs(price) / (abs(price) + 100.0)


def _expected_value(probability: float, price: float) -> float:
    profit = price / 100.0 if price > 0 else 100.0 / abs(price)
    return probability * profit - (1.0 - probability)


def _quarter_kelly(probability: float, price: float) -> float:
    profit = price / 100.0 if price > 0 else 100.0 / abs(price)
    return max(0.0, (profit * probability - (1.0 - probability)) / profit / 4.0)


def _paired_market_fields(
    odds: dict[str, dict] | None,
    probabilities: dict[str, float],
) -> dict:
    if not odds or set(probabilities) - set(odds):
        return {
            "odds_status": "missing_odds", "odds_snapshot_ts": None,
            "best_book": None, "implied_probability": None,
            "no_vig_probability": None, "edge": None, "ev": None,
            "kelly": None, "recommended_side": None,
            "recommended_probability": None,
        }
    raw = {side: _implied_probability(float(odds[side]["price"])) for side in probabilities}
    denominator = sum(raw.values())
    no_vig = {side: value / denominator for side, value in raw.items()}
    candidates = {
        side: {
            "implied_probability": raw[side],
            "no_vig_probability": no_vig[side],
            "edge": probabilities[side] - no_vig[side],
            "ev": _expected_value(probabilities[side], float(odds[side]["price"])),
            "kelly": _quarter_kelly(probabilities[side], float(odds[side]["price"])),
        }
        for side in probabilities
    }
    side = max(candidates, key=lambda value: candidates[value]["ev"])
    snapshots = [value["snapshot_ts"] for value in odds.values()]
    return {
        "odds_status": "ok",
        "odds_snapshot_ts": max(snapshots).isoformat(),
        "best_book": odds[side]["metadata"].get("book_title") or odds[side]["book"],
        **candidates[side],
        "recommended_side": side,
        "recommended_probability": probabilities[side],
    }


def _build_moneyline_rows(df: pd.DataFrame, as_of_ts: datetime, odds_map: dict) -> list[dict]:
    """Transform moneyline DataFrame into Supabase row format."""
    rows = []
    for _, row in df.iterrows():
        game_pk = int(row["game_pk"])
        odds = odds_map.get((game_pk, "moneyline"))
        market = _paired_market_fields(
            odds,
            {"home": float(row["home_win_prob"]), "away": float(row["away_win_prob"])},
        )
        
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
                **market,
                "home_price": odds["home"]["price"] if odds and "home" in odds else None,
                "away_price": odds["away"]["price"] if odds and "away" in odds else None,
            }
        )
    return rows


def _build_runline_rows(
    df: pd.DataFrame,
    as_of_ts: datetime,
    odds_map: dict,
    *,
    margin_residuals,
) -> list[dict]:
    """Transform run-line DataFrame into Supabase row format."""
    rows = []
    for _, row in df.iterrows():
        game_pk = int(row["game_pk"])
        odds = odds_map.get((game_pk, "run_line"))
        home_line = odds.get("home", {}).get("line") if odds else None
        if home_line is not None:
            home_probability = float(
                cover_probability_from_residuals(
                    np.asarray([float(row["predicted_margin"])]),
                    np.asarray(margin_residuals, dtype=float),
                    threshold=-float(home_line),
                )[0]
            )
            market = _paired_market_fields(
                odds, {"home": home_probability, "away": 1.0 - home_probability}
            )
        else:
            market = _paired_market_fields(
                None,
                {"home": float(row["p_home_cover_15"]), "away": float(row["p_away_cover_plus_15"])},
            )
        
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
                "predicted_margin": float(row["predicted_margin"]),
                **market,
                "home_runline_price": odds["home"]["price"] if odds and "home" in odds else None,
                "away_runline_price": odds["away"]["price"] if odds and "away" in odds else None,
                "home_runline_line": home_line,
            }
        )
    return rows


def _build_totals_rows(
    df: pd.DataFrame,
    as_of_ts: datetime,
    odds_map: dict,
    *,
    residual_sigma: float,
) -> list[dict]:
    """Transform totals DataFrame into Supabase row format."""
    rows = []
    for _, row in df.iterrows():
        game_pk = int(row["game_pk"])
        odds = odds_map.get((game_pk, "total"))
        total_line = odds.get("over", {}).get("line") if odds else None
        if total_line is not None:
            z = (float(total_line) - float(row["predicted_total"])) / (residual_sigma * math.sqrt(2.0))
            over_probability = 0.5 * math.erfc(z)
            market = _paired_market_fields(
                odds, {"over": over_probability, "under": 1.0 - over_probability}
            )
        else:
            market = _paired_market_fields(None, {"over": 0.5, "under": 0.5})
        
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
                **market,
                "total_line": total_line,
                "over_price": odds["over"]["price"] if odds and "over" in odds else None,
                "under_price": odds["under"]["price"] if odds and "under" in odds else None,
            }
        )
    return rows


def _upsert_rows(conn, rows: list[dict], table: str = "mlb_research_predictions") -> None:
    """Upsert heterogeneous market rows directly through Postgres."""
    if not rows:
        return
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(sorted(row)), []).append(row)
    with conn.cursor() as cur:
        for columns, group in grouped.items():
            identifiers = sql.SQL(", ").join(map(sql.Identifier, columns))
            placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
            updates = sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(column), sql.Identifier(column))
                for column in columns
                if column != "prediction_id"
            )
            query = sql.SQL(
                "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT (prediction_id) DO UPDATE SET {}"
            ).format(sql.Identifier(table), identifiers, placeholders, updates)
            cur.executemany(query, [tuple(row[column] for column in columns) for row in group])
    conn.commit()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
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
        moneyline_rows = _build_moneyline_rows(result.moneyline, as_of_ts, odds_map)
        runline_rows = _build_runline_rows(
            result.run_line,
            as_of_ts,
            odds_map,
            margin_residuals=runline_artifact["margin_residuals"],
        )
        totals_rows = _build_totals_rows(
            result.totals,
            as_of_ts,
            odds_map,
            residual_sigma=float(totals_artifact["probability_method"]["validation_residual_rmse_sigma"]),
        )
        all_rows = moneyline_rows + runline_rows + totals_rows
        ml_with_odds = sum(1 for row in moneyline_rows if row["odds_status"] == "ok")
        rl_with_odds = sum(1 for row in runline_rows if row["odds_status"] == "ok")
        tot_with_odds = sum(1 for row in totals_rows if row["odds_status"] == "ok")
        print(f"\nPrepared {len(all_rows)} research prediction rows:")
        print(f"  Moneyline: {len(moneyline_rows)} ({ml_with_odds} with odds)")
        print(f"  Run-line: {len(runline_rows)} ({rl_with_odds} with odds)")
        print(f"  Totals: {len(totals_rows)} ({tot_with_odds} with odds)")
        if args.dry_run:
            print("Dry-run mode: skipping Supabase write.")
            return
        _upsert_rows(pg_conn, all_rows)
        print(f"Upserted {len(all_rows)} rows to mlb_research_predictions.")
    finally:
        pg_conn.close()


if __name__ == "__main__":
    main()
