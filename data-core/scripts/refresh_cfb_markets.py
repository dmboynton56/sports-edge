#!/usr/bin/env python3
"""Refresh daily CFB points, moneyline, spread, and total research markets."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from dotenv import load_dotenv
import pandas as pd
from psycopg.types.json import Jsonb
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_cfb_market_model import fetch_games  # noqa: E402
from src.models.cfb_market import (  # noqa: E402
    CfbMarketModel,
    FEATURE_COLUMNS,
    build_feature_frames,
    normal_probability_above,
)
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402


ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
ALIASES = {
    "massachusettsminutemen": "umassminutemen",
    "connecticuthuskies": "uconnhuskies",
    "southernmethodistmustangs": "smumustangs",
    "texaschristianhornedfrogs": "tcuhornedfrogs",
    "centralfloridaknights": "ucfknights",
    "brighamyoungcougars": "byucougars",
    "mississippirebels": "olemissrebels",
        "louisianastatetigers": "lsutigers",
}


def normalize_team(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return ALIASES.get(normalized, normalized)


def american_implied_probability(price: float) -> float:
    return 100.0 / (price + 100.0) if price > 0 else abs(price) / (abs(price) + 100.0)


def expected_value(probability: float, price: float) -> float:
    profit = price / 100.0 if price > 0 else 100.0 / abs(price)
    return probability * profit - (1.0 - probability)


def quarter_kelly(probability: float, price: float) -> float:
    profit = price / 100.0 if price > 0 else 100.0 / abs(price)
    if profit <= 0:
        return 0.0
    return max(0.0, (profit * probability - (1.0 - probability)) / profit / 4.0)


def fetch_odds(api_key: str) -> tuple[list[dict[str, Any]], datetime, int | None]:
    response = requests.get(
        ODDS_URL,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
        timeout=30,
    )
    response.raise_for_status()
    remaining = response.headers.get("x-requests-remaining")
    return response.json(), datetime.now(timezone.utc), int(remaining) if remaining else None


def match_odds_event(game: dict[str, Any], odds_events: list[dict[str, Any]]) -> dict[str, Any] | None:
    game_time = pd.Timestamp(game["game_time_utc"])
    home = normalize_team(game["home_team"])
    away = normalize_team(game["away_team"])
    best: tuple[float, dict[str, Any]] | None = None
    for event in odds_events:
        event_time = pd.Timestamp(event["commence_time"])
        if abs((event_time - game_time).total_seconds()) > 12 * 3600:
            continue
        home_score = SequenceMatcher(None, home, normalize_team(event.get("home_team"))).ratio()
        away_score = SequenceMatcher(None, away, normalize_team(event.get("away_team"))).ratio()
        score = (home_score + away_score) / 2.0
        if best is None or score > best[0]:
            best = (score, event)
    return best[1] if best and best[0] >= 0.72 else None


def parse_odds_rows(
    game: dict[str, Any],
    event: dict[str, Any],
    snapshot_ts: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    home_name = normalize_team(event.get("home_team"))
    away_name = normalize_team(event.get("away_team"))
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            market_key = market.get("key")
            if market_key not in {"h2h", "spreads", "totals"}:
                continue
            for outcome in market.get("outcomes", []):
                outcome_name = str(outcome.get("name") or "")
                normalized_outcome = normalize_team(outcome_name)
                if market_key == "totals":
                    selection = outcome_name.lower()
                    target_market = "total"
                elif normalized_outcome == home_name:
                    selection = "home"
                    target_market = "moneyline" if market_key == "h2h" else "spread"
                elif normalized_outcome == away_name:
                    selection = "away"
                    target_market = "moneyline" if market_key == "h2h" else "spread"
                else:
                    continue
                if selection not in {"home", "away", "over", "under"}:
                    continue
                price = float(outcome["price"])
                rows.append(
                    {
                        "event_id": game["event_id"],
                        "provider_event_id": str(event["id"]),
                        "book": str(book["key"]),
                        "book_title": str(book.get("title") or book["key"]),
                        "market": target_market,
                        "selection": selection,
                        "line": float(outcome["point"]) if outcome.get("point") is not None else None,
                        "price": price,
                        "implied_probability": american_implied_probability(price),
                        "last_update": market.get("last_update") or book.get("last_update"),
                        "snapshot_ts": snapshot_ts,
                        "raw_record": outcome,
                    }
                )
    return rows


def build_recommendations(
    scored: pd.DataFrame,
    odds_rows: list[dict[str, Any]],
    model: CfbMarketModel,
    prediction_ts: datetime,
) -> list[dict[str, Any]]:
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in odds_rows:
        by_game.setdefault(row["event_id"], []).append(row)
    recommendations: list[dict[str, Any]] = []
    for _, prediction in scored.iterrows():
        event_id = str(prediction["event_id"])
        prices = by_game.get(event_id, [])
        for market in ("moneyline", "spread", "total"):
            selections = ("home", "away") if market != "total" else ("over", "under")
            for selection in selections:
                candidates = [
                    row for row in prices
                    if row["market"] == market and row["selection"] == selection
                ]
                if not candidates:
                    continue
                best = max(candidates, key=lambda row: row["price"])
                line = best["line"]
                if market == "moneyline":
                    probability = float(prediction["home_win_probability"])
                    if selection == "away":
                        probability = 1.0 - probability
                elif market == "spread":
                    if line is None:
                        continue
                    mean = float(prediction["predicted_margin"])
                    if selection == "away":
                        mean = -mean
                    probability = normal_probability_above(mean, -float(line), model.margin_sigma)
                else:
                    if line is None:
                        continue
                    over = normal_probability_above(
                        float(prediction["predicted_total"]), float(line), model.total_sigma
                    )
                    probability = over if selection == "over" else 1.0 - over

                flags = ["no_historical_market_backtest"]
                prior_games = min(
                    int(prediction["home_games_before"]), int(prediction["away_games_before"])
                )
                if prior_games < 2:
                    flags.append("limited_current_season_sample")
                if market == "spread" and line is not None and abs(float(line)) >= 35:
                    flags.append("extreme_market_line")
                team = prediction["home_team"] if selection == "home" else prediction["away_team"]
                subject = (
                    f"{team} {float(line):+g}" if market == "spread"
                    else f"{str(selection).title()} {float(line):g}" if market == "total"
                    else str(team)
                )
                price = float(best["price"])
                implied = float(best["implied_probability"])
                edge = probability - implied
                ev = expected_value(probability, price)
                # Outcome validation alone cannot justify unconstrained betting
                # edges. Fail closed on model/market disagreements that sit
                # outside the range this v1 artifact can responsibly publish.
                if abs(edge) > 0.08 or abs(ev) > 0.20:
                    continue
                if market == "moneyline" and price > 400:
                    continue
                recommendations.append(
                    {
                        "event_id": event_id,
                        "model_version": model.model_version,
                        "market": market,
                        "selection": selection,
                        "subject": subject,
                        "book": best["book"],
                        "book_title": best["book_title"],
                        "line": line,
                        "price": price,
                        "model_probability": probability,
                        "implied_probability": implied,
                        "edge": edge,
                        "ev": ev,
                        "quarter_kelly": quarter_kelly(probability, price),
                        "confidence": min(0.82, 0.42 + 0.04 * prior_games),
                        "quality_flags": flags,
                        "prediction_ts": prediction_ts,
                        "odds_snapshot_ts": best["snapshot_ts"],
                    }
                )
    return recommendations


def sync_rows(
    conn,
    games: list[dict[str, Any]],
    scored: pd.DataFrame,
    odds_rows: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    model: CfbMarketModel,
    prediction_ts: datetime,
) -> None:
    with conn.cursor() as cur:
        for game in games:
            game_date = pd.Timestamp(game["game_time_utc"]).tz_convert("America/Denver").date()
            cur.execute(
                """
                INSERT INTO cfb_games (
                  event_id, season, week, game_time_utc, game_date, home_team_id,
                  away_team_id, home_team, away_team, venue, neutral_site, status,
                  home_score, away_score, raw_record, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                ON CONFLICT (event_id) DO UPDATE SET
                  season=EXCLUDED.season, week=EXCLUDED.week,
                  game_time_utc=EXCLUDED.game_time_utc, game_date=EXCLUDED.game_date,
                  home_team_id=EXCLUDED.home_team_id, away_team_id=EXCLUDED.away_team_id,
                  home_team=EXCLUDED.home_team, away_team=EXCLUDED.away_team,
                  venue=EXCLUDED.venue, neutral_site=EXCLUDED.neutral_site,
                  status=EXCLUDED.status, home_score=EXCLUDED.home_score,
                  away_score=EXCLUDED.away_score, raw_record=EXCLUDED.raw_record,
                  updated_at=NOW()
                """,
                (
                    game["event_id"], game["season"], game.get("week"), game["game_time_utc"],
                    game_date, game["home_team_id"], game["away_team_id"], game["home_team"],
                    game["away_team"], game.get("venue"), game.get("neutral_site", False),
                    game.get("status", "unknown"), game.get("home_score"), game.get("away_score"),
                    Jsonb(game.get("raw_record") or {}),
                ),
                prepare=False,
            )
        for _, row in scored.iterrows():
            prior_games = min(int(row["home_games_before"]), int(row["away_games_before"]))
            flags = ["no_historical_market_backtest"]
            if prior_games < 2:
                flags.append("limited_current_season_sample")
            cur.execute(
                """
                INSERT INTO cfb_team_predictions (
                  event_id, model_version, model_status, predicted_home_points,
                  predicted_away_points, predicted_margin, predicted_total,
                  home_win_probability, margin_sigma, total_sigma, confidence,
                  quality_flags, feature_snapshot, prediction_ts
                ) VALUES (%s,%s,'research',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (event_id, model_version) DO UPDATE SET
                  predicted_home_points=EXCLUDED.predicted_home_points,
                  predicted_away_points=EXCLUDED.predicted_away_points,
                  predicted_margin=EXCLUDED.predicted_margin,
                  predicted_total=EXCLUDED.predicted_total,
                  home_win_probability=EXCLUDED.home_win_probability,
                  margin_sigma=EXCLUDED.margin_sigma, total_sigma=EXCLUDED.total_sigma,
                  confidence=EXCLUDED.confidence, quality_flags=EXCLUDED.quality_flags,
                  feature_snapshot=EXCLUDED.feature_snapshot,
                  prediction_ts=EXCLUDED.prediction_ts
                """,
                (
                    row["event_id"], model.model_version, float(row["predicted_home_points"]),
                    float(row["predicted_away_points"]), float(row["predicted_margin"]),
                    float(row["predicted_total"]), float(row["home_win_probability"]),
                    model.margin_sigma, model.total_sigma, min(0.82, 0.42 + 0.04 * prior_games),
                    Jsonb(flags), Jsonb({name: float(row[name]) for name in FEATURE_COLUMNS}),
                    prediction_ts,
                ),
                prepare=False,
            )
        cur.executemany(
            """
                INSERT INTO cfb_odds_snapshots (
                  event_id, provider_event_id, book, book_title, market, selection,
                  line, price, implied_probability, last_update, snapshot_ts, raw_record
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            [
                (
                    row["event_id"], row["provider_event_id"], row["book"], row["book_title"],
                    row["market"], row["selection"], row["line"], row["price"],
                    row["implied_probability"], row["last_update"], row["snapshot_ts"],
                    Jsonb(row["raw_record"]),
                )
                for row in odds_rows
            ],
        )
        event_ids = [str(value) for value in scored["event_id"].tolist()]
        cur.execute(
            "DELETE FROM cfb_market_recommendations WHERE model_version=%s AND event_id=ANY(%s::text[])",
            (model.model_version, event_ids),
            prepare=False,
        )
        cur.executemany(
            """
                INSERT INTO cfb_market_recommendations (
                  event_id, model_version, market, selection, subject, book,
                  book_title, line, price, model_probability, implied_probability,
                  edge, ev, quarter_kelly, confidence, quality_flags,
                  prediction_ts, odds_snapshot_ts, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
            """,
            [
                (
                    row["event_id"], row["model_version"], row["market"], row["selection"],
                    row["subject"], row["book"], row["book_title"], row["line"], row["price"],
                    row["model_probability"], row["implied_probability"], row["edge"], row["ev"],
                    row["quarter_kelly"], row["confidence"], Jsonb(row["quality_flags"]),
                    row["prediction_ts"], row["odds_snapshot_ts"],
                )
                for row in recommendations
            ],
        )
    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh college-football team markets.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--lookahead-days", type=int, default=3)
    parser.add_argument("--history-seasons", type=int, default=2)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "cfb_team_v1.json")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    model = CfbMarketModel.load(args.model_path)
    history_start = date(args.date.year - args.history_seasons, 8, 1)
    history = fetch_games(history_start, args.date - timedelta(days=1))
    future = fetch_games(args.date, args.date + timedelta(days=args.lookahead_days))
    future = [game for game in future if not game["completed"]]
    _, future_features = build_feature_frames(history, future)
    if future_features.empty:
        print(json.dumps({"date": args.date.isoformat(), "games": 0, "ready": True}))
        return
    scored = model.predict(future_features)
    prediction_ts = datetime.now(timezone.utc)
    odds_events: list[dict[str, Any]] = []
    odds_rows: list[dict[str, Any]] = []
    remaining = None
    if not args.skip_odds:
        api_key = os.getenv("ODDS_API_KEY")
        if not api_key:
            raise RuntimeError("ODDS_API_KEY is required unless --skip-odds is used.")
        odds_events, odds_ts, remaining = fetch_odds(api_key)
        for game in future:
            event = match_odds_event(game, odds_events)
            if event:
                odds_rows.extend(parse_odds_rows(game, event, odds_ts))
    recommendations = build_recommendations(scored, odds_rows, model, prediction_ts)
    coverage = {
        market: len({row["event_id"] for row in odds_rows if row["market"] == market})
        for market in ("moneyline", "spread", "total")
    }
    published_coverage = {
        market: len({row["event_id"] for row in recommendations if row["market"] == market})
        for market in ("moneyline", "spread", "total")
    }
    summary = {
        "date": args.date.isoformat(),
        "games": len(future),
        "matched_odds_games": len({row["event_id"] for row in odds_rows}),
        "odds_rows": len(odds_rows),
        "recommendations": len(recommendations),
        "coverage": coverage,
        "published_guardrail_coverage": published_coverage,
        "odds_api_remaining": remaining,
        "model_version": model.model_version,
        "model_supportable_outcomes": bool(model.metrics.get("supportable")),
        "ready": bool(
            coverage["spread"] == len(future)
            and coverage["total"] == len(future)
            and coverage["moneyline"] >= int(0.75 * len(future))
        ),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    creds = load_supabase_credentials()
    if not creds["url"] or not creds["db_password"]:
        raise RuntimeError("Supabase URL and database password are required.")
    conn = create_pg_connection(
        creds["url"], creds["db_password"], host_override=creds.get("db_host"),
        port=creds["db_port"], database=creds["db_name"], user=creds["db_user"],
    )
    try:
        sync_rows(conn, future, scored, odds_rows, recommendations, model, prediction_ts)
    finally:
        conn.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
