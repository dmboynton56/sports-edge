#!/usr/bin/env python3
"""Strict readiness audit for the daily college-football research board."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cfb_market import CfbMarketModel  # noqa: E402
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402


def readiness(summary: dict) -> bool:
    games = int(summary["scheduled_games"])
    if games == 0:
        return True
    minimum_moneyline = int(0.75 * games)
    return bool(
        summary["model_supportable_outcomes"]
        and summary["predicted_games"] == games
        and summary["fresh_prediction_games"] == games
        and summary["fresh_spread_games"] == games
        and summary["fresh_total_games"] == games
        and summary["fresh_moneyline_games"] >= minimum_moneyline
        and summary["stale_recommendations"] == 0
        and summary["guardrail_violations"] == 0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CFB schedule, prediction, and odds coverage.")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--lookahead-days", type=int, default=3)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "cfb_team_v1.json")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    model = CfbMarketModel.load(args.model_path)
    end = args.date + timedelta(days=args.lookahead_days)
    creds = load_supabase_credentials()
    conn = create_pg_connection(
        creds["url"], creds["db_password"], host_override=creds.get("db_host"),
        port=creds["db_port"], database=creds["db_name"], user=creds["db_user"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH slate AS (
                  SELECT event_id FROM cfb_games WHERE game_date BETWEEN %s AND %s
                    AND game_time_utc >= NOW() - INTERVAL '3 hours'
                ), latest_odds AS (
                  SELECT DISTINCT event_id, market
                  FROM cfb_odds_snapshots
                  WHERE event_id IN (SELECT event_id FROM slate)
                    AND snapshot_ts >= NOW() - INTERVAL '24 hours'
                )
                SELECT
                  (SELECT count(*) FROM slate),
                  (SELECT count(DISTINCT event_id) FROM cfb_team_predictions
                    WHERE event_id IN (SELECT event_id FROM slate) AND model_version=%s),
                  (SELECT count(DISTINCT event_id) FROM cfb_team_predictions
                    WHERE event_id IN (SELECT event_id FROM slate) AND model_version=%s
                      AND prediction_ts >= NOW() - INTERVAL '24 hours'),
                  (SELECT count(DISTINCT event_id) FROM latest_odds WHERE market='moneyline'),
                  (SELECT count(DISTINCT event_id) FROM latest_odds WHERE market='spread'),
                  (SELECT count(DISTINCT event_id) FROM latest_odds WHERE market='total'),
                  (SELECT count(*) FROM cfb_market_recommendations
                    WHERE event_id IN (SELECT event_id FROM slate)
                      AND odds_snapshot_ts < NOW() - INTERVAL '24 hours'),
                  (SELECT count(*) FROM cfb_market_recommendations
                    WHERE event_id IN (SELECT event_id FROM slate)
                      AND (abs(edge) > .08 OR abs(ev) > .20 OR (market='moneyline' AND price > 400))),
                  (SELECT count(*) FROM cfb_market_recommendations
                    WHERE event_id IN (SELECT event_id FROM slate))
                """,
                (args.date, end, model.model_version, model.model_version),
                prepare=False,
            )
            row = cur.fetchone()
    finally:
        conn.close()
    summary = {
        "date": args.date.isoformat(),
        "end_date": end.isoformat(),
        "model_version": model.model_version,
        "model_supportable_outcomes": bool(model.metrics.get("supportable")),
        "scheduled_games": int(row[0]),
        "predicted_games": int(row[1]),
        "fresh_prediction_games": int(row[2]),
        "fresh_moneyline_games": int(row[3]),
        "fresh_spread_games": int(row[4]),
        "fresh_total_games": int(row[5]),
        "stale_recommendations": int(row[6]),
        "guardrail_violations": int(row[7]),
        "published_recommendations": int(row[8]),
    }
    summary["ready"] = readiness(summary)
    print(json.dumps(summary, indent=2 if args.json else None, sort_keys=True))
    if not summary["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
