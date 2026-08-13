#!/usr/bin/env python3
"""Sync game explanations from refresh cache into Supabase."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

DEFAULT_CACHE = ROOT / "notebooks" / "cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync game explanations to Supabase.")
    parser.add_argument("--league", required=True, choices=["NBA", "NFL"])
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSON file with explanation rows (default: notebooks/cache/{league}_explanations_latest.json)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _upsert_row(cur, row: dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO game_explanations (
          game_id, league, model_version, prediction_ts,
          top_features, injury_adjusted, home_injury_delta, away_injury_delta, base_vs_adjusted
        ) VALUES (
          %(game_id)s, %(league)s, %(model_version)s, %(prediction_ts)s,
          %(top_features)s::jsonb, %(injury_adjusted)s, %(home_injury_delta)s,
          %(away_injury_delta)s, %(base_vs_adjusted)s::jsonb
        )
        ON CONFLICT (game_id, model_version, prediction_ts)
        DO UPDATE SET
          top_features = EXCLUDED.top_features,
          injury_adjusted = EXCLUDED.injury_adjusted,
          home_injury_delta = EXCLUDED.home_injury_delta,
          away_injury_delta = EXCLUDED.away_injury_delta,
          base_vs_adjusted = EXCLUDED.base_vs_adjusted
        """,
        row,
    )


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    input_path = args.input or (DEFAULT_CACHE / f"{args.league.lower()}_explanations_latest.json")
    if not input_path.exists():
        print(f"No explanation cache at {input_path}; nothing to sync.")
        return

    payload = json.loads(input_path.read_text())
    rows = payload if isinstance(payload, list) else payload.get("rows", [])
    if not rows:
        print("Explanation cache empty.")
        return

    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )

    written = 0
    try:
        with conn.cursor() as cur:
            for raw in rows:
                row = {
                    "game_id": raw["game_id"],
                    "league": args.league,
                    "model_version": raw["model_version"],
                    "prediction_ts": raw.get("prediction_ts") or datetime.now(timezone.utc).isoformat(),
                    "top_features": json.dumps(raw.get("top_features") or []),
                    "injury_adjusted": bool(raw.get("injury_adjusted", False)),
                    "home_injury_delta": raw.get("home_injury_delta"),
                    "away_injury_delta": raw.get("away_injury_delta"),
                    "base_vs_adjusted": json.dumps(raw.get("base_vs_adjusted")),
                }
                if args.dry_run:
                    written += 1
                    continue
                _upsert_row(cur, row)
                written += 1
        if not args.dry_run:
            conn.commit()
    finally:
        conn.close()

    print(f"Synced {written} explanation rows for {args.league}.")


if __name__ == "__main__":
    main()
