#!/usr/bin/env python3
"""
Export free NFL closing spreads from nflverse/nflreadpy for ROI backtests.

nflverse `spread_line` is positive when the home team is favored. The rest of
this repo expects home-perspective book spreads, where a home favorite is
negative, so this script writes `line = -spread_line`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NFL spreads from nflverse schedules.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--games-path",
        default="data-core/notebooks/cache/nfl_backtest_2025_v1.csv",
        help="Existing repo backtest export used to enforce the target 285-game set.",
    )
    parser.add_argument(
        "--output",
        default="data-core/notebooks/cache/nfl_nflverse_spreads_2025.csv",
    )
    parser.add_argument(
        "--audit-output",
        default="data-core/notebooks/cache/nfl_nflverse_spreads_2025_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_games = pd.read_csv(args.games_path)
    target_games["game_id"] = target_games["game_id"].astype(str)

    schedules = nfl.load_schedules([args.season]).to_pandas()
    schedules["game_id"] = schedules["game_id"].astype(str)
    schedules = schedules[schedules["game_id"].isin(set(target_games["game_id"]))].copy()

    out = schedules[
        [
            "game_id",
            "gameday",
            "week",
            "home_team",
            "away_team",
            "spread_line",
            "home_spread_odds",
            "away_spread_odds",
            "home_moneyline",
            "away_moneyline",
            "total_line",
        ]
    ].rename(
        columns={
            "gameday": "game_date",
            "home_spread_odds": "price",
        }
    )
    out["book"] = "nflverse_pfr"
    out["market"] = "spread"
    out["line"] = -out["spread_line"].astype(float)
    out["source_spread_line"] = out["spread_line"]
    out = out[
        [
            "game_id",
            "game_date",
            "week",
            "home_team",
            "away_team",
            "book",
            "market",
            "line",
            "price",
            "away_spread_odds",
            "home_moneyline",
            "away_moneyline",
            "total_line",
            "source_spread_line",
        ]
    ].sort_values(["game_date", "game_id"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    missing = sorted(set(target_games["game_id"]) - set(out["game_id"]))
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "games_path": args.games_path,
        "requested_games": int(len(target_games)),
        "rows": int(len(out)),
        "missing_games": missing,
        "null_line_rows": int(out["line"].isna().sum()),
        "null_price_rows": int(out["price"].isna().sum()),
        "output": args.output,
        "source": "nflreadpy.load_schedules",
        "notes": [
            "nflverse spread_line is positive when the home team is favored.",
            "Exported line is home perspective: home favorite negative, home underdog positive.",
            "Book/source is nflverse/PFR aggregate, not a named sportsbook.",
        ],
    }
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved {len(out)} NFL spread rows to {output}")
    print(f"Saved audit to {audit_output}")


if __name__ == "__main__":
    main()
