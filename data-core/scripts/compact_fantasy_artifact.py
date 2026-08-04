#!/usr/bin/env python3
"""Remove repeated statlines from weekly fantasy rows in an artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(row: dict) -> dict:
    keep = {"player_id", "season", "position", "scope", "week", "projected_games", "points", "floor_points", "ceiling_points", "points_per_game", "overall_rank", "position_rank", "tier", "availability"}
    return {key: value for key, value in row.items() if key in keep}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    payload["weekly"] = {
        week: [compact(row) for row in rows]
        for week, rows in (payload.get("weekly") or {}).items()
    }
    args.artifact.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Compacted weekly rows in {args.artifact}")


if __name__ == "__main__":
    main()
