#!/usr/bin/env python3
"""Fail fast when the MLB HR Statcast blend is not healthy enough to publish."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEB_JSON = ROOT.parent / "web" / "public" / "data" / "mlb_home_runs.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MLB HR Statcast blend health metadata.")
    parser.add_argument("json_path", nargs="?", type=Path, default=DEFAULT_WEB_JSON)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=float(os.getenv("MLB_HR_STATCAST_MIN_COVERAGE", "0.50")),
        help="Minimum share of published candidates with ready Statcast features.",
    )
    parser.add_argument(
        "--allow-missing-artifact",
        action="store_true",
        help="Warn instead of failing when the torch blend artifact failed to load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    health = payload.get("statcastHealth")
    if not isinstance(health, dict):
        raise SystemExit(f"{args.json_path} is missing statcastHealth metadata.")

    if not health.get("enabled", False):
        print("Statcast blend is disabled for this artifact; skipping health gate.")
        return

    artifact_loaded = bool(health.get("artifactLoaded"))
    coverage = health.get("coverage")
    ready_rows = int(health.get("readyRows") or 0)
    total_rows = int(health.get("totalRows") or 0)
    coverage_value = float(coverage if coverage is not None else 0.0)

    print(
        "MLB HR Statcast health: "
        f"coverage={coverage_value:.3f} ready={ready_rows}/{total_rows} "
        f"artifactLoaded={artifact_loaded}"
    )

    if not artifact_loaded and not args.allow_missing_artifact:
        detail = health.get("artifactError") or "artifact load status was false"
        raise SystemExit(f"Statcast blend artifact did not load: {detail}")

    if coverage_value < args.min_coverage:
        raise SystemExit(
            f"Statcast coverage {coverage_value:.3f} is below floor {args.min_coverage:.3f}."
        )


if __name__ == "__main__":
    main()
