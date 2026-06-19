#!/usr/bin/env python3
"""Combine player-market artifacts into web/public/data/predictions.json."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from json_utils import dumps_strict  # noqa: E402
DEFAULT_PGA_JSON = ROOT / "web" / "public" / "data" / "pga_tournaments" / "current.json"
DEFAULT_MLB_JSON = ROOT / "web" / "public" / "data" / "mlb_home_runs.json"
DEFAULT_OUT = ROOT / "web" / "public" / "data" / "predictions.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export normalized web prediction feed.")
    parser.add_argument("--pga-json", type=Path, default=DEFAULT_PGA_JSON)
    parser.add_argument("--mlb-json", type=Path, default=DEFAULT_MLB_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-pga", action="store_true")
    parser.add_argument("--skip-mlb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pga = {} if args.skip_pga else _load_json(args.pga_json)
    mlb = {} if args.skip_mlb else _load_json(args.mlb_json)
    predictions = []
    predictions.extend(mlb.get("predictions") or [])
    predictions.extend(pga.get("normalizedMarkets") or [])
    gaps = []
    gaps.extend(mlb.get("gaps") or [])
    gaps.extend(pga.get("gaps") or [])
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "combined player market artifacts",
        "predictions": predictions,
        "gaps": gaps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps_strict(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out} with {len(predictions)} predictions")


if __name__ == "__main__":
    main()
