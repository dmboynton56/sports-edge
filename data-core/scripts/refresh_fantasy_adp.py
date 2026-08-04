#!/usr/bin/env python3
"""Refresh only the FantasyPros ADP fields on an existing artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "data-core"))
sys.path.insert(0, str(ROOT / "data-core" / "scripts"))

from generate_fantasy_projections import _join_adp, _load_adp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--artifact", type=Path, default=ROOT / "web" / "public" / "data" / "fantasy_projections.json")
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = _load_adp(args.season)
    _join_adp(payload.get("projections") or [], rows)
    for weekly in (payload.get("weekly") or {}).values():
        _join_adp(weekly or [], rows)
    payload["adp"] = rows
    payload["gaps"] = [gap for gap in payload.get("gaps", []) if not str(gap).startswith("FantasyPros ADP")]
    payload["generatedAt"] = datetime.now(timezone.utc).isoformat()
    args.artifact.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Refreshed ADP for {len(rows)} market rows in {args.artifact}")


if __name__ == "__main__":
    main()
