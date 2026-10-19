#!/usr/bin/env python3
"""Compact the free PropLine 14-day NFL graded sample into docs/analysis.

Read-only ingest. Does not call paid APIs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.nfl_historical_prop_lines import compact_graded_csv, write_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="nfl_graded_props_14d.csv or .csv.gz")
    args = parser.parse_args()
    rows = compact_graded_csv(args.source)
    path = write_cache(rows)
    print({"output": str(path), "n": len(rows)})


if __name__ == "__main__":
    main()
