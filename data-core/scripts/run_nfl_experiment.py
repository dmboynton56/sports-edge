#!/usr/bin/env python3
"""Run a frozen NFL v2 experiment against an immutable dataset export."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nfl_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "experiments" / "nfl_v2.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "experiments" / "nfl_v2",
    )
    parser.add_argument(
        "--allow-consumed-locked-test",
        action="store_true",
        help="Reproduce an already consumed locked test without treating it as new evidence.",
    )
    args = parser.parse_args()
    run_dir = run_experiment(
        args.config,
        args.dataset,
        args.output_root,
        allow_consumed_locked_test=args.allow_consumed_locked_test,
    )
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
