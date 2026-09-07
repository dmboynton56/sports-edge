#!/usr/bin/env python3
"""Evaluate graded 2026 NFL shadow predictions against promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nfl_shadow import evaluate_shadow_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "experiments" / "nfl_v2.yaml"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = evaluate_shadow_predictions(pd.read_csv(args.predictions), config["promotion_gates"])
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"Refusing to overwrite immutable shadow evaluation: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
