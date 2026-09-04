#!/usr/bin/env python3
"""Train and validate the NFL anytime-touchdown model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nflreadpy as nfl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.nfl_anytime_td import AnytimeTDModel, build_feature_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train calibrated NFL anytime TD model.")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--model-version", default="nfl-anytime-td-v1")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "models" / "nfl_anytime_td_v1.txt",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=ROOT / "models" / "nfl_anytime_td_v1_metrics.json",
    )
    return parser.parse_args()


def _to_pandas(value):
    if hasattr(value, "collect"):
        value = value.collect()
    return value.to_pandas() if hasattr(value, "to_pandas") else value


def main() -> None:
    args = parse_args()
    stats = _to_pandas(nfl.load_player_stats(args.seasons, summary_level="week"))
    stats = stats[stats["season_type"].eq("REG")].copy()
    schedule = _to_pandas(nfl.load_schedules(args.seasons))
    schedule = schedule[schedule["game_type"].eq("REG")].copy()
    features = build_feature_frame(stats, schedule)
    model, metrics = AnytimeTDModel.fit(features, model_version=args.model_version)
    model.save(args.model_path, args.metadata_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["supportable"]:
        raise SystemExit("Holdout gate failed; refusing to promote anytime-TD model")


if __name__ == "__main__":
    main()
