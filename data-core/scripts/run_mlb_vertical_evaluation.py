#!/usr/bin/env python3
"""Run the production-oriented MLB evaluation and edge artifact.

Examples:

    PYTHONPATH=data-core python data-core/scripts/run_mlb_vertical_evaluation.py

The default paths point at the refreshed feature store and free public
moneyline exports.  Additional odds files can be supplied with repeated
``--odds-path`` flags; all are normalized and deduplicated by game_pk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.mlb_vertical import evaluate_mlb_vertical, save_edges, write_json


DEFAULT_FEATURES = ROOT / "notebooks" / "cache" / "mlb_feature_store_v2_2021_2026.parquet"
DEFAULT_SUMMARY = ROOT / "notebooks" / "cache" / "mlb_vertical_evaluation.json"
DEFAULT_EDGES = ROOT / "notebooks" / "cache" / "mlb_vertical_edges.csv"
DEFAULT_ODDS = [
    ROOT / "notebooks" / "cache" / "mlb_oddspapi_moneylines_2026_ytd.csv",
    ROOT / "notebooks" / "cache" / "mlb_checkbestodds_moneylines_2025_2026_aug.csv",
    ROOT / "notebooks" / "cache" / "mlb_fantasydata_moneylines_2026_ytd_aug.csv",
    ROOT / "notebooks" / "cache" / "mlb_free_moneylines_2025_2026.csv",
]
DEFAULT_HR_PREDICTIONS = ROOT / "notebooks" / "cache" / "mlb_home_run_predictions_evaluated.csv"
DEFAULT_HR_MODEL = ROOT / "models" / "mlb_hr_model_v1.joblib"
DEFAULT_HR_ROWS = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows.csv"
DEFAULT_HR_METRICS = ROOT / "models" / "mlb_hr_model_v1_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all MLB model markets and produce edge rows.")
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--odds-path", type=Path, action="append", help="Free moneyline CSV/parquet; repeat for multiple sources.")
    parser.add_argument("--validation-season", type=int, default=2025)
    parser.add_argument("--test-season", type=int, default=2026)
    parser.add_argument("--as-of-date", default=None, help="Quality-check cutoff; defaults to the current UTC date.")
    parser.add_argument("--hr-predictions", type=Path, default=DEFAULT_HR_PREDICTIONS)
    parser.add_argument("--hr-model", type=Path, default=DEFAULT_HR_MODEL)
    parser.add_argument("--hr-training-rows", type=Path, default=DEFAULT_HR_ROWS)
    parser.add_argument("--hr-metrics", type=Path, default=DEFAULT_HR_METRICS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--edges-output", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--web-summary-output", type=Path, help="Optional copy for web/public/data consumption.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.features_path.exists():
        raise SystemExit(f"Missing feature store: {args.features_path}")
    features = pd.read_parquet(args.features_path)
    odds_paths = args.odds_path if args.odds_path is not None else DEFAULT_ODDS
    summary, edges = evaluate_mlb_vertical(
        features,
        odds_paths=odds_paths,
        validation_season=args.validation_season,
        test_season=args.test_season,
        as_of_date=args.as_of_date,
        hr_predictions_path=args.hr_predictions,
        hr_model_path=args.hr_model,
        hr_training_rows_path=args.hr_training_rows,
        hr_metrics_path=args.hr_metrics,
    )
    write_json(args.summary_output, summary)
    save_edges(args.edges_output, edges)
    if args.web_summary_output:
        write_json(args.web_summary_output, summary)
    print(f"Wrote MLB vertical summary: {args.summary_output}")
    print(f"Wrote {len(edges):,} MLB edge rows: {args.edges_output}")
    print(f"Production status: {summary['production_status']}")
    for market, payload in summary["markets"].items():
        gate = payload.get("quality_gate", {})
        print(f"  {market}: {gate.get('status')} ({payload.get('test_rows', 0):,} test rows)")
    print(f"  moneyline odds coverage: {summary['odds'].get('coverage')}")


if __name__ == "__main__":
    main()
