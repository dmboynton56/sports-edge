"""
Export the current model-performance history hub.

This script consolidates measured sport-level metrics into machine-readable
JSON plus a markdown summary. It intentionally records missing odds access as a
status instead of filling market ROI with invented values.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def build_history(cache_dir: Path) -> dict[str, Any]:
    cbb_cv = _load_json(cache_dir / "cbb_expanding_cv_2016_2025.json")
    mlb_metrics = _load_json(cache_dir / "mlb_backtest_metrics_2025.json")
    mlb_ytd_metrics = _load_json(cache_dir / "mlb_backtest_metrics_2026_ytd.json")
    mlb_odds_audit = _load_json(cache_dir / "mlb_oddspapi_moneylines_2026_ytd_audit.json")
    oddspapi_validation = _load_json(cache_dir / "oddspapi_validation_audit.json")
    nba_oddspapi_audit = _load_json(cache_dir / "nba_oddspapi_spreads_2026_tail_audit.json")
    nfl_oddspapi_audit = _load_json(cache_dir / "nfl_oddspapi_spreads_2025_audit.json")
    nba_bq = _load_json(cache_dir / "nba_backtest_2025_v3_metrics.json")
    nfl_bq = _load_json(cache_dir / "nfl_backtest_2025_v1_metrics.json")
    nba_ats = _load_json(cache_dir / "nba_supabase_ats_summary_2025.json")
    nfl_ats = _load_json(cache_dir / "nfl_supabase_ats_summary_2025.json")
    nba_default = nba_bq.get("default_strategy", {})
    nba_best = nba_bq.get("best_sweep", {})
    oddspapi_requests = sum(
        int(audit.get("api_requests", 0))
        for audit in (
            oddspapi_validation,
            mlb_odds_audit,
            nba_oddspapi_audit,
            nfl_oddspapi_audit,
        )
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "oddspapi": {
            "api_version": oddspapi_validation.get("api_version", "v4"),
            "validation_status": oddspapi_validation.get("status"),
            "validation_match_rate": oddspapi_validation.get("match_rate"),
            "cumulative_api_requests": oddspapi_requests,
        },
        "sports": [
            {
                "sport": "NBA",
                "model_version": "v3",
                "season": "2025-26",
                "market": "spread",
                "data_source": "BigQuery backtest + Supabase ATS",
                "sample": {
                    "completed_games": int(nba_bq.get("completed_games", 1174)),
                    "odds_joined_games": int(nba_bq.get("games_with_book_odds", 717))
                    + int(nba_oddspapi_audit.get("matched_rows", 0)),
                    "odds_rows": int(nba_bq.get("odds_rows", 767))
                    + int(nba_oddspapi_audit.get("matched_rows", 0)),
                    "supabase_graded_games": int(nba_ats.get("graded_games", 64)),
                    "oddspapi_tail_rows": int(nba_oddspapi_audit.get("matched_rows", 0)),
                },
                "metrics": {
                    "bigquery_default_bets": int(nba_default.get("n_bets", 591)),
                    "bigquery_default_wins": int(nba_default.get("n_wins", 301)),
                    "bigquery_default_accuracy": nba_default.get("accuracy", 0.5093062605752962),
                    "bigquery_default_roi": nba_default.get("roi", -0.005489001692047425),
                    "bigquery_best_sweep_bets": int(nba_best.get("n_bets", 112)),
                    "bigquery_best_sweep_edge_threshold": nba_best.get("edge_threshold", 0.5),
                    "bigquery_best_sweep_min_confidence": nba_best.get("min_confidence", 0.4),
                    "supabase_ats_wins": int(nba_ats.get("wins", 31)),
                    "supabase_ats_losses": int(nba_ats.get("losses", 33)),
                    "supabase_ats_pushes": int(nba_ats.get("pushes", 0)),
                    "supabase_ats_roi": nba_ats.get("flat_roi_at_minus_110", -0.07528409090909094),
                    "best_reported_sweep_roi": nba_best.get("roi", 0.08262499999999998),
                },
                "odds_status": "oddspapi_tail_patch_partial"
                if nba_oddspapi_audit.get("matched_rows")
                else "partial_historical_spread_odds",
                "artifact_refs": [
                    "docs/analysis/nba_performance_2026-05-21.md",
                    "notebooks/nba_ats_roi.ipynb",
                    "scripts/backtest_nba_spread.py",
                ],
                "gaps": [
                    "raw_nba_odds stale before OddsPapi tail patch for Feb-May 2026",
                    "Supabase season still has historical rows missing book_spread",
                ],
            },
            {
                "sport": "NFL",
                "model_version": "v1",
                "season": "2025",
                "market": "spread",
                "data_source": "Supabase ATS",
                "sample": {
                    "supabase_graded_games": int(nfl_ats.get("graded_games", 56)),
                    "odds_joined_games": int(nfl_ats.get("graded_games", 56)),
                    "bigquery_scored_games": int(nfl_bq.get("scored_games", 285)),
                    "oddspapi_spread_rows": int(nfl_oddspapi_audit.get("matched_rows", 0)),
                },
                "metrics": {
                    "bigquery_accuracy": nfl_bq.get("metrics", {}).get("accuracy"),
                    "bigquery_brier": nfl_bq.get("metrics", {}).get("brier"),
                    "bigquery_log_loss": nfl_bq.get("metrics", {}).get("log_loss"),
                    "bigquery_auc": nfl_bq.get("metrics", {}).get("roc_auc"),
                    "bigquery_spread_mae": nfl_bq.get("metrics", {}).get("spread_mae"),
                    "supabase_ats_wins": int(nfl_ats.get("wins", 27)),
                    "supabase_ats_losses": int(nfl_ats.get("losses", 29)),
                    "supabase_ats_pushes": int(nfl_ats.get("pushes", 0)),
                    "supabase_ats_roi": nfl_ats.get("flat_roi_at_minus_110", -0.07954545454545459),
                },
                "odds_status": "oddspapi_spread_archive_partial"
                if nfl_oddspapi_audit.get("matched_rows")
                else "partial_supabase_spread_odds",
                "artifact_refs": [
                    "docs/analysis/nfl_performance_2026-05-21.md",
                    "notebooks/nfl_ats_roi.ipynb",
                    "notebooks/cache/nfl_backtest_2025_v1_metrics.json",
                    "notebooks/cache/nfl_oddspapi_spreads_2025.csv",
                    "scripts/backfill_oddspapi_nfl_spreads.py",
                    "scripts/export_nfl_backtest_history.py",
                ],
                "gaps": [
                    "OddsPapi historical NFL coverage is recent-fixture limited on this key tier",
                    "Supabase has stale v1/v2/v3 version mix",
                    "Available ATS sample is not full season",
                ],
            },
            {
                "sport": "MLB",
                "model_version": "v3",
                "season": "2026 YTD",
                "market": "moneyline",
                "data_source": "MLB Stats API feature store; historical odds probe",
                "sample": {
                    "feature_rows": int(mlb_ytd_metrics.get("data_summary", {}).get("rows", 12427)),
                    "test_games": int(mlb_ytd_metrics.get("splits", {}).get("test_rows", 673)),
                    "odds_joined_games": int(mlb_ytd_metrics.get("odds_summary", {}).get("odds_rows", 0)),
                    "benchmark_2025_test_games": int(mlb_metrics.get("splits", {}).get("test_rows", 2350)),
                },
                "metrics": {
                    **mlb_ytd_metrics.get("selected_refit_test", {}),
                    "baseline_brier": mlb_ytd_metrics.get("baseline", {}).get("test", {}).get("brier"),
                    "baseline_log_loss": mlb_ytd_metrics.get("baseline", {}).get("test", {}).get("log_loss"),
                    "flat_roi": mlb_ytd_metrics.get("odds_summary", {}).get("flat_roi"),
                    "benchmark_2025_brier": mlb_metrics.get("selected_refit_test", {}).get("brier"),
                    "benchmark_2025_log_loss": mlb_metrics.get("selected_refit_test", {}).get("log_loss"),
                },
                "odds_status": "oddspapi_moneyline_partial"
                if mlb_ytd_metrics.get("odds_summary", {}).get("odds_rows")
                else mlb_odds_audit.get("status", "missing_historical_moneylines"),
                "artifact_refs": [
                    "docs/analysis/mlb_performance_2026-05-21.md",
                    "notebooks/mlb_model_training_backtest.ipynb",
                    "notebooks/mlb_moneyline_roi.ipynb",
                    "models/mlb_winner_model_v3.pkl",
                    "scripts/backtest_mlb_winners.py",
                    "scripts/backfill_oddspapi_moneylines.py",
                    "notebooks/cache/mlb_oddspapi_moneylines_2026_ytd.csv",
                ],
                "gaps": [
                    "OddsPapi historical MLB coverage is recent-window limited on this key tier",
                    "Full-season 2025 moneyline archive still requires additional quota/resume runs",
                ],
            },
            {
                "sport": "PGA",
                "model_version": "v2 artifacts evaluated on refreshed store",
                "season": "test >= 2025",
                "market": "outright/top placement",
                "data_source": "PGA feature store and Masters prediction cache",
                "sample": {
                    "test_rows_regression": 15172,
                    "feature_store_rows": 175743,
                    "masters_exact_name_join_rows": 39,
                },
                "metrics": {
                    "sg_lgbm_rmse": 2.406,
                    "sg_lgbm_mae": 1.488,
                    "sg_lgbm_spearman": 0.391,
                    "made_cut_brier": 0.200,
                    "top10_brier": 0.113,
                    "top20_brier": 0.157,
                    "win_brier": 0.0058,
                    "win_auc": 0.756,
                    "masters_winner_rank_sim_win": 9,
                },
                "odds_status": "masters_2026_pre_event_odds_cache_only",
                "artifact_refs": [
                    "docs/analysis/pga_performance_2026-05-21.md",
                    "notebooks/pga_model_evaluation.ipynb",
                    "notebooks/cache/pga_odds_masters_20260409.csv",
                ],
                "gaps": [
                    "No historical event-by-event sportsbook odds ROI history",
                    "Player name normalization limits post-mortems",
                    "ESPN ingest misses no-board/same-week collision events",
                ],
            },
            {
                "sport": "CBB",
                "model_version": "manual matchup artifacts",
                "season": "CV 2016-2025",
                "market": "tournament winner probability",
                "data_source": "Cached Kaggle MMLM-derived matchup feature store",
                "sample": {
                    "feature_rows": int(cbb_cv.get("feature_rows", 2002)),
                    "folds": int(cbb_cv.get("folds", 9)),
                    "validation_start": int(cbb_cv.get("validation_start", 2016)),
                    "validation_end": int(cbb_cv.get("validation_end", 2025)),
                },
                "metrics": {
                    "best_by_mean_log_loss": cbb_cv.get("best_by_mean_log_loss", "xgb"),
                    **cbb_cv.get("mean_metrics", {}).get(
                        cbb_cv.get("best_by_mean_log_loss", "xgb"),
                        {},
                    ),
                },
                "odds_status": "no_sportsbook_odds",
                "artifact_refs": [
                    "docs/analysis/cbb_performance_2026-05-21.md",
                    "notebooks/cbb_march_madness.ipynb",
                    "notebooks/cache/cbb_expanding_cv_2016_2025.json",
                    "scripts/export_cbb_cv_history.py",
                ],
                "gaps": [
                    "Raw Kaggle MMLM files are absent from the repo",
                    "2026 tournament labels are not present in the matchup feature store",
                    "No sportsbook odds or ROI history is available",
                ],
            },
        ],
    }


def write_markdown(history: dict[str, Any], path: Path) -> None:
    lines = [
        "# Sports Model Performance History",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "| Sport | Version | Season | Market | Sample | Primary metrics | Odds status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sport in history["sports"]:
        sample = sport["sample"]
        metrics = sport["metrics"]
        if sport["sport"] == "NBA":
            primary = f"ATS ROI {_pct(metrics['supabase_ats_roi'])}; BQ default ROI {_pct(metrics['bigquery_default_roi'])}"
            sample_text = f"{sample['completed_games']} completed; {sample['odds_joined_games']} BQ odds; {sample['supabase_graded_games']} Supabase graded"
        elif sport["sport"] == "NFL":
            primary = (
                f"ATS ROI {_pct(metrics['supabase_ats_roi'])}; "
                f"BQ AUC {metrics.get('bigquery_auc', 0):.4f}; "
                f"spread MAE {metrics.get('bigquery_spread_mae', 0):.2f}"
            )
            sample_text = f"{sample['bigquery_scored_games']} BQ scored; {sample['supabase_graded_games']} Supabase graded"
        elif sport["sport"] == "MLB":
            primary = (
                f"Brier {metrics.get('brier', 0):.4f}; log loss {metrics.get('log_loss', 0):.4f}; "
                f"AUC {metrics.get('roc_auc', 0):.4f}; ROI {_pct(metrics.get('flat_roi'))}"
            )
            sample_text = f"{sample['test_games']} test games; {sample['odds_joined_games']} odds rows"
        elif sport["sport"] == "CBB":
            primary = (
                f"{metrics.get('best_by_mean_log_loss', 'xgb').upper()} mean log loss "
                f"{metrics.get('log_loss', 0):.4f}; Brier {metrics.get('brier', 0):.4f}; "
                f"AUC {metrics.get('auc', 0):.4f}"
            )
            sample_text = f"{sample['folds']} folds; {sample['feature_rows']} matchup rows"
        else:
            primary = (
                f"SG Spearman {metrics['sg_lgbm_spearman']:.3f}; "
                f"made-cut Brier {metrics['made_cut_brier']:.3f}; win AUC {metrics['win_auc']:.3f}"
            )
            sample_text = f"{sample['test_rows_regression']} regression rows"
        lines.append(
            f"| {sport['sport']} | {sport['model_version']} | {sport['season']} | "
            f"{sport['market']} | {sample_text} | {primary} | {sport['odds_status']} |"
        )

    lines.extend(["", "## Blocking Gaps", ""])
    for sport in history["sports"]:
        for gap in sport["gaps"]:
            lines.append(f"- {sport['sport']}: {gap}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model performance history.")
    parser.add_argument("--cache-dir", default="data-core/notebooks/cache")
    parser.add_argument("--json-output", default="data-core/notebooks/cache/performance_history.json")
    parser.add_argument("--md-output", default="data-core/docs/PERFORMANCE_HISTORY.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = build_history(Path(args.cache_dir))
    json_path = Path(args.json_output)
    md_path = Path(args.md_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(history, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
