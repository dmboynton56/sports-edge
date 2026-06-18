#!/usr/bin/env python3
"""Export MLB HR baseline/PyTorch experiment metrics for the website."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_BASELINE = ROOT / "models" / "mlb_hr_model_v1_metrics.json"
DEFAULT_TORCH = ROOT / "models" / "mlb_hr_torch_model_v1_metrics.json"
DEFAULT_HANDED = ROOT / "models" / "mlb_hr_torch_handed_model_v1_metrics.json"
DEFAULT_STATCAST = ROOT / "models" / "mlb_hr_torch_statcast_model_v1_metrics.json"
DEFAULT_DAILY_OUTCOMES = ROOT / "notebooks" / "cache" / "mlb_home_run_prediction_outcome_metrics.json"
DEFAULT_OUT = REPO_ROOT / "web" / "public" / "data" / "mlb_hr_experiment.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_delta(new: dict[str, Any] | None, old: dict[str, Any] | None, key: str) -> float | None:
    if not new or not old:
        return None
    try:
        return float(new[key]) - float(old[key])
    except (KeyError, TypeError, ValueError):
        return None


def _compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    test = metrics.get("test") or {}
    return {
        "modelVersion": metrics.get("model_version"),
        "estimator": metrics.get("estimator"),
        "generatedAt": metrics.get("generatedAt"),
        "trainingWindow": metrics.get("training_window"),
        "featureColumns": metrics.get("feature_columns", []),
        "categoricalColumns": metrics.get("categorical_columns", []),
        "leakageControls": metrics.get("leakage_controls", []),
        "test": {
            "rows": test.get("rows"),
            "positiveRate": test.get("positive_rate"),
            "brier": test.get("brier"),
            "baselineBrier": test.get("baseline_brier"),
            "logLoss": test.get("log_loss"),
            "baselineLogLoss": test.get("baseline_log_loss"),
            "auc": test.get("auc"),
            "top10HitRate": test.get("top_10_hit_rate"),
            "top25HitRate": test.get("top_25_hit_rate"),
        },
    }


def _compact_blend(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics or not metrics.get("blend"):
        return None
    blend = metrics["blend"]
    test = blend.get("test") or {}
    return {
        "modelVersion": blend.get("name"),
        "estimator": "pytorch_heuristic_blend",
        "generatedAt": metrics.get("generatedAt"),
        "pytorchWeight": blend.get("pytorch_weight"),
        "heuristicWeight": blend.get("heuristic_weight"),
        "validationLogLoss": blend.get("validation_log_loss"),
        "test": {
            "rows": test.get("rows"),
            "positiveRate": test.get("positive_rate"),
            "brier": test.get("brier"),
            "logLoss": test.get("log_loss"),
            "auc": test.get("auc"),
            "top10HitRate": test.get("top_10_hit_rate"),
            "top25HitRate": test.get("top_25_hit_rate"),
        },
    }


def _compact_daily_outcomes(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    return {
        "generatedAt": metrics.get("generatedAt"),
        "status": metrics.get("status"),
        "predictionRows": metrics.get("prediction_rows"),
        "evaluatedRows": metrics.get("evaluated_rows"),
        "missingOutcomeRows": metrics.get("missing_outcome_rows"),
        "evaluatedDates": metrics.get("evaluated_dates", []),
        "modelVersions": metrics.get("model_versions", []),
        "modelProbability": metrics.get("model_probability"),
        "baselineProbability": metrics.get("baseline_probability"),
    }


def build_payload(
    baseline_path: Path,
    torch_path: Path,
    handed_metrics_path: Path,
    statcast_metrics_path: Path,
    daily_outcomes_path: Path,
) -> dict[str, Any]:
    baseline_raw = _read_json(baseline_path)
    torch_raw = _read_json(torch_path)
    handed_raw = _read_json(handed_metrics_path)
    statcast_raw = _read_json(statcast_metrics_path)
    daily_outcomes_raw = _read_json(daily_outcomes_path)
    baseline = _compact_metrics(baseline_raw)
    pytorch = _compact_metrics(torch_raw)
    pytorch_blend = _compact_blend(torch_raw)
    pytorch_handed = _compact_metrics(handed_raw)
    pytorch_handed_blend = _compact_blend(handed_raw)
    pytorch_statcast = _compact_metrics(statcast_raw)
    pytorch_statcast_blend = _compact_blend(statcast_raw)
    daily_outcomes = _compact_daily_outcomes(daily_outcomes_raw)
    baseline_test = baseline["test"] if baseline else None
    pytorch_test = pytorch["test"] if pytorch else None
    pytorch_blend_test = pytorch_blend["test"] if pytorch_blend else None
    pytorch_handed_blend_test = pytorch_handed_blend["test"] if pytorch_handed_blend else None
    pytorch_statcast_blend_test = pytorch_statcast_blend["test"] if pytorch_statcast_blend else None

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "market": "MLB batter home runs",
        "experimentStatus": "pytorch_pending" if pytorch is None else "pytorch_evaluated",
        "baselineMetricsPath": str(baseline_path),
        "pytorchMetricsPath": str(torch_path),
        "pytorchHandedMetricsPath": str(handed_metrics_path),
        "pytorchStatcastMetricsPath": str(statcast_metrics_path),
        "dailyOutcomeMetricsPath": str(daily_outcomes_path),
        "baseline": baseline,
        "pytorch": pytorch
        or {
            "modelVersion": "mlb-hr-torch-v1",
            "estimator": "pytorch_wide_deep",
            "status": "pending_training",
            "test": None,
        },
        "pytorchBlend": pytorch_blend,
        "pytorchHanded": pytorch_handed,
        "pytorchHandedBlend": pytorch_handed_blend,
        "pytorchStatcast": pytorch_statcast,
        "pytorchStatcastBlend": pytorch_statcast_blend,
        "dailyOutcomes": daily_outcomes,
        "comparison": {
            "brierDelta": _metric_delta(pytorch_test, baseline_test, "brier"),
            "logLossDelta": _metric_delta(pytorch_test, baseline_test, "logLoss"),
            "aucDelta": _metric_delta(pytorch_test, baseline_test, "auc"),
            "top10HitRateDelta": _metric_delta(pytorch_test, baseline_test, "top10HitRate"),
            "interpretation": "Lower Brier/log loss and higher AUC/top-K hit rate are better.",
        },
        "blendComparison": {
            "brierDelta": _metric_delta(pytorch_blend_test, baseline_test, "brier"),
            "logLossDelta": _metric_delta(pytorch_blend_test, baseline_test, "logLoss"),
            "aucDelta": _metric_delta(pytorch_blend_test, baseline_test, "auc"),
            "top10HitRateDelta": _metric_delta(pytorch_blend_test, baseline_test, "top10HitRate"),
            "top25HitRateDelta": _metric_delta(pytorch_blend_test, baseline_test, "top25HitRate"),
            "interpretation": "Blend weights are selected on the inner validation split, then scored once on the held-out test split.",
        },
        "handedBlendComparison": {
            "brierDelta": _metric_delta(pytorch_handed_blend_test, baseline_test, "brier"),
            "logLossDelta": _metric_delta(pytorch_handed_blend_test, baseline_test, "logLoss"),
            "aucDelta": _metric_delta(pytorch_handed_blend_test, baseline_test, "auc"),
            "top10HitRateDelta": _metric_delta(pytorch_handed_blend_test, baseline_test, "top10HitRate"),
            "top25HitRateDelta": _metric_delta(pytorch_handed_blend_test, baseline_test, "top25HitRate"),
            "interpretation": "Handedness enrichment adds batter/pitcher platoon context from MLB Stats API player metadata.",
        },
        "statcastBlendComparison": {
            "brierDelta": _metric_delta(pytorch_statcast_blend_test, baseline_test, "brier"),
            "logLossDelta": _metric_delta(pytorch_statcast_blend_test, baseline_test, "logLoss"),
            "aucDelta": _metric_delta(pytorch_statcast_blend_test, baseline_test, "auc"),
            "top10HitRateDelta": _metric_delta(pytorch_statcast_blend_test, baseline_test, "top10HitRate"),
            "top25HitRateDelta": _metric_delta(pytorch_statcast_blend_test, baseline_test, "top25HitRate"),
            "interpretation": "Statcast enrichment adds prior pitch-level contact quality and pitch-mix aggregates from Baseball Savant.",
        },
        "dataExpansion": [
            {
                "name": "Baseball Savant Statcast CSV",
                "url": "https://baseballsavant.mlb.com/csv-docs",
                "use": "Pitch-level pitch type, velocity, launch angle, exit velocity, batted-ball events, and outcomes.",
            },
            {
                "name": "pybaseball",
                "url": "https://github.com/jldbc/pybaseball",
                "use": "Python access to Baseball Savant, FanGraphs, and Baseball Reference data for reproducible feature backfills.",
            },
            {
                "name": "MLB Stats API boxscores",
                "url": "https://statsapi.mlb.com/api/",
                "use": "Current repo source for schedules, probable pitchers, lineups, batter outcomes, and starter history.",
            },
        ],
        "blogDraft": {
            "title": "Can a GPU model find better MLB home-run probabilities?",
            "summary": (
                "The current random-forest HR model is calibrated and leakage-aware. A validation-chosen "
                "PyTorch + heuristic blend improves Brier and log loss, a handedness-enriched blend improves "
                "top-10 ranking, and a Statcast-enriched blend has the best AUC/top-25 ranking so far."
            ),
            "publishWhen": [
                "Probability and ranking objectives are reported separately.",
                "Daily predictions are joined to completed-game outcomes.",
                "Statcast and handedness features are rerun on the same held-out split.",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLB HR experiment summary for website.")
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--torch-metrics", type=Path, default=DEFAULT_TORCH)
    parser.add_argument("--handed-metrics", type=Path, default=DEFAULT_HANDED)
    parser.add_argument("--statcast-metrics", type=Path, default=DEFAULT_STATCAST)
    parser.add_argument("--daily-outcomes", type=Path, default=DEFAULT_DAILY_OUTCOMES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(
        args.baseline_metrics,
        args.torch_metrics,
        args.handed_metrics,
        args.statcast_metrics,
        args.daily_outcomes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
