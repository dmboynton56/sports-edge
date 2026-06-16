#!/usr/bin/env python3
"""Generate tournament-level PGA predictions for configured events.

The script prefers the existing saved PGA model stack when those local artifacts
are present. In fresh clones where those large model files are not versioned, it
falls back to a deterministic feature-store baseline and labels the run as a
candidate model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_FEATURE_STORE = ROOT / "notebooks" / "cache" / "pga_feature_store_event_level.csv"
DEFAULT_MASTERS_PRED = ROOT / "notebooks" / "cache" / "masters_2026_predictions.csv"
DEFAULT_FIELD_JSON = ROOT / "src" / "data" / "fields" / "us_open_2026_field.json"
DEFAULT_OUT = ROOT / "notebooks" / "cache" / "us_open_2026_predictions.csv"
MODELS_DIR = ROOT / "models"
MODEL_REQUIRED = (
    "ridge_sg_model.joblib",
    "rf_sg_model.joblib",
    "lgbm_sg_model_v2.joblib",
    "xgb_sg_model_v2.joblib",
    "pytorch_tabular_v2.pth",
    "meta_ensemble_sg_v2.joblib",
)


def _load_field(path: Path) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        players = payload.get("players", [])
        names = [str(row.get("player") or "").strip() for row in players]
        names = [name for name in names if name]
        return payload, names, {str(row.get("player")): row for row in players}
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return {}, names, {name: {"player": name} for name in names}


def _norm_name(name: str) -> str:
    return " ".join(ch for ch in name.lower().replace(".", "").replace(",", "").split())


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _hash_jitter(name: str, scale: float = 0.015) -> float:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return (raw - 0.5) * 2 * scale


def _latest_feature_rows(feature_store: Path, names: list[str], as_of: str) -> pd.DataFrame:
    if not feature_store.exists():
        return pd.DataFrame()
    df = pd.read_csv(feature_store)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    as_of_ts = pd.Timestamp(as_of)
    df = df[df["start"] < as_of_ts].copy()
    if df.empty:
        return pd.DataFrame()
    df["_norm"] = df["name"].astype(str).map(_norm_name)
    wanted = {_norm_name(name): name for name in names}
    rows = []
    for norm, original in wanted.items():
        sub = df[df["_norm"] == norm]
        if sub.empty:
            continue
        row = sub.sort_values("start").iloc[-1].copy()
        row["field_player"] = original
        rows.append(row)
    return pd.DataFrame(rows)


def _baseline_sg_from_feature_row(row: pd.Series) -> float:
    pieces = [
        (row.get("prev_sg_form_20"), 0.30),
        (row.get("prev_sg_form_10"), 0.25),
        (row.get("prev_sg_form_5"), 0.15),
        (row.get("prev_avg_sg_round"), 0.25),
        (row.get("relative_skill_vs_field"), 0.05),
    ]
    total = 0.0
    weight = 0.0
    for value, w in pieces:
        f = _safe_float(value)
        if f is None:
            continue
        total += f * w
        weight += w
    if weight == 0:
        return _hash_jitter(str(row.get("field_player") or row.get("name") or ""))
    return float(np.clip(total / weight, -2.5, 2.5))


def _load_masters_fallback(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {_norm_name(str(row["player"])): row.to_dict() for _, row in df.iterrows() if row.get("player")}


def _run_mc(meta_sg: np.ndarray, n_sims: int, n_rounds: int, sg_std: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260618)
    scores = rng.normal(loc=-meta_sg, scale=sg_std, size=(n_sims, n_rounds, len(meta_sg))).sum(axis=1)
    ranks = np.argsort(scores, axis=1)
    win = np.zeros(len(meta_sg))
    top5 = np.zeros(len(meta_sg))
    top10 = np.zeros(len(meta_sg))
    top20 = np.zeros(len(meta_sg))
    for sim_ranks in ranks:
        win[sim_ranks[0]] += 1
        top5[sim_ranks[: min(5, len(meta_sg))]] += 1
        top10[sim_ranks[: min(10, len(meta_sg))]] += 1
        top20[sim_ranks[: min(20, len(meta_sg))]] += 1
    denom = float(n_sims)
    return win / denom, top5 / denom, top10 / denom, top20 / denom


def _baseline_predictions(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    field_meta, names, field_by_name = _load_field(args.field_file)
    feature_rows = _latest_feature_rows(args.feature_store, names, args.as_of)
    feature_by_norm = {
        _norm_name(str(row["field_player"])): row
        for _, row in feature_rows.iterrows()
    } if not feature_rows.empty else {}
    masters = _load_masters_fallback(args.masters_fallback)

    rows: list[dict[str, Any]] = []
    missing_feature = []
    masters_used = 0
    for name in names:
        norm = _norm_name(name)
        field_row = field_by_name.get(name, {})
        row = feature_by_norm.get(norm)
        if row is not None:
            sg = _baseline_sg_from_feature_row(row)
            starts_before = _safe_float(row.get("starts_before"), 0.0) or 0.0
            cut_rate = _safe_float(row.get("prev_cut_rate"), 0.58)
            source = "feature_store_baseline"
        elif norm in masters:
            master = masters[norm]
            sg = _safe_float(master.get("exp_sg_per_round"), 0.0) or 0.0
            cut_rate = _safe_float(master.get("best_calibrated_target_made_cut_prob"), 0.58)
            starts_before = 20.0
            masters_used += 1
            source = "masters_prediction_fallback"
        else:
            sg = -0.55 if field_row.get("amateur") else -0.25
            sg += _hash_jitter(name)
            cut_rate = 0.34 if field_row.get("amateur") else 0.50
            starts_before = 0.0
            missing_feature.append(name)
            source = "field_only_baseline"
        confidence = float(np.clip(0.35 + min(starts_before, 40.0) / 100.0, 0.25, 0.82))
        rows.append(
            {
                "player": name,
                "player_id": field_row.get("player_id"),
                "country": field_row.get("country"),
                "amateur": bool(field_row.get("amateur", False)),
                "exp_sg_per_round": float(np.clip(sg, -2.5, 2.5)),
                "baseline_cut_rate": float(np.clip(cut_rate if cut_rate is not None else 0.58, 0.05, 0.98)),
                "starts_before": int(starts_before),
                "confidence": confidence,
                "source": source,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit(f"No players loaded from {args.field_file}")

    sg_arr = out["exp_sg_per_round"].to_numpy(dtype=float)
    win_p, t5_p, t10_p, t20_p = _run_mc(sg_arr, args.n_sims, args.n_rounds, args.sg_std)
    make_cut = _sigmoid((sg_arr * 1.15) + ((out["baseline_cut_rate"].to_numpy(dtype=float) - 0.55) * 1.7))
    top10_blend = np.clip((t10_p * 0.75) + (_sigmoid(sg_arr * 1.2 - 1.4) * 0.25), 0.0, 1.0)
    top20_blend = np.clip((t20_p * 0.80) + (_sigmoid(sg_arr * 1.05 - 0.65) * 0.20), 0.0, 1.0)

    projected_score_to_par = -sg_arr * args.n_rounds
    out["sim_win_pct"] = 100.0 * win_p
    out["sim_top5_pct"] = 100.0 * t5_p
    out["sim_top10_pct"] = 100.0 * t10_p
    out["sim_top20_pct"] = 100.0 * t20_p
    out["best_calibrated_target_win_prob"] = np.clip(win_p, 0.0, 1.0)
    out["best_calibrated_target_top10_prob"] = top10_blend
    out["best_calibrated_target_top20_prob"] = top20_blend
    out["best_calibrated_target_made_cut_prob"] = np.clip(make_cut, 0.0, 1.0)
    out["best_calibrated_target_win_model"] = "mc_baseline"
    out["best_calibrated_target_top10_model"] = "mc_logit_baseline"
    out["best_calibrated_target_top20_model"] = "mc_logit_baseline"
    out["best_calibrated_target_made_cut_model"] = "cut_logit_baseline"
    out["projected_score_to_par"] = projected_score_to_par
    out["projected_total_strokes"] = (args.course_par * args.n_rounds) + projected_score_to_par
    out["quality_flags"] = out.apply(
        lambda row: json.dumps(
            [
                flag
                for flag in (
                    "amateur" if row.get("amateur") else None,
                    "low_history" if int(row.get("starts_before") or 0) < 5 else None,
                    "fallback_source" if row.get("source") != "feature_store_baseline" else None,
                )
                if flag
            ]
        ),
        axis=1,
    )
    out = out.sort_values("best_calibrated_target_win_prob", ascending=False).reset_index(drop=True)

    meta = {
        "event_key": args.tournament_key,
        "event_name": args.event_name,
        "season": args.season,
        "course": args.course_name,
        "course_par": args.course_par,
        "course_yardage": args.course_yardage,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "as_of": args.as_of,
        "n_players": len(out),
        "field_size": field_meta.get("field_size", len(names)),
        "n_sims": args.n_sims,
        "n_rounds": args.n_rounds,
        "model_version": "pga-baseline-v0",
        "production_status": "candidate",
        "prediction_method": "feature-store baseline with Monte Carlo placement probabilities",
        "feature_store": str(args.feature_store),
        "field_file": str(args.field_file),
        "missing_feature_players": missing_feature,
        "masters_fallback_players": masters_used,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_urls": field_meta.get("source_url"),
        "quality_notes": [
            "PGA model artifacts were unavailable; used deterministic baseline fallback.",
            "Course-fit is disabled for this non-Masters event unless full artifacts are present.",
        ],
    }
    return out, meta


def _pga_model_artifacts_available() -> bool:
    return all((MODELS_DIR / name).exists() for name in MODEL_REQUIRED)


def _try_existing_model(args: argparse.Namespace) -> bool:
    if args.baseline_only or not _pga_model_artifacts_available():
        return False
    legacy = ROOT / "scripts" / "predict_masters_tournament.py"
    cmd = [
        sys.executable,
        str(legacy),
        "--feature-store",
        str(args.feature_store),
        "--results-supplement",
        str(args.results_supplement),
        "--field-file",
        str(args.field_file),
        "--as-of",
        args.as_of,
        "--n-sims",
        str(args.n_sims),
        "--n-rounds",
        str(args.n_rounds),
        "--sg-std",
        str(args.sg_std),
        "--out-csv",
        str(args.out_csv),
        "--skip-importance",
    ]
    if args.tournament_key != "masters_2026":
        cmd.extend(["--course-fit-weight", "0"])
    subprocess.run(cmd, cwd=ROOT, check=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PGA tournament predictions.")
    parser.add_argument("--tournament-key", default="us_open_2026")
    parser.add_argument("--event-name", default="U.S. Open Championship")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--course-name", default="Shinnecock Hills Golf Club")
    parser.add_argument("--course-par", type=int, default=70)
    parser.add_argument("--course-yardage", type=int, default=7440)
    parser.add_argument("--start-date", default="2026-06-18")
    parser.add_argument("--end-date", default="2026-06-21")
    parser.add_argument("--as-of", default="2026-06-18")
    parser.add_argument("--field-file", type=Path, default=DEFAULT_FIELD_JSON)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--results-supplement", type=Path, default=ROOT / "src" / "data" / "archive" / "pga_results_espn_supplement.tsv")
    parser.add_argument("--masters-fallback", type=Path, default=DEFAULT_MASTERS_PRED)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-sims", type=int, default=50000)
    parser.add_argument("--n-rounds", type=int, default=4)
    parser.add_argument("--sg-std", type=float, default=2.5)
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    used_existing = _try_existing_model(args)
    if used_existing:
        df = pd.read_csv(args.out_csv)
        sg = pd.to_numeric(df["exp_sg_per_round"], errors="coerce").fillna(0.0)
        df["projected_score_to_par"] = -sg * args.n_rounds
        df["projected_total_strokes"] = (args.course_par * args.n_rounds) + df["projected_score_to_par"]
        df["quality_flags"] = "[]"
        df.to_csv(args.out_csv, index=False)
        meta_path = args.out_csv.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta.update(
            {
                "event_key": args.tournament_key,
                "event_name": args.event_name,
                "season": args.season,
                "course": args.course_name,
                "course_par": args.course_par,
                "course_yardage": args.course_yardage,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "model_version": "pga-v2-stack",
                "production_status": "candidate",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {len(df)} PGA model predictions to {args.out_csv}")
        return

    df, meta = _baseline_predictions(args)
    df.to_csv(args.out_csv, index=False)
    args.out_csv.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(df)} baseline PGA predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
