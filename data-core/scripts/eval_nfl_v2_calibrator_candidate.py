#!/usr/bin/env python3
"""Fit a PR #44 calibrator onto the frozen NFL v2 HGB and gate it on 2026.

Does not publish past predictions. Writes a candidate joblib only when the
promotion gate passes. Rollback env is NFL_V2_LIVE_VERSION.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_nfl_2026_week7 import load_team_games
from src.models.nfl_bucket_backtest import attach_schedule_prices, load_live_specs
from src.models.nfl_calibrator_promotion import (
    CANDIDATE_LIVE_MODEL_VERSION,
    LIVE_VERSION_ENV,
    evaluate_candidate,
    write_candidate_artifact,
)
from src.models.nfl_live import DEFAULT_ARTIFACT, LIVE_MODEL_VERSION, load_live_artifact
from src.models.nfl_v2 import audit_nfl_feature_store, build_nfl_feature_store, write_json


REPO_ROOT = ROOT.parent
DEFAULT_JSON = ROOT / "notebooks" / "cache" / "nfl_v2_calibrator_candidate.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "analysis" / "nfl-model-health-2026.md"
LEGACY_BACKUP = ROOT / "models" / "nfl_v2_live_20260906.joblib"
CANDIDATE_ARTIFACT = ROOT / "models" / "nfl_v2_live_20261019.joblib"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write-artifact", action="store_true", help="Write the candidate joblib if the gate passes.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    return parser.parse_args()


def _to_pandas(value) -> pd.DataFrame:
    if hasattr(value, "collect"):
        value = value.collect()
    return value.to_pandas() if hasattr(value, "to_pandas") else pd.DataFrame(value)


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    print(f"Loading nflverse schedules {seasons[0]}-{seasons[-1]}...", flush=True)
    schedules = _to_pandas(nfl.load_schedules(seasons))
    schedules["league"] = "NFL"
    return schedules


def build_features(schedules: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    team_games = load_team_games(seasons)
    print(f"Schedules={len(schedules)} team-games={len(team_games)}", flush=True)
    features = build_nfl_feature_store(schedules, team_games, include_unplayed=True)
    audit = audit_nfl_feature_store(features, raise_on_error=True)
    print(f"Feature store rows={audit['rows']} seasons={audit['seasons']}", flush=True)
    return attach_schedule_prices(features, schedules)


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def _metrics_table(label: str, side: dict) -> list[str]:
    base = side["baseline"]
    cand = side["candidate"]
    delta = side["delta"]
    return [
        f"### {label}",
        "",
        "| Metric | Frozen isotonic | Candidate | Δ (candidate − frozen) |",
        "| --- | ---: | ---: | ---: |",
        f"| n | {base.get('n') or side.get('n')} | {cand.get('n') or side.get('n')} | — |",
        f"| Brier | {_fmt(base['brier'])} | {_fmt(cand['brier'])} | {_fmt(delta['brier'])} |",
        f"| Log loss | {_fmt(base['log_loss'], 3)} | {_fmt(cand['log_loss'], 3)} | {_fmt(delta['log_loss'], 3)} |",
        f"| ECE (10-bin) | {_fmt(base['ece_10'])} | {_fmt(cand['ece_10'])} | {_fmt(delta['ece_10'])} |",
        f"| Margin MAE | {_fmt(base['mae'], 2)} | {_fmt(cand['mae'], 2)} | {_fmt(delta['mae'], 2)} |",
        f"| Unique p | {base['unique_probabilities']} | {cand['unique_probabilities']} | {delta['unique_probabilities']} |",
        "",
        (
            f"Brier paired Δ 95% CI {_fmt(side['delta_brier_ci']['lower_95'])} to "
            f"{_fmt(side['delta_brier_ci']['upper_95'])}. "
            f"Log-loss paired Δ 95% CI {_fmt(side['delta_log_loss_ci']['lower_95'], 3)} to "
            f"{_fmt(side['delta_log_loss_ci']['upper_95'], 3)}."
        ),
        "",
    ]


def render_health_section(payload: dict) -> str:
    gate = payload["gate"]
    holdout = payload["holdout_2026"]
    walk = payload["walk_forward_2021_2025"]
    toss = holdout["toss_up"]
    status = "SHIPPED" if gate["ship"] else "NOT SHIPPED"
    lines = [
        "",
        "## Smoother calibrator candidate (2026-10-19)",
        "",
        f"**Decision: {status}.** {gate['reason']}.",
        "",
        (
            f"Same frozen `hgb_small` + `ridge_a50` as `{payload['parent_model_version']}`. "
            f"Only the probability calibrator was a candidate, selected the PR #44 way on OOF "
            f"seasons {payload['oof_seasons']} (n={payload['oof_rows']}, 2026 never used to fit). "
            f"Selected: **{payload['selected_calibrator']}**."
        ),
        "",
        *_metrics_table("2026 weeks 1–6 holdout (n=92, never used in fitting)", holdout),
        *_metrics_table("2021–2025 walk-forward (isotonic vs auto)", walk),
        "### Toss-up band (±0.05) on 2026",
        "",
        (
            f"Frozen toss-up games: {toss['baseline_n']}. Candidate: {toss['candidate_n']}. "
            f"Entered {toss['entered']}, exited {toss['exited']}, stayed {toss['stayed']}, "
            f"net {toss['net']:+d}."
        ),
        "",
    ]
    if gate["ship"]:
        lines.extend(
            [
                f"Live default is now `{payload['candidate_model_version']}`. Past 2026 rows stay `{payload['parent_model_version']}`; future Daily / NFL Injury Refresh / posted-QB overrides load the new artifact.",
                "",
                f"Rollback (one line): `{payload['rollback_env']}={payload['rollback_value']}` (loads `data-core/models/nfl_v2_live_20260906.joblib`).",
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"Live serving is unchanged (`{payload['parent_model_version']}`). "
                    "The candidate joblib was not written. Daily Refresh, NFL Injury Refresh, "
                    "and posted-QB overrides still load the frozen isotonic artifact. "
                    f"Rollback switch `{LIVE_VERSION_ENV}` is unused because nothing shipped."
                ),
                "",
                "Reproduce:",
                "",
                "```bash",
                "PYTHONPATH=data-core python data-core/scripts/eval_nfl_v2_calibrator_candidate.py",
                "```",
                "",
            ]
        )
    candidates = (payload.get("calibrator_selection") or {}).get("candidates") or []
    if candidates:
        lines.extend(
            [
                "OOF calibrator selection on seasons through 2025 (negative Δ vs identity is better):",
                "",
                "| Method | Folds | Brier | Δ Brier | Log loss | Δ Log loss | Unique p |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in candidates:
            lines.append(
                f"| {row.get('method')} | {row.get('folds', 'n/a')} | {_fmt(row.get('brier'))} | "
                f"{_fmt(row.get('delta_brier'))} | {_fmt(row.get('log_loss'), 3)} | "
                f"{_fmt(row.get('delta_log_loss'), 3)} | {row.get('unique_probabilities', 'n/a')} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def upsert_health_section(path: Path, section: str) -> None:
    marker = "## Smoother calibrator candidate (2026-10-19)"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        prefix = existing.split(marker, 1)[0].rstrip()
        existing = prefix + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file)
    seasons = list(range(args.start_season, args.end_season + 1))
    schedules = load_schedules(seasons)
    features = build_features(schedules, seasons)
    frozen = load_live_artifact()
    specs = load_live_specs()
    print("Selecting calibrator on 2020–2025 OOF and scoring 2026...", flush=True)
    result = evaluate_candidate(features, frozen=frozen, specs=specs)
    payload = result["payload"]
    write_json(args.json_output, payload)
    section = render_health_section(payload)
    upsert_health_section(args.report, section)
    if payload["gate"]["ship"] and args.write_artifact:
        manifest = write_candidate_artifact(
            result["candidate_artifact"],
            path=CANDIDATE_ARTIFACT,
            frozen_backup=LEGACY_BACKUP,
            frozen_source=DEFAULT_ARTIFACT,
        )
        payload["candidate_artifact"] = str(CANDIDATE_ARTIFACT)
        payload["legacy_backup"] = str(LEGACY_BACKUP)
        payload["artifact_sha256"] = manifest.get("artifact_sha256")
        write_json(args.json_output, payload)
        print(f"Wrote {CANDIDATE_ARTIFACT} and backed up {LEGACY_BACKUP}", flush=True)
    print(
        json.dumps(
            {
                "ship": payload["gate"]["ship"],
                "reason": payload["gate"]["reason"],
                "selected": payload["selected_calibrator"],
                "holdout": payload["holdout_2026"]["delta"],
                "walk_forward": payload["walk_forward_2021_2025"]["delta"],
                "toss_up": payload["holdout_2026"]["toss_up"],
                "report": str(args.report),
                "json": str(args.json_output),
                "live_version_if_shipped": CANDIDATE_LIVE_MODEL_VERSION,
                "rollback": f"{LIVE_VERSION_ENV}={LIVE_MODEL_VERSION}",
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
