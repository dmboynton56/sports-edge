#!/usr/bin/env python3
"""Walk-forward margin-consistent NFL win probability vs frozen isotonic.

Fits residual sigma on prior-season OOF ridge residuals and a logistic stack
of Phi(-spread/sigma) plus the classifier logit. Ships only if a candidate is
no worse than frozen on walk-forward Brier and log loss (within CI) and
removes sign contradictions. Rollback env is NFL_V2_LIVE_VERSION.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_nfl_v2_calibrator_candidate import build_features, load_schedules
from src.models.nfl_bucket_backtest import json_safe, load_live_specs
from src.models.nfl_live import (
    DEFAULT_ARTIFACT,
    LEGACY_ARTIFACT,
    LIVE_MODEL_VERSION,
    LIVE_VERSION_ENV,
    PARENT_LIVE_MODEL_VERSION,
    load_live_artifact,
)
from src.models.nfl_margin_probability import (
    CANDIDATE_LIVE_MODEL_VERSION,
    DEFAULT_WINPROB_SIDECAR,
    attach_win_prob_config,
    evaluate_margin_winprob,
)
from src.models.nfl_v2 import write_json


DEFAULT_JSON = ROOT / "notebooks" / "cache" / "nfl_margin_winprob_eval.json"
DEFAULT_REPORT = ROOT.parent / "docs" / "analysis" / "nfl-model-health-2026.md"
METHODS = ("frozen_isotonic", "pr45_sigmoid", "margin_normal", "blend")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write-artifact", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    return parser.parse_args()


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def _metrics_row(label: str, metrics: dict) -> str:
    return (
        f"| {label} | {metrics.get('n')} | {_fmt(metrics.get('brier'))} | "
        f"{_fmt(metrics.get('log_loss'), 3)} | {_fmt(metrics.get('ece_10'))} | "
        f"{metrics.get('unique_probabilities')} | {metrics.get('sign_contradictions')} |"
    )


def _ci_line(vs_frozen: dict, name: str) -> str:
    side = vs_frozen[name]
    brier = side["delta_brier_ci"]
    log_loss = side["delta_log_loss_ci"]
    return (
        f"{name} vs frozen: Brier Δ 95% CI {_fmt(brier['lower_95'])} to {_fmt(brier['upper_95'])}; "
        f"log-loss Δ 95% CI {_fmt(log_loss['lower_95'], 3)} to {_fmt(log_loss['upper_95'], 3)}."
    )


def render_section(payload: dict) -> str:
    gate = payload["gate"]
    walk = payload["walk_forward_2021_2025"]
    holdout = payload["holdout_2026"]
    status = "SHIPPED" if gate["ship"] else "NOT SHIPPED"
    lines = [
        "",
        "## Margin-consistent win probability (2026-10-20)",
        "",
        f"**Decision: {status}.** {gate['reason']}.",
        "",
        (
            f"Frozen HGB + ridge stay `{payload['parent_model_version']}`. "
            f"Win probability candidates: Phi(-spread/σ) with walk-forward OOF residual σ="
            f"{_fmt(payload['live_sigma'], 2)}, and a logistic stack of that p plus the "
            "classifier logit. 2026 was never used to fit σ or the stack."
        ),
        "",
        "### Walk-forward 2021–2025",
        "",
        "| Method | n | Brier | Log loss | ECE | Unique p | Sign contradictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[_metrics_row(name, walk[name]) for name in METHODS],
        "",
        *[_ci_line(walk["vs_frozen"], name) for name in ("pr45_sigmoid", "margin_normal", "blend")],
        "",
        "### 2026 holdout",
        "",
        "| Method | n | Brier | Log loss | ECE | Unique p | Sign contradictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[_metrics_row(name, holdout[name]) for name in METHODS],
        "",
        *[_ci_line(holdout["vs_frozen"], name) for name in ("pr45_sigmoid", "margin_normal", "blend")],
        "",
        "Sign contradictions count games where p and the model spread disagree on the side "
        "outside the ±0.05 toss-up band. The serving guard falls back to the margin-implied p "
        "on any remaining disagreement, including if this candidate does not ship.",
        "",
    ]
    if gate["ship"]:
        lines.extend(
            [
                f"Live default is now `{payload['candidate_model_version']}` with "
                f"`win_prob.method={payload['win_prob_method_if_shipped']}`.",
                "",
                f"Rollback: `{payload['rollback_env']}={payload['rollback_value']}` "
                f"(loads `data-core/models/nfl_v2_live_20260906.joblib`).",
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    f"Live classifier stays `{payload['parent_model_version']}` isotonic. "
                    "The serving guard still replaces any out-of-band sign contradiction with "
                    f"Phi(-spread/σ), σ={_fmt(payload['live_sigma'], 2)}."
                ),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def upsert_health_section(path: Path, section: str) -> None:
    marker = "## Margin-consistent win probability (2026-10-20)"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        prefix = existing.split(marker, 1)[0].rstrip()
        existing = prefix + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")


def write_shipped_artifact(frozen: dict, result: dict) -> dict:
    method = result["gate"]["selected_method"]
    candidate = attach_win_prob_config(
        frozen,
        method=method,
        residual_sigma=float(result["live_sigma"]),
        blend=result["live_blend"],
        model_version=CANDIDATE_LIVE_MODEL_VERSION,
        parent_model_version=str(frozen.get("model_version") or PARENT_LIVE_MODEL_VERSION),
        gate=result["gate"],
    )
    if DEFAULT_ARTIFACT.exists() and not LEGACY_ARTIFACT.exists():
        shutil.copy2(DEFAULT_ARTIFACT, LEGACY_ARTIFACT)
        json_source = DEFAULT_ARTIFACT.with_suffix(".json")
        if json_source.exists():
            shutil.copy2(json_source, LEGACY_ARTIFACT.with_suffix(".json"))
    joblib.dump(candidate, DEFAULT_ARTIFACT)
    manifest = {
        key: value
        for key, value in candidate.items()
        if key not in {"probability_model", "margin_model", "calibrator"}
    }
    manifest["artifact_sha256"] = sha256(DEFAULT_ARTIFACT.read_bytes()).hexdigest()
    write_json(DEFAULT_ARTIFACT.with_suffix(".json"), json_safe(manifest))
    write_winprob_sidecar(candidate)
    return json_safe(manifest)


def write_winprob_sidecar(config: dict) -> dict:
    sidecar = json_safe(config.get("win_prob") or config)
    write_json(DEFAULT_WINPROB_SIDECAR, sidecar)
    return sidecar


def write_guard_only_artifact(frozen: dict, result: dict) -> dict:
    """Keep the frozen pickle; persist OOF sigma for the serving guard."""
    updated = attach_win_prob_config(
        frozen,
        method="isotonic",
        residual_sigma=float(result["live_sigma"]),
        blend=result["live_blend"],
        model_version=str(frozen.get("model_version") or PARENT_LIVE_MODEL_VERSION),
        parent_model_version=str(frozen.get("model_version") or PARENT_LIVE_MODEL_VERSION),
        gate=result["gate"],
    )
    return write_winprob_sidecar(updated)


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file)
    seasons = list(range(args.start_season, args.end_season + 1))
    schedules = load_schedules(seasons)
    features = build_features(schedules, seasons)
    frozen = load_live_artifact()
    specs = load_live_specs()
    print("Walk-forward margin win-prob vs frozen isotonic and PR #45 sigmoid...", flush=True)
    result = evaluate_margin_winprob(features, frozen=frozen, specs=specs)
    payload = result["payload"]
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["rollback_env"] = LIVE_VERSION_ENV
    payload["rollback_value"] = PARENT_LIVE_MODEL_VERSION
    payload["alters_live_predictions"] = bool(payload["gate"]["ship"])
    write_json(args.json_output, payload)
    section = render_section(payload)
    upsert_health_section(args.report, section)
    if args.write_artifact:
        if payload["gate"]["ship"]:
            manifest = write_shipped_artifact(frozen, result)
            payload["candidate_artifact"] = str(DEFAULT_ARTIFACT)
            payload["legacy_backup"] = str(LEGACY_ARTIFACT)
        else:
            manifest = write_guard_only_artifact(frozen, result)
            payload["candidate_artifact"] = None
            payload["legacy_backup"] = None
        payload["artifact_sha256"] = manifest.get("artifact_sha256")
        write_json(args.json_output, payload)
        print(f"Wrote live artifact config method={payload['gate']['selected_method'] or 'isotonic'}", flush=True)
    print(
        json.dumps(
            {
                "ship": payload["gate"]["ship"],
                "reason": payload["gate"]["reason"],
                "selected": payload["gate"]["selected_method"],
                "live_sigma": payload["live_sigma"],
                "walk_forward": {
                    name: {
                        "brier": payload["walk_forward_2021_2025"][name]["brier"],
                        "log_loss": payload["walk_forward_2021_2025"][name]["log_loss"],
                        "ece_10": payload["walk_forward_2021_2025"][name]["ece_10"],
                        "unique_probabilities": payload["walk_forward_2021_2025"][name]["unique_probabilities"],
                        "sign_contradictions": payload["walk_forward_2021_2025"][name]["sign_contradictions"],
                    }
                    for name in METHODS
                },
                "holdout_2026": {
                    name: {
                        "brier": payload["holdout_2026"][name]["brier"],
                        "log_loss": payload["holdout_2026"][name]["log_loss"],
                        "ece_10": payload["holdout_2026"][name]["ece_10"],
                        "unique_probabilities": payload["holdout_2026"][name]["unique_probabilities"],
                        "sign_contradictions": payload["holdout_2026"][name]["sign_contradictions"],
                    }
                    for name in METHODS
                },
                "report": str(args.report),
                "json": str(args.json_output),
                "live_version_if_shipped": CANDIDATE_LIVE_MODEL_VERSION,
                "current_live_version": LIVE_MODEL_VERSION,
                "rollback": f"{LIVE_VERSION_ENV}={PARENT_LIVE_MODEL_VERSION}",
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
