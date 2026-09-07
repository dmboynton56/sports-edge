#!/usr/bin/env python3
"""Fit and serialize the frozen NFL v2 production candidate through 2025."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.nfl_experiment import candidate_specs, fit_production_shadow_bundle
from src.models.nfl_live import DEFAULT_ARTIFACT, LIVE_MODEL_VERSION
from src.models.nfl_v2 import FEATURE_COLUMNS, audit_nfl_feature_store, dataframe_fingerprint, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "experiments" / "nfl_v2.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def _named_spec(config: dict, task: str, name: str) -> dict:
    matches = [spec for spec in candidate_specs(config, task) if spec["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {task} candidate named {name!r}.")
    return matches[0]


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selection = config["live_selection"]
    data = pd.read_parquet(args.dataset)
    audit_nfl_feature_store(data, raise_on_error=True)
    if int(data["season"].max()) != int(selection["trained_through"]):
        raise ValueError("Live dataset does not end at the frozen trained_through season.")

    probability_spec = _named_spec(config, "probability", selection["probability_candidate"])
    margin_spec = _named_spec(config, "margin", selection["margin_candidate"])
    bundle = fit_production_shadow_bundle(
        data,
        probability_spec,
        margin_spec,
        str(selection["calibration_method"]),
    )
    created_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        **bundle,
        "artifact_version": "nfl-v2-live-artifact-1",
        "model_version": LIVE_MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "dataset_fingerprint": dataframe_fingerprint(data),
        "config_fingerprint": sha256(args.config.read_bytes()).hexdigest(),
        "created_at": created_at,
        "selection": selection,
        "sklearn_version": importlib.metadata.version("scikit-learn"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    manifest = {key: value for key, value in artifact.items() if key not in {"probability_model", "margin_model", "calibrator"}}
    manifest["artifact_sha256"] = sha256(args.output.read_bytes()).hexdigest()
    write_json(args.output.with_suffix(".json"), manifest)
    print(json.dumps({"artifact": str(args.output), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
