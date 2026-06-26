from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.mlb_hr_torch_inference import (
    STATCAST_BLEND_MODEL_VERSION,
    apply_heuristic_blend,
    blend_weights,
    encode_categoricals,
    load_torch_hr_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "models" / "mlb_hr_torch_statcast_model_v1.pt"


def test_blend_weights_from_artifact() -> None:
    artifact = load_torch_hr_artifact(ARTIFACT)
    pytorch_weight, heuristic_weight = blend_weights(artifact)
    assert pytest_approx_sum(pytorch_weight + heuristic_weight, 1.0)
    assert 0.0 <= pytorch_weight <= 1.0
    assert 0.0 <= heuristic_weight <= 1.0


def test_apply_heuristic_blend() -> None:
    artifact = load_torch_hr_artifact(ARTIFACT)
    torch_probs = np.array([0.2, 0.4], dtype=float)
    heuristic_probs = np.array([0.1, 0.3], dtype=float)
    blended = apply_heuristic_blend(artifact, torch_probs, heuristic_probs)
    assert blended.shape == (2,)
    assert np.all(blended >= 0.001)
    assert np.all(blended <= 0.45)


def test_unknown_categorical_maps_to_zero() -> None:
    artifact = load_torch_hr_artifact(ARTIFACT)
    continuous_columns = artifact["continuous_columns"]
    categorical_columns = artifact["categorical_columns"]
    row = {col: 0.0 for col in continuous_columns}
    row.update({col: "__missing__" for col in categorical_columns})
    frame = pd.DataFrame([row])
    encoded = encode_categoricals(frame, artifact)
    assert encoded.shape == (1, len(categorical_columns))
    assert np.all(encoded == 0)


def test_served_blend_version_constant() -> None:
    assert STATCAST_BLEND_MODEL_VERSION == "mlb-hr-torch-statcast-v1-blend"


def pytest_approx_sum(value: float, target: float) -> bool:
    return abs(value - target) < 1e-9
