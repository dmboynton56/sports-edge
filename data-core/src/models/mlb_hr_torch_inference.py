"""Production inference helpers for PyTorch MLB HR models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STATCAST_BLEND_MODEL_VERSION = "mlb-hr-torch-statcast-v1-blend"


def load_torch_hr_artifact(path: Path) -> dict[str, Any]:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "state_dict" not in artifact:
        raise ValueError(f"Invalid PyTorch MLB HR artifact: {path}")
    required = {"continuous_columns", "categorical_columns", "category_mappings", "imputer", "scaler"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"PyTorch MLB HR artifact missing keys: {sorted(missing)}")
    return artifact


def encode_categoricals(frame: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    categorical_columns = artifact["categorical_columns"]
    mappings: dict[str, dict[str, int]] = artifact["category_mappings"]
    arrays: list[np.ndarray] = []
    for col in categorical_columns:
        mapping = mappings[col]
        encoded = frame[col].fillna("__missing__").astype(str).map(mapping).fillna(0).astype(np.int64)
        arrays.append(encoded.to_numpy())
    return np.vstack(arrays).T


def prepare_continuous(frame: pd.DataFrame, artifact: dict[str, Any]) -> np.ndarray:
    continuous_columns = artifact["continuous_columns"]
    imputer = artifact["imputer"]
    scaler = artifact["scaler"]
    imputed = imputer.transform(frame[continuous_columns])
    return scaler.transform(imputed).astype(np.float32)


def predict_torch_probs(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    import torch

    from src.models.mlb_hr_torch_model import MlbHrWideDeepNN

    if frame.empty:
        return np.array([], dtype=float)

    model_kwargs = artifact.get("model_kwargs") or {}
    cardinalities = model_kwargs.get("categorical_cardinalities") or artifact.get("categorical_cardinalities")
    num_continuous = model_kwargs.get("num_continuous") or len(artifact["continuous_columns"])
    embedding_dim = model_kwargs.get("embedding_dim", 16)

    model = MlbHrWideDeepNN(
        cardinalities,
        num_continuous,
        embedding_dim=embedding_dim,
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    cat_x = encode_categoricals(frame, artifact)
    cont_x = prepare_continuous(frame, artifact)
    device = torch.device("cpu")
    model = model.to(device)

    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(cat_x), batch_size):
            cat = torch.tensor(cat_x[start : start + batch_size], dtype=torch.long, device=device)
            cont = torch.tensor(cont_x[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(cat, cont)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.clip(np.concatenate(probs), 1e-6, 1 - 1e-6)


def blend_weights(artifact: dict[str, Any]) -> tuple[float, float]:
    blend = (artifact.get("metrics") or {}).get("blend") or {}
    pytorch_weight = float(blend.get("pytorch_weight", 1.0))
    heuristic_weight = float(blend.get("heuristic_weight", 1.0 - pytorch_weight))
    total = pytorch_weight + heuristic_weight
    if total <= 0:
        return 1.0, 0.0
    return pytorch_weight / total, heuristic_weight / total


def apply_heuristic_blend(
    artifact: dict[str, Any],
    torch_probs: np.ndarray,
    heuristic_probs: np.ndarray,
) -> np.ndarray:
    pytorch_weight, heuristic_weight = blend_weights(artifact)
    blended = (pytorch_weight * torch_probs) + (heuristic_weight * heuristic_probs)
    return np.clip(blended, 0.001, 0.45)
