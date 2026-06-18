"""Helpers for strict JSON artifacts consumed by the web app."""

from __future__ import annotations

import json
import math
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - scripts that need Timestamp/NA handling install pandas
    pd = None


def json_clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_clean(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_clean(child) for child in value]
    if pd is not None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        return json_clean(value.item())
    return value


def dumps_strict(payload: Any, **kwargs: Any) -> str:
    return json.dumps(json_clean(payload), allow_nan=False, **kwargs)
