from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from json_utils import dumps_strict, json_clean  # noqa: E402


def test_json_clean_converts_non_finite_numbers_to_null() -> None:
    payload = {
        "probability": float("nan"),
        "baseline": float("inf"),
        "rank": 3,
        "venue": float("nan"),
        "nested": [{"edge": float("-inf")}],
    }
    cleaned = json_clean(payload)
    assert cleaned == {
        "probability": None,
        "baseline": None,
        "rank": 3,
        "venue": None,
        "nested": [{"edge": None}],
    }


def test_dumps_strict_produces_valid_json() -> None:
    text = dumps_strict({"modelProbability": float("nan")}, sort_keys=True)
    parsed = json.loads(text)
    assert parsed == {"modelProbability": None}
