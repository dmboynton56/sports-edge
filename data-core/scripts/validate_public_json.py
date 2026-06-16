#!/usr/bin/env python3
"""Strictly validate public JSON artifacts used by the web app."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUBLIC_DATA = ROOT / "web" / "public" / "data"


def _walk_numbers(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_walk_numbers(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_numbers(child, f"{path}[{index}]"))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate web/public/data JSON files.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--public-data", type=Path, default=DEFAULT_PUBLIC_DATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.paths or sorted(args.public_data.rglob("*.json"))
    failures = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant {value}")))
            bad_numbers = _walk_numbers(payload)
            if bad_numbers:
                failures.append(f"{path}: non-finite numbers at {bad_numbers[:5]}")
            if isinstance(payload, dict) and path.name not in {"cbb_prob_matrix.json"}:
                if not (payload.get("generatedAt") or payload.get("generated_at")):
                    failures.append(f"{path}: missing generatedAt/generated_at")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(paths)} JSON files")


if __name__ == "__main__":
    main()
