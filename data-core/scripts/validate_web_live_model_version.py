#!/usr/bin/env python3
"""Fail closed when the dashboard preferred NFL version != the version just synced."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.live_model_versions import (
    WEB_LIVE_MODEL_VERSIONS_PATH,
    assert_sync_matches_dashboard,
    dashboard_model_version,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="NFL")
    parser.add_argument("--synced-version", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        assert_sync_matches_dashboard(args.league, args.synced_version)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    preferred = dashboard_model_version(args.league)
    print(
        f"ok dashboard prefers {preferred} from {WEB_LIVE_MODEL_VERSIONS_PATH.name}; "
        f"Daily Refresh synced {args.synced_version}"
    )


if __name__ == "__main__":
    main()
