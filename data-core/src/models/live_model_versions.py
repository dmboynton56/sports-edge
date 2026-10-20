"""Shared dashboard/pipeline live model versions.

The JSON lives under ``web/`` so the Vercel Next.js app can import it. Python
reads the same file so a version bump cannot publish to one side and blank the
other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

WEB_LIVE_MODEL_VERSIONS_PATH = (
    Path(__file__).resolve().parents[3] / "web" / "lib" / "data" / "live-model-versions.json"
)


def load_live_model_versions(path: Path | None = None) -> dict[str, str]:
    resolved = Path(path) if path is not None else WEB_LIVE_MODEL_VERSIONS_PATH
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Live model versions file is not an object: {resolved}")
    versions = {str(key): str(value).strip() for key, value in payload.items() if str(value).strip()}
    if not versions.get("NFL"):
        raise ValueError(f"Live model versions file is missing NFL: {resolved}")
    return versions


def dashboard_model_version(league: str, versions: Mapping[str, str] | None = None) -> str:
    table = versions if versions is not None else load_live_model_versions()
    version = table.get(league)
    if not version:
        raise KeyError(f"No dashboard model version for league {league!r}")
    return version


def assert_sync_matches_dashboard(league: str, synced_version: str | None) -> None:
    """Fail closed when Daily Refresh would sync a version the site does not read."""
    if league != "NFL":
        return
    preferred = dashboard_model_version("NFL")
    actual = str(synced_version or "").strip()
    if actual != preferred:
        raise ValueError(
            f"NFL sync version {actual!r} does not match dashboard preferred {preferred!r} "
            f"({WEB_LIVE_MODEL_VERSIONS_PATH})."
        )
