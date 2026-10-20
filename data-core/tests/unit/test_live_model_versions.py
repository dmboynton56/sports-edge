from __future__ import annotations

from pathlib import Path

import pytest

from src.models.live_model_versions import (
    WEB_LIVE_MODEL_VERSIONS_PATH,
    assert_sync_matches_dashboard,
    dashboard_model_version,
    load_live_model_versions,
)
from src.models.nfl_live import LIVE_MODEL_VERSION, PARENT_LIVE_MODEL_VERSION
from src.pipeline.refresh_nfl import LIVE_MODEL_VERSION as REFRESH_VERSION

REPO = Path(__file__).resolve().parents[3]
DAILY = REPO / ".github" / "workflows" / "daily-refresh.yml"
INJURY = REPO / ".github" / "workflows" / "nfl-injury-refresh.yml"
TEAM_MARKETS = REPO / "web" / "lib" / "data" / "team-markets.ts"


def test_web_preferred_version_equals_pipeline_and_daily_refresh_sync():
    versions = load_live_model_versions()
    preferred = dashboard_model_version("NFL", versions)
    assert WEB_LIVE_MODEL_VERSIONS_PATH.exists()
    assert preferred == LIVE_MODEL_VERSION == REFRESH_VERSION == versions["NFL"]
    assert versions["NFL_PARENT"] == PARENT_LIVE_MODEL_VERSION

    daily = DAILY.read_text(encoding="utf-8")
    injury = INJURY.read_text(encoding="utf-8")
    pin = f"--model-version {preferred}"
    assert pin in daily
    assert pin in injury
    assert "validate_web_live_model_version.py" in daily
    assert "validate_web_live_model_version.py" in injury
    assert f"--synced-version {preferred}" in daily
    assert f"--synced-version {preferred}" in injury

    team_markets = TEAM_MARKETS.read_text(encoding="utf-8")
    assert "live-model-versions.json" in team_markets
    assert "nfl-v2-live-20260906" not in team_markets


def test_nfl_sync_rejects_dashboard_version_mismatch():
    with pytest.raises(ValueError, match="dashboard preferred"):
        assert_sync_matches_dashboard("NFL", PARENT_LIVE_MODEL_VERSION)
    with pytest.raises(ValueError, match="dashboard preferred"):
        assert_sync_matches_dashboard("NFL", None)
    assert_sync_matches_dashboard("NFL", LIVE_MODEL_VERSION)
    assert_sync_matches_dashboard("NBA", "ignored")
