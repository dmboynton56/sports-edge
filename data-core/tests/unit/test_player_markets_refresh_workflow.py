from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "player-markets-refresh.yml"
DAILY_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "daily-refresh.yml"


def _step_names(path: Path, job: str) -> list[str]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [step["name"] for step in workflow["jobs"][job]["steps"]]


def test_player_markets_refresh_runs_bigquery_after_live_board_publish() -> None:
    names = _step_names(WORKFLOW_PATH, "refresh")

    assert names.index("Sync player markets to Supabase") < names.index("Sync player markets to BigQuery")
    assert names.index("Grade completed prior-slate MLB HR rows") < names.index("Sync player markets to BigQuery")
    assert names.index("Publish immutable MLB HR board rows") < names.index("Sync player markets to BigQuery")
    assert names.index("Finalize MLB HR board health") < names.index("Sync player markets to BigQuery")


def test_daily_refresh_runs_mlb_hr_bigquery_after_supabase() -> None:
    names = _step_names(DAILY_WORKFLOW_PATH, "refresh")

    assert names.index("Sync MLB HR Markets to Supabase") < names.index("Sync MLB HR Markets to BigQuery")
