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


def test_sync_market_odds_retries_transient_failures_without_continue_on_error() -> None:
    workflow = yaml.safe_load(DAILY_WORKFLOW_PATH.read_text(encoding="utf-8"))
    step = next(
        item
        for item in workflow["jobs"]["refresh"]["steps"]
        if item.get("name") == "Sync Market Odds"
    )
    assert "continue-on-error" not in step
    assert "sync_odds.py" in step["run"]
    assert "max_attempts=5" in step["run"]
    assert "OUT_OF_USAGE_CREDITS" in step["run"]


def _daily_step(name: str) -> dict:
    workflow = yaml.safe_load(DAILY_WORKFLOW_PATH.read_text(encoding="utf-8"))
    return next(
        item
        for item in workflow["jobs"]["refresh"]["steps"]
        if item.get("name") == name
    )


def test_cfb_market_refresh_does_not_fail_daily_refresh() -> None:
    step = _daily_step("Refresh college football team markets")
    assert step.get("continue-on-error") is True
    assert "refresh_cfb_markets.py" in step["run"]


def test_nfl_mlb_production_steps_still_hard_fail() -> None:
    for name in (
        "Generate NFL Predictions",
        "Generate MLB Predictions",
        "Sync Predictions to Supabase",
        "Sync Market Odds",
        "Refresh NFL anytime touchdown markets",
        "Repair Missing Book Spreads",
    ):
        step = _daily_step(name)
        assert "continue-on-error" not in step, name


def test_sync_final_scores_still_runs_after_market_failures() -> None:
    step = _daily_step("Sync Final Scores")
    assert "always()" in str(step.get("if", ""))
    assert "continue-on-error" not in step


def test_cfb_readiness_audit_does_not_fail_daily_refresh() -> None:
    workflow = yaml.safe_load(DAILY_WORKFLOW_PATH.read_text(encoding="utf-8"))
    step = next(
        item
        for item in workflow["jobs"]["refresh"]["steps"]
        if item.get("name") == "Post-sync validation"
    )
    collapsed = " ".join(step["run"].split())
    assert "audit_cfb_readiness.py" in collapsed
    assert "sports_edge_cfb_audit.json || true" in collapsed


def test_daily_research_step_wires_propline_fallback_secret() -> None:
    workflow = yaml.safe_load(DAILY_WORKFLOW_PATH.read_text(encoding="utf-8"))
    step = next(
        item
        for item in workflow["jobs"]["refresh"]["steps"]
        if item.get("name") == "Generate MLB Research Markets"
    )
    assert "PROPLINE_API_KEY" in step["env"]
    assert "ODDS_API_KEY" in step["env"]
    assert "fetch_mlb_game_odds.py" in step["run"]
    assert "sql/023_odds_api_usage.sql" in step["run"]
    assert "PropLine-first" in step["run"]


def test_pmr_and_daily_apply_shared_odds_budget_table() -> None:
    pmr = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    daily = yaml.safe_load(DAILY_WORKFLOW_PATH.read_text(encoding="utf-8"))
    pmr_runs = " ".join(
        step.get("run", "") for step in pmr["jobs"]["refresh"]["steps"] if isinstance(step.get("run"), str)
    )
    daily_runs = " ".join(
        step.get("run", "") for step in daily["jobs"]["refresh"]["steps"] if isinstance(step.get("run"), str)
    )
    assert "sql/023_odds_api_usage.sql" in pmr_runs
    assert "sql/023_odds_api_usage.sql" in daily_runs
