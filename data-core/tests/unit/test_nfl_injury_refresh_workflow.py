from pathlib import Path
import re

import yaml

REPO = Path(__file__).resolve().parents[3]
INJURY = REPO / ".github" / "workflows" / "nfl-injury-refresh.yml"
DAILY = REPO / ".github" / "workflows" / "daily-refresh.yml"
MIGRATION = Path(__file__).parents[2] / "sql" / "025_nfl_injury_prediction_snapshots.sql"


def test_injury_refresh_uses_cron_not_only_dispatch():
    text = INJURY.read_text(encoding="utf-8")
    # PyYAML 1.1 treats GitHub's top-level `on:` as boolean true.
    crons = set(re.findall(r'cron:\s*"([^"]+)"', text))
    assert crons == {"15 13 * * 4", "0 21 * * 4", "15 13 * * 0", "45 15 * * 0"}
    assert "workflow_dispatch:" in text
    workflow = yaml.safe_load(text)
    run = "\n".join(
        step.get("run", "") for step in workflow["jobs"]["refresh"]["steps"] if isinstance(step.get("run"), str)
    )
    assert "refresh_nfl_injury_news.py" in run
    assert "sql/025_nfl_injury_prediction_snapshots.sql" in run
    assert "--game-ids" in run
    assert "--skip-team-totals" in run
    assert "nfl-injury-" in run
    assert "--window-name thu-am" in run
    assert "--window-name sun-inactives" in run


def test_daily_snapshots_starters_before_nfl_predict_and_applies_025():
    workflow = yaml.safe_load(DAILY.read_text(encoding="utf-8"))
    steps = {step["name"]: step for step in workflow["jobs"]["refresh"]["steps"]}
    availability = steps["Refresh NFL roster and availability context"]["run"]
    predict = steps["Generate NFL Predictions"]["run"]
    apply_sql = steps["Apply dashboard serving migrations"]["run"]
    assert "refresh_nfl_injury_news.py" in availability
    assert "--snapshot" in availability
    assert "--mark-latest-used" in predict
    assert "sql/025_nfl_injury_prediction_snapshots.sql" in apply_sql
    names = [step["name"] for step in workflow["jobs"]["refresh"]["steps"]]
    assert names.index("Refresh NFL roster and availability context") < names.index("Generate NFL Predictions")


def test_snapshot_migration_has_used_for_prediction():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "create table if not exists nfl_injury_news_snapshots" in sql
    assert "used_for_prediction" in sql
    assert "status_fingerprint" in sql
    assert "provider_game_id" in sql
