from datetime import date, datetime, timezone

from scripts import plan_daily_refresh


def test_build_plan_skips_nfl_during_june_without_scheduled_games():
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 6, 8),
        lookback_days=1,
        lookahead_days=10,
        force_full_rebuild=False,
    )

    assert plan["run_mlb"] is True
    assert plan["run_nba"] is True
    assert plan["run_nfl"] is False
    assert plan["run_world_cup"] is True
    assert plan["nfl_season"] == 2025
    assert "offseason" in plan["nfl_reason"]
    assert plan["world_cup_season"] == 2026
    assert plan["world_cup_start_date"] == "2026-06-11"
    assert plan["world_cup_end_date"] == "2026-07-19"


def test_build_plan_force_full_rebuild_runs_every_league():
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 7, 15),
        lookback_days=0,
        lookahead_days=0,
        force_full_rebuild=True,
    )

    assert plan["run_mlb"] is True
    assert plan["run_nba"] is True
    assert plan["run_nfl"] is True
    assert plan["run_world_cup"] is True
    assert plan["run_market_odds"] is True


def test_build_plan_bigquery_games_activate_offseason_league(monkeypatch):
    def fake_count_games(project, league, start_date, end_date):
        return 2 if league == "NFL" else 0

    monkeypatch.setattr(plan_daily_refresh, "count_bigquery_games", fake_count_games)

    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 6, 8),
        lookback_days=0,
        lookahead_days=0,
        force_full_rebuild=False,
        project="example-project",
    )

    assert plan["run_nfl"] is True
    assert plan["nfl_scheduled_games"] == 2
    assert "scheduled games" in plan["nfl_reason"]


def test_build_plan_skips_world_cup_outside_tournament_window():
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 8, 15),
        lookback_days=0,
        lookahead_days=0,
        force_full_rebuild=False,
    )

    assert plan["run_world_cup"] is False
    assert plan["world_cup_season"] == 2026
    assert "offseason" in plan["world_cup_reason"]


def test_default_anchor_date_uses_denver_slate_boundary():
    # 2026-07-03 05:30 UTC is still 2026-07-02 at 23:30 in America/Denver.
    assert plan_daily_refresh.default_anchor_date(
        datetime(2026, 7, 3, 5, 30, tzinfo=timezone.utc)
    ) == date(2026, 7, 2)
