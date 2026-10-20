from datetime import date, datetime, timezone
import pytest

from scripts import plan_daily_refresh, predict_mlb_home_runs


def test_build_plan_skips_nfl_during_june_without_scheduled_games(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, "nfl_schedule_available", lambda _season: True)
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 6, 8),
        lookback_days=1,
        lookahead_days=10,
        force_full_rebuild=False,
    )

    assert plan["run_mlb"] is True
    assert plan["run_nba"] is True
    assert plan["run_nfl"] is False
    assert plan["run_cfb"] is False
    assert plan["run_world_cup"] is True
    assert plan["nfl_season"] == 2025
    assert "offseason" in plan["nfl_reason"]
    assert plan["world_cup_season"] == 2026
    assert plan["world_cup_start_date"] == "2026-06-11"
    assert plan["world_cup_end_date"] == "2026-07-19"


def test_build_plan_waits_for_unpublished_nfl_schedule(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, "nfl_provider_max_season", lambda: 2025)
    monkeypatch.setattr(plan_daily_refresh, "nfl_schedule_available", lambda _season: False)

    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 8, 6),
        lookback_days=0,
        lookahead_days=10,
        force_full_rebuild=False,
    )

    assert plan["nfl_season"] == 2026
    assert plan["nfl_provider_max_season"] == 2025
    assert plan["nfl_schedule_available"] is False
    assert plan["run_nfl"] is False
    assert plan["run_cfb"] is True
    assert plan["run_market_odds"] is False
    assert "schedule unavailable for season 2026" in plan["nfl_reason"]


def test_build_plan_activates_nfl_when_upcoming_schedule_is_available(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, "nfl_provider_max_season", lambda: 2025)
    monkeypatch.setattr(plan_daily_refresh, "nfl_schedule_available", lambda _season: True)

    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 8, 6),
        lookback_days=0,
        lookahead_days=10,
        force_full_rebuild=False,
    )

    assert plan["run_nfl"] is True
    assert plan["run_cfb"] is True
    assert plan["run_market_odds"] is False
    assert plan["run_nfl_predictions"] is False
    assert "schedule available" in plan["nfl_reason"]
    assert "historical features only" in plan["nfl_reason"]


def test_build_plan_force_full_rebuild_runs_supported_leagues(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, "nfl_provider_max_season", lambda: 2026)
    monkeypatch.setattr(plan_daily_refresh, "nfl_schedule_available", lambda _season: True)

    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 7, 15),
        lookback_days=0,
        lookahead_days=0,
        force_full_rebuild=True,
    )

    assert plan["run_mlb"] is True
    assert plan["run_nba"] is True
    assert plan["run_nfl"] is True
    assert plan["run_cfb"] is True
    assert plan["run_world_cup"] is True
    assert plan["run_market_odds"] is True


def test_build_plan_bigquery_games_activate_offseason_league(monkeypatch):
    def fake_count_games(project, league, start_date, end_date):
        return 2 if league == "NFL" else 0

    monkeypatch.setattr(plan_daily_refresh, "count_bigquery_games", fake_count_games)
    monkeypatch.setattr(plan_daily_refresh, "nfl_provider_max_season", lambda: 2025)
    monkeypatch.setattr(plan_daily_refresh, "nfl_schedule_available", lambda _season: False)

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


def test_build_plan_skips_world_cup_outside_tournament_window(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, "nfl_schedule_available", lambda _season: True)
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 8, 15),
        lookback_days=0,
        lookahead_days=0,
        force_full_rebuild=False,
    )

    assert plan["run_world_cup"] is False
    assert plan["run_cfb"] is True
    assert plan["world_cup_season"] == 2026
    assert "offseason" in plan["world_cup_reason"]


def test_default_anchor_date_uses_denver_slate_boundary():
    # 2026-07-16 05:30 UTC is still 2026-07-15 at 23:30 in America/Denver.
    boundary = datetime(2026, 7, 16, 5, 30, tzinfo=timezone.utc)

    assert plan_daily_refresh.default_anchor_date(boundary) == date(2026, 7, 15)
    assert predict_mlb_home_runs.default_slate_date(boundary) == date(2026, 7, 15)


def test_default_slate_date_after_denver_midnight():
    # 2026-07-16 06:30 UTC is 2026-07-16 at 00:30 in America/Denver.
    boundary = datetime(2026, 7, 16, 6, 30, tzinfo=timezone.utc)

    assert plan_daily_refresh.default_anchor_date(boundary) == date(2026, 7, 16)
    assert predict_mlb_home_runs.default_slate_date(boundary) == date(2026, 7, 16)


@pytest.mark.parametrize('day,predictions,odds,start,end', [
    ('2026-09-29', True, True, '2026-09-29', '2026-10-05'),
    ('2026-09-30', False, False, '2026-09-29', '2026-10-05'),
    ('2026-10-01', False, True, '2026-09-29', '2026-10-05'),
    ('2026-10-04', False, True, '2026-09-29', '2026-10-05'),
    ('2026-10-05', False, False, '2026-09-29', '2026-10-05'),
    ('2026-10-06', True, True, '2026-10-06', '2026-10-12'),
])
def test_nfl_weekly_predictions_and_month_start_odds_recovery(monkeypatch, day, predictions, odds, start, end):
    monkeypatch.setattr(plan_daily_refresh, 'nfl_schedule_available', lambda _: True)
    plan = plan_daily_refresh.build_plan(anchor=date.fromisoformat(day), lookback_days=1, lookahead_days=14, force_full_rebuild=False)
    assert plan['run_nfl_predictions'] is predictions
    assert plan['run_nfl_odds'] is odds
    assert (plan['nfl_start_date'], plan['nfl_end_date']) == (start, end)


def test_explicit_nfl_recovery_stays_in_same_cycle(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, 'nfl_schedule_available', lambda _: True)
    plan = plan_daily_refresh.build_plan(anchor=date(2026, 10, 3), lookback_days=1, lookahead_days=14, force_full_rebuild=False, force_nfl_refresh=True)
    assert plan['run_nfl_predictions'] is True
    assert plan['run_nfl_odds'] is True
    assert plan['nfl_end_date'] == '2026-10-05'
    assert plan['nfl_prediction_reason'] == 'force_nfl_refresh'


def test_nfl_predictions_stay_tuesday_only_when_freshness_is_unknown(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, 'nfl_schedule_available', lambda _: True)
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 10, 19),
        lookback_days=1,
        lookahead_days=14,
        force_full_rebuild=False,
    )
    assert plan['run_nfl_predictions'] is False
    assert plan['run_nfl_odds'] is False
    assert plan['run_espn_dk_odds'] is True
    assert plan['nfl_prediction_reason'] == 'prediction freshness unknown; tuesday-only'
    assert plan['nfl_forecast_start_date'] == '2026-10-13'
    assert plan['nfl_forecast_end_date'] == '2026-10-26'
    assert plan['nfl_forecast_window_days'] == 13


def test_nfl_predictions_run_midweek_when_upcoming_games_missing(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, 'nfl_schedule_available', lambda _: True)
    monkeypatch.setattr(plan_daily_refresh, 'count_bigquery_games', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        plan_daily_refresh,
        'fetch_nfl_prediction_freshness',
        lambda *args, **kwargs: {
            'upcoming_games': 14,
            'missing_game_ids': ['2026_07_NE_CHI'],
            'stale_game_ids': [],
            'missing_count': 14,
            'stale_count': 0,
            'min_game_date': '2026-10-22',
            'max_game_date': '2026-10-26',
            'stale_hours': 48,
            'model_version': 'nfl-v2-live-20260906',
        },
    )
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 10, 19),
        lookback_days=1,
        lookahead_days=14,
        force_full_rebuild=False,
        project='example-project',
    )
    assert plan['run_nfl_predictions'] is True
    assert plan['run_nfl_odds'] is False
    assert plan['run_espn_dk_odds'] is True
    assert 'no nfl-v2-live-20260906 prediction' in plan['nfl_prediction_reason']
    assert plan['nfl_missing_predictions'] == 14


def test_nfl_predictions_run_when_existing_rows_are_stale(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, 'nfl_schedule_available', lambda _: True)
    monkeypatch.setattr(plan_daily_refresh, 'count_bigquery_games', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        plan_daily_refresh,
        'fetch_nfl_prediction_freshness',
        lambda *args, **kwargs: {
            'upcoming_games': 1,
            'missing_game_ids': [],
            'stale_game_ids': ['2026_06_WAS_SF'],
            'missing_count': 0,
            'stale_count': 1,
            'min_game_date': '2026-10-19',
            'max_game_date': '2026-10-19',
            'stale_hours': 48,
            'model_version': 'nfl-v2-live-20260906',
        },
    )
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 10, 19),
        lookback_days=1,
        lookahead_days=14,
        force_full_rebuild=False,
        project='example-project',
    )
    assert plan['run_nfl_predictions'] is True
    assert 'older than 48h' in plan['nfl_prediction_reason']
    assert plan['nfl_stale_predictions'] == 1


def test_nfl_predictions_skip_midweek_when_upcoming_rows_are_fresh(monkeypatch):
    monkeypatch.setattr(plan_daily_refresh, 'nfl_schedule_available', lambda _: True)
    monkeypatch.setattr(plan_daily_refresh, 'count_bigquery_games', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        plan_daily_refresh,
        'fetch_nfl_prediction_freshness',
        lambda *args, **kwargs: {
            'upcoming_games': 14,
            'missing_game_ids': [],
            'stale_game_ids': [],
            'missing_count': 0,
            'stale_count': 0,
            'min_game_date': '2026-10-22',
            'max_game_date': '2026-10-26',
            'stale_hours': 48,
            'model_version': 'nfl-v2-live-20260906',
        },
    )
    plan = plan_daily_refresh.build_plan(
        anchor=date(2026, 10, 21),
        lookback_days=1,
        lookahead_days=14,
        force_full_rebuild=False,
        project='example-project',
    )
    assert date(2026, 10, 21).weekday() == 2
    assert plan['run_nfl_predictions'] is False
    assert plan['run_nfl_odds'] is False
    assert plan['nfl_prediction_reason'] == 'upcoming games have fresh predictions'


def test_should_run_nfl_predictions_helpers():
    missing = {
        'upcoming_games': 2,
        'missing_count': 2,
        'stale_count': 0,
        'model_version': 'nfl-v2-live-20260906',
        'stale_hours': 48,
    }
    assert plan_daily_refresh.should_run_nfl_predictions(
        run_nfl=True, anchor=date(2026, 10, 20), force_nfl_refresh=False, force_full_rebuild=False, freshness=missing
    )[0] is True
    assert plan_daily_refresh.should_run_nfl_predictions(
        run_nfl=True, anchor=date(2026, 10, 21), force_nfl_refresh=False, force_full_rebuild=False, freshness=None
    ) == (False, 'prediction freshness unknown; tuesday-only')
