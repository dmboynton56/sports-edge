from datetime import datetime, timezone

import pytest

from scripts.mlb_hr_board_contract import (
    american_to_decimal,
    american_to_implied_probability,
    classify_run,
    coverage_stats,
    edge,
    expected_value,
    flat_unit_result,
    is_priced_row,
    no_vig_probability,
    quarter_kelly,
    resolve_slate_date,
    select_latest_pregame_publication,
    summarize_flat_results,
)


PUBLISH = datetime(2026, 8, 11, 20, 30, tzinfo=timezone.utc)


def row(rank: int, *, priced: bool = True, event: str = "2026-08-12T00:00:00Z", player: str | None = None):
    return {
        "rank": rank,
        "player_id": player or f"p{rank}",
        "event_time": event,
        "book": "ExampleBook" if priced else None,
        "american_price": 120 if priced else None,
        "market_probability": 0.46 if priced else None,
        "odds_snapshot_ts": "2026-08-11T20:00:00Z" if priced else None,
    }


def test_denver_date_boundaries_and_dst_are_local():
    assert resolve_slate_date(datetime(2026, 3, 8, 6, 59, tzinfo=timezone.utc)).isoformat() == "2026-03-07"
    assert resolve_slate_date(datetime(2026, 3, 8, 7, 1, tzinfo=timezone.utc)).isoformat() == "2026-03-08"


def test_priced_contract_rejects_zero_future_and_stale_prices():
    assert is_priced_row(row(1), PUBLISH)
    assert not is_priced_row({**row(1), "american_price": 0}, PUBLISH)
    assert not is_priced_row({**row(1), "odds_snapshot_ts": "2026-08-11T21:00:00Z"}, PUBLISH)
    assert not is_priced_row({**row(1), "odds_snapshot_ts": "2026-08-11T18:00:00Z"}, PUBLISH)


def test_top25_coverage_uses_eligible_denominator_and_zero_is_null():
    rows = [row(index, priced=index <= 20) for index in range(1, 31)]
    stats = coverage_stats(rows, PUBLISH, now=datetime(2026, 8, 11, 20, tzinfo=timezone.utc))
    assert stats["top25_denominator"] == 25
    assert stats["top25_priced_count"] == 20
    assert stats["top25_coverage"] == 0.8
    assert coverage_stats([], PUBLISH)["top25_coverage"] is None


def test_started_and_near_start_candidates_do_not_enter_denominator():
    rows = [row(1, event="2026-08-11T20:01:00Z"), row(2, event="2026-08-11T20:04:00Z"), row(3)]
    stats = coverage_stats(rows, PUBLISH, now=datetime(2026, 8, 11, 20, tzinfo=timezone.utc))
    assert stats["top25_denominator"] == 1


def test_run_classification_includes_no_slate_and_partial():
    assert classify_run(has_slate=False, source_ok=True, predictions_valid=True, top25_coverage=None) == "no_slate"
    assert classify_run(has_slate=True, source_ok=True, predictions_valid=True, top25_coverage=0.79) == "partial"
    assert classify_run(has_slate=True, source_ok=True, predictions_valid=True, top25_coverage=0.8) == "healthy"
    assert classify_run(has_slate=True, source_ok=False, predictions_valid=True, top25_coverage=1) == "failed"


def test_latest_pregame_snapshot_wins_over_morning_run():
    rows = [
        {"published_at": "2026-08-11T12:30:00Z", "event_time": "2026-08-11T23:00:00Z", "run": "morning"},
        {"published_at": "2026-08-11T18:30:00Z", "event_time": "2026-08-11T23:00:00Z", "run": "afternoon"},
        {"published_at": "2026-08-11T23:01:00Z", "event_time": "2026-08-11T23:00:00Z", "run": "late"},
    ]
    assert select_latest_pregame_publication(rows, "2026-08-11T23:00:00Z")["run"] == "afternoon"


def test_price_math_and_voids():
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5
    assert round(american_to_implied_probability(150), 6) == round(100 / 250, 6)
    assert round(no_vig_probability(0.4, 0.5), 6) == round(4 / 9, 6)
    assert edge(0.35, 0.3) == pytest.approx(0.05)
    assert expected_value(0.35, 2.5) == pytest.approx(-0.125)
    assert quarter_kelly(0.6, 2.0) == pytest.approx(0.05)
    assert flat_unit_result({"american_price": 150, "actual_home_run": True, "actual_plate_appearances": 4}) == 1.5
    assert flat_unit_result({"american_price": 150, "actual_home_run": False, "actual_plate_appearances": 0}) is None
    assert flat_unit_result({"american_price": 150, "actual_home_run": True, "actual_plate_appearances": 4, "status": "postponed"}) is None


def test_results_keep_priced_roi_separate_from_model_accuracy():
    summary = summarize_flat_results(
        [
            {"american_price": 100, "actual_home_run": True, "actual_plate_appearances": 4},
            {"american_price": 100, "actual_home_run": False, "actual_plate_appearances": 4},
            {"american_price": None, "actual_home_run": True, "actual_plate_appearances": 4},
        ]
    )
    assert summary["priced_sample"] == 2
    assert summary["priced_hit_rate"] == 0.5
    assert summary["model_only_sample"] == 3
    assert summary["model_only_hit_rate"] == 2 / 3
