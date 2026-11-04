from scripts.validate_supabase_sync import strict_failures


def _healthy_report() -> dict:
    return {
        "recent_predictions": 15,
        "recent_games": 31,
        "recent_final_scores": 16,
        "orphan_predictions": 0,
        "mlb_window_games": 0,
        "mlb_recent_predictions": 0,
        "nba_slate_window_games": 0,
        "nba_slate_with_prediction": 0,
    }


def test_strict_failures_reds_empty_nba_publish_when_planner_counted_games():
    failures = strict_failures(_healthy_report(), nba_expected_games=14)

    assert failures == [
        "NBA serving window has 14 scheduled games but 0 published predictions."
    ]


def test_strict_failures_accepts_published_nba_slate():
    report = _healthy_report()
    report["nba_slate_window_games"] = 14
    report["nba_slate_with_prediction"] = 14

    assert strict_failures(report, nba_expected_games=14) == []


def test_strict_failures_ignores_empty_nba_when_planner_counted_none():
    assert strict_failures(_healthy_report(), nba_expected_games=0) == []
    assert strict_failures(_healthy_report(), nba_expected_games=None) == []
