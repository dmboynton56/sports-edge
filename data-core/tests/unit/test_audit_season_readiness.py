from scripts.audit_season_readiness import readiness_issues


def _healthy_report() -> dict:
    return {
        "league": "NFL",
        "scheduled_games": 2,
        "games_with_prediction": 2,
        "fresh_predictions": 2,
        "duplicate_rows": 0,
        "missing_book_spread": 0,
        "market_coverage": {
            market: {"complete_games": 2, "fresh_games": 2}
            for market in ("moneyline", "spread", "total")
        },
        "availability_reports": 4,
        "fresh_availability_reports": 4,
        "eligible_absences_missing_impact": 0,
        "anytime_td_prediction_games": 2,
        "anytime_td_odds_games": 2,
        "fresh_anytime_td_odds_games": 2,
        "qualified_anytime_td_rows": 3,
    }


def test_readiness_issues_accepts_complete_fresh_coverage():
    assert readiness_issues(_healthy_report()) == []


def test_readiness_issues_rejects_partial_predictions_odds_and_availability():
    report = _healthy_report()
    report["games_with_prediction"] = 1
    report["fresh_predictions"] = 0
    report["market_coverage"]["total"] = {"complete_games": 1, "fresh_games": 1}
    report["fresh_availability_reports"] = 3
    report["eligible_absences_missing_impact"] = 1

    issues = readiness_issues(report)

    assert "1 games missing predictions." in issues
    assert "2 games missing fresh predictions." in issues
    assert "1 games missing paired total odds." in issues
    assert "1 latest NFL availability reports are stale." in issues
    assert "1 eligible NFL absences are missing impact estimates." in issues
