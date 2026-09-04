from scripts.audit_mlb_research_readiness import readiness_issues


def _healthy_report() -> dict:
    return {
        "scheduled_games": 3,
        "duplicate_rows": 0,
        "non_research_rows": 0,
        "market_coverage": {
            market: {
                "predicted_games": 3,
                "fresh_prediction_games": 3,
                "price_eligible_games": 2,
                "fresh_priced_games": 2,
                "invalid_priced_rows": 0,
            }
            for market in ("moneyline", "run_line", "total")
        },
    }


def test_readiness_accepts_complete_research_feed():
    assert readiness_issues(_healthy_report()) == []


def test_readiness_rejects_empty_serving_slate():
    report = _healthy_report()
    report["scheduled_games"] = 0
    assert readiness_issues(report) == ["No MLB games found for the requested date."]


def test_readiness_rejects_missing_stale_or_invalid_rows():
    report = _healthy_report()
    report["market_coverage"]["moneyline"]["predicted_games"] = 2
    report["market_coverage"]["run_line"]["fresh_prediction_games"] = 1
    report["market_coverage"]["total"]["fresh_priced_games"] = 1
    report["market_coverage"]["total"]["invalid_priced_rows"] = 1
    report["non_research_rows"] = 1

    issues = readiness_issues(report)

    assert "1 games missing moneyline predictions." in issues
    assert "2 run_line predictions are stale or missing." in issues
    assert "1 price-eligible games missing fresh paired total odds." in issues
    assert "1 priced total rows violate the serving contract." in issues
    assert "1 rows are not labeled research." in issues
