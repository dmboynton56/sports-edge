from scripts.probe_nfl_prop_markets import should_spend_odds_api


def test_odds_api_spend_gate_preserves_contingency_and_nfl_allocation():
    ok, reason = should_spend_odds_api({"remaining": "437", "used": "63"}, nfl_spent=63, cost=5)
    assert ok is True
    blocked, why = should_spend_odds_api({"remaining": "437", "used": "63"}, nfl_spent=63, cost=161)
    assert blocked is False
    assert "allocation" in why.lower()
    starved, _ = should_spend_odds_api({"remaining": "40", "used": "460"}, nfl_spent=10, cost=5)
    assert starved is False
