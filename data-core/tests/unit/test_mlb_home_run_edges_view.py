from pathlib import Path


VIEW_SQL = Path(__file__).parents[2] / "sql" / "017_player_market_health_results.sql"


def test_mlb_hr_edges_view_uses_odds_api_player_key_normalization():
    sql = VIEW_SQL.read_text(encoding="utf-8").lower()

    assert "create extension if not exists unaccent with schema extensions" in sql
    assert "extensions.unaccent(lower(player_name))" in sql
    # The Odds API adapter removes punctuation rather than turning it into a
    # separator (for example, "crow-armstrong" -> "crowarmstrong") while
    # preserving normal word spaces.
    assert "'[^a-z0-9[:space:]]+'," in sql
