from pathlib import Path


VIEW_SQL = Path(__file__).parents[2] / "sql" / "017_player_market_health_results.sql"
SERVING_SQL = Path(__file__).parents[2] / "sql" / "015_player_market_serving_tables.sql"


def test_mlb_hr_edges_view_uses_odds_api_player_key_normalization():
    sql = VIEW_SQL.read_text(encoding="utf-8").lower()

    assert "create extension if not exists unaccent with schema extensions" in sql
    assert "extensions.unaccent(lower(player_name))" in sql
    # The Odds API adapter removes punctuation rather than turning it into a
    # separator (for example, "crow-armstrong" -> "crowarmstrong") while
    # preserving normal word spaces.
    assert "'[^a-z0-9[:space:]]+'," in sql


def test_mlb_hr_edges_view_uses_standard_and_alternate_half_run_markets():
    sql = VIEW_SQL.read_text(encoding="utf-8").lower()

    assert "market in ('batter_home_runs', 'batter_home_runs_alternate')" in sql
    assert "b.market as best_market" in sql
    assert "distinct on (game_date, game_id, normalized_player_name, line)" in sql


def test_legacy_serving_schema_drops_rebuilt_mlb_views_before_create_or_replace():
    sql = SERVING_SQL.read_text(encoding="utf-8").lower()

    assert sql.index("drop view if exists mlb_home_run_edges_latest") < sql.index("create table")
    assert sql.index("drop view if exists mlb_home_run_predictions_latest") < sql.index("create table")
