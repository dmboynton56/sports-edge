from pathlib import Path


MIGRATION = Path(__file__).parents[2] / "sql" / "024_odds_snapshots_team_total.sql"


def test_odds_snapshots_team_total_migration_widens_market_check():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "drop constraint if exists odds_snapshots_market_check" in sql
    assert "add constraint odds_snapshots_market_check" in sql
    assert "'team_total'" in sql
    assert "'spread'" in sql
    assert "'moneyline'" in sql
    assert "'total'" in sql
