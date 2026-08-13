from pathlib import Path


MIGRATION = Path(__file__).parents[2] / "supabase" / "migrations" / "20260811144616_mlb_hr_trusted_board.sql"


def test_trusted_board_migration_contains_serving_contract_guards():
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    assert "create table if not exists public.mlb_home_run_board_runs" in sql
    assert "create table if not exists public.mlb_home_run_board_rows" in sql
    assert "unique (run_id, model_version, game_id, player_id)" in sql
    assert "top25_coverage is null" in sql
    assert "security_invoker = true" in sql
    assert "enable row level security" in sql
    assert "revoke all privileges on table" in sql
    assert "grant select on public.mlb_home_run_results to anon, authenticated" in sql
    assert "grant select on public.mlb_home_run_board_latest to anon, authenticated" in sql
    assert "mlb_home_run_results_board_row_id_fkey" in sql


def test_published_board_contract_cannot_recompute_historical_edges():
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    assert "board.edge" in sql
    assert "board.ev" in sql
    assert "board.odds_snapshot_ts" in sql
