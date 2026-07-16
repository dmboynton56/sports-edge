# Codex task 03 completion

Implemented append-only model evaluation history persistence without changing the existing latest-only `model_evaluation_runs` schema or SQL identity.

## Files changed

- `data-core/sql/018_model_evaluation_history.sql`
  - Added the idempotent `model_evaluation_history` table and indexes.
  - Added public read RLS policy and `anon`/`authenticated` SELECT grants.
- `data-core/scripts/sync_evaluation_history_to_supabase.py`
  - Appends each evaluation to history with `ON CONFLICT ... DO NOTHING`.
  - Counts and reports `appended_history` rows.
  - Preserves the existing `model_evaluation_runs` DELETE/INSERT statements.
- `data-core/tests/unit/test_sync_evaluation_history.py`
  - Covers one history insert per evaluation, conflict behavior for repeated `generated_at`, and unchanged latest-table statements.
- `.github/workflows/performance-history-refresh.yml`
  - Applies migrations 009 and 018 before syncing.
- `.cursor/orchestration/logs/codex-03-done.md`
  - This completion record.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_sync_evaluation_history.py -q`
  - PASS: 4 passed.
- `PYTHONPATH=. .venv/bin/python -m pytest tests/ -q`
  - PASS: 121 passed, 6 existing sklearn artifact-version warnings.
- Migration content sanity command from the task packet
  - PASS: `ok 1532`.
- Python compile check for the changed script and unit test
  - PASS.
- Workflow/migration assertions for 018 application, idempotent DDL, RLS, and grants
  - PASS.
- `git diff --check`
  - PASS.
- Live Supabase migration/sync
  - Not run; optional verification was skipped to avoid applying changes to a live database from the local workspace.

No commit or push was performed.
