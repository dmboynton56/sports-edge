# Task 03 — Persist backtest/evaluation history over time (append-only)

Repo: `/home/dmboynton/projects/sports-edge`. Work only under `data-core/` and `.github/workflows/performance-history-refresh.yml`. Do NOT commit or push.
Independent of tasks 01/02/04/05 — parallel-safe.

## Goal

Today `sync_evaluation_history_to_supabase.py` DELETEs each evaluation identity and reinserts, and `sql/012` enforces `(league, model_name, model_version, evaluation_name)` as unique — so `model_evaluation_runs` only ever holds the latest snapshot and "backtest metrics over time" cannot accumulate. Add an append-only `model_evaluation_history` table that keeps every synced run, without changing the semantics of `model_evaluation_runs` or anything the dashboard/portfolio reads today.

## Context to read first

- `data-core/scripts/sync_evaluation_history_to_supabase.py` (esp. `sync_payloads`, lines ~240–310: DELETE + INSERT into `model_evaluation_runs`, then strategy rows)
- `data-core/sql/009_model_evaluation_tables.sql` (source schema), `data-core/sql/012_unique_evaluation_rows.sql` (identity indexes — do not touch)
- `data-core/scripts/apply_supabase_sql_files.py` (how migrations get applied)
- `.github/workflows/performance-history-refresh.yml` (lines ~70–75: applies `sql/009` then runs the sync)
- `data-core/tests/unit/test_sync_evaluation_history.py` (existing test patterns/fixtures)

## Exact changes

1. **`data-core/sql/018_model_evaluation_history.sql` (new migration — 018 is the next number).** Idempotent:

   ```sql
   create table if not exists model_evaluation_history (
     id uuid primary key default gen_random_uuid(),
     league text check (league in ('NFL', 'NBA', 'MLB', 'PGA', 'CBB')) not null,
     model_name text not null,
     model_version text not null,
     evaluation_name text not null,
     train_start_date date,
     train_end_date date,
     test_start_date date,
     test_end_date date,
     generated_at timestamptz not null,
     metrics jsonb not null default '{}'::jsonb,
     calibration jsonb not null default '{}'::jsonb,
     artifact_refs text[] not null default '{}',
     status text check (status in ('candidate', 'approved', 'rejected', 'archived')) not null default 'candidate',
     notes text,
     created_at timestamptz not null default now()
   );
   create unique index if not exists idx_model_eval_history_identity
     on model_evaluation_history(league, model_name, model_version, evaluation_name, generated_at);
   create index if not exists idx_model_eval_history_generated_at
     on model_evaluation_history(generated_at desc);
   ```

   Match the RLS/grant conventions used by 009/017 for read access (check how those files handle policies/`security_invoker`; the web app reads via PostgREST anon/service role — mirror whatever 017's tables do so the new table is readable the same way).

2. **`sync_evaluation_history_to_supabase.py`.** In `sync_payloads`, after the existing `model_evaluation_runs` insert for each evaluation, also insert the same row into `model_evaluation_history` with `ON CONFLICT (league, model_name, model_version, evaluation_name, generated_at) DO NOTHING`. Reuse the same `generated_at` value already computed per sync run. Keep the existing latest-only behavior for `model_evaluation_runs` byte-for-byte. Count and log appended history rows in the script's summary output.

3. **`.github/workflows/performance-history-refresh.yml`.** In the "Sync evaluation tables" step, apply the new migration: `python scripts/apply_supabase_sql_files.py sql/009_model_evaluation_tables.sql sql/018_model_evaluation_history.sql`. Change nothing else in the workflow (cron, publish step, secrets untouched).

4. **`data-core/tests/unit/test_sync_evaluation_history.py`.** Extend with the existing mock/fixture approach to assert: (a) one history insert per evaluation payload with the conflict clause, (b) running the sync twice with the same `generated_at` is a no-op for history, (c) `model_evaluation_runs` statements are unchanged.

## Constraints

- No changes to `games`, `model_predictions`, `odds_snapshots`, `model_evaluation_runs`, `strategy_backtest_results` schemas or to `sql/012`.
- Migration must be safe to re-apply (workflows apply migrations every run).
- No secrets in code; DB access via the existing connection helper already used in the script.
- No commits/pushes.

## Done definition

- `sql/018_model_evaluation_history.sql` exists and re-applies cleanly.
- Sync writes latest + history; repeat runs don't duplicate history rows.
- Workflow applies 018. Tests green.

## Verification

```bash
cd data-core
PYTHONPATH=. pytest tests/unit/test_sync_evaluation_history.py -q
PYTHONPATH=. pytest tests/ -q            # full suite still green
python - <<'EOF'                          # migration parses as SQL (cheap syntax sanity)
import pathlib; sql = pathlib.Path("sql/018_model_evaluation_history.sql").read_text(); assert "model_evaluation_history" in sql; print("ok", len(sql))
EOF
```

If local Supabase creds exist in the environment, optionally: `python scripts/apply_supabase_sql_files.py sql/018_model_evaluation_history.sql` then `python scripts/sync_evaluation_history_to_supabase.py` twice and confirm history row count is stable on the second run. Skip if no creds — do not fabricate results.
