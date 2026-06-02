-- Migration: add production model evaluation and strategy backtest tables.
--
-- These tables separate model-quality evidence from live predictions, so model
-- promotion can be based on reproducible calibration and ROI gates.

create table if not exists model_evaluation_runs (
  id uuid primary key default gen_random_uuid(),
  league text check (league in ('NFL', 'NBA', 'MLB', 'PGA', 'CBB')) not null,
  model_name text not null,
  model_version text not null,
  evaluation_name text not null,
  train_start_date date,
  train_end_date date,
  test_start_date date,
  test_end_date date,
  generated_at timestamptz not null default now(),
  metrics jsonb not null default '{}'::jsonb,
  calibration jsonb not null default '{}'::jsonb,
  artifact_refs text[] not null default '{}',
  status text check (status in ('candidate', 'approved', 'rejected', 'archived')) not null default 'candidate',
  notes text,
  created_at timestamptz not null default now()
);

create index if not exists idx_model_eval_runs_league_version
  on model_evaluation_runs(league, model_name, model_version);

create index if not exists idx_model_eval_runs_generated_at
  on model_evaluation_runs(generated_at desc);

create table if not exists strategy_backtest_results (
  id uuid primary key default gen_random_uuid(),
  evaluation_run_id uuid references model_evaluation_runs(id) on delete cascade,
  league text check (league in ('NFL', 'NBA', 'MLB', 'PGA', 'CBB')) not null,
  model_name text not null,
  model_version text not null,
  strategy_id text not null,
  market text not null,
  odds_source text,
  edge_threshold numeric,
  min_confidence numeric,
  sample_size int,
  bets int,
  wins int,
  losses int,
  pushes int,
  units numeric,
  roi numeric,
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_strategy_backtests_eval_run
  on strategy_backtest_results(evaluation_run_id);

create index if not exists idx_strategy_backtests_league_strategy
  on strategy_backtest_results(league, strategy_id, model_version);

alter table model_evaluation_runs enable row level security;
alter table strategy_backtest_results enable row level security;

drop policy if exists "public read model evaluation runs" on model_evaluation_runs;
drop policy if exists "public read strategy backtests" on strategy_backtest_results;

create policy "public read model evaluation runs"
  on model_evaluation_runs for select using (true);

create policy "public read strategy backtests"
  on strategy_backtest_results for select using (true);

comment on table model_evaluation_runs is 'Backtest and calibration evidence for model promotion decisions.';
comment on table strategy_backtest_results is 'Strategy-level betting simulation results linked to model evaluation runs.';
