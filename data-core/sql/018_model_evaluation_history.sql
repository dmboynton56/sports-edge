-- Migration: preserve append-only model evaluation history over time.

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

alter table model_evaluation_history enable row level security;

drop policy if exists "public read model evaluation history" on model_evaluation_history;

create policy "public read model evaluation history"
  on model_evaluation_history for select using (true);

grant select on model_evaluation_history to anon, authenticated;

comment on table model_evaluation_history is 'Append-only backtest and calibration evidence for model performance over time.';
