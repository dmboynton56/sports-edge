-- Migration: add player availability and injury impact estimate tables.
--
-- Availability reports capture source status near game time. Impact estimates
-- capture model-ready deltas that can be joined to predictions and backtests.

create table if not exists player_availability_reports (
  id uuid primary key default gen_random_uuid(),
  league text check (league in ('NFL', 'NBA', 'MLB', 'PGA', 'CBB')) not null,
  game_id uuid references games(id) on delete cascade,
  game_date date,
  team text not null,
  opponent text,
  player_name text not null,
  player_id text,
  position text,
  status text not null,
  report_ts timestamptz not null,
  source text not null,
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_player_availability_game
  on player_availability_reports(league, game_id, team);

create index if not exists idx_player_availability_date_team
  on player_availability_reports(league, game_date, team);

create unique index if not exists idx_player_availability_unique_report
  on player_availability_reports(
    league,
    coalesce(game_id, '00000000-0000-0000-0000-000000000000'::uuid),
    coalesce(game_date, '1900-01-01'::date),
    team,
    player_name,
    report_ts,
    source
  );

create table if not exists player_impact_estimates (
  id uuid primary key default gen_random_uuid(),
  league text check (league in ('NFL', 'NBA', 'MLB', 'PGA', 'CBB')) not null,
  season int,
  game_id uuid references games(id) on delete cascade,
  game_date date,
  team text not null,
  player_name text not null,
  player_id text,
  position text,
  metric_name text not null,
  player_value numeric,
  replacement_value numeric,
  usage_share numeric,
  team_delta numeric not null default 0,
  sample_size int,
  model_version text not null,
  estimated_at timestamptz not null default now(),
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_player_impact_estimates_game
  on player_impact_estimates(league, game_id, team);

create index if not exists idx_player_impact_estimates_date_team
  on player_impact_estimates(league, game_date, team);

create unique index if not exists idx_player_impact_unique_identity
  on player_impact_estimates(
    league,
    coalesce(season, 0),
    coalesce(game_id, '00000000-0000-0000-0000-000000000000'::uuid),
    coalesce(game_date, '1900-01-01'::date),
    team,
    player_name,
    metric_name,
    model_version
  );

alter table player_availability_reports enable row level security;
alter table player_impact_estimates enable row level security;

drop policy if exists "public read player availability reports" on player_availability_reports;
drop policy if exists "public read player impact estimates" on player_impact_estimates;

create policy "public read player availability reports"
  on player_availability_reports for select using (true);

create policy "public read player impact estimates"
  on player_impact_estimates for select using (true);

comment on table player_availability_reports is 'Source-level player availability reports for injury-aware predictions.';
comment on table player_impact_estimates is 'Model-ready player absence deltas for prediction features and backtests.';
