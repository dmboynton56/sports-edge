-- Snapshot of injury / posted-starter news used (or considered) for NFL forecasts.
-- Lets evals separate stale news from model error. Auto-applied like 024.

create table if not exists nfl_injury_news_snapshots (
  id uuid primary key default gen_random_uuid(),
  game_id uuid references games(id) on delete cascade,
  provider_game_id text,
  season int,
  week int,
  game_date date,
  home_team text not null,
  away_team text not null,
  home_starter_qb_id text,
  home_starter_qb_name text,
  away_starter_qb_id text,
  away_starter_qb_name text,
  home_expected_qb_id text,
  away_expected_qb_id text,
  key_injuries jsonb not null default '[]'::jsonb,
  status_fingerprint text not null,
  espn_home_spread numeric,
  book_spread numeric,
  source text not null,
  fallback_used boolean not null default false,
  fetched_at timestamptz not null,
  window_name text,
  used_for_prediction boolean not null default false,
  prediction_asof_ts timestamptz,
  model_version text,
  change_reasons text[] not null default '{}',
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_nfl_injury_news_game_fetched
  on nfl_injury_news_snapshots(game_id, fetched_at desc);

create index if not exists idx_nfl_injury_news_provider_fetched
  on nfl_injury_news_snapshots(provider_game_id, fetched_at desc);

create index if not exists idx_nfl_injury_news_used
  on nfl_injury_news_snapshots(used_for_prediction, fetched_at desc);

alter table nfl_injury_news_snapshots enable row level security;

drop policy if exists "public read nfl injury news snapshots" on nfl_injury_news_snapshots;
create policy "public read nfl injury news snapshots"
  on nfl_injury_news_snapshots for select using (true);

comment on table nfl_injury_news_snapshots is
  'ESPN/Sleeper injury and posted-starter snapshot per NFL game; used_for_prediction marks the row that fed a published forecast.';
