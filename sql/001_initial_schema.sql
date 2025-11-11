-- Sports-Edge: Initial Schema Migration
-- Creates tables, views, and RLS policies for sports betting analysis

-- Games table
create table if not exists games (
  id uuid primary key default gen_random_uuid(),
  league text check (league in ('NFL','NBA')) not null,
  season int not null,
  game_time_utc timestamptz not null,
  home_team text not null,
  away_team text not null,
  created_at timestamptz default now()
);

create index if not exists idx_games_league_season on games(league, season);
create index if not exists idx_games_date on games(game_time_utc);

-- Odds snapshots table
create table if not exists odds_snapshots (
  id uuid primary key default gen_random_uuid(),
  game_id uuid references games(id) on delete cascade,
  book text not null,
  market text check (market in ('spread','moneyline','total')) not null,
  line numeric,
  price numeric,
  snapshot_ts timestamptz not null default now(),
  created_at timestamptz default now()
);

create index if not exists idx_odds_game_id on odds_snapshots(game_id);
create index if not exists idx_odds_snapshot_ts on odds_snapshots(snapshot_ts);

-- Model predictions table
create table if not exists model_predictions (
  id uuid primary key default gen_random_uuid(),
  game_id uuid references games(id) on delete cascade,
  model_name text not null,
  model_version text not null,
  my_spread numeric,
  my_home_win_prob numeric,
  asof_ts timestamptz not null default now(),
  created_at timestamptz default now()
);

create index if not exists idx_preds_game_id on model_predictions(game_id);
create index if not exists idx_preds_asof_ts on model_predictions(asof_ts);
create index if not exists idx_preds_model_version on model_predictions(model_version);

-- Features table
create table if not exists features (
  id uuid primary key default gen_random_uuid(),
  game_id uuid references games(id) on delete cascade,
  feature_json jsonb not null,
  asof_ts timestamptz not null default now(),
  created_at timestamptz default now()
);

create index if not exists idx_features_game_id on features(game_id);
create index if not exists idx_features_asof_ts on features(asof_ts);

-- Model runs audit table
create table if not exists model_runs (
  id uuid primary key default gen_random_uuid(),
  league text,
  started_at timestamptz default now(),
  finished_at timestamptz,
  rows_written int,
  success boolean,
  error_text text,
  created_at timestamptz default now()
);

create index if not exists idx_runs_league on model_runs(league);
create index if not exists idx_runs_started_at on model_runs(started_at);

-- View: Latest odds + predictions per game for today
create or replace view games_today_enriched as
select
  g.id as game_id,
  g.league, 
  g.season, 
  g.game_time_utc,
  g.home_team, 
  g.away_team,
  o.line as book_spread,
  p.my_spread,
  (p.my_spread - o.line) as edge_pts,
  p.my_home_win_prob,
  p.model_version,
  p.asof_ts as prediction_ts,
  o.snapshot_ts as odds_ts
from games g
left join lateral (
  select line, snapshot_ts
  from odds_snapshots o
  where o.game_id = g.id and o.market = 'spread'
  order by snapshot_ts desc limit 1
) o on true
left join lateral (
  select my_spread, my_home_win_prob, model_version, asof_ts
  from model_predictions p
  where p.game_id = g.id
  order by asof_ts desc limit 1
) p on true
where g.game_time_utc::date = (now() at time zone 'America/Denver')::date
order by g.game_time_utc;

-- RLS Policies (public read-only)
alter table games enable row level security;
alter table odds_snapshots enable row level security;
alter table model_predictions enable row level security;
alter table features enable row level security;
alter table model_runs enable row level security;

-- Drop existing policies if they exist
drop policy if exists "public read games" on games;
drop policy if exists "public read odds" on odds_snapshots;
drop policy if exists "public read preds" on model_predictions;
drop policy if exists "public read features" on features;
drop policy if exists "public read runs" on model_runs;

-- Create read-only policies
create policy "public read games" on games for select using (true);
create policy "public read odds" on odds_snapshots for select using (true);
create policy "public read preds" on model_predictions for select using (true);
create policy "public read features" on features for select using (true);
create policy "public read runs" on model_runs for select using (true);

