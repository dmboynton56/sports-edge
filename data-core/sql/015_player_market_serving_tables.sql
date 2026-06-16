-- Sports Edge player/tournament market serving tables.
-- BigQuery or local model artifacts remain the source of truth; Supabase is the
-- public read cache used by the Next.js app.

create table if not exists pga_tournaments (
  id uuid primary key default gen_random_uuid(),
  event_key text not null unique,
  season int not null,
  name text not null,
  start_date date not null,
  end_date date not null,
  course text,
  par int,
  field_size int,
  status text not null default 'scheduled',
  source text,
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_pga_tournaments_dates
  on pga_tournaments(start_date, end_date);

create table if not exists pga_player_predictions (
  id uuid primary key default gen_random_uuid(),
  event_key text not null references pga_tournaments(event_key) on delete cascade,
  player_name text not null,
  player_id text,
  exp_sg_per_round numeric,
  make_cut_prob numeric check (make_cut_prob is null or (make_cut_prob >= 0 and make_cut_prob <= 1)),
  top5_prob numeric check (top5_prob is null or (top5_prob >= 0 and top5_prob <= 1)),
  top10_prob numeric check (top10_prob is null or (top10_prob >= 0 and top10_prob <= 1)),
  top20_prob numeric check (top20_prob is null or (top20_prob >= 0 and top20_prob <= 1)),
  win_prob numeric check (win_prob is null or (win_prob >= 0 and win_prob <= 1)),
  projected_total_strokes numeric,
  projected_score_to_par numeric,
  model_version text not null,
  prediction_ts timestamptz not null default now(),
  simulation_count int,
  confidence numeric check (confidence is null or (confidence >= 0 and confidence <= 1)),
  quality_flags jsonb not null default '[]'::jsonb,
  feature_snapshot jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create unique index if not exists idx_pga_player_predictions_unique
  on pga_player_predictions(event_key, player_name, model_version, prediction_ts);
create index if not exists idx_pga_player_predictions_win
  on pga_player_predictions(event_key, win_prob desc);

create table if not exists pga_odds_snapshots (
  id uuid primary key default gen_random_uuid(),
  event_key text not null references pga_tournaments(event_key) on delete cascade,
  player_name text not null,
  market text not null,
  book text not null,
  price numeric,
  implied_probability numeric check (implied_probability is null or (implied_probability >= 0 and implied_probability <= 1)),
  snapshot_ts timestamptz not null default now(),
  source text,
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_pga_odds_snapshots_event_market
  on pga_odds_snapshots(event_key, market, snapshot_ts desc);

create table if not exists mlb_home_run_predictions (
  id uuid primary key default gen_random_uuid(),
  game_id text not null,
  game_date date not null,
  event_time timestamptz,
  player_id text not null,
  player_name text not null,
  team text,
  opponent text,
  venue text,
  lineup_slot int,
  lineup_status text not null default 'projected',
  opposing_probable_pitcher text,
  hr_probability numeric not null check (hr_probability >= 0 and hr_probability <= 1),
  baseline_probability numeric check (baseline_probability is null or (baseline_probability >= 0 and baseline_probability <= 1)),
  rank int,
  confidence numeric check (confidence is null or (confidence >= 0 and confidence <= 1)),
  model_version text not null,
  prediction_ts timestamptz not null default now(),
  quality_flags jsonb not null default '[]'::jsonb,
  top_features jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create unique index if not exists idx_mlb_hr_predictions_unique
  on mlb_home_run_predictions(game_id, player_id, model_version, prediction_ts);
create index if not exists idx_mlb_hr_predictions_date_rank
  on mlb_home_run_predictions(game_date, rank);
create index if not exists idx_mlb_hr_predictions_prob
  on mlb_home_run_predictions(game_date, hr_probability desc);

create or replace view pga_player_predictions_latest
with (security_invoker = true) as
select distinct on (p.event_key, p.player_name, p.model_version)
  t.name as event_name,
  t.season,
  t.start_date,
  t.end_date,
  t.course,
  t.par,
  p.*
from pga_player_predictions p
join pga_tournaments t on t.event_key = p.event_key
order by p.event_key, p.player_name, p.model_version, p.prediction_ts desc;

create or replace view mlb_home_run_predictions_latest
with (security_invoker = true) as
select distinct on (game_date, game_id, player_id, model_version)
  *
from mlb_home_run_predictions
order by game_date, game_id, player_id, model_version, prediction_ts desc;

alter table pga_tournaments enable row level security;
alter table pga_player_predictions enable row level security;
alter table pga_odds_snapshots enable row level security;
alter table mlb_home_run_predictions enable row level security;

drop policy if exists "public read pga tournaments" on pga_tournaments;
drop policy if exists "public read pga player predictions" on pga_player_predictions;
drop policy if exists "public read pga odds snapshots" on pga_odds_snapshots;
drop policy if exists "public read mlb home run predictions" on mlb_home_run_predictions;

create policy "public read pga tournaments" on pga_tournaments for select using (true);
create policy "public read pga player predictions" on pga_player_predictions for select using (true);
create policy "public read pga odds snapshots" on pga_odds_snapshots for select using (true);
create policy "public read mlb home run predictions" on mlb_home_run_predictions for select using (true);

grant select on pga_tournaments to anon, authenticated;
grant select on pga_player_predictions to anon, authenticated;
grant select on pga_odds_snapshots to anon, authenticated;
grant select on mlb_home_run_predictions to anon, authenticated;
grant select on pga_player_predictions_latest to anon, authenticated;
grant select on mlb_home_run_predictions_latest to anon, authenticated;
