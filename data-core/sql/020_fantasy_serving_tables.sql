-- Public fantasy projection serving tables. BigQuery remains the source of
-- truth; the refresh job writes these tables with the service role.

create table if not exists fantasy_projection_runs (
  run_id text primary key,
  season integer not null,
  model_version text not null,
  scope text not null check (scope in ('preseason', 'week')),
  week integer not null default 0,
  scoring_profile text not null default 'full_ppr',
  generated_at timestamptz not null default now(),
  status text not null default 'candidate',
  metrics jsonb not null default '{}'::jsonb,
  gaps text[] not null default '{}',
  source_updated_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists fantasy_player_projections (
  id bigserial primary key,
  run_id text not null references fantasy_projection_runs(run_id) on delete cascade,
  player_id text not null,
  player_name text not null,
  position text not null,
  team text,
  season integer not null,
  scope text not null check (scope in ('preseason', 'week')),
  week integer not null default 0,
  projected_games numeric,
  statline jsonb not null default '{}'::jsonb,
  statline_low jsonb not null default '{}'::jsonb,
  statline_high jsonb not null default '{}'::jsonb,
  points numeric not null default 0,
  floor_points numeric not null default 0,
  ceiling_points numeric not null default 0,
  points_per_game numeric not null default 0,
  overall_rank integer,
  position_rank integer,
  tier integer,
  adp numeric,
  adp_rank numeric,
  adp_tier integer,
  adp_source text,
  confidence text not null default 'low',
  availability text not null default 'expected',
  explanation text[] not null default '{}',
  model_version text not null,
  updated_at timestamptz not null default now(),
  unique (run_id, player_id, season, scope, week)
);

create index if not exists fantasy_player_projections_lookup
  on fantasy_player_projections (season, scope, week, points desc);

create or replace view fantasy_player_projections_latest
with (security_invoker = true) as
select distinct on (p.player_id, p.season, p.scope, p.week)
  p.*
from fantasy_player_projections p
join fantasy_projection_runs r on r.run_id = p.run_id
where r.status <> 'blocked'
order by p.player_id, p.season, p.scope, p.week, r.generated_at desc;

alter table fantasy_projection_runs enable row level security;
alter table fantasy_player_projections enable row level security;

drop policy if exists "public read fantasy runs" on fantasy_projection_runs;
create policy "public read fantasy runs"
  on fantasy_projection_runs for select using (true);

drop policy if exists "public read fantasy projections" on fantasy_player_projections;
create policy "public read fantasy projections"
  on fantasy_player_projections for select using (true);

revoke insert, update, delete on fantasy_projection_runs from anon, authenticated;
revoke insert, update, delete on fantasy_player_projections from anon, authenticated;
