-- Sports Edge World Cup serving tables.
-- BigQuery remains the warehouse source of truth; these public tables are the
-- read cache for portfolio match cards, group probabilities, and bracket odds.

create table if not exists world_cup_matches (
  id uuid primary key default gen_random_uuid(),
  external_match_id text not null unique,
  tournament text not null default 'FIFA World Cup',
  season int not null default 2026,
  stage text not null,
  group_name text,
  kickoff_utc timestamptz,
  home_team text not null,
  away_team text not null,
  venue text,
  status text not null default 'scheduled',
  home_score int,
  away_score int,
  source text,
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_world_cup_matches_stage on world_cup_matches(stage);
create index if not exists idx_world_cup_matches_group on world_cup_matches(group_name);
create index if not exists idx_world_cup_matches_kickoff on world_cup_matches(kickoff_utc);

create table if not exists world_cup_match_predictions (
  id uuid primary key default gen_random_uuid(),
  match_id uuid not null references world_cup_matches(id) on delete cascade,
  model_name text not null default 'sports_edge_world_cup',
  model_version text not null,
  home_win_prob numeric not null check (home_win_prob >= 0 and home_win_prob <= 1),
  draw_prob numeric not null check (draw_prob >= 0 and draw_prob <= 1),
  away_win_prob numeric not null check (away_win_prob >= 0 and away_win_prob <= 1),
  home_knockout_win_prob numeric check (home_knockout_win_prob >= 0 and home_knockout_win_prob <= 1),
  away_knockout_win_prob numeric check (away_knockout_win_prob >= 0 and away_knockout_win_prob <= 1),
  projected_home_goals numeric,
  projected_away_goals numeric,
  prediction_ts timestamptz not null default now(),
  feature_snapshot jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_world_cup_predictions_match on world_cup_match_predictions(match_id);
create index if not exists idx_world_cup_predictions_model on world_cup_match_predictions(model_name, model_version);
create index if not exists idx_world_cup_predictions_ts on world_cup_match_predictions(prediction_ts desc);

create table if not exists world_cup_team_probabilities (
  id uuid primary key default gen_random_uuid(),
  team text not null,
  group_name text,
  model_name text not null default 'sports_edge_world_cup',
  model_version text not null,
  simulation_ts timestamptz not null default now(),
  simulations int not null check (simulations > 0),
  bracket_source text not null,
  rating numeric,
  round_of_32_prob numeric not null check (round_of_32_prob >= 0 and round_of_32_prob <= 1),
  round_of_16_prob numeric not null check (round_of_16_prob >= 0 and round_of_16_prob <= 1),
  quarterfinal_prob numeric not null check (quarterfinal_prob >= 0 and quarterfinal_prob <= 1),
  semifinal_prob numeric not null check (semifinal_prob >= 0 and semifinal_prob <= 1),
  final_prob numeric not null check (final_prob >= 0 and final_prob <= 1),
  champion_prob numeric not null check (champion_prob >= 0 and champion_prob <= 1),
  group_rank_probs jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create unique index if not exists idx_world_cup_team_probs_unique
  on world_cup_team_probabilities(team, model_name, model_version, simulation_ts);
create index if not exists idx_world_cup_team_probs_champion
  on world_cup_team_probabilities(champion_prob desc);
create index if not exists idx_world_cup_team_probs_group
  on world_cup_team_probabilities(group_name);

create table if not exists world_cup_model_runs (
  id uuid primary key default gen_random_uuid(),
  model_name text not null default 'sports_edge_world_cup',
  model_version text not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  rows_written int,
  simulations int,
  bracket_source text,
  success boolean,
  error_text text,
  created_at timestamptz not null default now()
);

create index if not exists idx_world_cup_model_runs_started
  on world_cup_model_runs(started_at desc);

create or replace view world_cup_matches_enriched
with (security_invoker = true) as
select
  m.id as match_id,
  m.external_match_id,
  m.tournament,
  m.season,
  m.stage,
  m.group_name,
  m.kickoff_utc,
  m.home_team,
  m.away_team,
  m.venue,
  m.status,
  m.home_score,
  m.away_score,
  p.model_name,
  p.model_version,
  p.home_win_prob,
  p.draw_prob,
  p.away_win_prob,
  p.home_knockout_win_prob,
  p.away_knockout_win_prob,
  p.projected_home_goals,
  p.projected_away_goals,
  p.prediction_ts
from world_cup_matches m
left join lateral (
  select *
  from world_cup_match_predictions p
  where p.match_id = m.id
  order by p.prediction_ts desc
  limit 1
) p on true;

alter table world_cup_matches enable row level security;
alter table world_cup_match_predictions enable row level security;
alter table world_cup_team_probabilities enable row level security;
alter table world_cup_model_runs enable row level security;

drop policy if exists "public read world cup matches" on world_cup_matches;
drop policy if exists "public read world cup predictions" on world_cup_match_predictions;
drop policy if exists "public read world cup team probabilities" on world_cup_team_probabilities;
drop policy if exists "public read world cup model runs" on world_cup_model_runs;

create policy "public read world cup matches" on world_cup_matches for select using (true);
create policy "public read world cup predictions" on world_cup_match_predictions for select using (true);
create policy "public read world cup team probabilities" on world_cup_team_probabilities for select using (true);
create policy "public read world cup model runs" on world_cup_model_runs for select using (true);

grant select on world_cup_matches to anon, authenticated;
grant select on world_cup_match_predictions to anon, authenticated;
grant select on world_cup_team_probabilities to anon, authenticated;
grant select on world_cup_model_runs to anon, authenticated;
grant select on world_cup_matches_enriched to anon, authenticated;
