-- Migration: persist per-game model explanations for dashboard serving.

create table if not exists game_explanations (
  id uuid primary key default gen_random_uuid(),
  game_id text not null,
  league text check (league in ('NFL', 'NBA', 'MLB', 'PGA', 'CBB')) not null,
  model_version text not null,
  prediction_ts timestamptz not null,
  top_features jsonb not null default '[]'::jsonb,
  injury_adjusted boolean not null default false,
  home_injury_delta numeric,
  away_injury_delta numeric,
  base_vs_adjusted jsonb,
  created_at timestamptz not null default now()
);

create unique index if not exists idx_game_explanations_unique
  on game_explanations(game_id, model_version, prediction_ts);

create index if not exists idx_game_explanations_league_ts
  on game_explanations(league, prediction_ts desc);

alter table game_explanations enable row level security;

drop policy if exists "public read game explanations" on game_explanations;
create policy "public read game explanations"
  on game_explanations for select using (true);

comment on table game_explanations is 'Top feature drivers and injury flags for dashboard game detail pages.';
