-- College-football schedule, prediction, odds, and research recommendation serving.

create table if not exists cfb_games (
  event_id text primary key,
  season int not null,
  week int,
  game_time_utc timestamptz not null,
  game_date date not null,
  home_team_id text not null,
  away_team_id text not null,
  home_team text not null,
  away_team text not null,
  venue text,
  neutral_site boolean not null default false,
  status text not null,
  home_score numeric,
  away_score numeric,
  raw_record jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create index if not exists idx_cfb_games_date on cfb_games(game_date, game_time_utc);

create table if not exists cfb_team_predictions (
  event_id text not null references cfb_games(event_id) on delete cascade,
  model_version text not null,
  model_status text not null default 'research',
  predicted_home_points numeric not null,
  predicted_away_points numeric not null,
  predicted_margin numeric not null,
  predicted_total numeric not null,
  home_win_probability numeric not null check (home_win_probability between 0 and 1),
  margin_sigma numeric not null,
  total_sigma numeric not null,
  confidence numeric check (confidence between 0 and 1),
  quality_flags jsonb not null default '[]'::jsonb,
  feature_snapshot jsonb not null default '{}'::jsonb,
  prediction_ts timestamptz not null,
  created_at timestamptz not null default now(),
  primary key (event_id, model_version)
);

create table if not exists cfb_odds_snapshots (
  id uuid primary key default gen_random_uuid(),
  event_id text not null references cfb_games(event_id) on delete cascade,
  provider_event_id text not null,
  book text not null,
  book_title text,
  market text not null check (market in ('moneyline', 'spread', 'total')),
  selection text not null check (selection in ('home', 'away', 'over', 'under')),
  line numeric,
  price numeric not null,
  implied_probability numeric not null check (implied_probability between 0 and 1),
  last_update timestamptz,
  snapshot_ts timestamptz not null,
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_cfb_odds_event_market
  on cfb_odds_snapshots(event_id, market, selection, snapshot_ts desc);

create table if not exists cfb_market_recommendations (
  event_id text not null references cfb_games(event_id) on delete cascade,
  model_version text not null,
  market text not null check (market in ('moneyline', 'spread', 'total')),
  selection text not null check (selection in ('home', 'away', 'over', 'under')),
  subject text not null,
  book text not null,
  book_title text,
  line numeric,
  price numeric not null,
  model_probability numeric not null check (model_probability between 0 and 1),
  implied_probability numeric not null check (implied_probability between 0 and 1),
  edge numeric not null,
  ev numeric not null,
  quarter_kelly numeric,
  confidence numeric check (confidence between 0 and 1),
  quality_flags jsonb not null default '[]'::jsonb,
  prediction_ts timestamptz not null,
  odds_snapshot_ts timestamptz not null,
  updated_at timestamptz not null default now(),
  primary key (event_id, model_version, market, selection)
);

create or replace view cfb_market_edges_latest
with (security_invoker = true) as
select
  r.event_id,
  g.season,
  g.week,
  g.game_date,
  g.game_time_utc,
  g.home_team,
  g.away_team,
  g.venue,
  g.neutral_site,
  r.model_version,
  p.model_status,
  p.predicted_home_points,
  p.predicted_away_points,
  p.predicted_margin,
  p.predicted_total,
  p.home_win_probability,
  r.market,
  r.selection,
  r.subject,
  r.book,
  r.book_title,
  r.line,
  r.price,
  r.model_probability,
  r.implied_probability,
  r.edge,
  r.ev,
  r.quarter_kelly,
  r.confidence,
  r.quality_flags,
  r.prediction_ts,
  r.odds_snapshot_ts,
  case
    when r.odds_snapshot_ts < now() - interval '24 hours' then 'stale'
    else 'priced'
  end as odds_status
from cfb_market_recommendations r
join cfb_games g on g.event_id = r.event_id
join cfb_team_predictions p
  on p.event_id = r.event_id and p.model_version = r.model_version
where g.game_time_utc >= now() - interval '3 hours';

create or replace view cfb_team_predictions_latest
with (security_invoker = true) as
select
  p.event_id,
  g.season,
  g.week,
  g.game_date,
  g.game_time_utc,
  g.home_team,
  g.away_team,
  g.venue,
  g.neutral_site,
  p.model_version,
  p.model_status,
  p.predicted_home_points,
  p.predicted_away_points,
  p.predicted_margin,
  p.predicted_total,
  p.home_win_probability,
  p.confidence,
  p.quality_flags,
  p.prediction_ts
from cfb_team_predictions p
join cfb_games g on g.event_id = p.event_id
where g.game_time_utc >= now() - interval '3 hours';

alter table cfb_games enable row level security;
alter table cfb_team_predictions enable row level security;
alter table cfb_odds_snapshots enable row level security;
alter table cfb_market_recommendations enable row level security;

drop policy if exists "public read cfb games" on cfb_games;
drop policy if exists "public read cfb predictions" on cfb_team_predictions;
drop policy if exists "public read cfb odds" on cfb_odds_snapshots;
drop policy if exists "public read cfb recommendations" on cfb_market_recommendations;
create policy "public read cfb games" on cfb_games for select using (true);
create policy "public read cfb predictions" on cfb_team_predictions for select using (true);
create policy "public read cfb odds" on cfb_odds_snapshots for select using (true);
create policy "public read cfb recommendations" on cfb_market_recommendations for select using (true);

grant select on cfb_games to anon, authenticated;
grant select on cfb_team_predictions to anon, authenticated;
grant select on cfb_odds_snapshots to anon, authenticated;
grant select on cfb_market_recommendations to anon, authenticated;
grant select on cfb_market_edges_latest to anon, authenticated;
grant select on cfb_team_predictions_latest to anon, authenticated;
