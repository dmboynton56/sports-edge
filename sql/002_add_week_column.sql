-- Migration: ensure season + week columns exist on games table for easier filtering

alter table games
  add column if not exists season int,
  add column if not exists week smallint check (week between 1 and 30);

comment on column games.season is 'Season year (e.g., 2025).';

create index if not exists idx_games_league_season_week on games(league, season, week);

comment on column games.week is 'League-specific week number (1-30).';

-- Refresh dependent view so callers can access the new column
create or replace view games_today_enriched as
select
  g.id as game_id,
  g.league,
  g.season,
  g.week,
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
