-- Migration: store league schedule date separately from UTC start time.
--
-- Late MLB/NBA/NFL games can start after midnight UTC while still belonging
-- to the prior local slate. The portfolio view should filter by schedule date,
-- not by the UTC timestamp date.

alter table games
  add column if not exists game_date date;

update games
set game_date = (game_time_utc at time zone 'America/Denver')::date
where game_date is null;

comment on column games.game_date is 'League/local schedule date used for daily serving windows.';

create index if not exists idx_games_league_game_date on games(league, game_date);

create or replace view games_today_enriched
with (security_invoker = true) as
select
  g.id as game_id,
  g.league,
  g.season,
  g.week,
  g.game_time_utc,
  g.home_team,
  g.away_team,
  coalesce(g.book_spread, o.line) as book_spread,
  p.my_spread,
  (p.my_spread - coalesce(g.book_spread, o.line)) as edge_pts,
  p.my_home_win_prob,
  p.model_version,
  p.asof_ts as prediction_ts,
  o.snapshot_ts as odds_ts,
  coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date) as game_date,
  g.home_probable_pitcher,
  g.away_probable_pitcher
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
where coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date)
  = (now() at time zone 'America/Denver')::date
order by g.game_time_utc;
