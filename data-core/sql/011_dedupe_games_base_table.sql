-- Migration: collapse exact duplicate games and prevent future repeats.
--
-- This intentionally deduplicates by league + serving date + teams + UTC start.
-- It does not collapse same-day doubleheaders with different start times.

with ranked_games as (
  select
    g.id,
    first_value(g.id) over game_group as keeper_id,
    row_number() over game_group as rn
  from games g
  window game_group as (
    partition by
      g.league,
      coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date),
      g.home_team,
      g.away_team,
      g.game_time_utc
    order by
      (select count(*) from model_predictions p where p.game_id = g.id) desc,
      (select count(*) from odds_snapshots o where o.game_id = g.id) desc,
      (select count(*) from features f where f.game_id = g.id) desc,
      g.created_at desc,
      g.id desc
  )
),
duplicate_games as (
  select id, keeper_id
  from ranked_games
  where rn > 1
)
update model_predictions p
set game_id = d.keeper_id
from duplicate_games d
where p.game_id = d.id;

with ranked_games as (
  select
    g.id,
    first_value(g.id) over game_group as keeper_id,
    row_number() over game_group as rn
  from games g
  window game_group as (
    partition by
      g.league,
      coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date),
      g.home_team,
      g.away_team,
      g.game_time_utc
    order by
      (select count(*) from model_predictions p where p.game_id = g.id) desc,
      (select count(*) from odds_snapshots o where o.game_id = g.id) desc,
      (select count(*) from features f where f.game_id = g.id) desc,
      g.created_at desc,
      g.id desc
  )
),
duplicate_games as (
  select id, keeper_id
  from ranked_games
  where rn > 1
)
update odds_snapshots o
set game_id = d.keeper_id
from duplicate_games d
where o.game_id = d.id;

with ranked_games as (
  select
    g.id,
    first_value(g.id) over game_group as keeper_id,
    row_number() over game_group as rn
  from games g
  window game_group as (
    partition by
      g.league,
      coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date),
      g.home_team,
      g.away_team,
      g.game_time_utc
    order by
      (select count(*) from model_predictions p where p.game_id = g.id) desc,
      (select count(*) from odds_snapshots o where o.game_id = g.id) desc,
      (select count(*) from features f where f.game_id = g.id) desc,
      g.created_at desc,
      g.id desc
  )
),
duplicate_games as (
  select id, keeper_id
  from ranked_games
  where rn > 1
)
update features f
set game_id = d.keeper_id
from duplicate_games d
where f.game_id = d.id;

with ranked_games as (
  select
    g.id,
    row_number() over game_group as rn
  from games g
  window game_group as (
    partition by
      g.league,
      coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date),
      g.home_team,
      g.away_team,
      g.game_time_utc
    order by
      (select count(*) from model_predictions p where p.game_id = g.id) desc,
      (select count(*) from odds_snapshots o where o.game_id = g.id) desc,
      (select count(*) from features f where f.game_id = g.id) desc,
      g.created_at desc,
      g.id desc
  )
)
delete from games g
using ranked_games r
where g.id = r.id
  and r.rn > 1;

create unique index if not exists idx_games_unique_serving_matchup_time
  on games (
    league,
    (coalesce(game_date, (game_time_utc at time zone 'America/Denver')::date)),
    home_team,
    away_team,
    game_time_utc
  );

create or replace view games_today_enriched
with (security_invoker = true) as
with scoped_games as (
  select
    g.*,
    coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date) as serving_date
  from games g
  where coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date)
    = (now() at time zone 'America/Denver')::date
),
ranked_games as (
  select
    g.*,
    row_number() over (
      partition by g.league, g.serving_date, g.home_team, g.away_team, g.game_time_utc
      order by
        (select count(*) from model_predictions p where p.game_id = g.id) desc,
        g.created_at desc,
        g.id desc
    ) as rn
  from scoped_games g
)
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
  g.serving_date as game_date,
  g.home_probable_pitcher,
  g.away_probable_pitcher
from ranked_games g
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
where g.rn = 1
order by g.game_time_utc;
