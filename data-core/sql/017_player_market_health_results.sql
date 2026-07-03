-- Player-market health and graded result surfaces.
--
-- Keep shared portfolio tables intact. These additions are dashboard-serving
-- caches and views only.

drop view if exists mlb_home_run_edges_latest;
drop view if exists mlb_home_run_predictions_latest;

alter table mlb_home_run_predictions
  add column if not exists statcast_coverage numeric check (statcast_coverage is null or (statcast_coverage >= 0 and statcast_coverage <= 1)),
  add column if not exists statcast_ready_rows int,
  add column if not exists statcast_total_rows int,
  add column if not exists statcast_artifact_loaded boolean;

create index if not exists idx_mlb_hr_predictions_statcast_health
  on mlb_home_run_predictions(game_date, statcast_coverage);

create table if not exists mlb_home_run_results (
  id uuid primary key default gen_random_uuid(),
  game_id text not null,
  game_date date not null,
  player_id text not null,
  player_name text not null,
  team text,
  opponent text,
  model_version text not null,
  prediction_ts timestamptz,
  rank int,
  top_k_bucket text,
  model_probability numeric check (model_probability is null or (model_probability >= 0 and model_probability <= 1)),
  actual_home_run boolean,
  actual_home_runs int,
  actual_plate_appearances int,
  evaluated_at timestamptz not null default now(),
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists idx_mlb_hr_results_unique
  on mlb_home_run_results(game_id, player_id, model_version, prediction_ts);
create index if not exists idx_mlb_hr_results_date_model
  on mlb_home_run_results(game_date, model_version, rank);

create table if not exists pga_prediction_results (
  id uuid primary key default gen_random_uuid(),
  event_key text not null,
  season int,
  player_name text not null,
  player_id text,
  model_version text not null,
  prediction_ts timestamptz,
  win_prob numeric check (win_prob is null or (win_prob >= 0 and win_prob <= 1)),
  top10_prob numeric check (top10_prob is null or (top10_prob >= 0 and top10_prob <= 1)),
  top20_prob numeric check (top20_prob is null or (top20_prob >= 0 and top20_prob <= 1)),
  make_cut_prob numeric check (make_cut_prob is null or (make_cut_prob >= 0 and make_cut_prob <= 1)),
  final_position text,
  final_position_numeric int,
  final_score text,
  made_cut boolean,
  top10_hit boolean,
  top20_hit boolean,
  winner_hit boolean,
  evaluated_at timestamptz not null default now(),
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists idx_pga_prediction_results_unique
  on pga_prediction_results(event_key, player_name, model_version, prediction_ts);
create index if not exists idx_pga_prediction_results_event
  on pga_prediction_results(event_key, final_position_numeric);

create or replace view game_prediction_results
with (security_invoker = true) as
with latest_predictions as (
  select distinct on (p.game_id, p.model_version)
    p.*
  from model_predictions p
  order by p.game_id, p.model_version, p.asof_ts desc
),
latest_spread_odds as (
  select distinct on (o.game_id)
    o.game_id,
    o.line,
    o.snapshot_ts
  from odds_snapshots o
  where o.market = 'spread'
  order by o.game_id, o.snapshot_ts desc
)
select
  g.id::text as game_id,
  g.league,
  g.season,
  g.week,
  coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date) as game_date,
  g.game_time_utc,
  g.home_team,
  g.away_team,
  g.home_score,
  g.away_score,
  (g.home_score - g.away_score) as actual_margin,
  coalesce(g.book_spread, o.line) as book_spread,
  p.my_spread,
  p.my_home_win_prob,
  p.model_name,
  p.model_version,
  p.asof_ts as prediction_ts,
  case
    when g.home_score is null or g.away_score is null or p.my_spread is null then null
    when abs((g.home_score - g.away_score) + p.my_spread) < 0.001 then 'push'
    when ((g.home_score - g.away_score) + p.my_spread) > 0 then 'win'
    else 'loss'
  end as spread_result,
  case
    when g.home_score is null or g.away_score is null or p.my_home_win_prob is null then null
    when ((p.my_home_win_prob >= 0.5) and (g.home_score > g.away_score))
      or ((p.my_home_win_prob < 0.5) and (g.away_score > g.home_score)) then 'win'
    else 'loss'
  end as winner_result,
  case
    when g.home_score is null or g.away_score is null or p.my_spread is null then null
    when abs((g.home_score - g.away_score) + p.my_spread) < 0.001 then 0
    when ((g.home_score - g.away_score) + p.my_spread) > 0 then 100.0 / 110.0
    else -1
  end as flat_ats_units
from games g
join latest_predictions p on p.game_id = g.id
left join latest_spread_odds o on o.game_id = g.id
where g.home_score is not null
  and g.away_score is not null;

create or replace view mlb_home_run_predictions_latest
with (security_invoker = true) as
select distinct on (game_date, game_id, player_id, model_version)
  *
from mlb_home_run_predictions
order by game_date, game_id, player_id, model_version, prediction_ts desc;

create or replace view mlb_home_run_edges_latest
with (security_invoker = true) as
with latest_predictions as (
  select distinct on (game_date, game_id, player_id, model_version)
    *,
    trim(regexp_replace(
      regexp_replace(lower(player_name), '(^|[ .])(jr|sr|iii|ii|iv|v)\.?($|[ .])', ' ', 'g'),
      '[^a-z0-9]+',
      ' ',
      'g'
    )) as normalized_player_name
  from mlb_home_run_predictions
  order by game_date, game_id, player_id, model_version, prediction_ts desc
),
latest_odds as (
  select distinct on (game_date, game_id, normalized_player_name, market, line, side, book)
    *
  from mlb_home_run_odds_snapshots
  where market = 'batter_home_runs'
    and line = 0.5
  order by game_date, game_id, normalized_player_name, market, line, side, book,
    snapshot_ts desc, last_update desc, created_at desc
),
paired_over as (
  select
    o.*,
    u.implied_probability as under_implied_probability,
    case
      when o.implied_probability is not null
        and u.implied_probability is not null
        and (o.implied_probability + u.implied_probability) > 0
      then o.implied_probability / (o.implied_probability + u.implied_probability)
      else null
    end as no_vig_probability
  from latest_odds o
  left join latest_odds u
    on u.game_date = o.game_date
   and u.game_id = o.game_id
   and u.normalized_player_name = o.normalized_player_name
   and u.market = o.market
   and u.line = o.line
   and u.book = o.book
   and lower(u.side) = 'under'
  where lower(o.side) = 'over'
),
best_over as (
  select distinct on (game_date, game_id, normalized_player_name, market, line)
    *
  from paired_over
  order by game_date, game_id, normalized_player_name, market, line, price desc nulls last, snapshot_ts desc
),
book_counts as (
  select
    game_date,
    game_id,
    normalized_player_name,
    market,
    line,
    count(distinct book) as odds_books_count,
    max(snapshot_ts) as odds_snapshot_ts
  from paired_over
  group by game_date, game_id, normalized_player_name, market, line
),
priced as (
  select
    p.*,
    b.book as best_book,
    b.book_title as best_book_title,
    b.price as best_price,
    b.implied_probability,
    b.no_vig_probability,
    b.snapshot_ts as best_snapshot_ts,
    c.odds_books_count,
    c.odds_snapshot_ts,
    case
      when b.price > 0 then (b.price / 100.0) + 1.0
      when b.price < 0 then (100.0 / abs(b.price)) + 1.0
      else null
    end as decimal_price
  from latest_predictions p
  left join best_over b
    on b.game_date = p.game_date
   and b.game_id = p.game_id
   and b.normalized_player_name = p.normalized_player_name
   and b.market = 'batter_home_runs'
   and b.line = 0.5
  left join book_counts c
    on c.game_date = p.game_date
   and c.game_id = p.game_id
   and c.normalized_player_name = p.normalized_player_name
   and c.market = 'batter_home_runs'
   and c.line = 0.5
)
select
  *,
  coalesce(no_vig_probability, implied_probability) as market_probability,
  case
    when best_price is null then null
    else hr_probability - coalesce(no_vig_probability, implied_probability)
  end as edge,
  case
    when decimal_price is null then null
    else hr_probability * decimal_price - 1.0
  end as ev,
  case
    when decimal_price is null or decimal_price <= 1.0 then null
    else greatest(((hr_probability * (decimal_price - 1.0) - (1.0 - hr_probability)) / (decimal_price - 1.0)) * 0.25, 0.0)
  end as kelly,
  case
    when best_price is null then 'missing_odds'
    when no_vig_probability is null then 'raw_implied'
    else 'ok'
  end as odds_status
from priced;

alter table mlb_home_run_results enable row level security;
alter table pga_prediction_results enable row level security;

drop policy if exists "public read mlb home run results" on mlb_home_run_results;
drop policy if exists "public read pga prediction results" on pga_prediction_results;

create policy "public read mlb home run results" on mlb_home_run_results for select using (true);
create policy "public read pga prediction results" on pga_prediction_results for select using (true);

grant select on mlb_home_run_results to anon, authenticated;
grant select on pga_prediction_results to anon, authenticated;
grant select on game_prediction_results to anon, authenticated;
grant select on mlb_home_run_predictions_latest to anon, authenticated;
grant select on mlb_home_run_edges_latest to anon, authenticated;
