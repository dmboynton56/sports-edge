-- Add batter HR recency context to MLB home run prediction serving rows.

alter table mlb_home_run_predictions
  add column if not exists games_since_last_hr int,
  add column if not exists last_hr_date date;

create or replace view mlb_home_run_predictions_latest
with (security_invoker = true) as
select distinct on (game_date, game_id, player_id, model_version)
  *
from mlb_home_run_predictions
where game_date = ((now() at time zone 'America/Denver')::date)
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
  where game_date = ((now() at time zone 'America/Denver')::date)
  order by game_date, game_id, player_id, model_version, prediction_ts desc
),
latest_odds as (
  select distinct on (game_date, game_id, normalized_player_name, market, line, side, book)
    *
  from mlb_home_run_odds_snapshots
  where game_date = ((now() at time zone 'America/Denver')::date)
    and market = 'batter_home_runs'
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
    on u.game_id = o.game_id
   and u.normalized_player_name = o.normalized_player_name
   and u.market = o.market
   and u.line = o.line
   and u.book = o.book
   and lower(u.side) = 'under'
  where lower(o.side) = 'over'
),
best_over as (
  select distinct on (game_id, normalized_player_name, market, line)
    *
  from paired_over
  order by game_id, normalized_player_name, market, line, price desc nulls last, snapshot_ts desc
),
book_counts as (
  select
    game_id,
    normalized_player_name,
    market,
    line,
    count(distinct book) as odds_books_count,
    max(snapshot_ts) as odds_snapshot_ts
  from paired_over
  group by game_id, normalized_player_name, market, line
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
    on b.game_id = p.game_id
   and b.normalized_player_name = p.normalized_player_name
   and b.market = 'batter_home_runs'
   and b.line = 0.5
  left join book_counts c
    on c.game_id = p.game_id
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
