-- Add batter HR recency context to MLB home run prediction serving rows.

drop view if exists mlb_home_run_edges_latest;
drop view if exists mlb_home_run_predictions_latest;

alter table mlb_home_run_predictions
  add column if not exists games_since_last_hr int,
  add column if not exists last_hr_date date,
  add column if not exists v1_probability numeric check (v1_probability is null or (v1_probability >= 0 and v1_probability <= 1)),
  add column if not exists v1_rank int,
  add column if not exists statcast_probability numeric check (statcast_probability is null or (statcast_probability >= 0 and statcast_probability <= 1)),
  add column if not exists statcast_rank int,
  add column if not exists statcast_available boolean,
  add column if not exists model_agreement text,
  add column if not exists consensus_score numeric,
  add column if not exists market_signal_rank int;

with latest_v1 as (
  select distinct on (game_date, game_id, player_id)
    *
  from mlb_home_run_predictions
  where model_version like 'mlb-hr-v1%'
  order by game_date, game_id, player_id, prediction_ts desc
),
latest_statcast as (
  select distinct on (game_date, game_id, player_id)
    *
  from mlb_home_run_predictions
  where model_version = 'mlb-hr-torch-statcast-v1-blend'
  order by game_date, game_id, player_id, prediction_ts desc
),
slate_counts as (
  select game_date, count(*)::int as candidate_count
  from latest_v1
  group by game_date
),
comparison as (
  select
    v.game_date,
    v.game_id,
    v.player_id,
    v.hr_probability as v1_probability,
    v.rank as v1_rank,
    s.hr_probability as statcast_probability,
    s.rank as statcast_rank,
    case
      when s.id is null then null
      else not (s.quality_flags ? 'statcast_features_unavailable')
    end as statcast_available,
    case
      when s.id is null then 'V1 only'
      when s.quality_flags ? 'statcast_features_unavailable' then 'Missing Statcast'
      when s.rank - v.rank <= -5 then 'Statcast boost'
      when s.rank - v.rank >= 5 then 'Statcast fade'
      else 'Consensus'
    end as model_agreement,
    (
      v.rank
      + coalesce(s.rank, c.candidate_count + 25)
      + c.candidate_count + 1
      + jsonb_array_length(v.quality_flags) * 3
      + case when s.id is not null and s.quality_flags ? 'statcast_features_unavailable' then 6 else 0 end
    )::numeric as consensus_score,
    c.candidate_count + 1 as market_signal_rank
  from latest_v1 v
  join slate_counts c on c.game_date = v.game_date
  left join latest_statcast s
    on s.game_date = v.game_date
   and s.game_id = v.game_id
   and s.player_id = v.player_id
)
update mlb_home_run_predictions p
set
  v1_probability = c.v1_probability,
  v1_rank = c.v1_rank,
  statcast_probability = c.statcast_probability,
  statcast_rank = c.statcast_rank,
  statcast_available = c.statcast_available,
  model_agreement = c.model_agreement,
  consensus_score = c.consensus_score,
  market_signal_rank = c.market_signal_rank
from comparison c
where p.game_date = c.game_date
  and p.game_id = c.game_id
  and p.player_id = c.player_id
  and p.model_version in ('mlb-hr-v1', 'mlb-hr-torch-statcast-v1-blend');

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

grant select on mlb_home_run_predictions_latest to anon, authenticated;
grant select on mlb_home_run_edges_latest to anon, authenticated;
