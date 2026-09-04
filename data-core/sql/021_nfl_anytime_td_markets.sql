-- NFL anytime-touchdown prediction and odds serving surface.

create table if not exists nfl_anytime_td_predictions (
  id uuid primary key default gen_random_uuid(),
  game_id uuid not null references games(id) on delete cascade,
  season int not null,
  week int not null,
  game_date date not null,
  player_id text not null,
  player_name text not null,
  normalized_player_name text not null,
  team text not null,
  opponent text not null,
  position text not null,
  td_probability numeric not null check (td_probability between 0 and 1),
  sample_games int not null default 0,
  model_version text not null,
  prediction_ts timestamptz not null,
  quality_flags jsonb not null default '[]'::jsonb,
  feature_snapshot jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_nfl_anytime_td_predictions_game
  on nfl_anytime_td_predictions(game_id, td_probability desc);
create index if not exists idx_nfl_anytime_td_predictions_date
  on nfl_anytime_td_predictions(game_date, td_probability desc);
create unique index if not exists idx_nfl_anytime_td_prediction_identity
  on nfl_anytime_td_predictions(game_id, player_id, model_version, prediction_ts);

create table if not exists nfl_anytime_td_odds_snapshots (
  id uuid primary key default gen_random_uuid(),
  game_id uuid not null references games(id) on delete cascade,
  provider_event_id text not null,
  player_name text not null,
  normalized_player_name text not null,
  market text not null default 'player_anytime_td',
  book text not null,
  book_title text,
  price numeric not null,
  implied_probability numeric not null check (implied_probability between 0 and 1),
  last_update timestamptz,
  snapshot_ts timestamptz not null,
  source text not null default 'the_odds_api',
  raw_record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_nfl_anytime_td_odds_player
  on nfl_anytime_td_odds_snapshots(game_id, normalized_player_name, snapshot_ts desc);
create index if not exists idx_nfl_anytime_td_odds_event
  on nfl_anytime_td_odds_snapshots(provider_event_id, snapshot_ts desc);

create or replace view nfl_anytime_td_edges_latest
with (security_invoker = true) as
with latest_predictions as (
  select distinct on (p.game_id, p.player_id)
    p.*
  from nfl_anytime_td_predictions p
  join games g on g.id = p.game_id
  where g.game_time_utc >= now() - interval '3 hours'
  order by p.game_id, p.player_id, p.prediction_ts desc
),
latest_book_prices as (
  select distinct on (o.game_id, o.normalized_player_name, o.book)
    o.*
  from nfl_anytime_td_odds_snapshots o
  order by o.game_id, o.normalized_player_name, o.book,
    o.snapshot_ts desc, o.last_update desc, o.created_at desc
),
best_prices as (
  select distinct on (o.game_id, o.normalized_player_name)
    o.*
  from latest_book_prices o
  order by o.game_id, o.normalized_player_name, o.price desc, o.snapshot_ts desc
)
select
  p.*,
  g.game_time_utc,
  g.home_team,
  g.away_team,
  o.book as best_book,
  o.book_title as best_book_title,
  o.price as best_price,
  o.implied_probability as market_probability,
  p.td_probability - o.implied_probability as edge,
  case
    when o.price is null then null
    when o.price > 0 then p.td_probability * (o.price / 100.0) - (1.0 - p.td_probability)
    else p.td_probability * (100.0 / abs(o.price)) - (1.0 - p.td_probability)
  end as ev,
  case
    when o.price is null then null
    else greatest(
      0.0,
      (
        (
          case when o.price > 0 then o.price / 100.0 else 100.0 / abs(o.price) end
        ) * p.td_probability - (1.0 - p.td_probability)
      ) /
      nullif(
        case when o.price > 0 then o.price / 100.0 else 100.0 / abs(o.price) end,
        0
      ) / 4.0
    )
  end as quarter_kelly,
  o.snapshot_ts as odds_snapshot_ts,
  case
    when o.id is null then 'missing'
    when o.snapshot_ts < now() - interval '36 hours' then 'stale'
    else 'priced'
  end as odds_status
from latest_predictions p
join games g on g.id = p.game_id
left join best_prices o
  on o.game_id = p.game_id
 and o.normalized_player_name = p.normalized_player_name;

alter table nfl_anytime_td_predictions enable row level security;
alter table nfl_anytime_td_odds_snapshots enable row level security;

drop policy if exists "public read nfl anytime td predictions" on nfl_anytime_td_predictions;
drop policy if exists "public read nfl anytime td odds" on nfl_anytime_td_odds_snapshots;
create policy "public read nfl anytime td predictions"
  on nfl_anytime_td_predictions for select using (true);
create policy "public read nfl anytime td odds"
  on nfl_anytime_td_odds_snapshots for select using (true);

grant select on nfl_anytime_td_predictions to anon, authenticated;
grant select on nfl_anytime_td_odds_snapshots to anon, authenticated;
grant select on nfl_anytime_td_edges_latest to anon, authenticated;
