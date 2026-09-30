-- Shared once-per-Denver-day The Odds API budget for MLB.
--
-- Player Market Refresh (HR props) and Daily Refresh (research ML / run-line /
-- totals) share this marker so they cannot each spend Starter-plan credits on
-- the same America/Denver calendar day.
--
-- One row per denver_date. `source` records which path claimed the session
-- (`mlb_hr` or `mlb_research`). PropLine usage is not recorded here — this
-- table is only stamped when The Odds API is actually called.

create table if not exists odds_api_usage (
  denver_date date primary key,
  used_at timestamptz not null default now(),
  source text not null check (source in ('mlb_hr', 'mlb_research')),
  notes text
);

comment on table odds_api_usage is
  'At most one The Odds API session per America/Denver day across MLB HR and research ML/run-line/totals.';

comment on column odds_api_usage.denver_date is
  'America/Denver calendar date the Odds API session was consumed.';

comment on column odds_api_usage.source is
  'Which MLB path claimed the day''s Odds budget: mlb_hr or mlb_research.';

alter table odds_api_usage enable row level security;

-- Account-wide credit reservations and response reuse. Backend access only;
-- raw provider responses and budget operations are not exposed to browsers.
create table if not exists odds_api_request_cache (
  request_key text primary key,
  budget_month date not null,
  source text not null check (source in ('nfl', 'nba', 'mlb', 'cfb', 'pga')),
  cost integer not null check (cost >= 0),
  status text not null check (status in ('reserved', 'complete', 'failed')),
  payload jsonb,
  headers jsonb not null default '{}'::jsonb,
  fetched_at timestamptz,
  created_at timestamptz not null default now()
);
create index if not exists idx_odds_api_request_month_source
  on odds_api_request_cache(budget_month, source);
alter table odds_api_request_cache enable row level security;
revoke all on odds_api_request_cache from anon, authenticated;
comment on table odds_api_request_cache is
  'Starter quota reservations and original provider snapshots; API keys are never stored.';
