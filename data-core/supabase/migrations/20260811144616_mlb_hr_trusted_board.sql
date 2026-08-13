-- Trusted MLB home-run board serving contract.
--
-- This migration is additive. Prediction and odds tables remain upstream
-- snapshots; published rows below are immutable snapshots of what the public
-- board actually served at a point in time.

create table if not exists public.mlb_home_run_board_runs (
  run_id uuid primary key default gen_random_uuid(),
  run_key text not null unique,
  slate_date date not null,
  model_version text not null,
  run_window text not null check (run_window in ('morning', 'afternoon', 'manual')),
  status text not null default 'running'
    check (status in ('running', 'healthy', 'partial', 'failed', 'no_slate')),
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  workflow_url text,
  gaps jsonb not null default '[]'::jsonb,
  validation_summary jsonb not null default '{}'::jsonb,
  total_candidates integer not null default 0 check (total_candidates >= 0),
  priced_candidates integer not null default 0 check (priced_candidates >= 0),
  top25_denominator integer not null default 0 check (top25_denominator between 0 and 25),
  top25_priced_count integer not null default 0 check (top25_priced_count >= 0),
  top25_coverage numeric check (top25_coverage is null or (top25_coverage >= 0 and top25_coverage <= 1)),
  prediction_ts timestamptz,
  odds_ts timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint mlb_hr_board_run_counts_check
    check (priced_candidates <= total_candidates and top25_priced_count <= top25_denominator),
  constraint mlb_hr_board_run_coverage_check
    check ((top25_denominator = 0 and top25_coverage is null)
      or (top25_denominator > 0 and top25_coverage is not null))
);

create index if not exists idx_mlb_hr_board_runs_slate_completed
  on public.mlb_home_run_board_runs(slate_date, completed_at desc nulls last, started_at desc);
create index if not exists idx_mlb_hr_board_runs_status
  on public.mlb_home_run_board_runs(status, slate_date, completed_at desc nulls last);

create table if not exists public.mlb_home_run_board_rows (
  board_row_id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.mlb_home_run_board_runs(run_id) on delete restrict,
  slate_date date not null,
  game_id text not null,
  player_id text not null,
  player_name text not null,
  team text,
  opponent text,
  venue text,
  event_time timestamptz,
  lineup_slot integer,
  lineup_status text,
  opposing_probable_pitcher text,
  model_version text not null,
  model_probability numeric not null check (model_probability >= 0 and model_probability <= 1),
  baseline_probability numeric check (baseline_probability is null or (baseline_probability >= 0 and baseline_probability <= 1)),
  rank integer check (rank is null or rank > 0),
  book text,
  american_price integer check (american_price is null or american_price <> 0),
  raw_market_probability numeric check (raw_market_probability is null or (raw_market_probability >= 0 and raw_market_probability <= 1)),
  no_vig_market_probability numeric check (no_vig_market_probability is null or (no_vig_market_probability >= 0 and no_vig_market_probability <= 1)),
  market_probability numeric check (market_probability is null or (market_probability >= 0 and market_probability <= 1)),
  edge numeric,
  ev numeric,
  quarter_kelly numeric check (quarter_kelly is null or quarter_kelly >= 0),
  odds_snapshot_ts timestamptz,
  odds_status text not null default 'missing_odds'
    check (odds_status in ('ok', 'raw_implied', 'missing_odds', 'stale', 'invalid')),
  odds_books_count integer check (odds_books_count is null or odds_books_count >= 0),
  quality_flags jsonb not null default '[]'::jsonb,
  statcast_available boolean,
  statcast_coverage numeric check (statcast_coverage is null or (statcast_coverage >= 0 and statcast_coverage <= 1)),
  prediction_ts timestamptz,
  published_at timestamptz not null default now(),
  raw_record jsonb not null default '{}'::jsonb,
  constraint mlb_hr_board_row_priced_contract check (
    (odds_status in ('ok', 'raw_implied')
      and book is not null and american_price is not null
      and market_probability is not null and odds_snapshot_ts is not null)
    or odds_status in ('missing_odds', 'stale', 'invalid')
  ),
  constraint mlb_hr_board_rows_unique_key unique (run_id, model_version, game_id, player_id)
);

create index if not exists idx_mlb_hr_board_rows_run_rank
  on public.mlb_home_run_board_rows(run_id, rank nulls last, event_time);
create index if not exists idx_mlb_hr_board_rows_slate
  on public.mlb_home_run_board_rows(slate_date, model_version, event_time);
create index if not exists idx_mlb_hr_board_rows_identity
  on public.mlb_home_run_board_rows(game_id, player_id, model_version, published_at desc);

alter table if exists public.mlb_home_run_results
  add column if not exists board_row_id uuid;

do $$
begin
  if to_regclass('public.mlb_home_run_results') is not null
    and not exists (
      select 1
      from pg_constraint
      where conname = 'mlb_home_run_results_board_row_id_fkey'
        and conrelid = 'public.mlb_home_run_results'::regclass
    ) then
    alter table public.mlb_home_run_results
      add constraint mlb_home_run_results_board_row_id_fkey
      foreign key (board_row_id)
      references public.mlb_home_run_board_rows(board_row_id)
      on delete set null;
  end if;
end;
$$;

create index if not exists idx_mlb_hr_results_board_row
  on public.mlb_home_run_results(board_row_id);

-- The latest completed run for each slate is the only row set exposed to the
-- public board. Historical runs remain queryable for audit through the base
-- table and are never recomputed from newer odds snapshots.
create or replace view public.mlb_home_run_board_run_health
with (security_invoker = true) as
select distinct on (slate_date)
  run_id,
  run_key,
  slate_date,
  model_version,
  run_window,
  status,
  started_at,
  completed_at,
  workflow_url,
  gaps,
  validation_summary,
  total_candidates,
  priced_candidates,
  top25_denominator,
  top25_priced_count,
  top25_coverage,
  prediction_ts,
  odds_ts
from public.mlb_home_run_board_runs
where status <> 'running'
order by slate_date, completed_at desc nulls last, started_at desc;

create or replace view public.mlb_home_run_board_latest
with (security_invoker = true) as
select
  r.run_key,
  r.slate_date as run_slate_date,
  r.run_window,
  r.status as run_status,
  r.completed_at as run_completed_at,
  r.prediction_ts as run_prediction_ts,
  r.odds_ts as run_odds_ts,
  r.gaps as run_gaps,
  r.total_candidates as run_total_candidates,
  r.priced_candidates as run_priced_candidates,
  r.top25_denominator as run_top25_denominator,
  r.top25_priced_count as run_top25_priced_count,
  r.top25_coverage as run_top25_coverage,
  b.*
from public.mlb_home_run_board_run_health r
join public.mlb_home_run_board_rows b on b.run_id = r.run_id;

create or replace view public.mlb_home_run_published_results
with (security_invoker = true) as
select
  result.*,
  board.run_id,
  run.run_key,
  board.published_at,
  board.book,
  board.american_price,
  board.raw_market_probability,
  board.no_vig_market_probability,
  board.market_probability,
  board.edge,
  board.ev,
  board.quarter_kelly,
  board.odds_snapshot_ts,
  board.odds_status,
  board.quality_flags as board_quality_flags
from public.mlb_home_run_results result
left join public.mlb_home_run_board_rows board
  on board.board_row_id = result.board_row_id
left join public.mlb_home_run_board_runs run
  on run.run_id = board.run_id;

alter table public.mlb_home_run_board_runs enable row level security;
alter table public.mlb_home_run_board_rows enable row level security;

drop policy if exists "public read MLB HR board runs" on public.mlb_home_run_board_runs;
drop policy if exists "public read MLB HR board rows" on public.mlb_home_run_board_rows;

create policy "public read MLB HR board runs"
  on public.mlb_home_run_board_runs for select using (true);
create policy "public read MLB HR board rows"
  on public.mlb_home_run_board_rows for select using (true);

-- Explicit read-only grants are intentional: new Supabase projects no longer
-- expose newly-created public tables through the Data API implicitly, while
-- existing projects may still have broad default ACLs on public objects.
revoke all privileges on table
  public.mlb_home_run_board_runs,
  public.mlb_home_run_board_rows,
  public.mlb_home_run_results,
  public.mlb_home_run_board_run_health,
  public.mlb_home_run_board_latest,
  public.mlb_home_run_published_results
from anon, authenticated;

-- Keep service_role and the database owner available to the workflow; only
-- public Data API roles are intentionally restricted to reads.
grant select on public.mlb_home_run_board_runs to anon, authenticated;
grant select on public.mlb_home_run_board_rows to anon, authenticated;
grant select on public.mlb_home_run_results to anon, authenticated;
grant select on public.mlb_home_run_board_run_health to anon, authenticated;
grant select on public.mlb_home_run_board_latest to anon, authenticated;
grant select on public.mlb_home_run_published_results to anon, authenticated;
