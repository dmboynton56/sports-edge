-- Failed refreshes are audit records, not public board candidates.  In
-- particular, an empty publish rejected by manage_mlb_hr_board_run.py is
-- finalized as failed and must leave the last good snapshot authoritative.
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
where status in ('healthy', 'partial', 'no_slate')
order by slate_date, completed_at desc nulls last, started_at desc;
