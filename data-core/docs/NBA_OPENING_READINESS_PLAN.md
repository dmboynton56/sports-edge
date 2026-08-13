# NBA Opening Readiness Plan

Generated: 2026-07-12

The 2025-26 NBA regular season typically opens in mid-October. Use this plan as the
go/no-go checklist for the Sports Edge dashboard and serving pipeline.

## Objective

Have opening-week NBA predictions ready, explainable, and safe to publish once
schedules, rosters, injury context, and market lines are stable.

## Target Timeline

### T-7 Days

- Confirm `plan_daily_refresh.py` returns `run_nba=true` for opening week dates.
- Dry-run `refresh_nba --injury-aware --include-explanations`.
- Verify opening-week games exist in BigQuery `raw_schedules`.
- Confirm feature snapshots include current-season context (not empty form windows).

### T-3 Days

- Verify market spreads attach via `sync_odds.py --league NBA`.
- Run `audit_season_readiness.py --league NBA --json`.
- Review top feature drivers on `/nba/[gameId]` pages.

### T-1 Day

- Run `validate_supabase_sync.py --strict`.
- Run `dashboard_predeploy_gate.py --league NBA`.
- Confirm Vercel `/nba` shows fresh badges for all published games.

## Commands

```bash
cd data-core
python scripts/plan_daily_refresh.py --date 2026-10-20 --lookahead-days 7
python -m src.pipeline.refresh_nba \
  --project "$GCP_PROJECT_ID" \
  --model-version v3 \
  --date 2026-10-22 \
  --injury-aware \
  --include-explanations
python scripts/sync_bq_to_supabase.py --project "$GCP_PROJECT_ID" --league NBA --append
python scripts/sync_explanations_to_supabase.py --league NBA
python scripts/audit_season_readiness.py --league NBA --json
python scripts/dashboard_predeploy_gate.py --league NBA
```

## Go/No-Go Criteria

Publish opening-week picks only if:

- Every published game has one current prediction row.
- Book spreads are present or explicitly labeled unavailable.
- Explanation rows exist for published games (or UI shows graceful empty state).
- `validate_supabase_sync.py --strict` passes.
- `audit_season_readiness.py` reports prediction coverage for the serving window.
