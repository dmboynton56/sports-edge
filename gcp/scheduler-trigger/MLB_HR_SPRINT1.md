# Trusted MLB HR scheduler

The player-market workflow is the canonical MLB home-run path. Configure a
single Cloud Scheduler job in Mountain Time so daylight-saving transitions are
handled by Cloud Scheduler rather than by UTC arithmetic:

```bash
gcloud scheduler jobs create http sports-edge-mlb-hr-refresh \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --schedule "30 6,14 * * *" \
  --time-zone "America/Denver" \
  --uri "$TRIGGER_URL/dispatch/player-markets-refresh" \
  --http-method POST \
  --headers "Content-Type=application/json" \
  --message-body '{"inputs":{"run_pga":"false","run_mlb_hr":"true","train_mlb_hr":"false","sync_supabase":"true","sync_bigquery":"true","run_window":"auto","apply_trusted_board_migration":"false"}}' \
  --oidc-service-account-email "$SCHEDULER_SA" \
  --oidc-token-audience "$TRIGGER_URL"
```

Use `apply_trusted_board_migration=true` only for the one staging/rollout run
that applies `data-core/supabase/migrations/20260811144616_mlb_hr_trusted_board.sql`.
After that, leave it false; the workflow remains idempotent and the migration
is managed as a versioned Supabase change.

## Release checklist

Use this sequence for the first staging and production rollout. Keep the web
flag explicitly disabled until the shadow-run gate is complete:

```text
[ ] Confirm the staging Supabase URL, database credentials, GitHub secrets,
    Cloud Run trigger URL, scheduler service account, and Discord webhook.
[ ] Apply the additive trusted-board migration to staging.
[ ] Run Supabase database/security advisors and inspect the current-board and
    recent-results query plans.
[ ] Exercise morning, afternoon, no-slate, stale-odds, provider-failure, and
    partial-pricing staging cases.
[ ] Confirm the board views expose only the latest completed run and that
    historical rows retain their original odds snapshot.
[ ] Run the scheduler in shadow mode with
    MLB_HR_TRUSTED_BOARD_ENABLED=false.
[ ] Record four consecutive healthy scheduled runs across two MLB game dates.
    Each run must have the correct Denver slate date, no duplicate candidate
    keys, no future odds timestamps, and top-25 pricing coverage >= 80%.
[ ] Enable the trusted board in the production web environment.
[ ] Verify the next grading run resolves >= 99% of eligible official rows;
    alert unresolved rows after 18 hours and leave them pending.
[ ] Record the rollback action: set MLB_HR_TRUSTED_BOARD_ENABLED=false.
```

The migration is intentionally additive. Do not remove or rewrite the legacy
prediction and odds snapshots during this rollout; they remain upstream and
audit sources.
