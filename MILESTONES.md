# Sports Edge Dashboard v2 — Milestones

## M0 — Orchestration + spec
- [x] `TASK.md` — epic tracker
- [x] `MILESTONES.md` — this file
- [x] `AGENTS.md` — role routing
- [x] `data-core/docs/DASHBOARD_V2_SPEC.md` — page requirements

**Acceptance:** All four files exist; agent roles documented.

## M1 — Trustworthy serving + Vercel online
- [x] `data-core/scripts/audit_season_readiness.py`
- [x] `web/lib/data/team-markets.ts`
- [x] `web/app/nba/page.tsx`, `web/app/nfl/page.tsx`
- [x] `web/app/api/nba-slate/route.ts`, `web/app/api/nfl-slate/route.ts`
- [x] `web/components/leagues/TeamSpreadBoard.tsx`
- [x] Nav updates in `AppShell.tsx`, `markets/page.tsx`
- [x] `web/.env.example`, `web/vercel.json`
- [x] `validate_supabase_sync.py` NBA/NFL slate parity checks

**Acceptance:** `npm run build` passes; `/nba` and `/nfl` render; audit script produces JSON.

## M2 — Game explanations
- [x] `data-core/sql/018_game_explanations.sql`
- [x] Predictor explanation export + `sync_explanations_to_supabase.py`
- [x] `web/lib/data/explanations.ts`
- [x] `web/app/nba/[gameId]/page.tsx`, `web/app/nfl/[gameId]/page.tsx`
- [x] `web/components/analysis/FeatureDrivers.tsx`

**Acceptance:** Game detail shows ≥5 feature drivers; injury badge when adjusted.

## M3 — Evaluation + strategy UI
- [x] Eval freshness check in daily workflow
- [x] `web/lib/data/evaluations.ts`
- [x] `web/app/models/page.tsx`
- [x] Enhanced `performance/page.tsx` with league tabs

**Acceptance:** `/models` shows NBA v3 and NFL v1 metrics from Supabase or JSON fallback.

## M4 — Injury-aware daily refresh
- [x] Injury sync in `daily-refresh.yml`
- [x] Injury wiring in `refresh_nba` / `refresh_nfl`
- [x] Freshness badges on slate + game detail base vs adjusted

**Acceptance:** Daily refresh uses injury adjustments when impact rows exist.

## M5 — Season readiness gates
- [x] `data-core/docs/NBA_OPENING_READINESS_PLAN.md`
- [x] Pre-deploy CI gate script
- [x] `CHANGELOG_DASHBOARD.md`

**Acceptance:** Go/no-go audit runnable for NFL Week 1 and NBA opener dry-runs.
