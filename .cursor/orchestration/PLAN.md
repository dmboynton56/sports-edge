# PLAN — 2026-07 round: Markets/Performance hierarchy + remaining hardening

Planner: Fable (claude-fable-5). Implementer: Codex (`gpt-5.6-sol`). Date: 2026-07-16.

## State assessment (what recon found vs the July plan)

The hardening plan (`.cursor/plans/2026-07-hardening-and-ops-dashboard.md`) is **mostly landed already**:

| Plan item | Status | Evidence |
|---|---|---|
| P1 odds on daily path | DONE | `daily-refresh.yml:235` runs `fetch_mlb_home_run_odds.py` |
| P3 loud degradation | DONE | `validate_mlb_hr_statcast_health.py` gates both workflows (PR #1); coverage surfaced in `MlbHomeRunBoard` |
| P5 `RUN_PGA` default | DONE | `player-markets-refresh.yml:38` uses `|| 'false'` |
| P6 registry + generic fetcher | DONE | `config/pga_tournaments.yaml` full 2026 season, `fetch_pga_field.py`, `generate_pga_registry.py` |
| P7 scoreboard retries | DONE | `live_leaderboard.py:181` retries + `EspnScoreboardError` |
| P8 Supabase-first PGA serving | DONE | `web/lib/data/player-markets.ts` `getPgaBoardData()` + `/api/pga-board` |
| P9 grading (MLB + PGA) | DONE | `daily-refresh.yml:377-382` (evaluate + `sync_mlb_home_run_results_to_supabase.py`), `pga-tournament-refresh.yml:149` (`grade_pga_predictions.py --sync-supabase`); tables in `sql/017` |
| P10 live odds in PGA workflow | DONE | `pga-tournament-refresh.yml` `--live-odds` flag |
| Phase 3.2 live grading + /results | DONE | `game_prediction_results` view (017), `web/app/results/page.tsx`, `web/lib/data/results.ts` |
| Phase 3.3 automate performance artifact | DONE | `performance-history-refresh.yml` (weekly: export + eval sync + git publish) |

**Real remaining gaps** this round addresses:

1. **UX shape (the bulk):** top nav still promotes PGA/CBB (`AppShell.tsx:44-52`); `/markets` is one flat feed table; `/performance` is a flat page fed only by static `web/public/data/performance_history.json` (`getPerformanceHistory` never queries Supabase); `/pga`, `/cbb`, `/markets/mlb-home-runs` sit at wrong depths.
2. **Backtest history is overwritten, not accumulated:** `sync_evaluation_history_to_supabase.py:253-259` DELETEs by `(league, model_name, model_version, evaluation_name)` then reinserts; `sql/012` enforces that identity as unique. "Metrics over time" cannot be built from `model_evaluation_runs` alone.
3. **P2 still open:** Statcast preload in `daily-refresh.yml:180` uses `refresh=False` — cache validated, never updated.
4. **P4 timezone test:** views coalesce Denver dates (017), but no boundary test proves prediction-write and view-filter agree at 23:30 Denver.
5. **Insights** hasn't absorbed graded/backtest findings (single hardcoded post).

## Packets (see `CODEX_TASKS/`)

| # | Packet | Owner | Wave | Depends on |
|---|---|---|---|---|
| 01 | Markets hierarchy + nav cleanup + route moves/redirects | Codex | 1 | — |
| 03 | Backtest history persistence (migration 018 + sync append) | Codex | 1 | — |
| 04 | Statcast trailing-window refresh + timezone boundary test | Codex | 1 | — |
| 02 | Performance sport→market hierarchy with windowed graded metrics | Codex | 2 | 01 (shared `markets-registry`) |
| 05 | Insights: graded-results & backtest roundup post | Codex | 2 | 02 preferred (links), not blocking |

Wave 1 packets are mutually disjoint (different files) and can run in parallel. Packet 02 imports the sport/market registry file that 01 creates. Packet 05 only links to routes from 01/02; content reads existing data helpers.

### Packet 01 — Markets hierarchy + nav (UI)

- Nav: `Overview, Markets, Performance, Insights, Data Quality`. PGA/CBB removed from top level. No Results slot this round (graded metrics fold into Performance; `/results` route stays alive).
- New `web/lib/markets-registry.ts` — single source of truth for sport→market tree, consumed by Markets now and Performance in packet 02.
- Route tree: `/markets` hub → `/markets/{nba,mlb,pga,nfl,nhl,cbb}`; `/markets/mlb/home-runs` (moved), `/markets/pga` (moved `/pga` board), `/markets/cbb` (moved `/cbb` bracket, de-emphasized). NFL/NHL scaffold pages with honest empty states.
- Redirects in `next.config.ts` for `/pga`, `/cbb`, `/markets/mlb-home-runs`.

### Packet 02 — Performance hierarchy (UI + web data layer)

- `/performance` hub mirrors the Markets sport tree; `/performance/{sport}` pages show **graded** results per market over selectable windows (7d / 30d / season / all) from Supabase (`game_prediction_results`, `mlb_home_run_results`, `pga_prediction_results`) plus **persisted backtest** rows (`model_evaluation_runs` latest + `model_evaluation_history` time series, tolerating the history table not existing until 03 deploys).
- Shared `supabaseRest` helper extracted to `web/lib/data/supabase.ts` for new code + `results.ts`; `player-markets.ts` left untouched (risk containment).
- Existing flat artifact view is preserved on the hub (production gates still useful).

### Packet 03 — Backtest history persistence (data-core)

- New migration `data-core/sql/018_model_evaluation_history.sql`: append-only `model_evaluation_history` (evaluation identity + `generated_at` in the unique key). `model_evaluation_runs` keeps latest-only semantics — zero risk to existing readers; no changes to `games`/`model_predictions`/`odds_snapshots`.
- `sync_evaluation_history_to_supabase.py` additionally appends each run to history (idempotent `ON CONFLICT DO NOTHING`).
- `performance-history-refresh.yml` applies 018.

### Packet 04 — Pipeline leftovers (data-core + workflow)

- Flip the daily-refresh Statcast preload to refresh the trailing window (`refresh=True`, configurable `MLB_HR_STATCAST_REFRESH_DAYS`, default 10) while keeping the full-season cache read-only; existing deadline/health gate stays the safety net.
- Timezone boundary unit test: prediction-side `game_date` convention must match the Denver-anchored serving views at the 23:30 Denver / 05:30 UTC boundary; fix the write side if it disagrees.

### Packet 05 — Insights roundup (UI)

- New insight post ("Graded results & backtest roundup — 2026 season") rendering live numbers from `getResultsData()` / `getPerformanceHistory()`, following the `insights/mlb-hr-pytorch` pattern; index updated.

## Risks

- **Route moves break inbound links** — mitigated by `next.config.ts` redirects and a repo-wide grep for old hrefs (in 01's done definition).
- **`/pga` page is a 1,100-line client component** — packet 01 moves it verbatim (extract to `web/components/pga/PgaBoard.tsx`, thin page wrapper); no logic edits allowed, so regressions are import-path-only.
- **Performance pages depend on Supabase env at build/runtime** — all fetches must degrade to gap badges/empty states exactly like `results.ts` does today (`supabaseConfigGaps` pattern); `npm run build` must succeed with no env vars.
- **History table doesn't exist until 018 is applied in prod** — packet 02 must treat a PostgREST 404/undefined-table response as "no history yet", not an error.
- **Workflow edits (04)** touch `daily-refresh.yml` — constraint repeated in-task: Discord notification steps and the `sports-edge-refresh` dispatch block must not move or change.
- **Merge order**: 01 → 02 share `markets-registry.ts`; run 02 after 01's diff exists. 03/04 are disjoint from everything.

## This round vs deferred

**Lands this round:** everything in packets 01–05.

**Deferred (explicitly not this round):**
- Real NFL/NHL market models/data — scaffold pages only.
- CBB promotion or new CBB data work (seasonal; bracket page just relocates).
- Results as a top-nav item (revisit once Performance hierarchy proves itself).
- Densifying PGA crons further (already covers Thu–Sun 18:00–06:00 UTC).
- BigQuery-side evaluation history mirror.
- Any change to `model_evaluation_runs` identity/semantics (history table sidesteps it).

## Review protocol

After Codex completes, Fable reviews against `ACCEPTANCE.md`, writes `REVIEW.md` with `APPROVE`/`REVISE`. No commits/pushes before `APPROVE`.
