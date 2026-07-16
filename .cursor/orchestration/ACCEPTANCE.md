# ACCEPTANCE — binary checklist for Fable review (2026-07 round)

Each item is pass/fail. All must pass for `APPROVE`.

## Navigation (packet 01)

- [ ] `web/components/dashboard/AppShell.tsx` `navItems` is exactly: Overview `/`, Markets `/markets`, Performance `/performance`, Insights `/insights`, Data Quality `/data-quality` — no `/pga`, no `/cbb` entries.
- [ ] No remaining `href="/pga"`, `href="/cbb"`, or `href="/markets/mlb-home-runs"` anywhere under `web/` (grep is clean).
- [ ] `next.config.ts` redirects: `/pga` → `/markets/pga`, `/cbb` → `/markets/cbb`, `/markets/mlb-home-runs` → `/markets/mlb/home-runs`.

## Markets hierarchy (packet 01)

- [ ] `web/lib/markets-registry.ts` exists and exports the sport→market tree (nba, mlb, pga, nfl, nhl, cbb) with emphasis levels; CBB marked seasonal/de-emphasized.
- [ ] `/markets` renders sport cards from the registry (CBB last and visually muted) — not just a flat feed table.
- [ ] `/markets/nba` renders the NBA-filtered prediction board.
- [ ] `/markets/mlb` lists MLB markets and links to `/markets/mlb/home-runs`; the HR board page renders there with unchanged behavior (`getMlbHomeRunBoardData` untouched).
- [ ] `/markets/pga` renders the full former `/pga` board (tabs: Round Outlook, Score Context, Odds & Edges, 2026 events, Recent form) still fed by `/api/pga-board`; PGA board logic not edited, only relocated.
- [ ] `/markets/nfl` and `/markets/nhl` exist as scaffolds with honest empty/de-emphasized states (no fake numbers).
- [ ] `/markets/cbb` renders the former `/cbb` bracket page; old `web/app/pga/page.tsx` and `web/app/cbb/page.tsx` are removed.

## Performance hierarchy (packet 02)

- [ ] `/performance` shows the same sport tree as Markets (from `markets-registry.ts`) plus the existing artifact/gates summary.
- [ ] Per-sport performance pages exist for nba, mlb, pga, nfl, nhl, cbb.
- [ ] NBA/NFL/MLB pages show graded ATS/winner metrics computed from Supabase `game_prediction_results`; MLB page also shows HR hit rates from `mlb_home_run_results`; PGA page shows top-10/top-20/winner hit rates from `pga_prediction_results`.
- [ ] Window selection works (7d / 30d / season / all) and changes the computed summaries.
- [ ] Backtest section renders `model_evaluation_runs` rows for the sport; `model_evaluation_history` series renders when present and degrades silently (gap badge, no crash) when the table is missing or empty.
- [ ] With no Supabase env vars set, `/performance/*` pages render with gap badges/empty states — `npm --prefix web run build` succeeds without env.
- [ ] `supabaseRest` shared helper lives in `web/lib/data/supabase.ts`; `results.ts` uses it; `player-markets.ts` is not modified in this packet.

## Backtest persistence (packet 03)

- [ ] `data-core/sql/018_model_evaluation_history.sql` exists, idempotent (`create table if not exists` / `create index if not exists`), unique key includes `generated_at`.
- [ ] `sync_evaluation_history_to_supabase.py` writes latest rows (unchanged behavior) AND appends history rows with `ON CONFLICT DO NOTHING`.
- [ ] `performance-history-refresh.yml` applies `sql/018_...` before syncing.
- [ ] No schema change to `games`, `model_predictions`, `odds_snapshots`, or to `model_evaluation_runs` identity (012 index untouched).
- [ ] `tests/unit/test_sync_evaluation_history.py` covers the history append (runs green).

## Pipeline leftovers (packet 04)

- [ ] Daily-refresh Statcast preload refreshes the trailing window (`refresh=True` for last `MLB_HR_STATCAST_REFRESH_DAYS` days, default 10) and leaves the older cache read-only.
- [ ] Preload still soft-fails with the existing WARNING path (workflow does not hard-fail on Savant outage; the downstream health gate remains the enforcement point).
- [ ] A unit test freezes time at 23:30 America/Denver (05:30 UTC next day) and asserts the prediction-side `game_date` matches the Denver date used by the serving views; write side fixed if it disagreed.
- [ ] Discord notification steps and the `sports-edge-refresh` repository dispatch in `daily-refresh.yml` are byte-identical to before.

## Insights (packet 05)

- [ ] New insight page under `web/app/insights/<slug>/page.tsx` rendering live graded/backtest numbers (no hardcoded stale stats); insights index lists it.

## Global verification

- [ ] `npm --prefix web run lint` passes.
- [ ] `npm --prefix web run build` passes (no Supabase env).
- [ ] `cd data-core && PYTHONPATH=. pytest tests/ -q` passes.
- [ ] No secrets in any diff; no commits/pushes made by Codex.
- [ ] Diff only touches files named in the task packets (plus lockfiles if unavoidable).
