# Task 02 — Performance sport→market hierarchy with windowed graded metrics

Repo: `/home/dmboynton/projects/sports-edge`. Work only under `web/`. Do NOT commit or push.
**Depends on Task 01** (imports `web/lib/markets-registry.ts`). Run after 01's diff exists.

## Goal

Make `/performance` mirror the Markets sport tree. Each sport page shows (a) graded live results per market over selectable windows and (b) persisted backtest/evaluation metrics from Supabase — replacing "static JSON only" as the sole performance source.

## Context to read first

- `web/lib/markets-registry.ts` (from Task 01 — sport tree source of truth)
- `web/lib/data/results.ts` (existing Supabase REST fetch of `game_prediction_results`, `mlb_home_run_results`, `pga_prediction_results`; row types at top; private `supabaseRest` at line ~68)
- `web/lib/data/supabase.ts` (env config helpers — `getSupabaseRuntimeConfig`, `getSupabaseMissingEnv`)
- `web/lib/data/performance.ts` (`getPerformanceHistory` — static JSON path; keep working)
- `web/app/performance/page.tsx`, `web/app/results/page.tsx` (current pages)
- `web/components/dashboard/{PerformanceTables,RoiChart,RoiChartClient,MetricCard,PageHeader}.tsx`
- `data-core/sql/017_player_market_health_results.sql` (result table/view columns), `data-core/sql/009_model_evaluation_tables.sql` (`model_evaluation_runs` columns: league, model_name, model_version, evaluation_name, test dates, generated_at, metrics jsonb, status)

## Exact changes

1. **Shared REST helper.** Move the `supabaseRest<T>(resource)` implementation from `results.ts` into `web/lib/data/supabase.ts` (exported), re-use it in `results.ts` and all new code. **Do not modify `web/lib/data/player-markets.ts`** (it has its own copy; leave it).

2. **`web/lib/data/results.ts` — windowing.** Extend so callers can compute summaries per window:
   - Export `type ResultsWindow = "7d" | "30d" | "season" | "all"` and a `filterByWindow(rows, window, dateField)` helper (season = since Aug 1 for NFL/CBB-style or Jan 1 — implement as "current calendar year" for simplicity and say so in the UI label, e.g. "2026").
   - Export per-league fetchers (or one fetch + grouping helpers) that return raw rows so sport pages can window them: game results filtered by `league`, HR results, PGA results. Raise fetch limits to 5000, keep `order=...desc`.
   - Keep `getResultsData()` behavior for `/results` and Overview unchanged (same output shape).
   - Add a weekly time-bucket helper producing `{ weekStart, wins, losses, pushes, hitRate, units }[]` for charts.

3. **`web/lib/data/evaluations.ts` (new).** Fetch persisted backtests:
   - `getEvaluationRuns(league?)` → `model_evaluation_runs` rows (select league, model_name, model_version, evaluation_name, test_start_date, test_end_date, generated_at, metrics, status; order generated_at desc, limit 200).
   - `getEvaluationHistory(league?)` → same shape from `model_evaluation_history` (created by Task 03; **may not exist in prod yet** — a PostgREST error/404/relation-missing response must return `{ rows: [], gaps: ["model_evaluation_history not available"] }`, never throw).
   - Both degrade to gaps when Supabase env is missing, mirroring `results.ts` (`getSupabaseMissingEnv`).
   - Pull display metrics out of the `metrics` jsonb defensively (numbers only; keys vary by league — reuse the key lists in `performance.ts` `numberMetric` calls, e.g. `accuracy`, `auc`, `brier`, `log_loss`, `flat_roi`, `supabase_ats_roi`).

4. **`web/app/performance/page.tsx` (rewrite as hub).** `PageHeader` + sport-card grid from `SPORTS` (same emphasis treatment as `/markets`: primary sports first, CBB muted/last), each card linking to `/performance/<slug>` and showing one headline graded stat when cheaply available (fine to omit stats on the hub). Below the grid, keep the existing artifact summary (MetricCards + `RoiChart` + `PerformanceTables` from `getPerformanceHistory()`) under a "Model artifacts & production gates" section.

5. **`web/app/performance/[sport]/page.tsx` (new, `export const dynamic = "force-dynamic"`).** Validate `params.sport` against the registry (`notFound()` otherwise). Accept `searchParams.window` (`7d|30d|season|all`, default `season`); window switcher rendered as link-buttons preserving the route. Per sport render one card per market:
   - **NBA / NFL:** ATS summary (W-L-P, hit rate, flat -110 ROI from `flat_ats_units`) and winner summary from `game_prediction_results` rows for that league, per model_version; weekly hit-rate/units chart using the bucket helper and the existing `RoiChartClient` recharts pattern (match its styling; no new chart libs).
   - **MLB:** winner summary from `game_prediction_results` (league MLB) + Home Runs card from `mlb_home_run_results`: hit rate by `top_k_bucket` and by model_version within the window.
   - **PGA:** from `pga_prediction_results`: top-10 hit rate, top-20 hit rate, winner hits vs expected (sum of win_prob), grouped by event within the window.
   - **NHL / CBB:** honest empty state ("no graded markets yet"); CBB may show `model_evaluation_runs` CV rows if present.
   - **Backtests section (all sports):** table of `getEvaluationRuns(league)` (evaluation_name, model_version, test range, key metrics, status) and, when `getEvaluationHistory` returns rows, a small metric-over-time chart (pick roi/accuracy when present). Gap badges (existing `Badge variant="missing"` pattern) when data is missing.
   - Cross-link each market card to its `/markets/...` counterpart via the registry.

## Constraints

- Server components + link-based window switching; only reuse existing client chart components (`RoiChartClient` pattern) for charts — no new dependencies.
- Every Supabase fetch must degrade to gaps/empty states; `npm run build` with zero env vars must pass and pages must render placeholders.
- Do not change `/results` page output, `getPerformanceHistory()` behavior, or `player-markets.ts`.
- League string in Supabase rows is uppercase (`NBA`, `MLB`, `PGA`...); registry slugs are lowercase — map explicitly.
- No commits/pushes; no files outside `web/`.

## Done definition

- `/performance` is a sport tree mirroring `/markets`; `/performance/{nba,mlb,pga,nfl,nhl,cbb}` all render.
- Window switcher changes computed numbers (verify with two windows on MLB HR data, which has daily rows).
- Backtest tables render from `model_evaluation_runs`; missing history table produces a gap badge, not an error.

## Verification

```bash
npm --prefix web run lint
npm --prefix web run build                       # no env vars — must pass
# with real env (web/.env.local if present):
npm --prefix web run dev                          # load /performance, /performance/mlb?window=7d, /performance/pga, /performance/nhl
```
