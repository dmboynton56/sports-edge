VERDICT: APPROVE

# Fable review — 2026-07 round (packets 01–05)

Reviewer: Fable (claude-fable-5), 2026-07-16. All verification was re-run locally against the working tree, not taken from Codex's done notes.

## Checklist results (ACCEPTANCE.md)

### Navigation (packet 01)

- [x] `AppShell.tsx` `navItems` is exactly Overview `/`, Markets `/markets`, Performance `/performance`, Insights `/insights`, Data Quality `/data-quality`; PGA/CBB entries and the `Trophy` import removed. Verified by reading the file.
- [x] Grep for `href="/pga"`, `href="/cbb"`, `href="/markets/mlb-home-runs"` under `web/` is clean (exit 1, no matches).
- [x] `next.config.ts` has all three redirects (`permanent: false`).

### Markets hierarchy (packet 01)

- [x] `web/lib/markets-registry.ts` exports the six-sport tree with `emphasis` levels; CBB is `seasonal` and listed last.
- [x] `/markets` renders registry-driven sport cards (primary row, then muted secondary row with CBB last) plus the retained pre-live board.
- [x] `/markets/nba` filters the production feed to NBA with an honest empty state.
- [x] `/markets/mlb` lists registry markets linking to `/markets/mlb/home-runs`; the HR board page is byte-identical to the old `markets/mlb-home-runs/page.tsx` (`diff` exit 0); `getMlbHomeRunBoardData` untouched.
- [x] `/markets/pga` is a thin wrapper over `web/components/pga/PgaBoard.tsx`, which is byte-identical to `HEAD:web/app/pga/page.tsx` (verified with `diff <(git show ...)`).
- [x] `/markets/nfl` and `/markets/nhl` are scaffolds with honest empty states, no fake numbers.
- [x] `/markets/cbb` is the former `/cbb` page plus only a seasonal-note paragraph (3-line diff); old `web/app/pga/page.tsx` and `web/app/cbb/page.tsx` deleted (git status `D`).

### Performance hierarchy (packet 02)

- [x] `/performance` renders the same registry sport tree plus the preserved artifact/gates section (MetricCards, RoiChart, PerformanceTables).
- [x] `/performance/[sport]` dynamic route covers all six slugs; unknown slugs 404 via `notFound()`.
- [x] NBA/NFL use `game_prediction_results` ATS/winner summaries; MLB adds `mlb_home_run_results` hit rates by top-k bucket; PGA shows per-event top-10/top-20/winner-actual-vs-expected from `pga_prediction_results`.
- [x] Window selection (7d/30d/season/all) is link-based `?window=` and feeds `filterByWindow` before summarization; default `season`.
- [x] Backtest section renders `model_evaluation_runs` filtered by league; `model_evaluation_history` returns `{rows: [], gaps: [...]}` when the table 404s (`supabaseRest` returns null on any non-OK response) — gap badge, no crash; history chart only renders when rows exist.
- [x] No-env build passes: `npm --prefix web run build` with all Supabase vars unset → all 18 routes generated, `/performance/[sport]` dynamic.
- [x] `supabaseRest` lives in `web/lib/data/supabase.ts` (with try/catch hardening); `results.ts` imports it; `player-markets.ts` has zero diff.

### Backtest persistence (packet 03)

- [x] `sql/018_model_evaluation_history.sql` exists; `create table if not exists`, `create [unique] index if not exists`; unique key is `(league, model_name, model_version, evaluation_name, generated_at)`; RLS + anon/authenticated SELECT grants match the 009/017 pattern.
- [x] `sync_evaluation_history_to_supabase.py` keeps the DELETE/INSERT latest-only path untouched and adds a history INSERT with `ON CONFLICT ... DO NOTHING`, reporting `appended_history`.
- [x] `performance-history-refresh.yml` applies `sql/018_...` alongside 009 before syncing.
- [x] No diffs to `games`/`model_predictions`/`odds_snapshots` schemas or to `sql/012` (not in the changeset at all).
- [x] `test_sync_evaluation_history.py` covers the append + conflict path; full suite green (see Global).

### Pipeline leftovers (packet 04)

- [x] New `preload_statcast_cache()` reads the older interval `refresh=False` and refreshes only the trailing `refresh_days` (workflow default 10 via `MLB_HR_STATCAST_REFRESH_DAYS`), sharing one deadline across both calls. Unit test pins the exact split (2026-03-18→07-05 read-only, 07-06→07-15 refresh).
- [x] The workflow step keeps the same try/except with the existing `WARNING: Statcast preload failed...` soft-fail; health gate unchanged downstream.
- [x] Timezone boundary tests assert both `plan_daily_refresh.default_anchor_date` and the new `predict_mlb_home_runs.default_slate_date` return the Denver date at 2026-07-16 05:30 UTC (= 23:30 Denver) and the 06:30 UTC mirror. Write side was indeed wrong (UTC `datetime.now(timezone.utc).date()`) and is now fixed to the Denver anchor; explicit `--date` unchanged.
- [x] `git diff .github/workflows/daily-refresh.yml` contains only the env-var hunk and the preload-step hunk — the Discord notification steps and the `sports-edge-refresh` dispatch block are byte-identical because no other hunks exist.

### Insights (packet 05)

- [x] `/insights/grading-roundup-2026` is `force-dynamic`, computes everything from `getResultsData()` / `getPerformanceHistory()` at request time (no hardcoded stats), with gap badges and empty states; the insights index lists it first.

### Global verification

- [x] `npm --prefix web run lint` — 0 errors, 5 pre-existing unused-var warnings (CBB/bracket/model-metric files).
- [x] `npm --prefix web run build` with Supabase env unset — passes.
- [x] `PYTHONPATH=. pytest tests/ -q` in data-core — 121 passed (6 pre-existing sklearn version warnings).
- [x] Secrets grep over the full diff — clean. No commits or pushes were made (all work is in the working tree).
- [x] Diff scope: every changed file is named in a packet. `AppShell.tsx` and `RoiChartClient.tsx` appear in packets 01/02 file lists, so the lint fixes there are in scope.

## Reconciled review notes from STATUS.md

- **Lint discrepancy (01 vs 02):** resolved. Packet 01 hit pre-existing `react-hooks/set-state-in-effect` errors; packet 02 fixed both (AppShell theme init deferred to a rAF, RoiChartClient's redundant mounted-state removed — its loader already disables SSR). Lint now reports 0 errors.
- **Packet 05 vs 02 ordering:** the roundup post links only to the `/performance` hub, `/results`, and `/markets`; no links are broken. Per-sport `/performance/{sport}` deep links now exist and could be added, but that was never an acceptance item.

## Non-blocking nits (fine to fold into a future round)

1. `grading-roundup-2026` could deep-link to `/performance/{sport}` now that those routes exist (packet 05 ran before 02).
2. The history INSERT in `sync_evaluation_history_to_supabase.py` leaves `train/test_*_date` NULL — consistent with the existing runs insert (the script never carried those fields), and the web layer tolerates nulls, but populating them would make the history table more useful for future train/test planning.
3. `filterByWindow`'s "season" = Jan 1 UTC of the current calendar year; the UI labels the badge honestly ("Season = current calendar year"), so acceptable, but a per-sport season anchor may be wanted eventually.

## Suggested commit message (do not commit — human/orchestrator does this)

```
Markets/Performance sport hierarchy, backtest history persistence, Statcast trailing refresh

- Lean top nav (Overview/Markets/Performance/Insights/Data Quality); PGA & CBB
  moved under /markets with redirects; markets-registry.ts as the sport→market tree
- /performance hub + /performance/{sport} pages: windowed graded results from
  Supabase result tables plus persisted evaluation runs and history series
- New append-only model_evaluation_history (sql/018) written by the eval sync
  with ON CONFLICT DO NOTHING; weekly workflow applies the migration
- Daily-refresh Statcast preload now refreshes the trailing window
  (MLB_HR_STATCAST_REFRESH_DAYS, default 10); HR prediction default date now
  uses the Denver slate anchor, with boundary tests
- Insights: 2026 grading & backtest roundup rendered from live data

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Co-Authored-By: Codex <noreply@openai.com>
```
