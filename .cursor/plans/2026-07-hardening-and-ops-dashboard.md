# Plan: Ops dashboard as primary serving surface + PGA/MLB-HR pipeline hardening

Repo: `sports-edge`. Round: 2026-07. Self-contained — cross-repo facts are inlined; do not read sibling repos.

## Goals

1. Make the ops dashboard (`web/`, deployed at https://sports-edge.drewboynton.com) the main surface for looking across leagues: predictions, graded results, metrics, success rates. The portfolio site stops being the live-predictions UI (its own plan handles that side).
2. Harden the PGA pipeline: detect new tournaments automatically, fetch the field for any tournament, update after each completed round.
3. Fix the MLB HR pipeline: make Statcast fetching and the Statcast-blend model actually work consistently instead of silently falling back to v1.

## Inlined cross-repo context

- The portfolio (`personal-portfolio` repo, drewboynton.com) reads the shared Supabase (`games`, `model_predictions`) via its own `/api/sports-edges` routes. After both plans execute, the portfolio only shows *overall* metrics (ATS %, ROI) on its deep dive and links here. Nothing in this plan should break `games`/`model_predictions`/`odds_snapshots` schemas — the portfolio's ATS route still reads them.
- The `repository_dispatch` event `sports-edge-refresh` → personal-portfolio (fired from workflows on success, using `PORTFOLIO_DISPATCH_TOKEN`) must keep working; it refreshes the portfolio's RAG docs.
- The portfolio plan will want a clean dashboard screenshot for its deep dive; nothing to do here except keep the Overview page presentable.

## Known problems this plan fixes (from recon, with file refs)

| # | Problem | Where |
|---|---|---|
| P1 | Daily refresh syncs MLB HR predictions but never fetches HR odds — `fetch_mlb_home_run_odds.py` only runs in `player-markets-refresh.yml`, so `mlb_home_run_edges_latest` is empty on daily-only days | `.github/workflows/daily-refresh.yml` |
| P2 | Statcast preload in CI uses `refresh=False` — cache is validated but never updated; stale Savant data or v1 fallback, warnings only | `daily-refresh.yml` (~lines 155–176), `src/models/mlb_hr_statcast_features.py` |
| P3 | Statcast/blend failures are silently absorbed: `_build_statcast_blend_predictions()` mirrors v1 with a buried `statcast_features_unavailable` flag; `_fallback_model_predictions()`; missing torch artifact makes the blend inert while looking populated | `data-core/scripts/predict_mlb_home_runs.py` |
| P4 | Timezone skew: pipeline anchors "today" in UTC (`plan_daily_refresh.py`) but Supabase views filter `game_date = America/Denver today` | `data-core/sql/015_player_market_serving_tables.sql` |
| P5 | `player-markets-refresh.yml` bug: `RUN_PGA: ${{ github.event.inputs.run_pga \|\| 'true' }}` — empty scheduler dispatch runs PGA even though the input default is `"false"` | `.github/workflows/player-markets-refresh.yml` |
| P6 | PGA registry has exactly one tournament (`us_open_2026`) and only one field fetcher (`fetch_usopen_field.py`) is wired in `_field_fetch_command()` | `data-core/config/pga_tournaments.yaml`, `src/pga/tournament_registry.py`, `scripts/refresh_pga_tournament.py` |
| P7 | `fetch_scoreboard()` swallows all exceptions → live phase silently skips mid-tournament updates | `data-core/src/pga/live_leaderboard.py` |
| P8 | Ops dashboard PGA page reads git-committed static JSON only (`web/public/data/pga_tournaments/current.json`), never Supabase; PGA workflow git-pushes JSON to main (race with deploys) | `web/app/pga/page.tsx`, `pga-tournament-refresh.yml` |
| P9 | No automated outcome grading for player markets; `evaluate_mlb_home_run_predictions.py` and `sync_evaluation_history_to_supabase.py` exist but aren't in any workflow; dashboard "performance" is backtest artifacts, not live results | `data-core/scripts/` |
| P10 | CI runs PGA predictions `--baseline-only`, scheduled runs default `--skip-odds` | `refresh_pga_tournament.py`, workflow args |

---

## Phase 1 — MLB HR pipeline hardening (highest pain)

1. **Odds in the daily path (P1):** add the `fetch_mlb_home_run_odds.py` step to `daily-refresh.yml`'s MLB HR job (same secrets as `player-markets-refresh.yml` uses; The Odds API). Alternatively consolidate: have daily-refresh dispatch/skip based on whether player-markets-refresh already ran that day — pick the simpler diff, but the invariant is: *any day HR predictions sync, odds sync too*.
2. **Statcast cache actually refreshes (P2):** flip the CI preload to `refresh=True` for the trailing window (last ~10 days is enough; full-season cache stays). Keep per-chunk CSV caching. Budget retries so a Savant outage doesn't hang the workflow (existing 408/425/429/5xx retry logic is fine; add an overall timeout).
3. **Make degradation loud (P3):**
   - `predict_mlb_home_runs.py`: after building the board, compute `statcast_coverage = rows with statcast_feature_ready / total rows`. Write it into the output JSON and CSV.
   - Threshold gate in the workflow: if coverage < a configurable floor (start at 0.5) or the torch artifact failed to load, the workflow step **fails** (or at minimum posts a distinct Discord alert via the existing `src/utils` Discord path used elsewhere in workflows) instead of shipping a silent v1 clone.
   - Surface `statcast_coverage`, `model_agreement` distribution, and artifact-load status on the ops dashboard MLB HR page (data already flows through `web/lib/data/player-markets.ts` — add the fields to the JSON/Supabase sync).
4. **Timezone unification (P4):** pick **one** slate-date convention. Recommendation: compute `game_date` in `America/Denver` at prediction time and write it explicitly, and change the Supabase views in a new migration (`data-core/sql/017_...`) to filter on that column without recomputing "today" — or make both sides UTC. Either way, prediction writes and view filters must agree; add a test that fails on skew (freeze time at 23:30 Denver / 05:30 UTC boundary).
5. **Lower the `statcast_feature_ready` friction:** the ≥10 batted-ball-events requirement for BOTH batter and pitcher knocks most rows out early-season. Make the threshold configurable and consider a partial-feature mode (batter-only) with its own quality flag rather than full fallback to v1.
6. **Remove dead `pybaseball` from `requirements.txt`** (zero imports in `data-core/`; production uses direct Savant CSV) to stop dependency drift confusion.
7. **Automated outcome grading (P9, MLB half):** add a nightly step (in `daily-refresh.yml`, after final-scores sync) that runs `evaluate_mlb_home_run_predictions.py` for yesterday's slate and upserts results into Supabase — new table `mlb_home_run_results` (prediction id/date/player, hit_hr bool, model, prob, top-k bucket) via `sync_player_markets_to_supabase.py` extension or a small new sync script. This is what lets the dashboard show real HR success rates.

## Phase 2 — PGA pipeline hardening

1. **Generic field fetcher (P6):** replace the US-Open-only `fetch_usopen_field.py` wiring with a generic ESPN-based fetcher: `scripts/fetch_pga_field.py --event-id <espn_event_id>` pulling the field from ESPN's golf event endpoints (the repo already talks to `site.api.espn.com/.../golf/pga/scoreboard` in `live_leaderboard.py`; the event competitors endpoint provides the field). `_field_fetch_command()` in `refresh_pga_tournament.py` dispatches on a new `field_source: espn` registry key, keeping per-tournament overrides possible.
2. **Season registry (P6):** extend `config/pga_tournaments.yaml` with the remaining 2026 season (name, espn_match patterns, start/end dates, priority). Add a `scripts/generate_pga_registry.py` helper that pulls the ESPN season schedule and emits/updates the YAML so future seasons are one command. `resolve_active_tournament()` already handles selection windows and phase ranking — it just needs entries to select from.
3. **New-tournament detection:** with the full-season registry, the existing date-window + ESPN-name-match logic in `resolve_active_tournament()` covers "understand when a new tournament starts." Add one guard: if the ESPN scoreboard shows an in-progress PGA event that matches **no** registry entry, log + Discord-alert loudly (registry drift detection).
4. **Per-round updates (already mostly built, make reliable):** `rounds_completed_from_leaderboard()` + the `pga_refresh_state.json` dedup key already implement round-triggered mid-tournament updates. Fixes:
   - **P7:** `fetch_scoreboard()` — on exception, retry (3x, backoff), then raise a typed error that the orchestrator turns into a Discord alert + nonzero exit during `live` phase (silent skip is fine in `pre`/`post`).
   - Tighten the workflow cron during live phases: current `0 2-6,13,17,21,23 * * *` misses round completions during afternoon/evening US time. Either densify the cron (hourly 18:00–06:00 UTC Thu–Sun) or have the Cloud Run bridge (`gcp/scheduler-trigger/app.py`) add a schedule — cheapest is the GHA cron change since the run no-ops cheaply when no new round is complete.
   - **P10:** drop `--baseline-only` in the scheduled workflow if runtime allows (measure once), and enable odds (`--skip-odds` off) at least for pre-tournament runs.
5. **Serving path (P8):** add PGA to the Supabase-first pattern the MLB HR board already uses: `web/lib/data/` gets a `getPgaBoardData()` that queries `pga_tournaments`/`pga_player_predictions`/`pga_odds_snapshots` (sync script `sync_player_markets_to_supabase.py` already writes them — make that sync a non-optional workflow step), falling back to the static JSON. This removes the git-push-to-main race as the load-bearing path; keep the JSON export as fallback + local dev.
6. **PGA outcome grading (P9, PGA half):** post-tournament phase already fetches results (`fetch_espn_pga_results.py`). Add a grading step comparing pre-tournament predictions (top-10/winner probabilities) to final finishes, writing to a `pga_prediction_results` Supabase table for dashboard display.

## Phase 3 — Ops dashboard as the main cross-league surface

1. **Overview page (`web/app/page.tsx`) becomes results-first:** today's slate across leagues (data already reachable via `getProductionPredictionFeed()`), rolling graded success rates (NBA/NFL ATS %, MLB winner %, HR top-k hit rate, PGA grading) from the new results tables, and pipeline health (last successful run per workflow, statcast coverage, odds freshness — the `deriveProductionGates()` machinery in `web/lib/data/performance.ts` is the starting point).
2. **Live grading in-dashboard:** port the ATS/winner grading logic that currently lives in the portfolio (`spreadHit`/`winnerHit` computed at read time from `games` + `model_predictions` scores/spreads in shared Supabase) into `web/lib/data/` so the dashboard computes the same numbers natively. Add a `/results` page: per-league season tables of wins/losses/pushes, flat -110 ROI, weekly breakdowns.
3. **Automate the performance artifact:** `export_performance_history.py` and `sync_evaluation_history_to_supabase.py` are manual today. Add both to a weekly workflow (or the tail of daily-refresh) so `/performance` and `model_evaluation_runs` stay current without notebook runs.
4. **Fix P5** (one-line): `RUN_PGA: ${{ github.event.inputs.run_pga || 'false' }}`.
5. Keep the Discord notifications and the `sports-edge-refresh` repository dispatch to the portfolio intact in every workflow you touch.

## Phase 4 — Tests & verification

1. Unit tests: registry resolution with multi-event weeks; `infer_phase()` boundaries; timezone slate-date test (Phase 1.4); statcast coverage gate; `_field_fetch_command()` dispatch.
2. Dry-run each modified workflow via `workflow_dispatch` with test inputs before trusting cron.
3. Verify Supabase views return rows at 23:30 Denver time; verify the dashboard MLB HR page shows coverage stats; verify PGA page renders from Supabase with JSON fallback (kill env vars locally to test fallback).
4. Do not change `games`/`model_predictions`/`odds_snapshots` schemas (portfolio depends on them). New tables/views get new migrations under `data-core/sql/` (next number: 017).

## Sequencing

Phase 1 and Phase 2 are independent — parallelize if using multiple agents. Phase 3 depends on the results tables from Phases 1.7 and 2.6. Phase 4 runs throughout.
