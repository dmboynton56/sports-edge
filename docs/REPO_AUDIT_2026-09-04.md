# Repository audit — September 4, 2026

## Assessment

The direction is sound: immutable pregame publication, durable grading, explicit data freshness, and a distinction between research signals and validated evidence address the production roadmap's central problem. The September status document is appropriately candid about weak MLB winner evidence and missing sportsbook-return validation for NFL/CFB. These are repository claims; this audit did not independently validate live production data or profitability.

The risk is scope expansion before consolidation. The serving code covers multiple sports, player markets, fantasy, mobile APIs, and several generations of MLB feeds. Prefer proving the publication → odds → result → evaluation loop for the trusted vertical before adding more surfaces. A healthy ingestion run is not evidence that a betting model is profitable.

## Changes made

- Vectorized team-strength win and point-difference calculations in `data-core/src/models/predictor.py` and moved historical timestamp normalization outside the per-game loop. Current-season, pregame cutoff, missing-score, venue, tie, unknown-team, and sample-standard-deviation conventions are preserved. The existing statistic uses population standard deviation (`np.std`, ddof=0), which remains unchanged.
- Removed two duplicate REST implementations from `player-markets.ts` and `mlb-research.ts`. The shared helper accepts a cache lifetime: these callers retain 60 seconds, existing callers retain 300 seconds. Failure-to-null behavior is preserved.
- Removed the unused `web/lib/data/predictions.ts` reader after checking references. The JSON artifact and browser fetch used by `MarketsTable` remain in place.
- Simplified the import smoke test: removed `assert True`, manual path injection, and exception wrapping; parameterized the actual import checks. Most tests sampled were meaningful behavior checks and should stay.
- Added a publication consistency guard: separately cached latest-run metadata and latest-board rows could refer to different runs during a refresh. Mixed run IDs now yield an unavailable board with no picks instead of a misleading healthy snapshot. This prevents mixed publication output; it does not make the two queries atomic.

## Performance evidence

Compared the committed and modified team-strength methods using `pandas.testing.assert_frame_equal` on a deterministic synthetic fixture: 32 games, 2,400 historical games, several seasons, timezone-aware dates, missing scores, and unknown teams. Minimum of three single-call timings:

| Version | Elapsed |
| --- | --- |
| Original | 0.4586 seconds |
| Optimized | 0.0600 seconds |

That is about 7.6× faster for this method on this fixture. It is not an end-to-end pipeline or production latency measurement. Historical filtering still happens per upcoming game; further indexing should be justified by profiling real refreshes.

## Remaining priorities

1. **Make publication reads consistent by construction.** `getMlbHomeRunBoardSnapshot` still reads two independently cached latest views sequentially. Prefer one query returning both metadata and rows, or a run-pinned immutable read. The new guard safely detects mixed results, but a cache rollover may temporarily hide the board.
2. **Isolate model dependency environments.** `.github/workflows/daily-refresh.yml` force-reinstalls scikit-learn 1.7.2 for NBA/NFL and then 1.6.1 for MLB. This adds install work and execution-order coupling. Use distinct jobs/environments with locked, artifact-compatible dependencies. The broad backend requirements also install notebook and plotting packages in pipeline jobs; splitting production and development requirements merits measured follow-up.
3. **Consolidate serving modules by responsibility.** `player-markets.ts` still exceeds 1,000 lines and combines trusted MLB snapshots, older edge/prediction fallbacks, PGA data, and the cross-sport feed. `PgaBoard.tsx` exceeds 1,100 lines. Extract cohesive sport/serving contracts when changing them; avoid replacing duplication with one-line pass-through wrappers. Four other local REST helpers remain, with different exception behavior, so mechanically replacing all of them would change outage handling.
4. **Reconcile operational documentation.** `TASK.md` says M5 completed in July while `PRODUCTION_ROADMAP.md` retains June “started” and “in progress” statuses. `DATA_AND_MODEL_STATUS.md` contains the more current September evidence. Link one current operational checklist from the handoff and mark the older roadmap as historical or update its gates.
5. **Strengthen tests at actual boundaries.** Existing financial-math, freshness, injury, and publication tests protect useful contracts. SQL-text tests in the trusted-board migration and edges-view suites check strings, not database behavior; retain them until disposable-database migration/query tests cover those guarantees. Do not delete them merely to lower test count.

## Validation and limits

- Full backend suite: 275 passed.
- Frontend suite after the publication guard: 23 passed.
- TypeScript and ESLint checks passed.
- Production Next.js build passed before the final publication-guard edit; that edit subsequently passed the frontend suite and TypeScript/ESLint checks.
- No model retraining, production database mutation, or deployment performed. No browser interaction or end-to-end live data audit was performed.
- This was a targeted source, test, workflow, and architecture audit, not an exhaustive security assessment or a line-by-line review of every model and historical script.

Changes are recorded on `codex/repo-performance-audit`.
