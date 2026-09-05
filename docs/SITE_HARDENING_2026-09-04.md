# Site and pipeline improvements — September 4, 2026

Five changes build on the initial repository audit. They improve existing serving and inference paths without promoting a new model or changing betting supportability claims.

## 1. Read one immutable MLB HR publication

`web/lib/data/player-markets.ts` now selects the latest run metadata and fetches `mlb_home_run_board_rows` by that exact run ID. A newer publication appearing between requests no longer changes which rows are returned. Removed joined run metadata from the row contract; metadata has one owner.

Unavailable, stale, and no-slate runs skip the candidate query. A failed query or a candidate count different from the finalized publication is withheld. The existing run-ID guard remains as defense against inconsistent payloads. The query retains the existing 500-row limit; a publication above that limit will be withheld rather than silently truncated.

This uses the already-migrated public-read immutable table, so no database migration is needed. The local production build successfully read the connected publication. At the time of browser verification, all games had already started, so the board correctly hid the candidates while retaining publication health.

## 2. Bound data-request time and isolate source failures

All REST fetches in `web/lib/data` now use one helper. Removed five additional duplicate clients across team markets, NFL TD, CFB, explanations, and evaluations. Each existing cache lifetime remains 60 or 300 seconds.

Each request has an eight-second abort deadline. HTTP errors, transport errors, malformed JSON, and non-array responses return the existing unavailable sentinel, allowing callers to show their existing gaps instead of throwing through the entire cross-sport page. There are no automatic retries multiplying request latency. Sequential fallback chains can still take more than eight seconds overall.

React request-scoped caching retains deduplication during Server Component renders, because adding an abort signal opts out of Next.js native fetch memoization. It does not add a global cache or retain failures between renders. Route Handlers still rely on Next.js persistent fetch caching rather than React render memoization.

## 3. Make results grouping scale linearly

`web/lib/data/results.ts` no longer copies the entire accumulated group every time a game, MLB HR, or PGA result is appended. MLB HR model-only classification uses set membership instead of repeatedly scanning all priced rows. Existing grouping order, units, ROI, void handling, ungraded handling, and model/league boundaries remain unchanged.

The connected MLB HR result payload shrank from 1,730,399 to 512,554 bytes (70.4%) with the same 1,000 returned rows by selecting only consumed fields. Next.js had logged a 2,308,504-byte serialized cache entry for the old query, exceeding its 2 MB limit. The projected query fits the cache.

Deterministic MLB HR benchmark, minimum of five runs, identical JSON summaries before and after:

| Rows | Original | Updated | Speedup |
| --- | --- | --- | --- |
| 5,000 | 13.63 ms | 0.66 ms | 20.6× |
| 20,000 | 576.38 ms | 2.41 ms | 239.3× |

Fixture: one model/bucket, alternating priced and model-only rows, a hit every fourth row, and a void every eleventh row. The 20,000-row case demonstrates scaling beyond the current query limit; these are summary-function timings, not network or whole-page speedups.

## 4. Separate artifact-specific scikit-learn runtimes

The daily workflow creates `.venv-nba-nfl` with shared base dependencies and installs the version in `data-core/requirements-nba-nfl.txt`. NBA/NFL prediction steps explicitly use that interpreter. All other steps retain the base interpreter and MLB's scikit-learn 1.6.1. The workflow no longer upgrades and downgrades the base installation between sports, and asserts both expected versions before inference.

This is targeted isolation of the incompatible scikit-learn dependency, not a fully locked environment per model. Other dependencies remain shared. The pip cache tracks both requirement files.

Local verification loaded the NFL v1 artifact (37 features) and NBA v3 artifact (46 features) under 1.7.2, then exercised winner and spread estimators with synthetic inputs and checked finite outputs. The base project interpreter remained at 1.6.1. This verifies compatibility and execution, not predictive quality or out-of-time model performance. The local test used Python 3.12; the GitHub workflow uses 3.11 and has not been dispatched.

## 5. Keep results and performance tables readable on phones

A 390-pixel browser screenshot exposed overlapping table headers and values. Results and performance tables now use automatic column sizing and minimum widths inside the existing horizontal scroll containers. The 10-column summary retains 960 pixels of readable table space; smaller tables retain at least 640 pixels. Cards can shrink within grid layouts, so horizontal scrolling stays inside the table instead of widening the page. Final September 5 verification measured a 390-pixel document at a 390-pixel viewport, with the 960-pixel summary table scrolling within its 316-pixel container. No new JavaScript or client-side data request is needed. Added a screen-reader description to the mobile navigation drawer to remove its Radix accessibility warning.

## Verification

- Backend: 275 tests passed.
- Frontend: 31 tests passed, including publication pinning, truncation, skipped failed-run reads, timeout cancellation, upstream isolation, malformed payloads, and result accounting.
- TypeScript, ESLint, and production build passed.
- Daily workflow YAML parsed; every embedded shell step passed `bash -n`.
- 23 public routes returned HTTP 200 and passed a real-browser smoke traversal with no uncaught page errors. This covered the overview, markets hub, every sport board, results, models, performance, fantasy, data quality, NBA/NFL slates, and both insight articles. It does not test every dynamic game ID or every possible data state.
- Desktop overview → Markets navigation was exercised. Mobile navigation → Results was checked at 390 × 844. The drawer description warning was addressed in the final UI pass.
- Optional whole-repo anti-slop lint still reports nine pre-existing violations in `team-markets.ts`, `fantasy-draft.ts`, `PgaBoard.tsx`, and `mlb-research.ts`. The new shared helper and the other changed data modules pass their targeted check.

No production writes, model promotion, workflow dispatch, or deployment occurred. Changes are recorded on `codex/repo-performance-audit`, alongside the initial audit improvements.

## Next engineering priorities

Use a fixed evaluation dataset and archived contemporaneous odds to test whether a candidate improves calibration and realized returns before expanding the model roster. Separately, move growing results-history summaries into a paginated/aggregate database contract: current 5,000-row requests remain bounded and the connected server returned only 1,000 rows, so they should not be described as unlimited lifetime results. Lock common pipeline dependencies once artifact compatibility is explicitly tested across sports.
