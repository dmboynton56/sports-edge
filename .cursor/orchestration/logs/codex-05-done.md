# Codex Task 05 — Done

## Files changed

- `web/app/insights/grading-roundup-2026/page.tsx`
  - Added the force-dynamic grading and backtest roundup insight.
  - Renders graded summaries and missing-data badges from `getResultsData()`.
  - Renders per-sport ROI, calibration coverage, and production-status counts from `getPerformanceHistory()`.
  - Uses explicit empty states when result or performance rows are unavailable.
  - Links to `/performance`, `/results`, and `/markets`; no per-sport performance links were added because those routes are absent.
- `web/app/insights/page.tsx`
  - Added the roundup to the existing insight card array with the `BarChart3` icon.
- `.cursor/orchestration/logs/codex-05-done.md`
  - Recorded implementation and verification results.

## Verification

- `npm --prefix web run lint` — failed on pre-existing, out-of-scope `react-hooks/set-state-in-effect` errors in:
  - `web/components/dashboard/AppShell.tsx:89`
  - `web/components/dashboard/RoiChartClient.tsx:21`
  - The command also reported five pre-existing warnings outside the packet files.
- `cd web && npx eslint app/insights/page.tsx app/insights/grading-roundup-2026/page.tsx` — passed.
- `git diff --check` — passed.
- `npm --prefix web run build` with all Supabase URL and anon-key variables unset — passed.
  - Build output lists `/insights/grading-roundup-2026` as a dynamic route.
- `npm --prefix web run dev -- --hostname 127.0.0.1` plus HTTP requests — passed.
  - `/insights` returned HTTP 200 and listed the new post.
  - `/insights/grading-roundup-2026` returned HTTP 200.
  - With no Supabase variables, the post rendered both missing-environment badges and the explicit `No graded rows yet.` state.
  - The post rendered the persisted backtest snapshot and only valid next-step links.

No commits or pushes were made.
