# Task 05 — Insights: graded results & backtest roundup post

Repo: `/home/dmboynton/projects/sports-edge`. Work only under `web/app/insights/`. Do NOT commit or push.
Wave 2: run alongside/after Task 02 (it links to `/performance/*` routes created there), but it must not import anything Task 02 creates — use only helpers that exist on `main` today.

## Goal

Insights should absorb notable backtest/grading findings instead of everything living on Overview. Add one new insight post that presents the season's graded results and persisted backtest status with **live** numbers (rendered at request time from existing data helpers), following the established insight-page pattern.

## Context to read first

- `web/app/insights/page.tsx` (index — `posts` array pattern)
- `web/app/insights/mlb-hr-pytorch/page.tsx` (the pattern to follow: prose + live data sections)
- `web/lib/data/results.ts` — `getResultsData()` (graded ATS/winner/HR/PGA summaries + gaps)
- `web/lib/data/performance.ts` — `getPerformanceHistory()` (artifact metrics, production gates)
- `web/lib/data/mlb-hr-experiment.ts` (used by the existing insight; reuse if it helps)
- `web/components/dashboard/{PageHeader,MetricCard}.tsx`, `web/components/ui/{card,table,badge}.tsx`

## Exact changes

1. **`web/app/insights/grading-roundup-2026/page.tsx` (new, server component, `export const dynamic = "force-dynamic"`).** Structure:
   - `PageHeader`: title "2026 Grading & Backtest Roundup", description saying numbers are computed live from graded result tables.
   - Section "What's now graded automatically": short prose (2–3 sentences) stating that NBA/NFL/MLB ATS+winner grading, MLB HR outcome grading, and PGA placement grading run in the daily/tournament workflows and persist to Supabase.
   - Section "Graded results to date": table from `getResultsData().summaries` (league, market, model, sample, W-L-P, hit rate, flat ROI) — reuse the `/results` page rendering approach; show `data.gaps` as `Badge variant="missing"` when present.
   - Section "Backtest snapshot": headline metrics per sport from `getPerformanceHistory()` (best measured ROI, calibration availability, production status counts) as `MetricCard`s or a compact table.
   - Section "Where to look next": links to `/performance` and the per-sport pages (`/performance/mlb`, `/performance/pga`, ...) and `/markets`. Plain `next/link`.
   - No hardcoded statistics anywhere — every number comes from the helpers; when a helper returns nothing, render an explicit "no graded rows yet" state.
2. **`web/app/insights/page.tsx`.** Add the post to the `posts` array (pick a lucide icon already imported in the codebase style, e.g. `BarChart3`), keeping the existing card grid untouched.

## Constraints

- Only touch the two files above. No changes to `web/lib/data/*` or components.
- No new dependencies; match existing Tailwind/shadcn usage.
- Must build and render (with empty-state placeholders) when no Supabase env vars are set.
- No commits/pushes.

## Done definition

- `/insights` lists the new post; `/insights/grading-roundup-2026` renders live tables/metrics with graceful empty states.

## Verification

```bash
npm --prefix web run lint
npm --prefix web run build     # no env vars — must pass
npm --prefix web run dev       # load /insights and the new post; verify no hardcoded numbers
```
