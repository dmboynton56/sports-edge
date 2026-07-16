# Codex task 02 completion

## Files changed

- `web/lib/data/supabase.ts` — exported the shared defensive `supabaseRest` helper.
- `web/lib/data/results.ts` — added raw per-league/market fetchers, 5,000-row limits, result windows, exported summaries, and weekly result buckets while preserving `getResultsData()`'s output shape.
- `web/lib/data/evaluations.ts` — added persisted evaluation-run/history fetches and defensive display-metric normalization; missing `model_evaluation_history` returns an empty row set plus the required gap.
- `web/app/performance/page.tsx` — converted Performance into the registry-backed sport hub and retained the artifact/gates section.
- `web/app/performance/[sport]/page.tsx` — added the dynamic sport performance route, link-based windows, registry market cross-links, graded market summaries, persisted backtests, empty states, and history charts.
- `web/components/dashboard/PerformanceTrendChart.tsx` — added the client-only chart loader.
- `web/components/dashboard/PerformanceTrendChartClient.tsx` — added weekly hit-rate/units and evaluation metric-over-time chart rendering with the existing Recharts visual pattern.
- `web/components/dashboard/RoiChartClient.tsx` — removed a redundant mounted-state effect because its existing loader already disables SSR.
- `web/components/dashboard/AppShell.tsx` — deferred existing theme-state hydration by one animation frame to satisfy the current React hooks lint rule without changing navigation.

`web/lib/data/player-markets.ts` was not modified.

## Verification

- `npm --prefix web run lint` — passed (0 errors; 5 pre-existing unused-variable warnings in CBB/bracket/model-metric files).
- `env -u NEXT_PUBLIC_SUPABASE_URL -u SUPABASE_URL -u NEXT_PUBLIC_SUPABASE_ANON_KEY -u SUPABASE_ANON_KEY -u SUPABASE_SERVICE_ROLE_KEY -u SUPABASE_DB_PASSWORD -u supabaseDBpass npm --prefix web run build` — passed, including TypeScript and static generation.
- No-env production runtime smoke test — `/performance`, `/performance/nba`, `/performance/mlb?window=7d`, `/performance/pga`, `/performance/nfl`, `/performance/nhl`, and `/performance/cbb` returned HTTP 200; `/performance/not-a-sport` returned HTTP 404.
- `git diff --check` — passed.
- No Supabase `.env` file was present, so live-data comparison of MLB 7-day versus 30-day values was not available locally. Both links feed the implemented `filterByWindow` helper over raw daily `game_date` rows; no-env routes render their gap badges and empty states successfully.
- No commit or push was performed.
