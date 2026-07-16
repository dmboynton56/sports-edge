Implemented Codex task packet 02.

Highlights:

- Added registry-backed `/performance` hub and dynamic `/performance/[sport]` pages.
- Added 7d/30d/current-year/all windowing and weekly result buckets.
- Added NBA, NFL, MLB, and PGA graded metrics.
- Added Supabase evaluation runs/history with missing-history tolerance.
- Shared `supabaseRest`; left `player-markets.ts` untouched.
- Added Recharts performance trend charts.
- No-env routes render empty states and gap badges.

Verification:

- `npm --prefix web run lint` passed with 0 errors.
- No-env `npm --prefix web run build` passed.
- All six sport routes returned HTTP 200; invalid sport returned 404.
- `git diff --check` passed.
- No commit or push performed.

Completion log: [codex-02-done.md](/home/dmboynton/projects/sports-edge/.cursor/orchestration/logs/codex-02-done.md)