Implemented Codex task packet 05.

- Added live `/insights/grading-roundup-2026` page with graded results, backtest snapshots, gaps, and empty states.
- Added the roundup card to `/insights`.
- Used `/performance`, `/results`, and `/markets` because per-sport performance routes are missing.
- Build passed without Supabase environment variables.
- Both routes returned HTTP 200 in development.
- Targeted ESLint passed. Repository-wide lint remains blocked by two unrelated existing errors in `AppShell.tsx` and `RoiChartClient.tsx`.
- No commits or pushes made.

Full verification log: [codex-05-done.md](/home/dmboynton/projects/sports-edge/.cursor/orchestration/logs/codex-05-done.md)