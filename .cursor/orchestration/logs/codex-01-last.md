Implemented Codex task packet 01 exactly:

- Added Markets sport registry and hierarchy pages.
- Reduced navigation to the five required items.
- Relocated PGA, CBB, and MLB home-run routes.
- Added temporary redirects for legacy URLs.
- Swept stale links.
- No commit or push performed.

Verification:

- No-Supabase production build: passed.
- All new routes: HTTP `200`.
- Legacy redirects: HTTP `307` to correct destinations.
- Full lint: blocked by two pre-existing hook-rule errors, documented in the completion note.
- Targeted lint for new files: passed.

Completion note: [codex-01-done.md](/home/dmboynton/projects/sports-edge/.cursor/orchestration/logs/codex-01-done.md)