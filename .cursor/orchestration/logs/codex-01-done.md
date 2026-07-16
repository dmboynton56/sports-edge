# Codex packet 01 completion

Implemented the Markets sport hierarchy, lean navigation, nested sport routes, legacy redirects, and PGA/CBB/MLB HR route moves. No commit or push was made.

## Files changed

- `web/lib/markets-registry.ts`
- `web/components/dashboard/AppShell.tsx`
- `web/components/pga/PgaBoard.tsx` (moved verbatim from `web/app/pga/page.tsx`)
- `web/app/markets/page.tsx`
- `web/app/markets/nba/page.tsx`
- `web/app/markets/mlb/page.tsx`
- `web/app/markets/mlb/home-runs/page.tsx` (moved from `web/app/markets/mlb-home-runs/page.tsx`)
- `web/app/markets/pga/page.tsx`
- `web/app/markets/nfl/page.tsx`
- `web/app/markets/nhl/page.tsx`
- `web/app/markets/cbb/page.tsx` (moved from `web/app/cbb/page.tsx`; seasonal note added)
- `web/next.config.ts`
- Removed old route files/folders: `web/app/pga/`, `web/app/cbb/`, `web/app/markets/mlb-home-runs/`

## Verification

- `npm --prefix web run lint` — failed on pre-existing `react-hooks/set-state-in-effect` errors in `AppShell.tsx` theme initialization and `RoiChartClient.tsx`; also reported five pre-existing warnings. Packet-created/relocated non-CBB files pass targeted ESLint.
- `env -u NEXT_PUBLIC_SUPABASE_URL -u NEXT_PUBLIC_SUPABASE_ANON_KEY -u SUPABASE_URL -u SUPABASE_ANON_KEY npm --prefix web run build` — passed; all required Markets routes were generated with no Supabase env vars.
- `npm --prefix web run dev -- --hostname 127.0.0.1 --port 3101` + `curl` smoke checks — all required new routes returned `200`; `/pga`, `/cbb`, and `/markets/mlb-home-runs` returned `307` to their new destinations.
- Stale href sweep under `web/` — clean; only the intentional legacy sources remain in `web/next.config.ts`.
- PGA relocation comparison with `git show HEAD:web/app/pga/page.tsx` — byte-for-byte identical.
