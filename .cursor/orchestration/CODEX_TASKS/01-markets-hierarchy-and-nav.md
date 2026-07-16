# Task 01 — Markets sport→market hierarchy, nav cleanup, route moves

Repo: `/home/dmboynton/projects/sports-edge`. Work only under `web/`. Do NOT commit or push.

## Goal

Restructure the dashboard IA: lean top nav (no PGA/CBB tabs), `/markets` becomes a sport → market hierarchy, and the root-level `/pga` and `/cbb` pages move under `/markets`. Old URLs redirect.

## Context to read first

- `web/components/dashboard/AppShell.tsx` (nav lives in `navItems`, lines ~44–52)
- `web/app/markets/page.tsx` (current flat hub)
- `web/app/markets/mlb-home-runs/page.tsx` (moves to `/markets/mlb/home-runs`)
- `web/app/pga/page.tsx` (~1,100-line client component; moves under `/markets/pga`)
- `web/app/cbb/page.tsx` (bracket page; moves under `/markets/cbb`)
- `web/lib/data/player-markets.ts` (exports `getProductionPredictionFeed`, `getMlbHomeRunBoardData`, `getMlbHomeRunModelLabel` — do not modify this file)
- `web/components/dashboard/MarketsTable.tsx`, `PageHeader.tsx`, `web/components/ui/*` (existing patterns to match)
- `web/next.config.ts`

## Exact changes

1. **`web/lib/markets-registry.ts` (new).** Export a typed registry used by Markets now (and Performance in a later task):

   ```ts
   export type MarketEntry = { slug: string; label: string; href: string; status: "live" | "scaffold"; description: string };
   export type SportEntry = { slug: "nba" | "mlb" | "pga" | "nfl" | "nhl" | "cbb"; label: string; emphasis: "primary" | "scaffold" | "seasonal"; description: string; markets: MarketEntry[] };
   export const SPORTS: SportEntry[];
   export function getSport(slug: string): SportEntry | undefined;
   ```

   Content: NBA (primary; market: "Spread & winner board" → `/markets/nba`, live), MLB (primary; "Home runs" → `/markets/mlb/home-runs` live, "Winners" → `/markets/mlb` live), PGA (primary; "Tournament board — win/top-10/top-20" → `/markets/pga`, live), NFL (scaffold; "Spread & winner board" → `/markets/nfl`), NHL (scaffold; no models yet → `/markets/nhl`), CBB (seasonal; "Tournament bracket sim" → `/markets/cbb`). Keep descriptions short and factual.

2. **`web/components/dashboard/AppShell.tsx`.** `navItems` becomes exactly: Overview `/`, Markets `/markets`, Performance `/performance`, Insights `/insights`, Data Quality `/data-quality` (this order). Remove the PGA and CBB entries and the now-unused `Trophy` import if nothing else uses it.

3. **`web/app/markets/page.tsx` (rewrite).** `PageHeader` + a sport-card grid driven by `SPORTS` (each card: label, description, market links with `status` shown as a muted badge for scaffolds). Render `emphasis: "primary"` sports first and full-size; render `scaffold`/`seasonal` sports last in a visually muted, smaller row (CBB last). Below the grid, keep the existing "Pre-Live Model Board" card with `MarketsTable` fed by `getProductionPredictionFeed()` unchanged.

4. **Sport pages (new folders under `web/app/markets/`):**
   - `nba/page.tsx`: server component; fetch `getProductionPredictionFeed()`, filter `feed.predictions` to NBA rows (inspect the feed row shape for the league/sport field before filtering), render `PageHeader` + `MarketsTable` with the filtered rows and feed gaps. Empty state text if no NBA rows (off-season).
   - `mlb/page.tsx`: `PageHeader`, market cards from the registry's MLB entry (link to `/markets/mlb/home-runs`), plus `MarketsTable` filtered to MLB rows for winner markets.
   - `mlb/home-runs/page.tsx`: move `web/app/markets/mlb-home-runs/page.tsx` here verbatim; delete the old folder.
   - `pga/page.tsx`: move the entire component from `web/app/pga/page.tsx`. Preferred shape: extract the client component into `web/components/pga/PgaBoard.tsx` (verbatim — **no logic edits**, it keeps fetching `/api/pga-board`) and make the page a thin wrapper rendering it. Delete `web/app/pga/`.
   - `nfl/page.tsx`: like NBA but filtered to NFL; honest empty state ("No live NFL markets — models resume in season").
   - `nhl/page.tsx`: `PageHeader` + a single card stating no NHL models are wired yet (scaffold; no fake data).
   - `cbb/page.tsx`: move `web/app/cbb/page.tsx` verbatim (it's a self-contained client bracket page); delete the old folder. Add a muted note that CBB is seasonal/low-priority.

5. **`web/next.config.ts`.** Add `redirects()`: `/pga` → `/markets/pga`, `/cbb` → `/markets/cbb`, `/markets/mlb-home-runs` → `/markets/mlb/home-runs` (all `permanent: false`).

6. **Sweep old hrefs.** `grep -rn '"/pga"\|"/cbb"\|/markets/mlb-home-runs' web/app web/components web/lib` and update every hit (e.g. the buttons in the old markets hub, any Overview links).

## Constraints

- Do not modify `web/lib/data/*`, `web/app/api/pga-board/route.ts`, or any component logic in the moved PGA/CBB pages — moves are relocations, not rewrites.
- Match existing patterns: `PageHeader`, shadcn `Card`/`Badge`/`Button`, Tailwind tokens already in use. No new dependencies.
- `/results` route stays as-is (not in nav, not removed).
- No commits, no pushes, no changes outside `web/`.

## Done definition

- Nav shows exactly the five items; PGA/CBB reachable only via Markets.
- All routes render: `/markets`, `/markets/{nba,mlb,pga,nfl,nhl,cbb}`, `/markets/mlb/home-runs`; old URLs redirect.
- `grep -rn '"/pga"\|"/cbb"\|mlb-home-runs' web/app web/components web/lib` returns only the redirect config and the new `mlb/home-runs` path.

## Verification

```bash
npm --prefix web run lint
npm --prefix web run build   # must pass with NO Supabase env vars set
```

Then `npm --prefix web run dev` and manually load `/`, `/markets`, `/markets/pga`, `/markets/mlb/home-runs`, `/markets/cbb`, `/pga` (expect redirect), `/cbb` (expect redirect).
