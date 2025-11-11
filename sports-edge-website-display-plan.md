# Sports-Edge: Site Access & Display Plan (Next.js + Supabase)

> Goal: Show **today’s games** with **book spread vs our spread** and **home win%** directly on the portfolio, keeping the UI fast, secure, and decoupled from the modeling job.

---

## 1) Server API Route (Next.js App Router)
Create `app/api/sports-edges/route.ts`:

```ts
import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

export async function GET() {
  const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY! // server-only; NEVER expose to client
  )

  const { data, error } = await supabase
    .from('games_today_enriched')
    .select('*')
    .order('game_time_utc', { ascending: true })

  if (error) return NextResponse.json({ error: error.message }, { status: 500 })
  return NextResponse.json(data, {
    headers: { 'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=120' }
  })
}
```

**Why**: server-only secrets, minimal JSON, easy caching, and your React card simply fetches from `/api/sports-edges`.

---

## 2) Client Component (Project Card)
`components/SportsEdgeCard.tsx`

```tsx
'use client'
import { useEffect, useState } from 'react'

type EdgeRow = {
  league: string
  season: number
  game_time_utc: string
  home_team: string
  away_team: string
  book_spread: number | null
  my_spread: number | null
  edge_pts: number | null
  my_home_win_prob: number | null
  model_version: string | null
  asof_ts: string | null
}

export default function SportsEdgeCard() {
  const [rows, setRows] = useState<EdgeRow[]>([])
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/sports-edges')
      .then(r => r.ok ? r.json() : r.json().then(j => Promise.reject(j.error)))
      .then(setRows)
      .catch(e => setErr(String(e)))
  }, [])

  if (err) return <div className="rounded-xl border p-4 text-red-600 text-sm">Error: {err}</div>

  return (
    <div className="rounded-2xl border p-4">
      <div className="mb-2 text-sm opacity-70">Today’s Edges (model vs books)</div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {rows.map((r, i) => (
          <div key={i} className="rounded-xl border p-3">
            <div className="text-xs opacity-60">
              {r.league} • {new Date(r.game_time_utc).toLocaleString()}
            </div>
            <div className="text-sm font-medium mt-1">
              {r.away_team} @ {r.home_team}
            </div>
            <div className="text-xs mt-1">
              Book: {r.book_spread?.toFixed(1) ?? '—'} | Ours: {r.my_spread?.toFixed(1) ?? '—'}
            </div>
            <div className={`text-xs mt-1 ${Math.abs(r.edge_pts ?? 0) >= 1.0 ? 'text-emerald-600' : 'opacity-70'}`}>
              Edge: {r.edge_pts?.toFixed(1) ?? '—'} pts • Home win: {r.my_home_win_prob != null ? (100*r.my_home_win_prob).toFixed(1) + '%' : '—'}
            </div>
            <div className="text-[10px] opacity-60 mt-1">
              Updated: {r.asof_ts ? new Date(r.asof_ts).toLocaleTimeString() : '—'} • v{r.model_version ?? '—'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
```

Place `<SportsEdgeCard />` inside your **Work/Projects** section card.

---

## 3) Optional: Dedicated Page
If you want deeper charts/tables, create `app/projects/sports-edge/page.tsx` that SSR-fetches `/api/sports-edges` and renders a table with sorting/filtering.

---

## 4) Performance & Reliability
- **Caching**: 60s CDN cache on API route; client re-fetch on focus for freshness.
- **Fallback**: show placeholder “No games today” if array empty.
- **Partial data**: gracefully handle `null` spreads/preds when upstream fetch lags.
- **Timezone**: format using America/Denver for display; DB stores UTC.

---

## 5) Security
- Enable **RLS** with read-only policies for public.
- Only the API route uses the **service role key**; the client never does.
- For admin-only routes (e.g., backfills), guard with auth middleware.

---

## 6) QA Checklist
- [ ] API route returns 200 with array (today or empty).
- [ ] Card renders at least one row on a game day.
- [ ] Edge sign logic confirmed: `edge_pts = my_spread - book_spread`.
- [ ] Accessibility: readable at mobile widths; semantic headings.
- [ ] No PII or secrets in client bundles.
