# Sports Edge Mobile API Contract

The native app consumes the versioned Next.js facade under `/api/mobile/v1`.
The facade owns Supabase normalization so the Swift client never needs a
service credential or duplicated query logic.

## Envelope

Every successful response is an `APIEnvelope<T>`:

```json
{
  "schemaVersion": "1.0",
  "generatedAt": "2026-08-12T18:30:00.000Z",
  "data": {},
  "gaps": [],
  "freshness": {
    "status": "fresh",
    "source": "supabase",
    "updatedAt": "2026-08-12T18:15:00.000Z",
    "ageSeconds": 900
  }
}
```

`freshness.status` is one of `fresh`, `stale`, `missing`, or `offline`.
`source` is one of `supabase`, `static_json`, `mixed`, `fixture`, or
`unavailable`. The `gaps` array is user-visible data quality context, not a
transport error.

## Routes

- `GET /api/mobile/v1/home` — top NBA/NFL edges and league summaries.
- `GET /api/mobile/v1/markets/{league}` — normalized team or player market
  rows for `NBA`, `NFL`, `MLB`, or `PGA`.
- `GET /api/mobile/v1/games/{league}/{id}` — a team prediction plus persisted
  explanation for an NBA or NFL game.
- `GET /api/mobile/v1/performance` — normalized performance history and
  production gates.
- `GET /api/mobile/v1/insights` — model evaluations, strategy evidence, and
  data-quality summaries.

All fields use lower camel case. Additive fields are allowed within a schema
version; changing the meaning or type of a field requires a new version.

## Native navigation

The SwiftUI shell is a `TabView` with Home, Markets, Performance, Insights,
and Settings tabs. Each tab owns a `NavigationStack`. Market rows push a
native game detail route, and `sportsedge://game/{league}/{id}` deep links
select Markets and push the same route. Settings exposes fixture/live mode,
tracked leagues, sorting, and the read-only product boundary.
