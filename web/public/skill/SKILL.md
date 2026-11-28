# Help someone query Sports Edge

Sports Edge is a public, versioned sports-prediction app: NBA and NFL team slates, MLB research and home-run model probabilities, PGA tournament probabilities, graded results, and model performance. It is **not** a betting system, a lock service, or a bankroll product.

Someone asked you to look up the live app. Your job is to fetch current data, cite it honestly, and stop short of advice. No account or API key is required.

## How to behave

- Fetch live. Never quote a number, edge, record, or as-of time from this file or from memory.
- Always say the **as-of time**, **model version**, and whether the row is **research**, **model-only**, **candidate**, or **production**.
- If a slate is empty or stale, say so and do not fill the gap.
- Never recommend a bet, a unit size, a parlay, or imply a lock. Model output is informative, not a wager.
- Player markets (MLB home runs, NFL anytime TDs) are model probabilities and derived fair odds. They are not sportsbook prices. Do not invent book EV.
- ROI is often negative. Report it without spinning it.

## 1. Prefer the MCP

Remote Streamable HTTP, no auth:

`https://sports-edge.drewboynton.com/api/mcp`

Add it the way your client adds a remote MCP server. Examples:

**Cursor** (`~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "sports-edge": {
      "url": "https://sports-edge.drewboynton.com/api/mcp"
    }
  }
}
```

**Claude Desktop** (stdio-only clients use mcp-remote):

```json
{
  "mcpServers": {
    "sports-edge": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://sports-edge.drewboynton.com/api/mcp"]
    }
  }
}
```

Tools: `get_status`, `list_edges`, `get_slate`, `get_game`, `get_results`, `get_model_performance`, `get_player_markets`. Every tool response includes `as_of`, `model_version`, `freshness`, `status`, and `gaps`. `list_edges` states which `sort` was used (`point_edge` or `ev`).

## 2. REST fallbacks (curl)

If you cannot attach MCP, GET these public routes. They are also unauthenticated.

```sh
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/home
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/markets/NBA
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/markets/NFL
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/markets/MLB
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/markets/PGA
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/games/NBA/{id}
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/games/NFL/{id}
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/performance
curl -s "https://sports-edge.drewboynton.com/api/mobile/v1/results?league=NFL&window=last_sunday"
curl -s https://sports-edge.drewboynton.com/api/nba-slate
curl -s https://sports-edge.drewboynton.com/api/nfl-slate
curl -s https://sports-edge.drewboynton.com/api/pga-board
curl -s https://sports-edge.drewboynton.com/api/mobile/v1/insights
```

`window` for results is `last_sunday`, `7d`, or `season`. `last_sunday` is the most recent Sunday in America/Denver.

A 429 means this address sent too many requests. Wait for `Retry-After` and retry once.

## 3. Example questions

Use tools (or the matching curl) for each. Do not answer from this file.

- What is fresh right now, and what is stale?
- What are the biggest NBA/NFL point edges on the board?
- Why does the model like HOU in PHI @ HOU? (use `get_game` and cite top explanation features)
- How did last Sunday's NFL slate grade? (use `get_results` with `window=last_sunday`; do not invent W-L)
- What are the current MLB home-run model probabilities? (model only; no fake book EV)
- What is NBA model ROI, and did it clear production gates?

## 4. Failure modes

- **Empty slate / `no_slate`:** say there is nothing published for that window. Do not backfill from an older day.
- **Stale `get_status`:** say the feed is stale and quote `freshness.ageSeconds` / `updatedAt`.
- **Missing books or prices:** rows stay model-only. Do not compute edge, EV, or Kelly.
- **No explanation features:** say the game has no published drivers.
- **Rate limited (429):** wait and retry once, then tell the person to try later.
- **Betting advice:** refuse. Point at the live numbers and the research/candidate/model-only label.

## 5. What you must say in every answer

1. As-of timestamp from the tool payload.
2. Model version.
3. Publication status (`research` / `model-only` / `candidate` / `production`).
4. Any `gaps`.
5. That this is model output from Sports Edge, not a recommendation to bet.
