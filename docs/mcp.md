# Sports Edge MCP

Read-only Model Context Protocol endpoint for the public dashboard. Agents can inspect live slates, graded results, and model performance without Odds API, BigQuery, or service-role credentials.

- **MCP (Streamable HTTP, stateless, no auth):** `https://sports-edge.drewboynton.com/api/mcp`
- **Skill:** `https://sports-edge.drewboynton.com/skill/SKILL.md`
- **Stack:** `mcp-handler` v2 + `@modelcontextprotocol/server` (MCP SDK v2). The v1 `@modelcontextprotocol/sdk` package is not used.

Sports Edge is a versioned prediction app, not a betting system. Tools fail closed: missing books stay model-only, empty slates stay empty, and player markets never invent sportsbook EV.

## Tools

| Tool | What it returns |
| --- | --- |
| `get_status` | Freshness, gaps, last publish per league |
| `list_edges` | Ranked rows. `sort=point_edge` (default) or `sort=ev`. Response includes the sort that was used. `limit` ≤ 25 |
| `get_slate` | NBA / NFL / MLB / PGA slate. Date windows are capped |
| `get_game` | NBA/NFL line, probabilities, top explanation features, injury flags. Lookup by id or team |
| `get_results` | Graded W-L, units, game rows. `window`: `last_sunday` \| `7d` \| `season` |
| `get_model_performance` | Metrics, gates, backtests, ROI (often negative) |
| `get_player_markets` | MLB HR or NFL anytime-TD **model** probabilities. No fake book EV |

Every payload includes `as_of`, `model_version`, `freshness`, `status` (`research` / `model-only` / `candidate` / `production`), and `gaps`. Raw feature vectors and `model_runs` are not exposed.

REST equivalents live under `/api/mobile/v1/*` plus `/api/nba-slate`, `/api/nfl-slate`, and `/api/pga-board`. Graded results: `GET /api/mobile/v1/results?league=NFL&window=last_sunday`.

## Cursor

`~/.cursor/mcp.json` or project `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "sports-edge": {
      "url": "https://sports-edge.drewboynton.com/api/mcp"
    }
  }
}
```

Local:

```json
{
  "mcpServers": {
    "sports-edge": {
      "url": "http://localhost:3000/api/mcp"
    }
  }
}
```

## Claude Desktop

Claude Desktop talks stdio, so wrap the remote URL with `mcp-remote`:

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

## Rate limiting

`/api/mcp` and `/api/mobile/*` share an in-process limiter (60 requests / IP / minute) in Next.js 16 `proxy.ts`. This is best-effort on Vercel: each isolate has its own map. Optional extra protection: a Vercel WAF rule on `/api/mcp` and `/api/mobile/*` at the same 60 req/min budget. No extra paid service is required for the in-app limiter.

## Local

```sh
cd web
npm install
npm run dev
npm run mcp:smoke -- http://localhost:3000
```

Evals (fixture snapshots, no live data):

```sh
cd web
npm run eval:mcp
```

`llm-rubric` cases run only when `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set. CI runs the deterministic assertions only.
