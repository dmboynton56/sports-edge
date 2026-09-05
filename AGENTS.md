# Sports Edge — agent notes

Useful sports-analytics app (predictions, boards, freshness, research). **Not** a profitable betting strategy. Do not invent edges, trading loops, or LLM eval harnesses (no Langfuse / Promptfoo).

## Repo

| Path | Owns |
| --- | --- |
| `data-core/` | Python pipeline, SQL, scripts, pytest |
| `web/` | Next.js dashboard — canonical public surface (Vercel root `web/`) |
| `.github/workflows/` | Daily Refresh, Player Market Refresh, other crons |

Do not edit `.cursor/plans/*.plan.md`. Do not revert landed work: UI makeover, Odds-once-per-day, BQ isolation, MLB research audit.

## Secrets

- Local scripts: `data-core/.env` only. Never commit it; never put secrets elsewhere.
- CI: GitHub Actions secrets (`ODDS_API_KEY`, `PROPLINE_API_KEY`, `SUPABASE_*`, `GCP_SERVICE_ACCOUNT_KEY`, `GCP_PROJECT_ID`, `DISCORD_WEBHOOK_URL`).
- Dashboard: Vercel `NEXT_PUBLIC_SUPABASE_*`.
- GCP key JSON is gitignored (`learned-pier-*.json`). `.github/workflows/verify-key-hygiene.yml` enforces that.

## Refresh ownership

| Workflow | Owns | When |
| --- | --- | --- |
| Daily Refresh (`.github/workflows/daily-refresh.yml`) | League slates, **research MLB** (ML / run line / totals + `audit_mlb_research_readiness`) | Morning cron ~7:05 AM MT |
| Player Market Refresh (`.github/workflows/player-markets-refresh.yml`) | **MLB HR odds** + HR board / player markets | Afternoon cron **after 2pm MT** (`15 20 * * *` ≈ 2:15 PM MT) |

HR odds = PMR. Research MLB = Daily. `run_mlb_hr` on Daily is a deprecated escape hatch — do not make it canonical again.

Odds API is **once per Denver day**. Missing / empty / quota → PropLine (`PROPLINE_API_KEY`). Fail closed if both fail: no invented prices, no fake EV.

Ops detail: `.cursor/skills/sports-edge-ops/SKILL.md`.

## Fail-closed

Missing books or prices → model-only rows, no edge/EV. Missing slate/source → `failed` / `no_slate`. Do not paper over gaps.

## How to work

Default implementation model: **grok-4.6**. Do not prescribe Sonnet or Composer as defaults.

| Role | Tool | Use |
| --- | --- | --- |
| Implement / UI | Cursor | Default path. Multi-file edits, Next.js, Vercel, wiring. |
| Grind | Codex (`codex exec`) | Optional: SQL migrations, pytest, mechanical packets. |
| Harsh review | Cursor slash skills | Before merge — slash-only, not auto. |

Do not dump review rubrics into always-on rules. Skills load name + description until activated.

## Before merge

Run both thermos **explicitly** (`disable-model-invocation`; they will not auto-fire):

- `/thermo-nuclear-review`
- `/thermo-nuclear-code-quality-review`
