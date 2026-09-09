# Epic: Sports Edge Dashboard v2

## Current step
- **Milestone:** M5 — Season readiness gates (complete)
- **Owner:** cursor
- **Status:** completed

## Goal
Make `web/` the canonical public dashboard on Vercel with NBA/NFL slates, game explanations, eval/strategy UI, and injury-aware daily refresh.

## Handoff notes
- Serving layer: `sports-edge/web` (not personal-portfolio)
- Deploy: Vercel production with `NEXT_PUBLIC_SUPABASE_*`
- v1 excludes: auth/paywall, portfolio sync, static NBA/NFL JSON fallback
- Vercel root directory: `sports-edge/web`
- Pre-deploy gate: `data-core/scripts/dashboard_predeploy_gate.py`

## Spend log
| Date | Tool | Model | Task | Notes |
|------|------|-------|------|-------|
| 2026-07-12 | Cursor | Composer | M0–M5 full implementation | Dashboard v2 plan execution |

## September maintenance follow-up
- Repository audit and five site/pipeline improvements are implemented on `codex/repo-performance-audit`; not merged or deployed.
- Verification and remaining work: [site hardening report](docs/SITE_HARDENING_2026-09-04.md).
- Current production evidence: [Data and Model Status](data-core/docs/DATA_AND_MODEL_STATUS.md). The June production roadmap retains historical milestone statuses.
