# Agent Role Routing — Sports Edge Dashboard v2

Use this file when delegating work across Cursor, Claude Code, and Codex.

## Roles

| Role | Tool | Model tier | Owns |
|------|------|------------|------|
| Planner / architect | `claude -p` | Sonnet | Spec review, injury pipeline design, gate definitions |
| Grinder | `codex exec` | gpt-5.6-sol | SQL migrations, Python scripts, pytest, mechanical review |
| UI + integration | Cursor Composer | Composer | Next.js pages, components, Vercel wiring |
| Deep review | `claude -p` | Sonnet | Pre-merge review against `PRODUCTION_ROADMAP.md` |

## Per-milestone loop

```bash
# 1. Plan (Claude)
claude -p "Read TASK.md + data-core/docs/DASHBOARD_V2_SPEC.md. Steps for M{N}. No edits." < /dev/null

# 2. Backend (Codex)
codex exec -C data-core -s workspace-write -c approval_policy=on-request "..." < /dev/null

# 3. UI (Cursor) — multi-file edits in IDE

# 4. Review
codex exec review --uncommitted < /dev/null
```

## Git isolation
- Branch per milestone: `dashboard/m1-serving`, `dashboard/m2-explanations`, etc.
- Update `TASK.md` handoff after each milestone merge.

## Repo boundaries
- `data-core/` — Python pipeline, SQL, scripts, tests
- `web/` — Next.js dashboard (canonical public surface)
- Do not edit `.cursor/plans/*.plan.md`

## Key docs
- `data-core/docs/DASHBOARD_V2_SPEC.md`
- `data-core/docs/PRODUCTION_ROADMAP.md`
- `data-core/docs/NFL_WEEK_1_READINESS_PLAN.md`
- `data-core/docs/DATA_AND_MODEL_STATUS.md`
