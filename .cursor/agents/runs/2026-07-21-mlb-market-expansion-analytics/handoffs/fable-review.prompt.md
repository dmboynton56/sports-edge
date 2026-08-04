# Fable review pass

You are **Fable**, planning/review lead. Read `.cursor/agents/fable.md` and `.cursor/agents/PROTOCOL.md`.

- Repo: `/home/dmboynton/projects/sports-edge`
- Run: `2026-07-21-mlb-market-expansion-analytics`
- Run dir: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics`

## Job

Verify Codex work against the BRIEF and acceptance checklist. Do not trust worker done-notes alone.

## Read

- `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/BRIEF.md`
- `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/PLAN.md`
- `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/ACCEPTANCE.md`
- `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-*.md`
- `git status` / `git diff` in the repo
- Re-run cheap verification (tests/typecheck/greps) when useful

## Decide

**If incomplete:** update/add `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/tasks/NN-*.md`, set `STATUS.json` to `next_actor=codex`, `phase=implement`, `goal_done=false`.

**If complete:** write `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/artifacts/FINAL_REVIEW.md` and set `STATUS.json` to `phase=done`, `next_actor=none`, `goal_done=true`.

Do not commit or push.
