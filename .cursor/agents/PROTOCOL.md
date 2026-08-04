# Handoff protocol

Filesystem is the bus. Terminals stream live output (via `tee`). Artifacts under `.cursor/agents/runs/<run-id>/` are source of truth.

## Run layout

```
runs/<run-id>/
  BRIEF.md              # human / dispatcher intent
  PLAN.md               # Fable plan
  ACCEPTANCE.md         # Fable review checklist
  STATUS.json           # machine-readable phase pointer
  tasks/NN-<slug>.md    # self-contained Codex packets
  handoffs/
    CODEX_DONE-NN.md    # worker completion notes
  artifacts/            # plans, reviews, research dumps
  logs/
    fable-plan.log
    fable-review.log
    codex-NN.log
    monitor.md          # phase transitions (append-only)
```

## STATUS.json schema

```json
{
  "run_id": "2026-07-20-mvp",
  "phase": "plan|implement|review|done",
  "next_actor": "fable|codex|dispatcher|none",
  "active_tasks": ["01", "03"],
  "completed_tasks": ["02"],
  "goal_done": false,
  "notes": "one-line status"
}
```

Only Fable sets `goal_done: true`. Workers never declare the overall goal done.

## Loop

1. **Dispatcher** — `new-run.sh`, fill `BRIEF.md`, set `next_actor=fable`, `phase=plan`.
2. **Fable plan** — explore repo; write `PLAN.md`, `ACCEPTANCE.md`, `tasks/NN-*.md`; set `next_actor=codex`, `phase=implement`. No product code.
3. **Codex workers** — one terminal per packet (parallel if file-disjoint). Each writes `handoffs/CODEX_DONE-NN.md`.
4. **Fable review** — verify against `ACCEPTANCE.md` + git diff. Either more tasks (`next_actor=codex`) or `goal_done=true`.
5. **Dispatcher** — report outcome to the user; do not commit unless asked.

## Task packet requirements (Codex)

Every `tasks/NN-*.md` must include:

- Goal (1–3 sentences)
- Context paths to read first
- Exact changes (files + behavior)
- Constraints / out-of-scope
- Done definition (verifiable)
- "Do not commit or push"

## Terminal visibility

Launchers **must** stream to the terminal (`tee`). Do not redirect away from stdout. Dispatcher starts long jobs with background shells so the user can watch the terminal panel.
