# CLI agent orchestration (MetricProof)

Cursor = control center. Claude Fable = brain. Codex Sol = workers. Live terminals via tee'd launchers.

## Quick start

```bash
# 1. New run
.cursor/agents/bin/new-run.sh mvp-scaffold
# → prints run-id like 2026-07-20-mvp-scaffold

# 2. Edit the brief
$EDITOR .cursor/agents/runs/<run-id>/BRIEF.md

# 3. Plan (watch this terminal)
.cursor/agents/bin/fable.sh <run-id> plan

# 4. Workers — prefer separate terminals for parallel packets
.cursor/agents/bin/codex.sh <run-id> 01
.cursor/agents/bin/codex.sh <run-id> 02

# 5. Review
.cursor/agents/bin/fable.sh <run-id> review

# Status
.cursor/agents/bin/status.sh <run-id>
```

Blocking convenience (less ideal for watching parallel workers):

```bash
.cursor/agents/bin/orchestrate.sh <run-id> loop
```

## Docs

- [`ROLES.md`](ROLES.md) — who does what
- [`PROTOCOL.md`](PROTOCOL.md) — handoffs + STATUS schema
- [`fable.md`](fable.md) / [`codex-worker.md`](codex-worker.md) — role cards

## Env overrides

| Var | Default |
|---|---|
| `FABLE_MODEL` | `fable` |
| `FABLE_EFFORT` | `high` |
| `CODEX_MODEL` | `gpt-5.6-sol` |
