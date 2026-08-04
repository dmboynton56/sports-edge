# Agent roles — MetricProof CLI orchestration

Cursor is the **control center**. You talk in the Agents Window. Claude Fable and Codex Sol do the real work in visible terminals.

| Role | Who | CLI | Model | Job |
|---|---|---|---|---|
| **Dispatcher** | Cursor agent | (spawns shells) | cheap default | Write BRIEF, launch terminals, watch `STATUS.json` + logs, report to you |
| **Brain** | Claude Code | `claude` | `fable` | Plan, decompose, review, decide done/not-done |
| **Worker** | Codex | `codex exec` | `gpt-5.6-sol` | Implement + test assigned packets |

## Cost rule

- Fable only for planning gates and review gates (short, high-leverage).
- Sol for the long implement/test loops (parallel packets when file-disjoint).
- Cursor never bulk-implements product code under this protocol.

## Role cards

- Brain: [`fable.md`](fable.md)
- Worker: [`codex-worker.md`](codex-worker.md)
- Handoff contract: [`PROTOCOL.md`](PROTOCOL.md)

## Launchers (always tee so terminals stay readable)

```bash
.cursor/agents/bin/fable.sh  <run-id> plan|review [prompt-file]
.cursor/agents/bin/codex.sh  <run-id> <task-id> [prompt-file]
.cursor/agents/bin/status.sh <run-id>
.cursor/agents/bin/new-run.sh <slug>
.cursor/agents/bin/orchestrate.sh <run-id> plan|implement|review|loop
```
