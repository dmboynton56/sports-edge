Implemented Codex task packet 04 exactly.

Key outcomes:

- Statcast preload now refreshes only the trailing configurable window, defaulting to 10 days.
- Historical Statcast cache remains `refresh=False`.
- The existing shared deadline and soft-fail warning behavior are preserved.
- MLB HR’s default slate date now uses `America/Denver`; explicit `--date` behavior is unchanged.
- Added 23:30 and 00:30 Denver boundary tests.
- Discord notification and `sports-edge-refresh` dispatch blocks remain byte-identical.
- No commits or pushes made.

Verification:

- Targeted tests: `11 passed`
- Full suite: `121 passed`
- Workflow YAML: valid
- `git diff --check`: clean

Completion log: [codex-04-done.md](/home/dmboynton/projects/sports-edge/.cursor/orchestration/logs/codex-04-done.md)