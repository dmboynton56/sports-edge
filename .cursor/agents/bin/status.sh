#!/usr/bin/env bash
# Print run STATUS + recent monitor lines + log tips.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

[[ $# -ge 1 ]] || { echo "Usage: $0 <run-id>" >&2; exit 1; }
RUN_ID="$1"
RUN_DIR="$(require_run "$RUN_ID")"

echo "== run: $RUN_ID =="
echo "dir: $RUN_DIR"
echo

if [[ -f "$RUN_DIR/STATUS.json" ]]; then
  echo "-- STATUS.json --"
  cat "$RUN_DIR/STATUS.json"
  echo
else
  echo "(no STATUS.json yet)"
  echo
fi

if [[ -f "$RUN_DIR/logs/monitor.md" ]]; then
  echo "-- monitor (last 20) --"
  tail -n 20 "$RUN_DIR/logs/monitor.md"
  echo
fi

echo "-- artifacts --"
ls -1 "$RUN_DIR"/{BRIEF,PLAN,ACCEPTANCE}.md 2>/dev/null || true
ls -1 "$RUN_DIR/tasks"/*.md 2>/dev/null || echo "(no tasks yet)"
echo
echo "-- logs --"
ls -1 "$RUN_DIR/logs"/*.log 2>/dev/null || echo "(no logs yet)"
echo
echo "Watch live: open the terminal that ran fable.sh / codex.sh"
echo "Or: tail -f $RUN_DIR/logs/<name>.log"
