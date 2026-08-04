#!/usr/bin/env bash
# Convenience phases. Prefer launching fable/codex in separate terminals for live monitoring.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

usage() {
  cat >&2 <<EOF
Usage: $0 <run-id> <command>

Commands:
  plan       Run Fable planning pass (blocking, tee'd)
  implement  Run all tasks/NN-*.md sequentially via Codex
  review     Run Fable review pass
  loop       plan → implement → review (blocking)

For parallel workers + live watching, open separate terminals:
  $AGENTS_ROOT/bin/codex.sh <run-id> 01
  $AGENTS_ROOT/bin/codex.sh <run-id> 02
EOF
  exit 1
}

[[ $# -ge 2 ]] || usage
RUN_ID="$1"
CMD="$2"
RUN_DIR="$(require_run "$RUN_ID")"

run_implement() {
  shopt -s nullglob
  tasks=("$RUN_DIR/tasks"/[0-9][0-9]-*.md)
  shopt -u nullglob
  if [[ ${#tasks[@]} -eq 0 ]]; then
    echo "error: no tasks in $RUN_DIR/tasks" >&2
    exit 1
  fi
  for task in "${tasks[@]}"; do
    base="$(basename "$task")"
    tid="${base%%-*}"
    monitor "$RUN_DIR" "orchestrate implement → task $tid ($base)"
    "$SCRIPT_DIR/codex.sh" "$RUN_ID" "$tid" "$task"
  done
}

case "$CMD" in
  plan)
    "$SCRIPT_DIR/fable.sh" "$RUN_ID" plan
    ;;
  implement)
    run_implement
    ;;
  review)
    "$SCRIPT_DIR/fable.sh" "$RUN_ID" review
    ;;
  loop)
    monitor "$RUN_DIR" "=== ORCHESTRATION LOOP START ==="
    "$SCRIPT_DIR/fable.sh" "$RUN_ID" plan
    run_implement
    "$SCRIPT_DIR/fable.sh" "$RUN_ID" review
    monitor "$RUN_DIR" "=== ORCHESTRATION LOOP END ==="
    "$SCRIPT_DIR/status.sh" "$RUN_ID"
    ;;
  *)
    usage
    ;;
esac
