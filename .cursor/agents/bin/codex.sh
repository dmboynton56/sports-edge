#!/usr/bin/env bash
# Launch Codex Sol worker with live terminal output (tee'd to run logs).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

usage() {
  echo "Usage: $0 <run-id> <task-id> [prompt-or-task-file]" >&2
  echo "  task-id: NN  (matches tasks/NN-*.md)" >&2
  exit 1
}

[[ $# -ge 2 ]] || usage
RUN_ID="$1"
TASK_ID="$2"
PROMPT_OR_TASK="${3:-}"

RUN_DIR="$(require_run "$RUN_ID")"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/handoffs" "$RUN_DIR/tasks"

TASK_FILE=""
if [[ -n "$PROMPT_OR_TASK" && -f "$PROMPT_OR_TASK" ]]; then
  if [[ "$(basename "$PROMPT_OR_TASK")" == *.prompt.md ]]; then
    RENDERED="$PROMPT_OR_TASK"
  else
    TASK_FILE="$PROMPT_OR_TASK"
  fi
else
  # Prefer exact tasks/NN-*.md
  shopt -s nullglob
  matches=("$RUN_DIR/tasks/${TASK_ID}"-*.md)
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "error: no task file matching $RUN_DIR/tasks/${TASK_ID}-*.md" >&2
    exit 1
  fi
  TASK_FILE="${matches[0]}"
fi

LOG="$RUN_DIR/logs/codex-${TASK_ID}.log"
SESSION_FILE="$RUN_DIR/handoffs/codex-${TASK_ID}.session.id"

if [[ -z "${RENDERED:-}" ]]; then
  RENDERED="$RUN_DIR/handoffs/codex-${TASK_ID}.prompt.md"
  sed \
    -e "s|{{REPO_ROOT}}|$REPO_ROOT|g" \
    -e "s|{{RUN_DIR}}|$RUN_DIR|g" \
    -e "s|{{RUN_ID}}|$RUN_ID|g" \
    -e "s|{{AGENTS_ROOT}}|$AGENTS_ROOT|g" \
    -e "s|{{TASK_ID}}|$TASK_ID|g" \
    -e "s|{{TASK_FILE}}|$TASK_FILE|g" \
    "$AGENTS_ROOT/prompts/codex-impl.md" > "$RENDERED"
  {
    echo
    echo "## Attached task packet"
    echo
    cat "$TASK_FILE"
  } >> "$RENDERED"
fi

monitor "$RUN_DIR" "Codex task $TASK_ID starting — model=$CODEX_MODEL"
monitor "$RUN_DIR" "task=${TASK_FILE:-custom} log=$LOG"

{
  echo "======== $(ts) codex $TASK_ID start ========"
  echo "repo=$REPO_ROOT run=$RUN_ID model=$CODEX_MODEL"
  echo
} | tee -a "$LOG"

set +e
if [[ -f "$SESSION_FILE" && -s "$SESSION_FILE" ]]; then
  (
    codex exec resume "$(cat "$SESSION_FILE")" \
      -C "$REPO_ROOT" \
      --skip-git-repo-check \
      -s workspace-write \
      --dangerously-bypass-approvals-and-sandbox \
      "$(cat "$RENDERED")" \
      < /dev/null
  ) 2>&1 | tee -a "$LOG"
else
  (
    codex exec \
      -m "$CODEX_MODEL" \
      -C "$REPO_ROOT" \
      --skip-git-repo-check \
      -s workspace-write \
      --dangerously-bypass-approvals-and-sandbox \
      "$(cat "$RENDERED")" \
      < /dev/null
  ) 2>&1 | tee -a "$LOG"
fi
status=${PIPESTATUS[0]}
set -e

# Best-effort session id capture for resume
sid=$(rg -o 'session id:[[:space:]]*[0-9a-f-]+' "$LOG" 2>/dev/null | awk '{print $3}' | tail -1 || true)
if [[ -n "${sid:-}" ]]; then
  echo "$sid" > "$SESSION_FILE"
  monitor "$RUN_DIR" "Codex task $TASK_ID session=$sid"
fi

{
  echo
  echo "======== $(ts) codex $TASK_ID end status=$status ========"
} | tee -a "$LOG"

monitor "$RUN_DIR" "Codex task $TASK_ID finished status=$status → $LOG"
exit "$status"
