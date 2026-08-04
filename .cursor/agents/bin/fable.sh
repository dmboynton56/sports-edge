#!/usr/bin/env bash
# Launch Claude Fable with live terminal output (tee'd to run logs).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

usage() {
  echo "Usage: $0 <run-id> plan|review [prompt-file]" >&2
  exit 1
}

[[ $# -ge 2 ]] || usage
RUN_ID="$1"
MODE="$2"
PROMPT_FILE="${3:-}"

RUN_DIR="$(require_run "$RUN_ID")"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/handoffs" "$RUN_DIR/artifacts" "$RUN_DIR/tasks"

case "$MODE" in
  plan)
    LOG="$RUN_DIR/logs/fable-plan.log"
    DEFAULT_PROMPT="$AGENTS_ROOT/prompts/fable-plan.md"
    PHASE_NOTE="Fable plan starting"
    ;;
  review)
    LOG="$RUN_DIR/logs/fable-review.log"
    DEFAULT_PROMPT="$AGENTS_ROOT/prompts/fable-review.md"
    PHASE_NOTE="Fable review starting"
    ;;
  *)
    usage
    ;;
esac

PROMPT_FILE="${PROMPT_FILE:-$DEFAULT_PROMPT}"
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "error: prompt not found: $PROMPT_FILE" >&2
  exit 1
fi

# Materialize prompt with run paths substituted.
RENDERED="$RUN_DIR/handoffs/fable-${MODE}.prompt.md"
sed \
  -e "s|{{REPO_ROOT}}|$REPO_ROOT|g" \
  -e "s|{{RUN_DIR}}|$RUN_DIR|g" \
  -e "s|{{RUN_ID}}|$RUN_ID|g" \
  -e "s|{{AGENTS_ROOT}}|$AGENTS_ROOT|g" \
  "$PROMPT_FILE" > "$RENDERED"

monitor "$RUN_DIR" "$PHASE_NOTE — model=$FABLE_MODEL effort=$FABLE_EFFORT"
monitor "$RUN_DIR" "prompt=$RENDERED log=$LOG"

{
  echo "======== $(ts) fable $MODE start ========"
  echo "repo=$REPO_ROOT run=$RUN_ID model=$FABLE_MODEL"
  echo
} | tee -a "$LOG"

set +e
(
  cd "$REPO_ROOT"
  claude -p \
    --model "$FABLE_MODEL" \
    --effort "$FABLE_EFFORT" \
    --dangerously-skip-permissions \
    --output-format text \
    "$(cat "$RENDERED")" \
    < /dev/null
) 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

{
  echo
  echo "======== $(ts) fable $MODE end status=$status ========"
} | tee -a "$LOG"

monitor "$RUN_DIR" "Fable $MODE finished status=$status → $LOG"
exit "$status"
