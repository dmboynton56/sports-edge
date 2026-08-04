#!/usr/bin/env bash
# Shared helpers for MetricProof CLI orchestration.
set -euo pipefail

export PATH="/home/dmboynton/.nvm/versions/node/v24.14.0/bin:${PATH:-}"

AGENTS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$AGENTS_ROOT/../.." && pwd)"
RUNS_DIR="$AGENTS_ROOT/runs"

FABLE_MODEL="${FABLE_MODEL:-fable}"
FABLE_EFFORT="${FABLE_EFFORT:-high}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.6-sol}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

require_run() {
  local run_id="$1"
  local run_dir="$RUNS_DIR/$run_id"
  if [[ ! -d "$run_dir" ]]; then
    echo "error: run not found: $run_dir" >&2
    echo "create one with: $AGENTS_ROOT/bin/new-run.sh <slug>" >&2
    exit 1
  fi
  echo "$run_dir"
}

monitor() {
  local run_dir="$1"
  local msg="$2"
  mkdir -p "$run_dir/logs"
  echo "[$(ts)] $msg" | tee -a "$run_dir/logs/monitor.md"
}

write_status() {
  local run_dir="$1"
  local phase="$2"
  local next_actor="$3"
  local notes="$4"
  local run_id
  run_id="$(basename "$run_dir")"
  cat > "$run_dir/STATUS.json" <<EOF
{
  "run_id": "$run_id",
  "phase": "$phase",
  "next_actor": "$next_actor",
  "active_tasks": [],
  "completed_tasks": [],
  "goal_done": false,
  "notes": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$notes")
}
EOF
}
