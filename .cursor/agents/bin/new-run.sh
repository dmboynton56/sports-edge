#!/usr/bin/env bash
# Create a new orchestration run from the template.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

[[ $# -ge 1 ]] || { echo "Usage: $0 <slug>" >&2; exit 1; }
SLUG="$1"
# sanitize slug
SLUG="$(echo "$SLUG" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9-]+/-/g; s/^-+//; s/-+$//')"
RUN_ID="$(date -u +%Y-%m-%d)-${SLUG}"
RUN_DIR="$RUNS_DIR/$RUN_ID"

if [[ -e "$RUN_DIR" ]]; then
  echo "error: run already exists: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"/{tasks,handoffs,logs,artifacts}
cp "$RUNS_DIR/_template/BRIEF.md" "$RUN_DIR/BRIEF.md"
cp "$RUNS_DIR/_template/STATUS.json" "$RUN_DIR/STATUS.json"

# Fill run_id into STATUS
python3 - "$RUN_DIR/STATUS.json" "$RUN_ID" <<'PY'
import json, sys
path, run_id = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
data["run_id"] = run_id
data["notes"] = f"Run created {run_id}. Fill BRIEF.md then launch fable plan."
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

# Stamp BRIEF header
{
  echo "<!-- run_id: $RUN_ID -->"
  echo
  cat "$RUN_DIR/BRIEF.md"
} > "$RUN_DIR/BRIEF.md.tmp"
mv "$RUN_DIR/BRIEF.md.tmp" "$RUN_DIR/BRIEF.md"

monitor "$RUN_DIR" "Run created: $RUN_ID"
echo "$RUN_ID"
echo "Edit: $RUN_DIR/BRIEF.md"
echo "Then: $AGENTS_ROOT/bin/fable.sh $RUN_ID plan"
