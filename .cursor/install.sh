#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for Sports Edge.
# Runs after the repo is checked out. Prepares the Next.js dashboard (web/)
# and the Python data pipeline (data-core/).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> web: installing Node dependencies (npm ci)"
cd "${repo_root}/web"
npm ci

echo "==> data-core: creating/refreshing Python venv"
cd "${repo_root}/data-core"
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest

echo "==> install complete"
