#!/usr/bin/env python3
"""Pre-deploy gate for Sports Edge dashboard v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dashboard pre-deploy readiness gate.")
    parser.add_argument("--league", action="append", choices=["NBA", "NFL"], default=["NBA", "NFL"])
    parser.add_argument("--skip-strict-validation", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []

    if not args.skip_strict_validation:
        code, output = run([sys.executable, "scripts/validate_supabase_sync.py", "--strict"])
        print(output)
        if code != 0:
            failures.append("validate_supabase_sync.py --strict failed")

    for league in args.league:
        code, output = run(
            [sys.executable, "scripts/audit_season_readiness.py", "--league", league, "--json"]
        )
        print(output)
        if code != 0:
            failures.append(f"audit_season_readiness.py --league {league} failed")
            continue
        try:
            report = json.loads(output.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            failures.append(f"Could not parse audit JSON for {league}")
            continue
        if report.get("scheduled_games", 0) > 0 and report.get("games_with_prediction", 0) == 0:
            failures.append(f"{league} has scheduled games but zero predictions")

    if failures:
        for failure in failures:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("Dashboard pre-deploy gate passed.")


if __name__ == "__main__":
    main()
