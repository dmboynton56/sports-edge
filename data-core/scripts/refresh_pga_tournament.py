#!/usr/bin/env python3
"""Refresh registry-driven PGA tournament predictions and serving JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pga.live_leaderboard import fetch_live_leaderboard, fetch_scoreboard, rounds_completed_from_leaderboard  # noqa: E402
from src.pga.tournament_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    PgaTournament,
    event_status_for_phase,
    infer_phase,
    load_registry,
    resolve_active_tournament,
)


STATE_PATH = ROOT / "notebooks" / "cache" / "pga_refresh_state.json"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _existing_midtournament_state(tournament: PgaTournament) -> str | None:
    meta_path = tournament.midtournament_csv.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return meta.get("round_state_key")


def _field_fetch_command(tournament: PgaTournament) -> list[str] | None:
    if tournament.field_fetcher == "fetch_usopen_field":
        return [
            sys.executable,
            "scripts/fetch_usopen_field.py",
            "--json-out",
            str(tournament.field_json),
            "--text-out",
            str(tournament.field_text),
        ]
    return None


def ensure_field(tournament: PgaTournament, *, force: bool, skip_fetch: bool, dry_run: bool) -> None:
    if tournament.field_json.exists() and not force:
        print(f"Field exists: {tournament.field_json}")
        return
    if skip_fetch:
        if not tournament.field_json.exists():
            raise SystemExit(f"Missing field file and --skip-field-fetch was set: {tournament.field_json}")
        return
    cmd = _field_fetch_command(tournament)
    if not cmd:
        if tournament.field_json.exists():
            return
        raise SystemExit(f"No field fetcher is configured for {tournament.key}; expected {tournament.field_json}")
    _run(cmd, dry_run=dry_run)


def run_pretournament_predictions(tournament: PgaTournament, args: argparse.Namespace, *, dry_run: bool) -> None:
    if tournament.predictions_csv.exists() and not args.force_pre:
        print(f"Pre-tournament predictions exist: {tournament.predictions_csv}")
        return
    ensure_field(tournament, force=args.force_field, skip_fetch=args.skip_field_fetch, dry_run=dry_run)
    cmd = [
        sys.executable,
        "scripts/predict_pga_tournament.py",
        "--tournament-key",
        tournament.key,
        "--event-name",
        tournament.name,
        "--season",
        str(tournament.season),
        "--course-name",
        tournament.course,
        "--course-par",
        str(tournament.par),
        "--start-date",
        tournament.start_date.isoformat(),
        "--end-date",
        tournament.end_date.isoformat(),
        "--as-of",
        tournament.start_date.isoformat(),
        "--field-file",
        str(tournament.field_json),
        "--out-csv",
        str(tournament.predictions_csv),
        "--n-rounds",
        str(tournament.total_rounds),
    ]
    if tournament.yardage is not None:
        cmd.extend(["--course-yardage", str(tournament.yardage)])
    if args.baseline_only:
        cmd.append("--baseline-only")
    _run(cmd, dry_run=dry_run)


def run_midtournament_update(
    tournament: PgaTournament,
    *,
    leaderboard: dict[str, Any],
    args: argparse.Namespace,
    dry_run: bool,
) -> bool:
    rounds_completed = rounds_completed_from_leaderboard(leaderboard, total_rounds=tournament.total_rounds)
    if rounds_completed <= 0:
        print("No completed round state yet; skipping mid-tournament simulation.")
        return False

    state_key = f"{tournament.key}:R{rounds_completed}"
    state = _load_state()
    last_state = (
        state.get(tournament.key, {}).get("last_midtournament_state_key")
        or _existing_midtournament_state(tournament)
    )
    if last_state == state_key and not args.force_mid:
        print(f"Mid-tournament simulation already processed for {state_key}; skipping.")
        return False

    if not tournament.predictions_csv.exists():
        run_pretournament_predictions(tournament, args, dry_run=dry_run)

    cmd = [
        sys.executable,
        "scripts/update_pga_midtournament.py",
        "--tournament-key",
        tournament.key,
        "--event-name",
        tournament.name,
        "--course-par",
        str(tournament.par),
        "--cut-size",
        str(tournament.cut_size),
        "--cut-after-round",
        str(tournament.cut_after_round),
        "--total-rounds",
        str(tournament.total_rounds),
        "--pred-csv",
        str(tournament.predictions_csv),
        "--out-csv",
        str(tournament.midtournament_csv),
        "--n-sims",
        str(args.n_sims),
    ]
    for pattern in tournament.espn_match:
        cmd.extend(["--espn-match", pattern])
    _run(cmd, dry_run=dry_run)

    state[tournament.key] = {
        "last_midtournament_state_key": state_key,
        "updated_at": _now_utc().isoformat(),
    }
    if not dry_run:
        _write_state(state)
    return True


def run_post_results_fetch(tournament: PgaTournament, args: argparse.Namespace, *, dry_run: bool) -> None:
    if args.skip_results_fetch:
        print("Skipping post-tournament ESPN results fetch.")
        return
    cmd = [
        sys.executable,
        "scripts/fetch_espn_pga_results.py",
        "--season",
        str(tournament.season),
        "--as-of",
        _now_utc().isoformat(),
    ]
    _run(cmd, dry_run=dry_run)


def export_dashboard(
    tournament: PgaTournament,
    *,
    phase: str,
    args: argparse.Namespace,
    dry_run: bool,
) -> None:
    if not tournament.predictions_csv.exists() and phase != "post":
        run_pretournament_predictions(tournament, args, dry_run=dry_run)
    cmd = [
        sys.executable,
        "scripts/export_pga_tournament_dashboard.py",
        "--pred-csv",
        str(tournament.predictions_csv),
        "--out",
        str(tournament.public_json),
        "--current-out",
        str(tournament.current_json),
        "--tournament-key",
        tournament.key,
        "--event-name",
        tournament.name,
        "--season",
        str(tournament.season),
        "--course-name",
        tournament.course,
        "--course-par",
        str(tournament.par),
        "--start-date",
        tournament.start_date.isoformat(),
        "--end-date",
        tournament.end_date.isoformat(),
        "--status",
        event_status_for_phase(phase),
    ]
    if tournament.yardage is not None:
        cmd.extend(["--course-yardage", str(tournament.yardage)])
    if tournament.odds_key:
        cmd.extend(["--odds-key", tournament.odds_key])
    if phase != "pre" and tournament.midtournament_csv.exists():
        cmd.extend(["--midtournament-csv", str(tournament.midtournament_csv)])
    for pattern in tournament.espn_match:
        cmd.extend(["--espn-match", pattern])
    if args.skip_odds:
        cmd.append("--skip-odds")
    if args.live_odds:
        cmd.append("--live-odds")
    if args.skip_leaderboard:
        cmd.append("--skip-leaderboard")
    _run(cmd, dry_run=dry_run)


def sync_outputs(tournament: PgaTournament, args: argparse.Namespace, *, dry_run: bool) -> None:
    if args.sync_bigquery:
        cmd = [
            sys.executable,
            "scripts/sync_player_markets_to_bigquery.py",
            "--pga-json",
            str(tournament.current_json),
            "--skip-mlb",
        ]
        if args.project:
            cmd.extend(["--project", args.project])
        _run(cmd, dry_run=dry_run)
    if args.sync_supabase:
        _run(
            [
                sys.executable,
                "scripts/sync_player_markets_to_supabase.py",
                "--pga-json",
                str(tournament.current_json),
                "--skip-mlb",
            ],
            dry_run=dry_run,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh active PGA tournament automation.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--tournament-key", default="")
    parser.add_argument("--as-of", default="", help="YYYY-MM-DD anchor date. Defaults to current UTC date.")
    parser.add_argument("--force-phase", choices=["pre", "live", "post"], default="")
    parser.add_argument("--force-field", action="store_true")
    parser.add_argument("--force-pre", action="store_true")
    parser.add_argument("--force-mid", action="store_true")
    parser.add_argument("--skip-field-fetch", action="store_true")
    parser.add_argument("--skip-results-fetch", action="store_true")
    parser.add_argument("--skip-leaderboard", action="store_true")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--live-odds", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--n-sims", type=int, default=50000)
    parser.add_argument("--sync-supabase", action="store_true")
    parser.add_argument("--sync-bigquery", action="store_true")
    parser.add_argument("--project", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)
    anchor = args.as_of or _now_utc().date().isoformat()
    scoreboard = None if args.skip_leaderboard else fetch_scoreboard()
    tournament = resolve_active_tournament(
        registry,
        tournament_key=args.tournament_key or None,
        as_of=anchor,
        scoreboard=scoreboard,
    )
    if not tournament:
        print(f"No PGA tournament is active for automation window at {anchor}; no-op.")
        return

    leaderboard = None
    if not args.skip_leaderboard:
        leaderboard = fetch_live_leaderboard(espn_match=tournament.espn_match, scoreboard=scoreboard)

    phase = infer_phase(
        tournament,
        as_of=anchor,
        leaderboard=leaderboard,
        force_phase=args.force_phase or None,
    )
    print(f"Resolved PGA tournament: {tournament.key} ({tournament.name})")
    print(f"Refresh phase: {phase}")

    if phase == "pre":
        run_pretournament_predictions(tournament, args, dry_run=args.dry_run)
    elif phase == "live":
        if not tournament.predictions_csv.exists():
            run_pretournament_predictions(tournament, args, dry_run=args.dry_run)
        if leaderboard:
            run_midtournament_update(tournament, leaderboard=leaderboard, args=args, dry_run=args.dry_run)
        else:
            print("No matched ESPN leaderboard available; skipping mid-tournament update.")
    elif phase == "post":
        if not tournament.predictions_csv.exists():
            run_pretournament_predictions(tournament, args, dry_run=args.dry_run)
        run_post_results_fetch(tournament, args, dry_run=args.dry_run)

    export_dashboard(tournament, phase=phase, args=args, dry_run=args.dry_run)
    sync_outputs(tournament, args, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
