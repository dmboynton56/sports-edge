#!/usr/bin/env python3
"""Refresh registry-driven PGA tournament predictions and serving JSON."""

from __future__ import annotations

import argparse
import json
import tempfile
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pga.live_leaderboard import (  # noqa: E402
    EspnScoreboardError,
    event_matches,
    fetch_leaderboard_event,
    fetch_scoreboard,
    rounds_completed_from_leaderboard,
)
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


def _should_skip_leaderboard(phase: str, explicit: bool, *, has_snapshot: bool = False) -> bool:
    """Skip live retrieval outside live/post phases unless a snapshot is supplied."""

    return explicit or phase == "pre" or (phase == "post" and not has_snapshot)


def _should_preserve_last_good_output(phase: str, leaderboard: dict[str, Any] | None) -> bool:
    """Return whether a live/post refresh must no-op to protect the last good data."""

    return phase in {"live", "post"} and not leaderboard


def _write_leaderboard_snapshot(event: dict[str, Any], leaderboard: dict[str, Any]) -> Path:
    """Write one normalized ESPN artifact shared by mid-update/export/results."""

    competition = (event.get("competitions") or [{}])[0]
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="sports-edge-pga-",
        delete=False,
    )
    path = Path(handle.name)
    try:
        json.dump(
            {"event": event, "competition": competition, "leaderboard": leaderboard},
            handle,
            sort_keys=True,
        )
    finally:
        handle.close()
    return path


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
    # Check for latest round-specific file first
    for round_no in range(tournament.total_rounds, 0, -1):
        meta_path = tournament.midtournament_csv.with_name(
            f"{tournament.key}_midtournament_R{round_no}.meta.json"
        )
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if meta.get("score_source") == "espn_completed_round_scores_v1":
                return meta.get("round_state_key")
    
    # Fall back to legacy single-file name
    meta_path = tournament.midtournament_csv.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta.get("score_source") != "espn_completed_round_scores_v1":
        return None
    return meta.get("round_state_key")


def _field_fetch_command(tournament: PgaTournament) -> list[str] | None:
    if tournament.field_source == "espn":
        if not tournament.espn_event_id:
            raise SystemExit(f"{tournament.key} uses field_source=espn but has no espn_event_id.")
        cmd = [
            sys.executable,
            "scripts/fetch_pga_field.py",
            "--event-id",
            tournament.espn_event_id,
            "--event-key",
            tournament.key,
            "--event-name",
            tournament.name,
            "--season",
            str(tournament.season),
            "--course",
            tournament.course,
            "--par",
            str(tournament.par),
            "--start-date",
            tournament.start_date.isoformat(),
            "--end-date",
            tournament.end_date.isoformat(),
            "--json-out",
            str(tournament.field_json),
            "--text-out",
            str(tournament.field_text),
        ]
        if tournament.yardage is not None:
            cmd.extend(["--yardage", str(tournament.yardage)])
        return cmd
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


def _live_event_unmatched(registry, scoreboard: dict[str, Any] | None, *, as_of: date | None = None) -> str | None:
    today = as_of or _now_utc().date()
    for event in (scoreboard or {}).get("events") or []:
        comp = (event.get("competitions") or [{}])[0]
        status_type = (comp.get("status") or {}).get("type") or (event.get("status") or {}).get("type") or {}
        state = str(status_type.get("state") or "").lower()
        completed = bool(status_type.get("completed"))
        starts = str(event.get("date") or "")[:10]
        ends = str(event.get("endDate") or event.get("date") or "")[:10]
        in_event_window = False
        try:
            in_event_window = starts <= today.isoformat() <= ends
        except TypeError:
            in_event_window = False
        if state not in {"in", "pre"} and completed and not in_event_window:
            continue
        best_match = max(
            (event_matches(event, tournament.espn_match) for tournament in registry.tournaments),
            default=0,
        )
        if best_match <= 0:
            return str(event.get("name") or event.get("shortName") or event.get("id"))
    return None


def ensure_field(
    tournament: PgaTournament,
    *,
    force: bool,
    skip_fetch: bool,
    dry_run: bool,
    allow_unavailable: bool = False,
) -> bool:
    if tournament.field_json.exists() and not force:
        print(f"Field exists: {tournament.field_json}")
        return True
    if skip_fetch:
        if not tournament.field_json.exists():
            raise SystemExit(f"Missing field file and --skip-field-fetch was set: {tournament.field_json}")
        return True
    cmd = _field_fetch_command(tournament)
    if not cmd:
        if tournament.field_json.exists():
            return True
        raise SystemExit(f"No field fetcher is configured for {tournament.key}; expected {tournament.field_json}")
    try:
        _run(cmd, dry_run=dry_run)
    except subprocess.CalledProcessError:
        if allow_unavailable and not tournament.field_json.exists():
            print(
                f"WARNING: ESPN has not published a usable field for {tournament.key}; "
                "deferring pre-tournament predictions until the next refresh."
            )
            return False
        raise
    return tournament.field_json.exists() or dry_run


def run_pretournament_predictions(
    tournament: PgaTournament,
    args: argparse.Namespace,
    *,
    dry_run: bool,
    allow_unavailable_field: bool = False,
) -> bool:
    if tournament.predictions_csv.exists() and not args.force_pre:
        print(f"Pre-tournament predictions exist: {tournament.predictions_csv}")
        return True
    field_ready = ensure_field(
        tournament,
        force=args.force_field,
        skip_fetch=args.skip_field_fetch,
        dry_run=dry_run,
        allow_unavailable=allow_unavailable_field,
    )
    if not field_ready:
        return False
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
    return True


def run_midtournament_update(
    tournament: PgaTournament,
    *,
    leaderboard: dict[str, Any],
    leaderboard_snapshot: Path | None = None,
    args: argparse.Namespace,
    dry_run: bool,
) -> bool:
    rounds_completed = rounds_completed_from_leaderboard(leaderboard, total_rounds=tournament.total_rounds)
    if rounds_completed <= 0:
        print("No completed round state yet; skipping mid-tournament simulation.")
        return False

    state_key = f"{tournament.key}:R{rounds_completed}"
    state = _load_state()
    tournament_state = state.get(tournament.key, {})
    last_state = None
    if tournament_state.get("score_source") == "espn_completed_round_scores_v1":
        last_state = tournament_state.get("last_midtournament_state_key")
    last_state = last_state or _existing_midtournament_state(tournament)
    if last_state == state_key and not args.force_mid:
        print(f"Mid-tournament simulation already processed for {state_key}; skipping.")
        return False

    if not tournament.predictions_csv.exists():
        run_pretournament_predictions(tournament, args, dry_run=dry_run)

    # Use round-specific filename instead of overwriting single file
    midtournament_csv = tournament.midtournament_csv.with_name(
        f"{tournament.key}_midtournament_R{rounds_completed}.csv"
    )
    
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
        str(midtournament_csv),
        "--n-sims",
        str(args.n_sims),
    ]
    for pattern in tournament.espn_match:
        cmd.extend(["--espn-match", pattern])
    if tournament.espn_event_id:
        cmd.extend(["--espn-event-id", tournament.espn_event_id])
    if leaderboard_snapshot:
        cmd.extend(["--leaderboard-json", str(leaderboard_snapshot)])
    _run(cmd, dry_run=dry_run)

    state[tournament.key] = {
        "last_midtournament_state_key": state_key,
        "score_source": "espn_completed_round_scores_v1",
        "updated_at": _now_utc().isoformat(),
    }
    if not dry_run:
        _write_state(state)
    return True


def run_post_results_fetch(
    tournament: PgaTournament,
    args: argparse.Namespace,
    *,
    leaderboard_snapshot: Path | None = None,
    dry_run: bool,
) -> None:
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
    if leaderboard_snapshot:
        cmd.extend(["--event-json", str(leaderboard_snapshot)])
    _run(cmd, dry_run=dry_run)


def export_dashboard(
    tournament: PgaTournament,
    *,
    phase: str,
    args: argparse.Namespace,
    leaderboard_snapshot: Path | None = None,
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
    if phase != "pre":
        # Find the latest round-specific midtournament CSV
        latest_midtournament = None
        for round_no in range(tournament.total_rounds, 0, -1):
            candidate = tournament.midtournament_csv.with_name(
                f"{tournament.key}_midtournament_R{round_no}.csv"
            )
            if candidate.exists():
                latest_midtournament = candidate
                break
        # Fall back to legacy single-file name
        if latest_midtournament is None and tournament.midtournament_csv.exists():
            latest_midtournament = tournament.midtournament_csv
        if latest_midtournament:
            cmd.extend(["--midtournament-csv", str(latest_midtournament)])
    for pattern in tournament.espn_match:
        cmd.extend(["--espn-match", pattern])
    if tournament.espn_event_id:
        cmd.extend(["--espn-event-id", tournament.espn_event_id])
    if leaderboard_snapshot:
        cmd.extend(["--leaderboard-json", str(leaderboard_snapshot)])
    if args.skip_odds:
        cmd.append("--skip-odds")
    if args.live_odds:
        cmd.append("--live-odds")
    if _should_skip_leaderboard(
        phase,
        args.skip_leaderboard,
        has_snapshot=leaderboard_snapshot is not None,
    ):
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
    scoreboard: dict[str, Any] = {}
    try:
        if not args.skip_leaderboard:
            scoreboard = fetch_scoreboard()
    except EspnScoreboardError as exc:
        print(f"WARNING: {exc}")
        # The Core event adapter below is the fallback. Keep an empty Site
        # payload as a sentinel so we do not make the same blocked request a
        # second time in this refresh.
        scoreboard = {}
    tournament = resolve_active_tournament(
        registry,
        tournament_key=args.tournament_key or None,
        as_of=anchor,
        scoreboard=scoreboard,
    )
    if not tournament:
        unmatched = _live_event_unmatched(registry, scoreboard, as_of=date.fromisoformat(anchor))
        if unmatched:
            raise SystemExit(f"ESPN has an active PGA event with no registry match: {unmatched}")
        print(f"No PGA tournament is active for automation window at {anchor}; no-op.")
        return

    phase_hint = infer_phase(
        tournament,
        as_of=anchor,
        force_phase=args.force_phase or None,
    )
    event: dict[str, Any] | None = None
    leaderboard = None
    if not args.skip_leaderboard and phase_hint != "pre":
        try:
            event, leaderboard = fetch_leaderboard_event(
                espn_match=tournament.espn_match,
                espn_event_id=tournament.espn_event_id,
                scoreboard=scoreboard,
            )
        except EspnScoreboardError as exc:
            print(f"WARNING: ESPN Site/Core leaderboard unavailable: {exc}")

    phase = infer_phase(
        tournament,
        as_of=anchor,
        leaderboard=leaderboard,
        force_phase=args.force_phase or None,
    )
    print(f"Resolved PGA tournament: {tournament.key} ({tournament.name})")
    print(f"Refresh phase: {phase}")

    if _should_preserve_last_good_output(phase, leaderboard):
        print(
            "WARNING: ESPN leaderboard is unavailable from both Site and Core APIs; "
            "preserving the last committed public dashboard and warehouse rows."
        )
        return

    snapshot_path: Path | None = None
    if event is not None and leaderboard is not None:
        snapshot_path = _write_leaderboard_snapshot(event, leaderboard)

    try:
        if phase == "pre":
            if not run_pretournament_predictions(
                tournament,
                args,
                dry_run=args.dry_run,
                allow_unavailable_field=True,
            ):
                return
        elif phase == "live":
            if not tournament.predictions_csv.exists():
                run_pretournament_predictions(tournament, args, dry_run=args.dry_run)
            rounds_done = rounds_completed_from_leaderboard(leaderboard, total_rounds=tournament.total_rounds) if leaderboard else 0
            sims_ran = run_midtournament_update(
                tournament,
                leaderboard=leaderboard,
                leaderboard_snapshot=snapshot_path,
                args=args,
                dry_run=args.dry_run,
            )
            # Validation: if a round is complete, the sim MUST have run
            if rounds_done > 0 and not sims_ran and not args.dry_run:
                raise SystemExit(
                    f"VALIDATION FAILURE: {rounds_done} completed round(s) detected but midtournament simulation did not run. "
                    "Refusing to publish stale pre-tournament predictions as if they were current."
                )
        elif phase == "post":
            if not tournament.predictions_csv.exists():
                run_pretournament_predictions(tournament, args, dry_run=args.dry_run)
            # Run midtournament update in post phase to capture final round
            if leaderboard:
                run_midtournament_update(
                    tournament,
                    leaderboard=leaderboard,
                    leaderboard_snapshot=snapshot_path,
                    args=args,
                    dry_run=args.dry_run,
                )
            run_post_results_fetch(
                tournament,
                args,
                leaderboard_snapshot=snapshot_path,
                dry_run=args.dry_run,
            )

        export_dashboard(
            tournament,
            phase=phase,
            args=args,
            leaderboard_snapshot=snapshot_path,
            dry_run=args.dry_run,
        )
        sync_outputs(tournament, args, dry_run=args.dry_run)
        
        # Summary log for debugging
        rounds_done = rounds_completed_from_leaderboard(leaderboard, total_rounds=tournament.total_rounds) if leaderboard else 0
        current_round = leaderboard.get("currentRound") if leaderboard else None
        
        # Check if predictions were updated (by looking for midtournament file)
        predictions_updated = False
        scheffler_win = None
        if phase in {"live", "post"} and rounds_done > 0:
            for round_no in range(tournament.total_rounds, 0, -1):
                midtournament_path = tournament.midtournament_csv.with_name(
                    f"{tournament.key}_midtournament_R{round_no}.csv"
                )
                if midtournament_path.exists():
                    predictions_updated = True
                    break
            if not predictions_updated and tournament.midtournament_csv.exists():
                predictions_updated = True
        
        # Try to extract Scheffler's win prob from the published JSON
        if tournament.current_json.exists():
            try:
                import json
                with open(tournament.current_json) as f:
                    pub_data = json.load(f)
                    for pred in pub_data.get("predictions", []):
                        if pred.get("player") == "Scottie Scheffler":
                            scheffler_win = pred.get("best_calibrated_target_win_prob")
                            break
            except Exception:
                pass
        
        print(
            f"REFRESH SUMMARY: round={current_round} phase={phase} "
            f"rounds_completed={rounds_done} sims_ran={'yes' if predictions_updated else 'no'} "
            f"predictions_updated={'yes' if predictions_updated else 'no'} "
            f"scheffler_win={scheffler_win:.5f if scheffler_win is not None else 'N/A'}"
        )
    finally:
        if snapshot_path:
            snapshot_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
