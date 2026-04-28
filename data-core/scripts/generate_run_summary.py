#!/usr/bin/env python3
"""
Build GitHub Actions and Discord summaries for the Sports Edge refresh.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("generate_run_summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Sports Edge run summaries.")
    parser.add_argument(
        "--mode",
        choices=("normal", "dry-run"),
        default="normal",
        help="Refresh mode used for this run.",
    )
    parser.add_argument(
        "--status",
        choices=("success", "failure", "cancelled"),
        default="success",
        help="Current workflow status.",
    )
    parser.add_argument("--run-url", default="", help="GitHub Actions run URL.")
    parser.add_argument("--repository", default="", help="GitHub repository name.")
    parser.add_argument("--workflow", default="", help="GitHub workflow name.")
    parser.add_argument(
        "--validation-json",
        type=Path,
        help="Optional JSON output from validate_supabase_sync.py.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional markdown summary output path.",
    )
    parser.add_argument(
        "--discord-json",
        type=Path,
        help="Optional Discord webhook payload output path.",
    )
    parser.add_argument(
        "--prediction-hours",
        type=int,
        default=48,
        help="Window for recent predictions (default: 48h).",
    )
    parser.add_argument(
        "--score-lookback-days",
        type=int,
        default=7,
        help="Lookback window for game/final-score counts (default: 7).",
    )
    parser.add_argument(
        "--odds-hours",
        type=int,
        default=24,
        help="Window for odds snapshot counts (default: 24h).",
    )
    parser.add_argument(
        "--allow-query-failures",
        action="store_true",
        help="Emit a degraded summary instead of failing when Supabase cannot be queried.",
    )
    return parser.parse_args()


def load_validation_report(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "No validation JSON path was provided."
    if not path.exists():
        return None, f"Validation JSON was not found at {path}."
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"Validation JSON at {path} could not be parsed: {exc}."
    if not isinstance(data, dict):
        return None, f"Validation JSON at {path} was not an object."
    return data, None


def fetch_supabase_snapshot(
    prediction_hours: int,
    score_lookback_days: int,
    odds_hours: int,
) -> dict[str, Any]:
    from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

    creds = load_supabase_credentials()
    missing = []
    if not creds.get("url"):
        missing.append("SUPABASE_URL")
    if not creds.get("db_password"):
        missing.append("SUPABASE_DB_PASSWORD or supabaseDBpass")
    if missing:
        raise RuntimeError(
            "Missing Supabase connection environment: " + ", ".join(missing)
        )

    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )

    snapshot: dict[str, Any] = {
        "prediction_hours": prediction_hours,
        "score_lookback_days": score_lookback_days,
        "odds_hours": odds_hours,
        "predictions_by_league": {},
        "games_by_league": {},
        "final_scores_by_league": {},
        "latest_game_time_by_league": {},
        "odds_by_league": {},
        "latest_odds_snapshot_ts": None,
    }

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(g.league, 'unknown') AS league, COUNT(*)
                FROM model_predictions p
                LEFT JOIN games g ON g.id = p.game_id
                WHERE p.asof_ts >= NOW() - (%s || ' hours')::interval
                GROUP BY COALESCE(g.league, 'unknown')
                ORDER BY league
                """,
                (prediction_hours,),
            )
            snapshot["predictions_by_league"] = {
                str(league): int(count) for league, count in cur.fetchall()
            }

            cur.execute(
                """
                SELECT league, COUNT(*), MAX(game_time_utc)
                FROM games
                WHERE game_time_utc::date >= CURRENT_DATE - (%s || ' days')::interval
                GROUP BY league
                ORDER BY league
                """,
                (score_lookback_days,),
            )
            for league, count, latest_game_time in cur.fetchall():
                snapshot["games_by_league"][str(league)] = int(count)
                snapshot["latest_game_time_by_league"][str(league)] = iso_or_none(
                    latest_game_time
                )

            cur.execute(
                """
                SELECT league, COUNT(*)
                FROM games
                WHERE game_time_utc::date >= CURRENT_DATE - (%s || ' days')::interval
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                GROUP BY league
                ORDER BY league
                """,
                (score_lookback_days,),
            )
            snapshot["final_scores_by_league"] = {
                str(league): int(count) for league, count in cur.fetchall()
            }

            cur.execute(
                """
                SELECT COALESCE(g.league, 'unknown') AS league, COUNT(*), MAX(o.snapshot_ts)
                FROM odds_snapshots o
                LEFT JOIN games g ON g.id = o.game_id
                WHERE o.snapshot_ts >= NOW() - (%s || ' hours')::interval
                GROUP BY COALESCE(g.league, 'unknown')
                ORDER BY league
                """,
                (odds_hours,),
            )
            for league, count, latest_snapshot_ts in cur.fetchall():
                snapshot["odds_by_league"][str(league)] = {
                    "count": int(count),
                    "latest_snapshot_ts": iso_or_none(latest_snapshot_ts),
                }

            latest_odds = [
                item["latest_snapshot_ts"]
                for item in snapshot["odds_by_league"].values()
                if item.get("latest_snapshot_ts")
            ]
            snapshot["latest_odds_snapshot_ts"] = max(latest_odds) if latest_odds else None
    finally:
        conn.close()

    return snapshot


def iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def int_value(data: dict[str, Any] | None, key: str) -> int | None:
    if not data or key not in data:
        return None
    try:
        return int(data[key])
    except (TypeError, ValueError):
        return None


def total_count(values: dict[str, Any]) -> int:
    total = 0
    for value in values.values():
        if isinstance(value, dict):
            total += int(value.get("count") or 0)
        else:
            total += int(value or 0)
    return total


def format_by_league(values: dict[str, Any], empty_text: str = "None found") -> str:
    if not values:
        return empty_text
    parts: list[str] = []
    for league in sorted(values):
        value = values[league]
        count = value.get("count") if isinstance(value, dict) else value
        parts.append(f"{league}: {count}")
    return "\n".join(parts)


def format_by_league_inline(values: dict[str, Any], empty_text: str = "None found") -> str:
    return format_by_league(values, empty_text=empty_text).replace("\n", ", ")


def build_warnings(
    mode: str,
    status: str,
    validation: dict[str, Any] | None,
    validation_warning: str | None,
    snapshot: dict[str, Any] | None,
    query_error: str | None,
) -> list[str]:
    warnings: list[str] = []
    if mode == "dry-run":
        warnings.append(
            "Dry run: Supabase write and strict validation steps were skipped."
        )
    if validation_warning and mode != "dry-run":
        warnings.append(validation_warning)
    if query_error:
        warnings.append(f"Supabase summary query failed: {query_error}")

    if validation:
        recent_predictions = int_value(validation, "recent_predictions")
        recent_games = int_value(validation, "recent_games")
        recent_final_scores = int_value(validation, "recent_final_scores")
        final_missing_scores = int_value(validation, "final_missing_scores")
        orphan_predictions = int_value(validation, "orphan_predictions")

        if status == "success" and recent_predictions is not None and recent_predictions <= 0:
            warnings.append("No recent model_predictions rows were reported.")
        if (
            status == "success"
            and recent_games is not None
            and recent_games > 0
            and recent_final_scores is not None
            and recent_final_scores <= 0
        ):
            warnings.append("Recent games exist, but no final scores were reported.")
        if final_missing_scores is not None and final_missing_scores > 0:
            warnings.append(f"{final_missing_scores} final games are missing scores.")
        if orphan_predictions is not None and orphan_predictions > 0:
            warnings.append(f"{orphan_predictions} orphan predictions were reported.")

    if snapshot and status == "success" and mode == "normal":
        if total_count(snapshot.get("predictions_by_league", {})) <= 0:
            warnings.append("No recent predictions found in the Supabase summary.")
        if total_count(snapshot.get("odds_by_league", {})) <= 0:
            warnings.append("No recent odds snapshots found in the Supabase summary.")

    return warnings


def build_summary(
    *,
    mode: str,
    status: str,
    run_url: str,
    repository: str,
    workflow: str,
    validation: dict[str, Any] | None,
    validation_warning: str | None,
    snapshot: dict[str, Any] | None,
    query_error: str | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    warnings = build_warnings(
        mode=mode,
        status=status,
        validation=validation,
        validation_warning=validation_warning,
        snapshot=snapshot,
        query_error=query_error,
    )
    return {
        "status": status,
        "mode": mode,
        "run_url": run_url,
        "repository": repository,
        "workflow": workflow,
        "generated_at": generated_at,
        "validation": validation,
        "validation_warning": validation_warning,
        "supabase": snapshot,
        "query_error": query_error,
        "warnings": warnings,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    validation = summary.get("validation") or {}
    snapshot = summary.get("supabase") or {}
    warnings = summary.get("warnings") or []

    lines = [
        "### Sports Edge Refresh Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Generated at: `{summary['generated_at']}`",
    ]
    if summary.get("run_url"):
        lines.append(f"- Run: {summary['run_url']}")
    if summary.get("repository"):
        lines.append(f"- Repository: `{summary['repository']}`")
    if summary.get("workflow"):
        lines.append(f"- Workflow: `{summary['workflow']}`")

    lines.extend(["", "#### Validation", "", "| Check | Count |", "| --- | ---: |"])
    for key in (
        "recent_predictions",
        "recent_games",
        "recent_final_scores",
        "final_missing_scores",
        "orphan_predictions",
    ):
        value = validation.get(key, "n/a")
        lines.append(f"| `{key}` | {value} |")

    if summary.get("mode") == "dry-run" and not validation:
        lines.extend(["", "Validation note: strict validation skipped for dry run."])
    elif summary.get("validation_warning") and not validation:
        lines.extend(["", f"Validation note: {summary['validation_warning']}"])

    lines.extend(["", "#### Supabase Snapshot", ""])
    if snapshot:
        predictions_by_league = format_by_league_inline(
            snapshot.get("predictions_by_league", {})
        )
        games_by_league = format_by_league_inline(snapshot.get("games_by_league", {}))
        final_scores_by_league = format_by_league_inline(
            snapshot.get("final_scores_by_league", {})
        )
        odds_by_league = format_by_league_inline(snapshot.get("odds_by_league", {}))
        lines.extend(
            [
                f"- Prediction window: `{snapshot.get('prediction_hours')}h`",
                f"- Score lookback: `{snapshot.get('score_lookback_days')}d`",
                f"- Odds window: `{snapshot.get('odds_hours')}h`",
                f"- Predictions by league: {predictions_by_league}",
                f"- Games by league: {games_by_league}",
                f"- Final scores by league: {final_scores_by_league}",
                f"- Odds snapshots by league: {odds_by_league}",
                f"- Latest odds snapshot: `{snapshot.get('latest_odds_snapshot_ts') or 'n/a'}`",
            ]
        )
    else:
        snapshot_error = summary.get("query_error") or "not queried"
        lines.append(f"Supabase snapshot unavailable: {snapshot_error}")

    if warnings:
        lines.extend(["", "#### Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines) + "\n"


def compact_validation_text(validation: dict[str, Any] | None) -> str:
    if not validation:
        return "Validation payload unavailable."
    labels = [
        ("recent_predictions", "Predictions"),
        ("recent_games", "Games"),
        ("recent_final_scores", "Final scores"),
        ("final_missing_scores", "Final missing scores"),
        ("orphan_predictions", "Orphan predictions"),
    ]
    return "\n".join(f"{label}: {validation.get(key, 'n/a')}" for key, label in labels)


def compact_snapshot_text(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return "Supabase snapshot unavailable."
    return "\n".join(
        [
            "Predictions: "
            + format_by_league(snapshot.get("predictions_by_league", {})),
            "Games: " + format_by_league(snapshot.get("games_by_league", {})),
            "Final scores: "
            + format_by_league(snapshot.get("final_scores_by_league", {})),
            "Odds snapshots: " + format_by_league(snapshot.get("odds_by_league", {})),
            f"Latest odds: {snapshot.get('latest_odds_snapshot_ts') or 'n/a'}",
        ]
    )


def build_discord_payload(summary: dict[str, Any]) -> dict[str, Any]:
    status = summary["status"]
    mode = summary["mode"]
    title_status = {
        "success": "succeeded",
        "failure": "failed",
        "cancelled": "was cancelled",
    }[status]
    color = {
        "success": 0x2ECC71,
        "failure": 0xE74C3C,
        "cancelled": 0x95A5A6,
    }[status]
    warnings = summary.get("warnings") or []
    fields = [
        {"name": "Mode", "value": mode, "inline": True},
        {"name": "Repository", "value": summary.get("repository") or "n/a", "inline": True},
        {"name": "Workflow", "value": summary.get("workflow") or "n/a", "inline": True},
        {
            "name": "Validation",
            "value": compact_validation_text(summary.get("validation"))[:1024],
            "inline": False,
        },
        {
            "name": "Supabase snapshot",
            "value": compact_snapshot_text(summary.get("supabase"))[:1024],
            "inline": False,
        },
    ]
    if warnings:
        fields.append(
            {
                "name": "Warnings",
                "value": "\n".join(f"- {warning}" for warning in warnings)[:1024],
                "inline": False,
            }
        )

    run_url = summary.get("run_url")
    description = f"Run: {run_url}" if run_url else "Run URL unavailable."
    return {
        "content": f"**Sports-Edge Refresh {title_status}**",
        "embeds": [
            {
                "title": f"Daily refresh {status}",
                "description": description,
                "color": color,
                "fields": fields,
                "timestamp": summary["generated_at"],
            }
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    validation, validation_warning = load_validation_report(args.validation_json)
    snapshot = None
    query_error = None
    try:
        snapshot = fetch_supabase_snapshot(
            prediction_hours=args.prediction_hours,
            score_lookback_days=args.score_lookback_days,
            odds_hours=args.odds_hours,
        )
    except Exception as exc:  # pragma: no cover - depends on live credentials.
        query_error = str(exc)
        LOGGER.warning("Could not query Supabase summary: %s", exc)
        if not args.allow_query_failures:
            raise

    summary = build_summary(
        mode=args.mode,
        status=args.status,
        run_url=args.run_url,
        repository=args.repository,
        workflow=args.workflow,
        validation=validation,
        validation_warning=validation_warning,
        snapshot=snapshot,
        query_error=query_error,
    )

    markdown = render_markdown(summary)
    discord_payload = build_discord_payload(summary)

    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown)
    if args.discord_json:
        write_json(args.discord_json, discord_payload)

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
