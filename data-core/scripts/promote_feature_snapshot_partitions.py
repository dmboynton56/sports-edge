#!/usr/bin/env python3
"""Validate and atomically promote staged feature-snapshot partitions.

The command is read-only unless --apply is supplied. It is deliberately scoped
to the three known contaminated partitions from the 2026-09-06 audit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery


ALLOWED_PARTITIONS = (("NBA", 2025), ("NFL", 2025), ("NFL", 2026))


def _query(client: bigquery.Client, sql: str):
    return client.query(sql).to_dataframe()


def audit_stage(client: bigquery.Client, stage_table: str, raw_schedule_table: str | None = None) -> dict:
    partitions = _query(
        client,
        f"""
        SELECT league, season, COUNT(*) AS row_count, COUNT(DISTINCT game_id) AS games,
               COUNT(*) - COUNT(DISTINCT game_id) AS duplicate_rows,
               COUNTIF(game_id IS NULL OR game_date IS NULL) AS null_keys
        FROM `{stage_table}`
        GROUP BY league, season
        ORDER BY league, season
        """,
    )
    actual = {(str(row.league), int(row.season)) for row in partitions.itertuples()}
    expected = set(ALLOWED_PARTITIONS)
    errors = []
    if actual != expected:
        errors.append(f"partition mismatch: expected={sorted(expected)}, actual={sorted(actual)}")
    for row in partitions.itertuples():
        if int(row.row_count) != int(row.games) or int(row.duplicate_rows) or int(row.null_keys):
            errors.append(
                f"invalid grain for {row.league} {row.season}: rows={row.row_count}, "
                f"games={row.games}, duplicates={row.duplicate_rows}, null_keys={row.null_keys}"
            )
    lineage = None
    if raw_schedule_table:
        lineage_frame = _query(
            client,
            f"""
            WITH schedule_keys AS (
              SELECT CAST(game_id AS STRING) AS game_id,
                     ARRAY_AGG(DISTINCT UPPER(league) IGNORE NULLS) AS leagues
              FROM `{raw_schedule_table}`
              GROUP BY game_id
            )
            SELECT
              COUNTIF(schedule_keys.game_id IS NULL) AS missing_schedule_keys,
              COUNTIF(schedule_keys.game_id IS NOT NULL AND
                      UPPER(stage.league) NOT IN UNNEST(schedule_keys.leagues)) AS cross_league_rows
            FROM `{stage_table}` AS stage
            LEFT JOIN schedule_keys USING (game_id)
            """,
        )
        lineage = lineage_frame.iloc[0].to_dict()
        if int(lineage["missing_schedule_keys"]) or int(lineage["cross_league_rows"]):
            errors.append(
                "schedule lineage failed: "
                f"missing={lineage['missing_schedule_keys']}, cross_league={lineage['cross_league_rows']}"
            )
    return {
        "stage_table": stage_table,
        "partitions": partitions.to_dict("records"),
        "errors": errors,
        "schedule_lineage": lineage,
        "ready": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--stage-table", required=True)
    parser.add_argument(
        "--target-table",
        default=None,
        help="Defaults to PROJECT.sports_edge_curated.feature_snapshots.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".env",
    )
    args = parser.parse_args()
    load_dotenv(args.env_file)
    target = args.target_table or f"{args.project}.sports_edge_curated.feature_snapshots"
    client = bigquery.Client(project=args.project)
    audit = audit_stage(
        client,
        args.stage_table,
        f"{args.project}.sports_edge_raw.raw_schedules",
    )
    print(json.dumps(audit, indent=2, default=str))
    if not audit["ready"]:
        raise SystemExit("Staged feature partitions failed validation; target was not changed.")
    if not args.apply:
        print("Dry run complete. Re-run with --apply only after reviewing the exact partition counts above.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{args.project}.sports_edge_curated.feature_snapshots_backup_{stamp}"
    predicate = "(league = 'NBA' AND season = 2025) OR (league = 'NFL' AND season IN (2025, 2026))"
    client.query(f"CREATE TABLE `{backup}` AS SELECT * FROM `{target}` WHERE {predicate}").result()
    transaction = f"""
    BEGIN TRANSACTION;
      DELETE FROM `{target}` WHERE {predicate};
      INSERT INTO `{target}` SELECT * FROM `{args.stage_table}`;
    COMMIT TRANSACTION;
    """
    try:
        client.query(transaction).result()
    except Exception:
        print(f"Promotion failed. Backup remains available at {backup}.")
        raise
    print(json.dumps({"promoted": True, "target": target, "backup": backup, "stage": args.stage_table}, indent=2))


if __name__ == "__main__":
    main()
