#!/usr/bin/env python3
"""Remove old NBA/NFL serving rows after verifying BigQuery retains them.

BigQuery is the source of truth for sports history.  The shared Supabase
project only needs the current NBA/NFL seasons for serving.  This script is
dry-run by default and requires ``--apply`` for deletion.  It deletes parent
``games`` rows; the schema's ``ON DELETE CASCADE`` relationships remove the
associated predictions, features, odds, and game-level reports together.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import psycopg
from dotenv import load_dotenv

from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


LOGGER = logging.getLogger("prune_supabase_historical_predictions")
LEAGUES = ("NBA", "NFL")
PROJECT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class SeasonTarget:
    league: str
    keep_season: int


def season_year_for(league: str, anchor: date) -> int:
    if league == "NBA":
        return anchor.year if anchor.month >= 10 else anchor.year - 1
    if league == "NFL":
        return anchor.year if anchor.month >= 8 else anchor.year - 1
    raise ValueError(f"Unsupported league: {league}")


def current_targets(anchor: date | None = None) -> tuple[SeasonTarget, ...]:
    anchor = anchor or datetime.now(ZoneInfo("America/Denver")).date()
    return tuple(SeasonTarget(league, season_year_for(league, anchor)) for league in LEAGUES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        help="GCP project containing sports_edge_curated.model_predictions; required with --apply.",
    )
    parser.add_argument(
        "--keep-season",
        action="append",
        metavar="LEAGUE=SEASON",
        help="Override a keep boundary, e.g. NBA=2025. May be supplied twice.",
    )
    parser.add_argument("--apply", action="store_true", help="Delete eligible Supabase games and cascaded rows.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser.parse_args()


def parse_targets(values: list[str] | None) -> tuple[SeasonTarget, ...]:
    targets = {target.league: target.keep_season for target in current_targets()}
    for value in values or []:
        try:
            league, season_text = value.split("=", 1)
            league = league.upper()
            season = int(season_text)
        except ValueError as exc:
            raise ValueError(f"Invalid --keep-season value: {value!r}; expected LEAGUE=SEASON") from exc
        if league not in LEAGUES or season < 2000 or season > 2100:
            raise ValueError(f"Invalid --keep-season value: {value!r}")
        targets[league] = season
    return tuple(SeasonTarget(league, targets[league]) for league in LEAGUES)


def _validate_project(project: str) -> None:
    if not PROJECT_RE.fullmatch(project):
        raise ValueError("--project must be a plain GCP project identifier")


def verify_bigquery_retention(project: str, targets: tuple[SeasonTarget, ...]) -> dict[str, tuple[int, int]]:
    """Verify that BigQuery retains source games and prediction rows to delete."""
    from google.cloud import bigquery

    _validate_project(project)
    client = bigquery.Client(project=project)
    query = f"""
        SELECT
          league,
          COUNT(DISTINCT game_id) AS games,
          COUNT(*) AS predictions
        FROM `{project}.sports_edge_curated.model_predictions`
        WHERE (league = 'NBA' AND season < @nba_keep)
           OR (league = 'NFL' AND season < @nfl_keep)
        GROUP BY league
    """
    keep = {target.league: target.keep_season for target in targets}
    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("nba_keep", "INT64", keep["NBA"]),
            bigquery.ScalarQueryParameter("nfl_keep", "INT64", keep["NFL"]),
        ]
    )
    result = {
        row["league"]: (int(row["games"]), int(row["predictions"]))
        for row in client.query(query, job_config=config).result()
    }
    missing = [league for league in LEAGUES if result.get(league, (0, 0))[0] == 0]
    if missing:
        raise RuntimeError(
            "BigQuery verification found no historical game rows for: "
            + ", ".join(missing)
            + "; refusing to delete from Supabase."
        )
    return result


def _connect():
    credentials = load_supabase_credentials()
    if not credentials["db_password"]:
        raise RuntimeError("SUPABASE_DB_PASSWORD or supabaseDBpass is required")
    return create_pg_connection(
        supabase_url=credentials["url"],
        password=credentials["db_password"],
        host_override=credentials.get("db_host"),
        port=credentials["db_port"],
        database=credentials["db_name"],
        user=credentials["db_user"],
    )


def prune(*, targets: tuple[SeasonTarget, ...], apply: bool) -> dict[str, int]:
    conn = _connect()
    summary: dict[str, int] = {}
    try:
        with conn.cursor() as cur:
            for target in targets:
                cur.execute(
                    "SELECT COUNT(*) FROM public.games WHERE league = %s AND season < %s",
                    (target.league, target.keep_season),
                )
                count = int(cur.fetchone()[0])
                summary[target.league] = count
                LOGGER.info(
                    "%s: %d games before season %d (%s); cascaded predictions/features/odds are included",
                    target.league,
                    count,
                    target.keep_season,
                    "apply" if apply else "dry-run",
                )
                if apply and count:
                    cur.execute(
                        "DELETE FROM public.games WHERE league = %s AND season < %s",
                        (target.league, target.keep_season),
                    )
        if apply:
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except psycopg.Error:
            pass
        raise
    finally:
        conn.close()
    return summary


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    targets = parse_targets(args.keep_season)
    LOGGER.info("Keeping current serving seasons: %s", {target.league: target.keep_season for target in targets})
    if args.apply and not args.project:
        raise RuntimeError("--project is required with --apply so BigQuery retention can be verified")
    if args.project:
        bq_summary = verify_bigquery_retention(args.project, targets)
        LOGGER.info("BigQuery retention verified: %s", bq_summary)
    prune(targets=targets, apply=args.apply)


if __name__ == "__main__":
    main()
