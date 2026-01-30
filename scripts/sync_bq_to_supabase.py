#!/usr/bin/env python3
"""
Sync curated BigQuery model predictions into the Supabase tables that power the site.

Usage:
    python scripts/sync_bq_to_supabase.py --project learned-pier-478122-p7 --league NFL --week 11
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from google.cloud import bigquery

from pathlib import Path
import os
import sys

# Ensure project root on sys.path for importing helper functions.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.supabase_pg import (
    create_pg_connection,
    game_map_key,
    load_supabase_credentials,
    upsert_games_pg,
)


MODEL_NAME = "bq_pipeline"
LOGGER = logging.getLogger("sync_bq_to_supabase")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load model_predictions rows from BigQuery (sports_edge_curated) and push them into Supabase."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="GCP project that owns the sports_edge_curated dataset (e.g., learned-pier-478122-p7).",
    )
    parser.add_argument(
        "--league",
        choices=["NFL", "NBA"],
        default="NFL",
        help="League to sync (default: NFL).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Limit to a specific season (optional; inferred from game_date when omitted).",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Limit to a specific league week number (e.g., 11 for Week 11).",
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Earliest game_date to sync (YYYY-MM-DD). Defaults to today when no week filter provided.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=10,
        help="Number of days after start-date to include (default: 10). Ignored if --end-date or --week supplied.",
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Latest game_date to sync (YYYY-MM-DD). Overrides --window-days.",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Restrict to a specific model_version (default: latest per game).",
    )
    parser.add_argument(
        "--model-number",
        default=None,
        help="Restrict to a specific model_number as stored in BigQuery (e.g., v1, v2).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append predictions without deleting existing Supabase rows (default: replace per game/model_version).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    LOGGER.debug("Logging configured at %s level", level.upper())


def _resolve_date_filters(args: argparse.Namespace) -> Tuple[Optional[date], Optional[date]]:
    start = args.start_date
    end = args.end_date
    if start and end and end < start:
        raise ValueError("end-date must be on/after start-date")
    if start and not end:
        end = start + timedelta(days=args.window_days)
    if not start and end:
        raise ValueError("start-date is required when end-date is provided")
    if not start and not end and args.week is None:
        start = datetime.now(tz=timezone.utc).date()
        end = start + timedelta(days=args.window_days)
    LOGGER.debug(
        "Resolved date window -> start: %s, end: %s (week=%s, season=%s)",
        start,
        end,
        args.week,
        args.season,
    )
    return start, end


def fetch_latest_predictions(
    client: bigquery.Client,
    project: str,
    league: str,
    start_date: Optional[date],
    end_date: Optional[date],
    week: Optional[int] = None,
    season: Optional[int] = None,
    model_version: Optional[str] = None,
    model_number: Optional[str] = None,
) -> pd.DataFrame:
    """Return one prediction per game_id joined with the latest feature snapshot metadata."""
    model_version_filter = "AND p.model_version = @model_version" if model_version else ""
    model_number_filter = "AND p.model_number = @model_number" if model_number else ""
    season_filter = "AND p.season = @season" if season is not None else ""
    season_week_filter = "AND p.season_week = @season_week" if week is not None else ""
    date_filter = ""
    if start_date and end_date:
        date_filter = "AND f.game_date BETWEEN @start_date AND @end_date"
    
    query = f"""
        WITH latest_preds AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (PARTITION BY p.game_id, p.model_version ORDER BY p.prediction_ts DESC) AS rn
            FROM `{project}.sports_edge_curated.model_predictions` AS p
            WHERE p.league = @league
            {model_version_filter}
            {model_number_filter}
            {season_filter}
            {season_week_filter}
        ),
        latest_features AS (
            SELECT
                f.game_id,
                f.league,
                f.season,
                f.game_date,
                f.home_team,
                f.away_team,
                f.week_number,
                ROW_NUMBER() OVER (PARTITION BY f.game_id ORDER BY f.as_of_ts DESC) AS rn
            FROM `{project}.sports_edge_curated.feature_snapshots` AS f
            WHERE f.league = @league
        )
        SELECT
            p.prediction_id,
            p.game_id,
            p.league,
            COALESCE(p.season, f.season) AS season,
            COALESCE(p.season_week, f.week_number) AS season_week,
            f.game_date,
            f.week_number,
            f.home_team,
            f.away_team,
            p.model_version,
            p.predicted_spread,
            p.home_win_prob,
            p.prediction_ts
        FROM latest_preds AS p
        JOIN latest_features AS f
          ON p.game_id = f.game_id
        WHERE p.rn = 1
          AND f.rn = 1
          {date_filter}
        ORDER BY f.game_date, f.home_team
    """
    params = [
        bigquery.ScalarQueryParameter("league", "STRING", league),
    ]
    if start_date and end_date:
        params.extend(
            [
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            ]
        )
    if model_version:
        params.append(bigquery.ScalarQueryParameter("model_version", "STRING", model_version))
    if week is not None:
        params.append(bigquery.ScalarQueryParameter("season_week", "INT64", week))
    if season is not None:
        params.append(bigquery.ScalarQueryParameter("season", "INT64", season))
    if model_number:
        params.append(bigquery.ScalarQueryParameter("model_number", "STRING", model_number))
    LOGGER.debug(
        "Running BigQuery fetch | league=%s, season=%s, week=%s, start=%s, end=%s, model_version=%s, model_number=%s",
        league,
        season,
        week,
        start_date,
        end_date,
        model_version,
        model_number,
    )
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    df = client.query(query, job_config=job_config).to_dataframe()
    LOGGER.debug("BigQuery returned %d rows", len(df))
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"], utc=True).dt.tz_localize(None)
    df["prediction_ts"] = pd.to_datetime(df["prediction_ts"], utc=True, errors="coerce")
    if LOGGER.isEnabledFor(logging.DEBUG):
        preview = df[["game_id", "season", "season_week", "home_team", "away_team", "model_version"]].head()
        LOGGER.debug("Preview of fetched predictions:\n%s", preview.to_string(index=False))
    return df


def _prepare_games(preds: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate games and coerce week/time columns before upserting."""
    games = (
        preds[["league", "season", "home_team", "away_team", "game_date", "season_week"]]
        .drop_duplicates(subset=["league", "season", "home_team", "away_team", "game_date"])
        .rename(columns={"season_week": "week"})
    )
    games["week"] = games["week"].astype("Int64")
    games["game_time_utc"] = pd.to_datetime(games["game_date"], utc=True)
    LOGGER.debug("Prepared %d unique games for upsert.", len(games))
    return games


def insert_predictions_from_bq(
    conn,
    predictions: pd.DataFrame,
    game_id_map: dict,
    replace_existing: bool = True,
) -> Tuple[int, int]:
    """Insert each BigQuery prediction row into Supabase."""
    insert_sql = """
        insert into model_predictions (game_id, model_name, model_version, my_spread, my_home_win_prob, asof_ts)
        values (%s, %s, %s, %s, %s, %s)
    """
    delete_sql = "delete from model_predictions where game_id = %s and model_version = %s"
    inserted = 0
    skipped = 0
    with conn.cursor() as cur:
        for _, row in predictions.iterrows():
            key = game_map_key(row["home_team"], row["away_team"], row["game_date"])
            game_id = game_id_map.get(key)
            if not game_id:
                LOGGER.debug("No Supabase game_id for %s -> skipping prediction", key)
                skipped += 1
                continue
            
            # Clean values for psycopg
            def _clean(val):
                if pd.isna(val):
                    return None
                if hasattr(val, "to_pydatetime"):
                    return val.to_pydatetime()
                return val

            spread = float(row["predicted_spread"]) if pd.notna(row["predicted_spread"]) else None
            win_prob = float(row["home_win_prob"]) if pd.notna(row["home_win_prob"]) else None
            asof_ts = row["prediction_ts"]
            if pd.isna(asof_ts):
                asof_ts = datetime.now(tz=timezone.utc)
            if spread is not None:
                spread = -spread
            
            model_version = _clean(row["model_version"])
            
            if replace_existing:
                LOGGER.debug("Deleting existing predictions for game_id=%s version=%s", game_id, model_version)
                cur.execute(delete_sql, (game_id, model_version), prepare=False)
            cur.execute(
                insert_sql,
                (
                    game_id,
                    MODEL_NAME,
                    model_version,
                    spread,
                    win_prob,
                    _clean(asof_ts),
                ),
                prepare=False,
            )
            inserted += 1
    LOGGER.info("Inserted %d predictions (skipped %d missing games).", inserted, skipped)
    conn.commit()
    return inserted, skipped


def send_discord_alerts(preds: pd.DataFrame, league: str) -> None:
    """Format and send predictions to Discord via webhook."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        LOGGER.warning("DISCORD_WEBHOOK_URL not found; skipping Discord alert.")
        return

    if preds.empty:
        return

    # Group by date to handle multiple days if necessary
    preds["game_date_str"] = preds["game_date"].dt.strftime("%m-%d-%Y")
    dates = preds["game_date_str"].unique()

    for d in dates:
        day_preds = preds[preds["game_date_str"] == d]
        
        message_lines = [f"**{league} GAMES {d}**"]
        
        for _, row in day_preds.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            win_prob = row["home_win_prob"]
            spread = row["predicted_spread"]
            
            if win_prob > 0.5:
                winner = home
                prob = win_prob
            else:
                winner = away
                prob = 1 - win_prob
            
            # Confidence is absolute distance from 50% * 2
            confidence = abs(prob - 0.5) * 2
            
            # Formatting spread (standard notation: negative for favorite)
            # In our data, predicted_spread is home_margin (positive = home win)
            # So negative home_margin means away is favorite. 
            # Standard spread = -predicted_spread
            display_spread = -spread if spread is not None else 0
            spread_sign = "-" if display_spread <= 0 else "+"
            spread_val = abs(display_spread)
            
            line = (
                f"{home} vs {away}:\n"
                f"Win prob: **{winner} {prob:.1%}** ({confidence:.0%} confidence)\n"
                f"**{home} {spread_sign}{spread_val:.1f}**\n"
            )
            message_lines.append(line)

        full_message = "\n".join(message_lines)
        
        try:
            response = requests.post(
                webhook_url,
                json={"content": full_message},
                timeout=10
            )
            response.raise_for_status()
            LOGGER.info("Sent Discord alert for %s games on %s", league, d)
        except Exception as e:
            LOGGER.error("Failed to send Discord alert: %s", e)


def main() -> None:
    load_dotenv()
    args = _parse_args()
    _configure_logging(args.log_level)
    start_date, end_date = _resolve_date_filters(args)
    LOGGER.info(
        "Starting sync | project=%s league=%s season=%s week=%s model_version=%s model_number=%s append=%s",
        args.project,
        args.league,
        args.season,
        args.week,
        args.model_version,
        args.model_number,
        args.append,
    )

    client = bigquery.Client(project=args.project)
    preds = fetch_latest_predictions(
        client,
        args.project,
        args.league,
        start_date,
        end_date,
        week=args.week,
        season=args.season,
        model_version=args.model_version,
        model_number=args.model_number,
    )

    if preds.empty:
        filters = []
        if args.week is not None:
            filters.append(f"Week {args.week}")
        if args.season is not None:
            filters.append(f"Season {args.season}")
        if args.model_number is not None:
            filters.append(f"ModelNumber {args.model_number}")
        if start_date and end_date:
            filters.append(f"dates {start_date} to {end_date}")
        filter_text = ", ".join(filters) if filters else "requested window"
        LOGGER.warning("No matching BigQuery predictions found for %s.", filter_text)
        return

    LOGGER.info("Found %d predictions from BigQuery", len(preds))
    if args.week is not None:
        LOGGER.info("  Week filter: %s", args.week)
    if args.season is not None:
        LOGGER.info("  Season filter: %s", args.season)
    if args.model_number is not None:
        LOGGER.info("  Model number filter: %s", args.model_number)
    if start_date and end_date:
        LOGGER.info(
            "  Date range: %s to %s", preds["game_date"].min().date(), preds["game_date"].max().date()
        )
    LOGGER.info("  Model versions: %s", preds["model_version"].unique().tolist())

    games_df = _prepare_games(preds)
    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    try:
        LOGGER.info("Upserting %d games into Supabase", len(games_df))
        game_id_map = upsert_games_pg(conn, games_df)
        inserted, skipped = insert_predictions_from_bq(conn, preds, game_id_map, replace_existing=not args.append)
        LOGGER.info(
            "Supabase sync complete | inserted=%d skipped=%d replaced_existing=%s",
            inserted,
            skipped,
            not args.append,
        )
        
        # Send Discord alerts for the predictions we just synced
        if inserted > 0:
            send_discord_alerts(preds, args.league)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
