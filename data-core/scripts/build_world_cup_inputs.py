#!/usr/bin/env python3
"""Build World Cup model input CSVs from repeatable source extracts.

This script is the ingestion bridge for the v0 World Cup model. It can use a
cached ESPN scoreboard payload or fetch ESPN directly for fixtures/results, then
blend optional FIFA rankings, World Football Elo, recent results, World Cup
history, player form, and futures odds into the team-ratings CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.world_cup_sources import (
    build_team_rating_inputs,
    derive_round_of_32_slots,
    extract_espn_team_form,
    fetch_espn_world_cup_fixtures,
    fetch_world_football_elo,
    normalize_fixtures_for_model,
    parse_espn_world_cup_scoreboard,
    write_world_cup_inputs,
)


def _read_optional_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return pd.read_csv(path)


def _load_fixtures(args: argparse.Namespace) -> pd.DataFrame:
    if args.espn_scoreboard_json:
        payload = json.loads(args.espn_scoreboard_json.read_text())
        return parse_espn_world_cup_scoreboard(payload, season=args.season)
    if args.fetch_espn:
        return fetch_espn_world_cup_fixtures(
            start_date=args.start_date,
            end_date=args.end_date,
            season=args.season,
            timeout=args.timeout,
        )
    if args.fixtures_csv:
        return pd.read_csv(args.fixtures_csv)
    raise ValueError("Provide --fixtures-csv, --espn-scoreboard-json, or --fetch-espn")


def _load_world_elo(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    if args.world_elo_csv:
        return pd.read_csv(args.world_elo_csv)
    if args.fetch_world_elo:
        return fetch_world_football_elo()
    return None


def _write_bigquery(df: pd.DataFrame, *, project: str, dataset: str, table: str, write_disposition: str) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Wrote {len(df):,} rows to {table_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build World Cup prediction input CSVs.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--fixtures-csv", type=Path, help="Existing normalized/manual fixtures CSV.")
    parser.add_argument("--teams-csv", type=Path, help="Seed teams CSV with team and optional group columns.")
    parser.add_argument("--espn-scoreboard-json", type=Path, help="Cached ESPN scoreboard JSON payload.")
    parser.add_argument("--fetch-espn", action="store_true", help="Fetch fixtures/results from ESPN scoreboard.")
    parser.add_argument("--start-date", help="Fetch start date, YYYY-MM-DD. Used with --fetch-espn.")
    parser.add_argument("--end-date", help="Fetch end date, YYYY-MM-DD. Used with --fetch-espn.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--fifa-rankings-csv", type=Path)
    parser.add_argument("--world-elo-csv", type=Path)
    parser.add_argument("--fetch-world-elo", action="store_true")
    parser.add_argument("--recent-results-csv", type=Path)
    parser.add_argument("--world-cup-history-csv", type=Path)
    parser.add_argument("--player-form-csv", type=Path)
    parser.add_argument("--market-odds-csv", type=Path)
    parser.add_argument("--as-of-date", help="Exclude recent-result rows after this date.")
    parser.add_argument("--lookback-matches", type=int, default=10)
    parser.add_argument("--output-teams-csv", type=Path, default=Path("data-core/notebooks/cache/world_cup_team_ratings.csv"))
    parser.add_argument("--output-fixtures-csv", type=Path, default=Path("data-core/notebooks/cache/world_cup_fixtures.csv"))
    parser.add_argument("--output-round-of-32-slots-json", type=Path)
    parser.add_argument("--write-bigquery", action="store_true")
    parser.add_argument("--project", help="GCP project for --write-bigquery.")
    parser.add_argument("--write-disposition", default="WRITE_TRUNCATE", choices=["WRITE_APPEND", "WRITE_TRUNCATE"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures_raw = _load_fixtures(args)
    fixtures = normalize_fixtures_for_model(fixtures_raw)
    teams_seed = _read_optional_csv(args.teams_csv)
    recent_results = _read_optional_csv(args.recent_results_csv)
    team_form = None if recent_results is not None else extract_espn_team_form(fixtures_raw)
    ratings = build_team_rating_inputs(
        teams=teams_seed,
        fixtures=fixtures,
        fifa_rankings=_read_optional_csv(args.fifa_rankings_csv),
        world_elo=_load_world_elo(args),
        recent_results=recent_results,
        team_form=team_form,
        world_cup_history=_read_optional_csv(args.world_cup_history_csv),
        player_form=_read_optional_csv(args.player_form_csv),
        market_odds=_read_optional_csv(args.market_odds_csv),
        season=args.season,
        as_of_date=args.as_of_date,
        lookback_matches=args.lookback_matches,
    )
    write_world_cup_inputs(
        team_ratings=ratings,
        fixtures=fixtures,
        teams_csv=args.output_teams_csv,
        fixtures_csv=args.output_fixtures_csv,
        round_of_32_slots_json=args.output_round_of_32_slots_json,
    )
    print(f"Wrote {len(ratings)} teams to {args.output_teams_csv}")
    print(f"Wrote {len(fixtures)} fixtures to {args.output_fixtures_csv}")
    if args.output_round_of_32_slots_json:
        print(f"Wrote {len(derive_round_of_32_slots(fixtures))} Round of 32 slots to {args.output_round_of_32_slots_json}")

    if args.write_bigquery:
        if not args.project:
            raise ValueError("--project is required with --write-bigquery")
        _write_bigquery(
            fixtures_raw,
            project=args.project,
            dataset="sports_edge_raw",
            table="raw_wc_fixtures",
            write_disposition=args.write_disposition,
        )
        _write_bigquery(
            ratings,
            project=args.project,
            dataset="sports_edge_curated",
            table="wc_team_ratings",
            write_disposition=args.write_disposition,
        )


if __name__ == "__main__":
    main()
