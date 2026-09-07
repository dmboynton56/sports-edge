#!/usr/bin/env python3
"""Export an immutable, point-in-time NFL v2 modeling dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import nflreadpy as nfl
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.nfl_v2 import (
    PBP_AGGREGATE_SQL,
    audit_nfl_feature_store,
    build_nfl_feature_store,
    dataframe_fingerprint,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "curated")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    return parser.parse_args()


def _to_pandas(value) -> pd.DataFrame:
    return value.to_pandas() if hasattr(value, "to_pandas") else pd.DataFrame(value)


def _schedule_fingerprint(frame: pd.DataFrame) -> str:
    fields = [
        field for field in ("game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score")
        if field in frame
    ]
    return dataframe_fingerprint(frame, fields)


def main() -> None:
    args = _parse_args()
    if args.start_season > args.end_season:
        raise SystemExit("--start-season must be no later than --end-season")
    load_dotenv(args.env_file)
    seasons = list(range(args.start_season, args.end_season + 1))
    extracted_at = datetime.now(timezone.utc)

    schedules = _to_pandas(nfl.load_schedules(seasons))
    schedules["league"] = "NFL"
    completed = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()].copy()

    client = bigquery.Client(project=args.project)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_season", "INT64", args.start_season),
            bigquery.ScalarQueryParameter("end_season", "INT64", args.end_season),
        ]
    )
    team_games = client.query(PBP_AGGREGATE_SQL.format(project=args.project), job_config=job_config).to_dataframe()
    features = build_nfl_feature_store(completed, team_games)
    audit = audit_nfl_feature_store(features, raise_on_error=True)
    pbp_coverage = float(len(team_games) / max(1, 2 * len(completed)))
    if pbp_coverage < 0.99:
        raise SystemExit(f"PBP team-game coverage gate failed: {pbp_coverage:.1%} < 99.0%")
    fingerprint = dataframe_fingerprint(features)
    source_fingerprint = sha256(
        (_schedule_fingerprint(completed) + dataframe_fingerprint(team_games)).encode("utf-8")
    ).hexdigest()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"nfl_v2_{args.start_season}_{args.end_season}_{fingerprint[:8]}_{source_fingerprint[:8]}"
    dataset_path = args.output_dir / f"{stem}.parquet"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    if dataset_path.exists() or manifest_path.exists():
        raise SystemExit(f"Immutable export already exists: {stem}")
    features.to_parquet(dataset_path, index=False)
    manifest = {
        "artifact_version": "nfl-v2-dataset-1",
        "created_at": extracted_at.isoformat(),
        "dataset_path": str(dataset_path.resolve()),
        "data_fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint,
        "sources": {
            "schedules": "nflreadpy.load_schedules",
            "play_by_play": f"{args.project}.sports_edge_raw.raw_pbp (league-filtered aggregate)",
        },
        "seasons": seasons,
        "rows": int(len(features)),
        "schedule_rows": int(len(completed)),
        "team_game_pbp_rows": int(len(team_games)),
        "columns": [{"name": column, "dtype": str(features[column].dtype)} for column in features],
        "maximum_source_timestamp": {
            "schedule_game_date": pd.to_datetime(completed["gameday"]).max().isoformat(),
            "pbp_ingested_at": pd.to_datetime(team_games["source_max_ingested_at"], utc=True).max().isoformat(),
        },
        "audit": audit,
        "coverage": {
            "pbp_team_game": pbp_coverage,
            "closing_market_probability": float(features["closing_market_home_prob"].notna().mean()),
        },
        "excluded_feature_families": {
            "injuries": "No verified historical point-in-time coverage audit.",
            "roster_continuity": "No verified historical point-in-time coverage audit.",
            "coaching": "No verified historical point-in-time coverage audit.",
            "weather": "No verified historical point-in-time coverage audit.",
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"dataset": str(dataset_path), "manifest": str(manifest_path), "rows": len(features)}, indent=2))


if __name__ == "__main__":
    main()
