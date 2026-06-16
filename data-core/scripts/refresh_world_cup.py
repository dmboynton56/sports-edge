#!/usr/bin/env python3
"""Refresh World Cup inputs, predictions, warehouse rows, and serving cache."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Optional

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
    normalize_fifa_rankings,
    normalize_fixtures_for_model,
    normalize_world_football_elo,
    parse_espn_world_cup_scoreboard,
    write_world_cup_inputs,
)
from scripts.predict_world_cup import (
    build_payload,
    load_fixtures,
    load_round_of_32_slots,
    load_team_ratings,
)
from scripts.sync_world_cup_to_supabase import sync_payload


MODEL_NAME = "sports_edge_world_cup"
DEFAULT_MODEL_VERSION = "world-cup-v0-live"
DEFAULT_OUTPUT_DIR = ROOT / "notebooks" / "cache" / "world_cup"
TOURNAMENT_WINDOWS = {
    2026: (date(2026, 6, 11), date(2026, 7, 19)),
}


@dataclass(frozen=True)
class OutputPaths:
    teams_csv: Path
    fixtures_csv: Path
    round_of_32_slots_json: Path
    predictions_json: Path


def tournament_window_for_season(season: int) -> tuple[date, date]:
    """Return the fixture fetch window for a World Cup season."""

    return TOURNAMENT_WINDOWS.get(season, (date(season, 6, 1), date(season, 7, 31)))


def build_output_paths(output_dir: Path, model_version: str) -> OutputPaths:
    safe_model_version = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_version).strip("_")
    if not safe_model_version:
        safe_model_version = DEFAULT_MODEL_VERSION
    return OutputPaths(
        teams_csv=output_dir / "world_cup_team_ratings.csv",
        fixtures_csv=output_dir / "world_cup_fixtures.csv",
        round_of_32_slots_json=output_dir / "world_cup_round_of_32_slots.json",
        predictions_json=output_dir / f"world_cup_predictions_{safe_model_version}.json",
    )


def _read_optional_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    return pd.read_csv(path)


def _load_fixtures(args: argparse.Namespace, *, start_date: date, end_date: date) -> pd.DataFrame:
    if args.fixtures_csv:
        return pd.read_csv(args.fixtures_csv)
    if args.espn_scoreboard_json:
        payload = json.loads(args.espn_scoreboard_json.read_text())
        return parse_espn_world_cup_scoreboard(payload, season=args.season)
    return fetch_espn_world_cup_fixtures(
        start_date=start_date,
        end_date=end_date,
        season=args.season,
        timeout=args.timeout,
    )


def _load_world_elo(args: argparse.Namespace) -> Optional[pd.DataFrame]:
    if args.world_elo_csv:
        return pd.read_csv(args.world_elo_csv)
    if args.skip_world_elo:
        return None
    return fetch_world_football_elo(timeout=args.timeout)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _json_records(df: Optional[pd.DataFrame]) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.where(pd.notna(df), None).to_dict(orient="records")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hash(*frames: Optional[pd.DataFrame]) -> str:
    return _stable_hash([_json_records(frame) for frame in frames])[:16]


def _timestamp(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _raw_record(row: dict[str, Any]) -> str:
    raw = row.get("raw_record")
    if raw:
        return str(raw)
    return json.dumps(row, sort_keys=True, default=str)


def _rank_probs_for_team(payload: dict[str, Any], team: str, group_name: Optional[str]) -> dict[str, Any]:
    if not group_name:
        return {}
    for row in payload.get("groupRankProbabilities", {}).get(str(group_name), []):
        if row.get("team") == team:
            return {
                key: row.get(key)
                for key in ("rank_1", "rank_2", "rank_3", "rank_4")
                if key in row
            }
    return {}


def _raw_fixture_rows(fixtures_raw: pd.DataFrame, *, run_ts: datetime) -> pd.DataFrame:
    rows = []
    for row in _json_records(fixtures_raw):
        match_id = row.get("external_match_id") or row.get("match_id")
        rows.append(
            {
                "external_match_id": None if match_id is None else str(match_id),
                "season": row.get("season"),
                "tournament": row.get("tournament") or "FIFA World Cup",
                "stage": row.get("stage"),
                "group_name": row.get("group_name") or row.get("group"),
                "kickoff_utc": _timestamp(row.get("kickoff_utc")),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "venue": row.get("venue"),
                "status": row.get("status"),
                "home_score": row.get("home_score"),
                "away_score": row.get("away_score"),
                "neutral_site": row.get("neutral_site"),
                "source": row.get("source") or "world_cup_refresh",
                "ingested_at": run_ts,
                "raw_record": _raw_record(row),
            }
        )
    return pd.DataFrame(rows)


def _rating_rows(
    team_ratings: pd.DataFrame,
    *,
    season: int,
    model_version: str,
    rating_ts: datetime,
    source_hash: str,
) -> pd.DataFrame:
    data = team_ratings.copy()
    data["rating_ts"] = rating_ts
    data["season"] = season
    data["model_version"] = model_version
    data["source_hash"] = source_hash
    data = data.rename(columns={"group": "group_name"})
    return data[
        [
            "rating_ts",
            "season",
            "team",
            "group_name",
            "fifa_rank",
            "elo",
            "form_points_per_game",
            "form_goal_diff_per_game",
            "world_cup_experience_score",
            "star_player_score",
            "host_boost",
            "market_rating",
            "model_version",
            "source_hash",
        ]
    ]


def _prediction_rows(payload: dict[str, Any], *, model_name: str = MODEL_NAME) -> pd.DataFrame:
    season = int(payload.get("season", 2026))
    model_version = str(payload["modelVersion"])
    rows = []
    for match in payload.get("matches", []):
        prediction_ts = match.get("prediction_ts") or payload.get("updatedAt")
        rows.append(
            {
                "prediction_id": _stable_hash([model_name, model_version, match.get("match_id"), prediction_ts])[:32],
                "match_id": str(match.get("match_id")),
                "prediction_ts": _timestamp(prediction_ts),
                "season": season,
                "stage": match.get("stage"),
                "group_name": match.get("group"),
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "model_name": model_name,
                "model_version": model_version,
                "home_win_prob": match.get("home_win_prob"),
                "draw_prob": match.get("draw_prob"),
                "away_win_prob": match.get("away_win_prob"),
                "home_knockout_win_prob": match.get("home_knockout_win_prob"),
                "away_knockout_win_prob": match.get("away_knockout_win_prob"),
                "projected_home_goals": match.get("projected_home_goals"),
                "projected_away_goals": match.get("projected_away_goals"),
                "feature_json": json.dumps(match, sort_keys=True, default=str),
            }
        )
    return pd.DataFrame(rows)


def _team_probability_rows(payload: dict[str, Any], *, model_name: str = MODEL_NAME) -> pd.DataFrame:
    season = int(payload.get("season", 2026))
    model_version = str(payload["modelVersion"])
    simulation_ts = _timestamp(payload.get("updatedAt"))
    rows = []
    for row in payload.get("teamProbabilities", []):
        group_name = row.get("group")
        rows.append(
            {
                "simulation_ts": simulation_ts,
                "season": season,
                "team": row.get("team"),
                "group_name": group_name,
                "model_name": model_name,
                "model_version": model_version,
                "simulations": payload.get("simulations"),
                "bracket_source": payload.get("bracketSource"),
                "rating": row.get("rating"),
                "group_prob": row.get("group_prob"),
                "round_of_32_prob": row.get("round_of_32"),
                "round_of_16_prob": row.get("round_of_16"),
                "quarterfinal_prob": row.get("quarterfinal"),
                "semifinal_prob": row.get("semifinal"),
                "final_prob": row.get("final"),
                "champion_prob": row.get("champion"),
                "group_rank_probs": json.dumps(
                    _rank_probs_for_team(payload, str(row.get("team")), group_name),
                    sort_keys=True,
                    default=str,
                ),
            }
        )
    return pd.DataFrame(rows)


def _simulation_run_rows(
    payload: dict[str, Any],
    *,
    rows_written: int,
    model_name: str = MODEL_NAME,
    success: bool = True,
    error_text: Optional[str] = None,
) -> pd.DataFrame:
    started_at = _timestamp(payload.get("updatedAt"))
    return pd.DataFrame(
        [
            {
                "run_id": _stable_hash([model_name, payload.get("modelVersion"), payload.get("updatedAt")])[:32],
                "started_at": started_at,
                "finished_at": _now_utc(),
                "season": payload.get("season", 2026),
                "model_name": model_name,
                "model_version": payload.get("modelVersion"),
                "simulations": payload.get("simulations"),
                "bracket_source": payload.get("bracketSource"),
                "rows_written": rows_written,
                "success": success,
                "error_text": error_text,
            }
        ]
    )


def _raw_world_elo_rows(world_elo: Optional[pd.DataFrame], *, run_ts: datetime) -> pd.DataFrame:
    if world_elo is None or world_elo.empty:
        return pd.DataFrame()
    normalized = normalize_world_football_elo(world_elo)
    if "elo_rank" not in normalized.columns:
        normalized["elo_rank"] = pd.NA
    normalized["snapshot_date"] = run_ts.date()
    normalized["source"] = "world_football_elo"
    normalized["ingested_at"] = run_ts
    normalized["raw_record"] = normalized.apply(lambda row: json.dumps(row.to_dict(), default=str), axis=1)
    return normalized[["snapshot_date", "team", "elo", "elo_rank", "source", "ingested_at", "raw_record"]]


def _raw_fifa_ranking_rows(fifa_rankings: Optional[pd.DataFrame], *, run_ts: datetime) -> pd.DataFrame:
    if fifa_rankings is None or fifa_rankings.empty:
        return pd.DataFrame()
    normalized = normalize_fifa_rankings(fifa_rankings)
    if "fifa_points" not in normalized.columns:
        normalized["fifa_points"] = pd.NA
    normalized["ranking_date"] = run_ts.date()
    normalized["source"] = "manual_or_cached_csv"
    normalized["ingested_at"] = run_ts
    normalized["raw_record"] = normalized.apply(lambda row: json.dumps(row.to_dict(), default=str), axis=1)
    return normalized[["ranking_date", "team", "fifa_rank", "fifa_points", "source", "ingested_at", "raw_record"]]


def build_bigquery_frames(
    *,
    fixtures_raw: pd.DataFrame,
    team_ratings: pd.DataFrame,
    payload: dict[str, Any],
    world_elo: Optional[pd.DataFrame] = None,
    fifa_rankings: Optional[pd.DataFrame] = None,
    run_ts: Optional[datetime] = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    run_ts = run_ts or _now_utc()
    source_hash = _source_hash(fixtures_raw, team_ratings, world_elo, fifa_rankings)
    prediction_rows = _prediction_rows(payload)
    team_probability_rows = _team_probability_rows(payload)
    rows_written = len(prediction_rows) + len(team_probability_rows)
    return {
        ("sports_edge_raw", "raw_wc_fixtures"): _raw_fixture_rows(fixtures_raw, run_ts=run_ts),
        ("sports_edge_raw", "raw_world_elo"): _raw_world_elo_rows(world_elo, run_ts=run_ts),
        ("sports_edge_raw", "raw_fifa_rankings"): _raw_fifa_ranking_rows(fifa_rankings, run_ts=run_ts),
        ("sports_edge_curated", "wc_team_ratings"): _rating_rows(
            team_ratings,
            season=int(payload.get("season", 2026)),
            model_version=str(payload["modelVersion"]),
            rating_ts=run_ts,
            source_hash=source_hash,
        ),
        ("sports_edge_curated", "wc_match_predictions"): prediction_rows,
        ("sports_edge_curated", "wc_team_probabilities"): team_probability_rows,
        ("sports_edge_curated", "wc_simulation_runs"): _simulation_run_rows(
            payload,
            rows_written=rows_written,
        ),
    }


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


def write_bigquery_frames(
    frames: dict[tuple[str, str], pd.DataFrame],
    *,
    project: str,
    write_disposition: str,
) -> dict[str, int]:
    counts = {}
    for (dataset, table), frame in frames.items():
        counts[f"{dataset}.{table}"] = len(frame)
        if frame.empty:
            continue
        _write_bigquery(
            frame,
            project=project,
            dataset=dataset,
            table=table,
            write_disposition=write_disposition,
        )
    return counts


def _sync_supabase(payload: dict[str, Any], *, dry_run: bool) -> dict[str, int]:
    if dry_run:
        return sync_payload(None, payload, dry_run=True)

    from dotenv import load_dotenv
    from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials

    load_dotenv(ROOT / ".env")
    creds = load_supabase_credentials()
    missing = {
        "SUPABASE_URL": creds["url"],
        "SUPABASE_DB_PASSWORD or supabaseDBpass": creds["db_password"],
    }
    missing_keys = [key for key, value in missing.items() if not value]
    if missing_keys:
        raise RuntimeError(f"Missing Supabase credentials: {', '.join(missing_keys)}")

    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    try:
        return sync_payload(conn, payload)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh World Cup prediction artifacts.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date())
    parser.add_argument("--end-date", type=lambda value: datetime.strptime(value, "%Y-%m-%d").date())
    parser.add_argument("--fixtures-csv", type=Path)
    parser.add_argument("--teams-csv", type=Path)
    parser.add_argument("--espn-scoreboard-json", type=Path)
    parser.add_argument("--fifa-rankings-csv", type=Path)
    parser.add_argument("--world-elo-csv", type=Path)
    parser.add_argument("--skip-world-elo", action="store_true")
    parser.add_argument("--recent-results-csv", type=Path)
    parser.add_argument("--world-cup-history-csv", type=Path)
    parser.add_argument("--player-form-csv", type=Path)
    parser.add_argument("--market-odds-csv", type=Path)
    parser.add_argument("--as-of-date")
    parser.add_argument("--lookback-matches", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--n-sims", type=int, default=50000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--write-bigquery", action="store_true")
    parser.add_argument("--project", help="GCP project for --write-bigquery.")
    parser.add_argument("--write-disposition", default="WRITE_APPEND", choices=["WRITE_APPEND", "WRITE_TRUNCATE"])
    parser.add_argument("--sync-supabase", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_start, default_end = tournament_window_for_season(args.season)
    start_date = args.start_date or default_start
    end_date = args.end_date or default_end
    run_ts = _now_utc()
    seed = args.seed if args.seed is not None else args.season * 10000 + 611
    paths = build_output_paths(args.output_dir, args.model_version)

    fixtures_raw = _load_fixtures(args, start_date=start_date, end_date=end_date)
    fixtures = normalize_fixtures_for_model(fixtures_raw)
    teams_seed = _read_optional_csv(args.teams_csv)
    fifa_rankings = _read_optional_csv(args.fifa_rankings_csv)
    world_elo = _load_world_elo(args)
    recent_results = _read_optional_csv(args.recent_results_csv)
    team_form = None if recent_results is not None else extract_espn_team_form(fixtures_raw)

    ratings = build_team_rating_inputs(
        teams=teams_seed,
        fixtures=fixtures,
        fifa_rankings=fifa_rankings,
        world_elo=world_elo,
        recent_results=recent_results,
        team_form=team_form,
        world_cup_history=_read_optional_csv(args.world_cup_history_csv),
        player_form=_read_optional_csv(args.player_form_csv),
        market_odds=_read_optional_csv(args.market_odds_csv),
        season=args.season,
        as_of_date=args.as_of_date or run_ts.date().isoformat(),
        lookback_matches=args.lookback_matches,
    )

    write_world_cup_inputs(
        team_ratings=ratings,
        fixtures=fixtures,
        teams_csv=paths.teams_csv,
        fixtures_csv=paths.fixtures_csv,
        round_of_32_slots_json=paths.round_of_32_slots_json,
    )

    payload = build_payload(
        teams=load_team_ratings(paths.teams_csv),
        fixtures=load_fixtures(paths.fixtures_csv),
        season=args.season,
        model_version=args.model_version,
        n_sims=args.n_sims,
        seed=seed,
        round_of_32_slots=load_round_of_32_slots(paths.round_of_32_slots_json),
    )

    paths.predictions_json.parent.mkdir(parents=True, exist_ok=True)
    paths.predictions_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(ratings)} teams, {len(fixtures)} fixtures, "
        f"{len(derive_round_of_32_slots(fixtures))} Round-of-32 slots, and "
        f"{len(payload['matches'])} predictions to {args.output_dir}"
    )
    print(f"Prediction JSON: {paths.predictions_json}")

    if args.write_bigquery:
        if not args.project:
            raise ValueError("--project is required with --write-bigquery")
        frames = build_bigquery_frames(
            fixtures_raw=fixtures_raw,
            team_ratings=ratings,
            payload=payload,
            world_elo=world_elo,
            fifa_rankings=fifa_rankings,
            run_ts=run_ts,
        )
        if args.dry_run:
            counts = {f"{dataset}.{table}": len(frame) for (dataset, table), frame in frames.items()}
            print(json.dumps({"bigquery_dry_run": counts}, sort_keys=True))
        else:
            counts = write_bigquery_frames(
                frames,
                project=args.project,
                write_disposition=args.write_disposition,
            )
            print(json.dumps({"bigquery": counts}, sort_keys=True))

    if args.sync_supabase:
        result = _sync_supabase(payload, dry_run=args.dry_run)
        key = "supabase_dry_run" if args.dry_run else "supabase"
        print(json.dumps({key: result}, sort_keys=True))


if __name__ == "__main__":
    main()
