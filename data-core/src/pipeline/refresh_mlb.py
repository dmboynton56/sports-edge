#!/usr/bin/env python3
"""
Daily MLB refresh script.

Fetches the public MLB schedule for a rolling window, scores home-win
probabilities with the saved v3 winner artifact, and writes the rows needed by
the shared BigQuery-to-Supabase serving path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from src.data.mlb_fetcher import fetch_mlb_schedule
from src.models.mlb_winner_model import build_mlb_prediction_features


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "mlb_winner_model_v3.pkl"
FEATURE_SNAPSHOT_COLUMNS = [
    "game_id",
    "league",
    "season",
    "game_date",
    "as_of_ts",
    "home_team",
    "away_team",
    "home_win",
    "home_margin",
    "rest_home",
    "rest_away",
    "b2b_home",
    "b2b_away",
    "is_3in4_home",
    "is_3in4_away",
    "opp_strength_home_season",
    "opp_strength_away_season",
    "home_team_win_pct",
    "away_team_win_pct",
    "home_team_point_diff",
    "away_team_point_diff",
    "rest_differential",
    "win_pct_differential",
    "point_diff_differential",
    "opp_strength_differential",
    "is_3in4_differential",
    "week_number",
    "month",
    "is_playoff",
    "form_home_epa_off_3",
    "form_home_epa_off_5",
    "form_home_epa_off_10",
    "form_home_epa_def_3",
    "form_home_epa_def_5",
    "form_home_epa_def_10",
    "form_away_epa_off_3",
    "form_away_epa_off_5",
    "form_away_epa_off_10",
    "form_away_epa_def_3",
    "form_away_epa_def_5",
    "form_away_epa_def_10",
    "form_epa_off_diff_3",
    "form_epa_off_diff_5",
    "form_epa_off_diff_10",
    "form_epa_def_diff_3",
    "form_epa_def_diff_5",
    "form_epa_def_diff_10",
    "form_home_net_rating_3",
    "form_home_net_rating_5",
    "form_home_net_rating_10",
    "form_away_net_rating_3",
    "form_away_net_rating_5",
    "form_away_net_rating_10",
    "form_net_rating_diff_3",
    "form_net_rating_diff_5",
    "form_net_rating_diff_10",
    "feature_version",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MLB home-win predictions.")
    parser.add_argument("--project", required=True, help="GCP project ID.")
    parser.add_argument("--model-version", default="v3", help="Model version tag.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to MLB winner artifact.")
    parser.add_argument(
        "--date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=None,
        help="Anchor date in YYYY-MM-DD. Default: today UTC.",
    )
    parser.add_argument("--season", type=int, default=None, help="MLB season override.")
    parser.add_argument("--lookback-days", type=int, default=0, help="Days before anchor date to score.")
    parser.add_argument("--lookahead-days", type=int, default=0, help="Days after anchor date to score.")
    parser.add_argument("--min-prior-games", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write BigQuery rows.")
    return parser.parse_args()


def _game_id(game_pk: object) -> str:
    return f"MLB_{int(game_pk)}"


def _date_window(anchor: date, lookback_days: int, lookahead_days: int) -> tuple[date, date]:
    if lookback_days < 0 or lookahead_days < 0:
        raise ValueError("lookback-days and lookahead-days must be non-negative.")
    return anchor - timedelta(days=lookback_days), anchor + timedelta(days=lookahead_days)


def _json_safe(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _raw_record(row: pd.Series) -> str:
    payload = {
        "game_pk": int(row["game_pk"]),
        "game_datetime": _json_safe(row.get("game_datetime")),
        "status": _json_safe(row.get("status")),
        "venue_id": _json_safe(row.get("venue_id")),
        "venue_name": _json_safe(row.get("venue_name")),
        "home_team_id": _json_safe(row.get("home_team_id")),
        "home_team_abbr": _json_safe(row.get("home_team_abbr")),
        "home_probable_pitcher_id": _json_safe(row.get("home_probable_pitcher_id")),
        "home_probable_pitcher": _json_safe(row.get("home_probable_pitcher")),
        "away_team_id": _json_safe(row.get("away_team_id")),
        "away_team_abbr": _json_safe(row.get("away_team_abbr")),
        "away_probable_pitcher_id": _json_safe(row.get("away_probable_pitcher_id")),
        "away_probable_pitcher": _json_safe(row.get("away_probable_pitcher")),
    }
    return json.dumps(payload, sort_keys=True)


def _build_raw_schedules(schedule: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    window = schedule[
        (schedule["game_date"].dt.date >= start_date)
        & (schedule["game_date"].dt.date <= end_date)
    ].copy()
    if window.empty:
        return pd.DataFrame()

    raw = pd.DataFrame(
        {
            "game_id": window["game_pk"].map(_game_id),
            "league": "MLB",
            "season": window["season"].astype(int),
            "week": None,
            "game_date": window["game_date"].dt.date,
            "home_team": window["home_team"],
            "away_team": window["away_team"],
            "home_score": pd.to_numeric(window["home_score"], errors="coerce").astype("Int64"),
            "away_score": pd.to_numeric(window["away_score"], errors="coerce").astype("Int64"),
            "venue": window["venue_name"],
            "result": window["status"],
            "raw_record": window.apply(_raw_record, axis=1),
        }
    )
    return raw.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def _score_window(
    schedule: pd.DataFrame,
    *,
    start_date: date,
    end_date: date,
    artifact: dict,
    min_prior_games: int,
) -> pd.DataFrame:
    feature_frames = []
    for game_day in pd.date_range(start=start_date, end=end_date, freq="D"):
        game_date = game_day.date()
        games_to_score = schedule[schedule["game_date"].dt.date == game_date].copy()
        if games_to_score.empty:
            continue

        history = schedule[
            (schedule["game_date"].dt.date < game_date)
            & schedule["home_score"].notna()
            & schedule["away_score"].notna()
        ].copy()
        if history.empty:
            continue

        features = build_mlb_prediction_features(
            history,
            games_to_score,
            min_prior_games=min_prior_games,
        )
        if not features.empty:
            feature_frames.append(features)

    if not feature_frames:
        return pd.DataFrame()

    features = pd.concat(feature_frames, ignore_index=True)
    feature_cols = artifact["feature_columns"]
    probabilities = artifact["model"].predict_proba(features[feature_cols])[:, 1]
    predictions = features[
        [
            "game_pk",
            "season",
            "game_date",
            "game_datetime",
            "away_team",
            "home_team",
        ]
    ].copy()
    predictions["game_id"] = predictions["game_pk"].map(_game_id)
    predictions["home_win_prob"] = probabilities
    return predictions.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)


def _build_feature_snapshots(
    raw_schedules: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    as_of_ts: datetime,
    feature_version: str,
) -> pd.DataFrame:
    if raw_schedules.empty or predictions.empty:
        return pd.DataFrame(columns=FEATURE_SNAPSHOT_COLUMNS)

    base = raw_schedules.merge(
        predictions[["game_id"]],
        on="game_id",
        how="inner",
    ).copy()
    snapshots = pd.DataFrame(
        {
            "game_id": base["game_id"],
            "league": "MLB",
            "season": base["season"].astype("Int64"),
            "game_date": pd.to_datetime(base["game_date"]).dt.date,
            "as_of_ts": as_of_ts,
            "home_team": base["home_team"],
            "away_team": base["away_team"],
            "home_win": (
                pd.to_numeric(base["home_score"], errors="coerce")
                > pd.to_numeric(base["away_score"], errors="coerce")
            ).where(
                pd.to_numeric(base["home_score"], errors="coerce").notna()
                & pd.to_numeric(base["away_score"], errors="coerce").notna(),
                None,
            ),
            "home_margin": (
                pd.to_numeric(base["home_score"], errors="coerce")
                - pd.to_numeric(base["away_score"], errors="coerce")
            ),
            "month": pd.to_datetime(base["game_date"]).dt.month.astype("Int64"),
            "is_playoff": False,
            "feature_version": feature_version,
        }
    )
    for column in FEATURE_SNAPSHOT_COLUMNS:
        if column not in snapshots.columns:
            snapshots[column] = None
    return snapshots[FEATURE_SNAPSHOT_COLUMNS]


def _build_model_predictions(predictions: pd.DataFrame, *, model_version: str, prediction_ts: datetime) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "prediction_id": predictions.apply(
                lambda row: f"{row['game_id']}_{model_version}_{prediction_ts.strftime('%Y%m%dT%H%M%S')}",
                axis=1,
            ),
            "game_id": predictions["game_id"],
            "league": "MLB",
            "season": predictions["season"].astype("Int64"),
            "season_week": None,
            "model_version": model_version,
            "predicted_spread": None,
            "home_win_prob": predictions["home_win_prob"].astype(float),
            "prediction_ts": prediction_ts,
        }
    )
    out["input_hash"] = out.apply(
        lambda row: hashlib.sha256(
            f"{row['game_id']}|{row['model_version']}|{row['home_win_prob']:.8f}".encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    return out


def _delete_window_rows(
    client: bigquery.Client,
    *,
    project: str,
    table: str,
    start_date: date,
    end_date: date,
    model_version: Optional[str] = None,
) -> None:
    version_clause = "AND model_version = @model_version" if model_version else ""
    date_column = "DATE(prediction_ts)" if table == "model_predictions" else "game_date"
    query = f"""
        DELETE FROM `{project}.sports_edge_curated.{table}`
        WHERE league = 'MLB'
          AND {date_column} BETWEEN @start_date AND @end_date
          {version_clause}
    """
    if table == "raw_schedules":
        query = f"""
            DELETE FROM `{project}.sports_edge_raw.raw_schedules`
            WHERE league = 'MLB'
              AND game_date BETWEEN @start_date AND @end_date
        """
    params: list[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
    ]
    if model_version:
        params.append(bigquery.ScalarQueryParameter("model_version", "STRING", model_version))
    client.query(query, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()


def _delete_prediction_games(client: bigquery.Client, project: str, game_ids: Iterable[str], model_version: str) -> None:
    ids = list(game_ids)
    if not ids:
        return
    query = f"""
        DELETE FROM `{project}.sports_edge_curated.model_predictions`
        WHERE league = 'MLB'
          AND game_id IN UNNEST(@game_ids)
          AND model_version = @model_version
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("game_ids", "STRING", ids),
            bigquery.ScalarQueryParameter("model_version", "STRING", model_version),
        ]
    )
    client.query(query, job_config=job_config).result()


def _load_dataframe(client: bigquery.Client, df: pd.DataFrame, table_id: str) -> None:
    if df.empty:
        return
    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND"),
    )
    job.result()


def _log_model_run(
    client: bigquery.Client,
    project: str,
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    rows_written: int,
    status: str,
    error_text: Optional[str] = None,
) -> None:
    table_id = f"{project}.sports_edge_curated.model_runs"
    df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "league": "MLB",
                "rows_written": rows_written,
                "status": status,
                "error_text": error_text,
            }
        ]
    )
    _load_dataframe(client, df, table_id)


def main() -> None:
    load_dotenv()
    args = _parse_args()
    client = bigquery.Client(project=args.project)

    anchor_date = args.date or datetime.now(tz=timezone.utc).date()
    start_date, end_date = _date_window(anchor_date, args.lookback_days, args.lookahead_days)
    season = args.season or anchor_date.year
    season_start = date(season, 3, 1)
    prediction_ts = datetime.now(tz=timezone.utc)
    run_id = f"mlb_{prediction_ts.strftime('%Y%m%dT%H%M%S')}"
    print(f"Building MLB predictions for {start_date} to {end_date}. Target season={season}.")

    with open(args.model_path, "rb") as handle:
        artifact = pickle.load(handle)

    schedule = fetch_mlb_schedule(
        season,
        start_date=season_start,
        end_date=end_date,
        include_uncompleted=True,
    )
    if schedule.empty:
        print(f"No MLB schedule rows found for season={season} through {end_date}. Exiting.")
        return

    raw_schedules = _build_raw_schedules(schedule, start_date, end_date)
    predictions = _score_window(
        schedule,
        start_date=start_date,
        end_date=end_date,
        artifact=artifact,
        min_prior_games=args.min_prior_games,
    )
    if predictions.empty:
        print("No MLB predictions were generated.")
        return

    model_predictions = _build_model_predictions(
        predictions,
        model_version=args.model_version,
        prediction_ts=prediction_ts,
    )
    feature_snapshots = _build_feature_snapshots(
        raw_schedules,
        predictions,
        as_of_ts=prediction_ts,
        feature_version=f"mlb-winner-{args.model_version}",
    )

    if args.dry_run:
        print(model_predictions[["game_id", "home_win_prob", "model_version"]].to_string(index=False))
        return

    started_at = datetime.now(tz=timezone.utc)
    try:
        _delete_window_rows(
            client,
            project=args.project,
            table="raw_schedules",
            start_date=start_date,
            end_date=end_date,
        )
        _delete_window_rows(
            client,
            project=args.project,
            table="feature_snapshots",
            start_date=start_date,
            end_date=end_date,
        )
        _delete_prediction_games(
            client,
            args.project,
            model_predictions["game_id"].dropna().tolist(),
            args.model_version,
        )
        _load_dataframe(client, raw_schedules, f"{args.project}.sports_edge_raw.raw_schedules")
        _load_dataframe(client, feature_snapshots, f"{args.project}.sports_edge_curated.feature_snapshots")
        _load_dataframe(client, model_predictions, f"{args.project}.sports_edge_curated.model_predictions")
        finished_at = datetime.now(tz=timezone.utc)
        _log_model_run(
            client,
            args.project,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            rows_written=len(model_predictions),
            status="SUCCESS",
        )
        print(
            f"Wrote {len(raw_schedules)} raw MLB games, {len(feature_snapshots)} feature rows, "
            f"and {len(model_predictions)} predictions."
        )
    except Exception as exc:
        finished_at = datetime.now(tz=timezone.utc)
        _log_model_run(
            client,
            args.project,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            rows_written=0,
            status="FAILED",
            error_text=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
