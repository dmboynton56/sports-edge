#!/usr/bin/env python3
"""Grade PGA tournament predictions against final ESPN result rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pga.live_leaderboard import normalize_name  # noqa: E402
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials  # noqa: E402


DEFAULT_PGA_JSON = REPO_ROOT / "web" / "public" / "data" / "pga_tournaments" / "current.json"
DEFAULT_RESULTS = ROOT / "src" / "data" / "archive" / "pga_results_espn_supplement.tsv"
DEFAULT_OUT = ROOT / "notebooks" / "cache" / "pga_prediction_results.csv"


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _position_numeric(value: Any) -> int | None:
    text = str(value or "").upper().strip()
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def _prob(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and pd.notna(value):
            return float(value)
    return None


def grade_predictions(pga_json: Path, results_tsv: Path) -> pd.DataFrame:
    payload = json.loads(pga_json.read_text(encoding="utf-8"))
    event = payload.get("event") or {}
    event_key = str(event.get("eventKey") or "")
    event_name = str(event.get("name") or "")
    start_date = str(event.get("startDate") or "")[:10]
    predictions = payload.get("predictions") or []
    model_version = str((payload.get("predictionMeta") or {}).get("model_version") or "pga-baseline-v0")
    prediction_ts = payload.get("generatedAt")

    if not predictions or not results_tsv.exists():
        return pd.DataFrame()

    results = pd.read_csv(results_tsv, sep="\t")
    if results.empty:
        return pd.DataFrame()
    results["start"] = pd.to_datetime(results["start"], errors="coerce").dt.strftime("%Y-%m-%d")
    tournament_norm = normalize_name(event_name)
    result_tournament_norm = results["tournament"].astype(str).map(normalize_name)
    scoped = results[
        (results["start"] == start_date)
        & (
            (result_tournament_norm == tournament_norm)
            | result_tournament_norm.str.contains(tournament_norm, regex=False)
            | result_tournament_norm.map(lambda value: value in tournament_norm)
        )
    ].copy()
    if scoped.empty:
        scoped = results[
            (results["start"] == start_date)
            & (results["tournament"].astype(str).map(normalize_name).str.contains(tournament_norm, regex=False))
        ].copy()
    if scoped.empty:
        return pd.DataFrame()

    scoped["_name_key"] = scoped["name"].astype(str).map(normalize_name)
    result_by_name = scoped.drop_duplicates("_name_key", keep="last").set_index("_name_key")
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        player = str(pred.get("player") or "")
        result = result_by_name.loc[normalize_name(player)] if normalize_name(player) in result_by_name.index else None
        if result is None:
            continue
        final_position = result.get("position")
        pos_num = _position_numeric(final_position)
        made_cut = pos_num is not None
        rows.append(
            {
                "event_key": event_key,
                "season": event.get("season"),
                "player_name": player,
                "player_id": pred.get("player_id"),
                "model_version": model_version,
                "prediction_ts": prediction_ts,
                "win_prob": _prob(pred, "best_calibrated_target_win_prob", "lr_target_win_prob"),
                "top10_prob": _prob(pred, "best_calibrated_target_top10_prob", "lr_target_top10_prob"),
                "top20_prob": _prob(pred, "best_calibrated_target_top20_prob", "lr_target_top20_prob"),
                "make_cut_prob": _prob(pred, "best_calibrated_target_made_cut_prob", "lr_target_made_cut_prob"),
                "final_position": final_position,
                "final_position_numeric": pos_num,
                "final_score": result.get("score"),
                "made_cut": made_cut,
                "top10_hit": pos_num is not None and pos_num <= 10,
                "top20_hit": pos_num is not None and pos_num <= 20,
                "winner_hit": pos_num == 1,
                "raw_record": result.where(pd.notna(result), None).to_dict(),
            }
        )
    return pd.DataFrame(rows)


def sync_supabase(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    creds = load_supabase_credentials()
    conn = create_pg_connection(
        supabase_url=creds["url"],
        password=creds["db_password"],
        host_override=creds.get("db_host"),
        port=creds["db_port"],
        database=creds["db_name"],
        user=creds["db_user"],
    )
    evaluated_at = datetime.now(timezone.utc)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            (
                _clean(row.get("event_key")),
                _clean(row.get("season")),
                _clean(row.get("player_name")),
                _clean(row.get("player_id")),
                _clean(row.get("model_version")),
                _clean(row.get("prediction_ts")),
                _clean(row.get("win_prob")),
                _clean(row.get("top10_prob")),
                _clean(row.get("top20_prob")),
                _clean(row.get("make_cut_prob")),
                _clean(row.get("final_position")),
                _clean(row.get("final_position_numeric")),
                _clean(row.get("final_score")),
                _clean(row.get("made_cut")),
                _clean(row.get("top10_hit")),
                _clean(row.get("top20_hit")),
                _clean(row.get("winner_hit")),
                evaluated_at,
                json.dumps(row.get("raw_record") or {}, default=str),
            )
        )
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                insert into pga_prediction_results (
                  event_key, season, player_name, player_id, model_version, prediction_ts,
                  win_prob, top10_prob, top20_prob, make_cut_prob,
                  final_position, final_position_numeric, final_score,
                  made_cut, top10_hit, top20_hit, winner_hit, evaluated_at, raw_record
                )
                values (
                  %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s,
                  %s, %s, %s,
                  %s, %s, %s, %s, %s, %s::jsonb
                )
                on conflict (event_key, player_name, model_version, prediction_ts)
                do update set
                  win_prob = excluded.win_prob,
                  top10_prob = excluded.top10_prob,
                  top20_prob = excluded.top20_prob,
                  make_cut_prob = excluded.make_cut_prob,
                  final_position = excluded.final_position,
                  final_position_numeric = excluded.final_position_numeric,
                  final_score = excluded.final_score,
                  made_cut = excluded.made_cut,
                  top10_hit = excluded.top10_hit,
                  top20_hit = excluded.top20_hit,
                  winner_hit = excluded.winner_hit,
                  evaluated_at = excluded.evaluated_at,
                  raw_record = excluded.raw_record,
                  updated_at = now()
                """,
                rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade PGA predictions against ESPN result supplement.")
    parser.add_argument("--pga-json", type=Path, default=DEFAULT_PGA_JSON)
    parser.add_argument("--results-tsv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sync-supabase", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    frame = grade_predictions(args.pga_json, args.results_tsv)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(frame)} PGA graded rows to {args.out_csv}")
    if args.sync_supabase:
        print(f"Synced {sync_supabase(frame)} PGA result rows to Supabase")


if __name__ == "__main__":
    main()
