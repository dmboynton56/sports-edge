"""
Export NBA/NFL ATS and ROI history from Supabase.

This uses the Supabase REST client so it works with SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY even when the direct Postgres password is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


WIN_PROFIT_AT_MINUS_110 = 100.0 / 110.0


def _paged_select(query, page_size: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        end = start + page_size - 1
        response = query.range(start, end).execute()
        chunk = response.data or []
        rows.extend(chunk)
        if len(chunk) < page_size:
            break
        start += page_size
    return rows


def _chunks(values: list[Any], size: int = 200) -> list[list[Any]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def load_scored_games(client, league: str, season: int) -> pd.DataFrame:
    query = (
        client.table("games")
        .select(
            "id,league,season,week,game_time_utc,home_team,away_team,"
            "book_spread,home_score,away_score"
        )
        .eq("league", league)
        .eq("season", season)
        .not_.is_("home_score", "null")
        .not_.is_("away_score", "null")
        .order("game_time_utc")
    )
    return pd.DataFrame(_paged_select(query))


def load_predictions_for_games(client, game_ids: list[str]) -> pd.DataFrame:
    if not game_ids:
        return pd.DataFrame()
    frames = []
    for chunk in _chunks(game_ids):
        query = (
            client.table("model_predictions")
            .select("game_id,my_spread,my_home_win_prob,model_version,asof_ts")
            .in_("game_id", chunk)
            .order("asof_ts", desc=True)
        )
        rows = _paged_select(query)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def latest_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    out = predictions.copy()
    out["asof_ts"] = pd.to_datetime(out["asof_ts"], utc=True)
    out = out.sort_values(["game_id", "asof_ts"], ascending=[True, False])
    return out.drop_duplicates(subset=["game_id"], keep="first")


def compute_ats(games: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    numeric_cols = ["home_score", "away_score", "my_spread", "book_spread", "my_home_win_prob"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["actual_margin"] = out["home_score"] - out["away_score"]
    out["cover_margin"] = out["actual_margin"] + out["my_spread"]
    out["ats_result"] = np.select(
        [out["cover_margin"].abs() < 1e-3, out["cover_margin"] > 0],
        ["push", "win"],
        default="loss",
    )
    out["edge_vs_book"] = out["my_spread"] - out["book_spread"]
    out["abs_edge_vs_book"] = out["edge_vs_book"].abs()
    return out


def summarize_ats(games: pd.DataFrame) -> dict[str, Any]:
    if games.empty:
        return {
            "graded_games": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "flat_roi_at_minus_110": None,
            "edge_buckets": [],
        }
    wins = int((games["ats_result"] == "win").sum())
    losses = int((games["ats_result"] == "loss").sum())
    pushes = int((games["ats_result"] == "push").sum())
    risked = wins + losses
    net_units = wins * WIN_PROFIT_AT_MINUS_110 - losses
    summary: dict[str, Any] = {
        "graded_games": int(len(games)),
        "date_min": str(pd.to_datetime(games["game_time_utc"]).min().date()),
        "date_max": str(pd.to_datetime(games["game_time_utc"]).max().date()),
        "latest_asof_ts": str(pd.to_datetime(games["asof_ts"], utc=True).max()),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "ats_pct": wins / risked if risked else None,
        "net_units_at_minus_110": net_units,
        "flat_roi_at_minus_110": net_units / risked if risked else None,
        "missing_book_spread": int(games["book_spread"].isna().sum()),
    }

    with_book = games[games["book_spread"].notna()].copy()
    if with_book.empty:
        summary["edge_buckets"] = []
        return summary

    with_book["edge_bucket"] = pd.cut(
        with_book["abs_edge_vs_book"],
        bins=[0, 1, 2, 3, 5, np.inf],
        labels=["<1", "1-2", "2-3", "3-5", "5+"],
        include_lowest=True,
        right=False,
    )
    bucket = (
        with_book.groupby("edge_bucket", observed=False)
        .agg(
            games=("game_id", "count"),
            wins=("ats_result", lambda s: int((s == "win").sum())),
            losses=("ats_result", lambda s: int((s == "loss").sum())),
            pushes=("ats_result", lambda s: int((s == "push").sum())),
            avg_edge=("edge_vs_book", "mean"),
            avg_cover_margin=("cover_margin", "mean"),
        )
        .reset_index()
    )
    bucket["risked_games"] = bucket["wins"] + bucket["losses"]
    bucket["ats_pct"] = bucket["wins"] / bucket["risked_games"].replace(0, np.nan)
    bucket["roi"] = (
        bucket["wins"] * WIN_PROFIT_AT_MINUS_110 - bucket["losses"]
    ) / bucket["risked_games"].replace(0, np.nan)
    summary["edge_buckets"] = bucket.where(pd.notna(bucket), None).to_dict(orient="records")
    return summary


def export_league(client, league: str, season: int, output_dir: Path) -> dict[str, Any]:
    games = load_scored_games(client, league, season)
    predictions = load_predictions_for_games(client, games["id"].astype(str).tolist() if not games.empty else [])
    latest = latest_predictions(predictions)
    if games.empty or latest.empty:
        merged = pd.DataFrame()
    else:
        merged = games.merge(latest, left_on="id", right_on="game_id", how="inner")
        merged = compute_ats(merged)

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = league.lower()
    games_path = output_dir / f"{slug}_supabase_ats_games_{season}.csv"
    summary_path = output_dir / f"{slug}_supabase_ats_summary_{season}.json"
    if not merged.empty:
        merged.to_csv(games_path, index=False)
    summary = {
        "league": league,
        "season": season,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "games_rows": int(len(games)),
        "prediction_rows": int(len(predictions)),
        "latest_prediction_rows": int(len(latest)),
        "joined_rows": int(len(merged)),
        "games_output": str(games_path) if not merged.empty else None,
        **summarize_ats(merged),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    if not merged.empty:
        print(f"Wrote {games_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Supabase ATS history.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--leagues", nargs="+", default=["NBA", "NFL"])
    parser.add_argument("--output-dir", default="data-core/notebooks/cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv("data-core/.env")
    load_dotenv(".env")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")

    client = create_client(url, key)
    summaries = [
        export_league(client, league.upper(), args.season, Path(args.output_dir))
        for league in args.leagues
    ]
    combined_path = Path(args.output_dir) / f"supabase_ats_history_{args.season}.json"
    combined_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
