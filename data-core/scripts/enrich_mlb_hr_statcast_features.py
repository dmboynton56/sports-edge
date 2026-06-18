#!/usr/bin/env python3
"""Add leakage-safe Statcast matchup features to MLB HR training rows.

The script downloads pitch-level Baseball Savant CSV data, builds batter and
pitcher aggregates using only games before each training-row date, then writes
an enriched training file for PyTorch experiments. It is intentionally separate
from the production random-forest pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows_enriched.csv"
DEFAULT_OUTPUT = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows_statcast.csv"
DEFAULT_CACHE = ROOT / "notebooks" / "cache" / "mlb_statcast_2026.csv"
DEFAULT_AUDIT = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows_statcast_audit.json"
SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

FASTBALL_TYPES = {"FF", "FA", "FT", "SI", "FC"}
BREAKING_TYPES = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "SC", "KN", "EP"}


def _date_arg(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(end, current + pd.Timedelta(days=days - 1))
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks


def _fetch_statcast_chunk(start: pd.Timestamp, end: pd.Timestamp, *, timeout: int) -> pd.DataFrame:
    response = requests.get(
        SAVANT_CSV_URL,
        params={
            "all": "true",
            "type": "details",
            "player_type": "batter",
            "game_date_gt": start.strftime("%Y-%m-%d"),
            "game_date_lt": end.strftime("%Y-%m-%d"),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    if not response.text.strip():
        return pd.DataFrame()
    from io import StringIO

    return pd.read_csv(StringIO(response.text))


def fetch_statcast(start: pd.Timestamp, end: pd.Timestamp, *, cache: Path, refresh: bool, chunk_days: int, timeout: int, sleep: float) -> pd.DataFrame:
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    frames = []
    chunk_dir = cache.with_suffix("")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
        chunk_path = chunk_dir / f"statcast_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.csv"
        print(f"Fetching Statcast {chunk_start.date()} through {chunk_end.date()}...", flush=True)
        if chunk_path.exists() and not refresh:
            frame = pd.read_csv(chunk_path)
        else:
            frame = _fetch_statcast_chunk(chunk_start, chunk_end, timeout=timeout)
            frame.to_csv(chunk_path, index=False)
        if not frame.empty:
            frames.append(frame)
        time.sleep(sleep)

    if not frames:
        raise ValueError("No Statcast rows returned for requested date range.")
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def _pitch_category(pitch_type: Any) -> str:
    text = str(pitch_type or "").upper()
    if text in FASTBALL_TYPES:
        return "fastball"
    if text in BREAKING_TYPES:
        return "breaking"
    if text in OFFSPEED_TYPES:
        return "offspeed"
    return "other"


def _prepare_statcast(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce")
    frame["pitcher"] = pd.to_numeric(frame["pitcher"], errors="coerce")
    frame["launch_speed"] = pd.to_numeric(frame.get("launch_speed"), errors="coerce")
    frame["launch_angle"] = pd.to_numeric(frame.get("launch_angle"), errors="coerce")
    frame = frame.dropna(subset=["game_date", "batter", "pitcher"]).copy()
    frame["batter"] = frame["batter"].astype(int)
    frame["pitcher"] = frame["pitcher"].astype(int)
    frame["pitch_category"] = frame["pitch_type"].map(_pitch_category)
    frame["pitch_count"] = 1.0
    frame["is_bbe"] = (frame["launch_speed"].notna() | frame["launch_angle"].notna()).astype(float)
    frame["ev_sum"] = frame["launch_speed"].fillna(0.0)
    frame["la_sum"] = frame["launch_angle"].fillna(0.0)
    frame["hard_hit"] = (frame["launch_speed"] >= 95.0).astype(float)
    frame["barrel_proxy"] = (
        (frame["launch_speed"] >= 98.0)
        & (frame["launch_angle"].between(26.0, 30.0, inclusive="both"))
    ).astype(float)
    frame["sweet_spot"] = frame["launch_angle"].between(8.0, 32.0, inclusive="both").astype(float)
    frame["fb_ld"] = frame["bb_type"].isin(["fly_ball", "line_drive"]).astype(float)
    frame["home_run_event"] = (frame["events"] == "home_run").astype(float)
    for category in ("fastball", "breaking", "offspeed"):
        frame[f"{category}_pitch"] = (frame["pitch_category"] == category).astype(float)
    return frame


def _daily_aggregates(frame: pd.DataFrame, player_col: str, prefix: str) -> pd.DataFrame:
    daily = (
        frame.groupby([player_col, "game_date"], as_index=False)
        .agg(
            pitch_count=("pitch_count", "sum"),
            bbe=("is_bbe", "sum"),
            ev_sum=("ev_sum", "sum"),
            la_sum=("la_sum", "sum"),
            hard_hit=("hard_hit", "sum"),
            barrel_proxy=("barrel_proxy", "sum"),
            sweet_spot=("sweet_spot", "sum"),
            fb_ld=("fb_ld", "sum"),
            home_run_event=("home_run_event", "sum"),
            fastball_pitch=("fastball_pitch", "sum"),
            breaking_pitch=("breaking_pitch", "sum"),
            offspeed_pitch=("offspeed_pitch", "sum"),
        )
        .sort_values([player_col, "game_date"])
    )
    cumulative_cols = [
        "pitch_count",
        "bbe",
        "ev_sum",
        "la_sum",
        "hard_hit",
        "barrel_proxy",
        "sweet_spot",
        "fb_ld",
        "home_run_event",
        "fastball_pitch",
        "breaking_pitch",
        "offspeed_pitch",
    ]
    grouped = daily.groupby(player_col, sort=False)
    for col in cumulative_cols:
        daily[f"{prefix}_{col}_cum"] = grouped[col].cumsum()
    return daily.rename(columns={player_col: "statcast_player_id", "game_date": "statcast_date"})[
        ["statcast_player_id", "statcast_date", *[f"{prefix}_{col}_cum" for col in cumulative_cols]]
    ]


def _merge_prior(rows: pd.DataFrame, prior: pd.DataFrame, *, row_id_col: str) -> pd.DataFrame:
    left = rows.sort_values("game_date").copy()
    right = prior.sort_values("statcast_date").copy()
    merged_parts = []
    for player_id, group in left.groupby(row_id_col, sort=False):
        player_prior = right[right["statcast_player_id"] == int(player_id)].copy()
        if player_prior.empty:
            merged_parts.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values("game_date"),
            player_prior.sort_values("statcast_date"),
            left_on="game_date",
            right_on="statcast_date",
            direction="backward",
            allow_exact_matches=False,
        ).drop(columns=["statcast_player_id", "statcast_date"], errors="ignore")
        merged_parts.append(merged)
    return pd.concat(merged_parts, ignore_index=True)


def _add_rate_features(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = rows.copy()
    pitch = out[f"{prefix}_pitch_count_cum"].replace(0, np.nan)
    bbe = out[f"{prefix}_bbe_cum"].replace(0, np.nan)
    out[f"{prefix}_statcast_pitches_lag"] = out[f"{prefix}_pitch_count_cum"]
    out[f"{prefix}_statcast_bbe_lag"] = out[f"{prefix}_bbe_cum"]
    out[f"{prefix}_avg_ev_lag"] = out[f"{prefix}_ev_sum_cum"] / bbe
    out[f"{prefix}_avg_la_lag"] = out[f"{prefix}_la_sum_cum"] / bbe
    out[f"{prefix}_hard_hit_rate_lag"] = out[f"{prefix}_hard_hit_cum"] / bbe
    out[f"{prefix}_barrel_proxy_rate_lag"] = out[f"{prefix}_barrel_proxy_cum"] / bbe
    out[f"{prefix}_sweet_spot_rate_lag"] = out[f"{prefix}_sweet_spot_cum"] / bbe
    out[f"{prefix}_fb_ld_rate_lag"] = out[f"{prefix}_fb_ld_cum"] / bbe
    out[f"{prefix}_statcast_hr_per_pitch_lag"] = out[f"{prefix}_home_run_event_cum"] / pitch
    out[f"{prefix}_fastball_share_lag"] = out[f"{prefix}_fastball_pitch_cum"] / pitch
    out[f"{prefix}_breaking_share_lag"] = out[f"{prefix}_breaking_pitch_cum"] / pitch
    out[f"{prefix}_offspeed_share_lag"] = out[f"{prefix}_offspeed_pitch_cum"] / pitch
    return out


def enrich_training_rows(rows: pd.DataFrame, statcast: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["game_date"] = pd.to_datetime(rows["game_date"], errors="coerce")
    prepared = _prepare_statcast(statcast)
    batter_prior = _daily_aggregates(prepared, "batter", "batter")
    pitcher_prior = _daily_aggregates(prepared, "pitcher", "pitcher")
    out = _merge_prior(rows, batter_prior, row_id_col="player_id")
    out = _merge_prior(out, pitcher_prior, row_id_col="opposing_starter_id")
    out = _add_rate_features(out, "batter")
    out = _add_rate_features(out, "pitcher")
    out["statcast_feature_ready"] = (
        (out["batter_statcast_bbe_lag"].fillna(0) >= 10)
        & (out["pitcher_statcast_bbe_lag"].fillna(0) >= 10)
    ).astype(float)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich MLB HR training rows with leakage-safe Statcast features.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--start-date", type=_date_arg, default=None)
    parser.add_argument("--end-date", type=_date_arg, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.input)
    rows["game_date"] = pd.to_datetime(rows["game_date"], errors="coerce")
    start = args.start_date or (rows["game_date"].min().normalize() - pd.Timedelta(days=35))
    end = args.end_date or rows["game_date"].max().normalize()
    statcast = fetch_statcast(
        start,
        end,
        cache=args.cache,
        refresh=args.refresh_cache,
        chunk_days=args.chunk_days,
        timeout=args.timeout,
        sleep=args.sleep,
    )
    enriched = enrich_training_rows(rows, statcast)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    audit = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "statcast_cache": str(args.cache),
        "rows": int(len(enriched)),
        "statcast_rows": int(len(statcast)),
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "feature_ready_rate": float(enriched["statcast_feature_ready"].mean()),
        "batter_bbe_missing_rate": float(enriched["batter_statcast_bbe_lag"].isna().mean()),
        "pitcher_bbe_missing_rate": float(enriched["pitcher_statcast_bbe_lag"].isna().mean()),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(enriched)} rows to {args.output}")
    print(f"Wrote audit to {args.audit_output}")


if __name__ == "__main__":
    main()
