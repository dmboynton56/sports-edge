"""Live Statcast and handedness feature builders for MLB HR torch scoring."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

FASTBALL_TYPES = {"FF", "FA", "FT", "SI", "FC"}
BREAKING_TYPES = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "SC", "KN", "EP"}

DEFAULT_STATCAST_CACHE = Path(__file__).resolve().parents[2] / "notebooks" / "cache" / "mlb_statcast_2026.csv"
DEFAULT_HANDEDNESS_CACHE = Path(__file__).resolve().parents[2] / "notebooks" / "cache" / "mlb_player_handedness_cache.json"


def _normalize_hand(value: Any) -> str:
    text = str(value or "U").upper().strip()
    if text in {"L", "R", "S"}:
        return text
    return "U"


def _meta_field(row: Any, field: str) -> Any:
    return row.get(field) if isinstance(row, dict) else None


def _batter_platoon_advantage(batter_hand: str, pitcher_hand: str) -> float:
    if batter_hand == "U" or pitcher_hand == "U":
        return 0.0
    if batter_hand == "S":
        return 1.0
    return 1.0 if batter_hand != pitcher_hand else 0.0


def _read_handedness_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_handedness_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _fetch_people(ids: list[int], *, timeout: int = 30) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    response = requests.get(
        PEOPLE_URL,
        params={"personIds": ",".join(str(value) for value in ids)},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    out: dict[str, dict[str, Any]] = {}
    for person in payload.get("people", []):
        player_id = str(person.get("id"))
        out[player_id] = {
            "player_id": int(person.get("id")),
            "full_name": person.get("fullName"),
            "bat_side": (person.get("batSide") or {}).get("code"),
            "pitch_hand": (person.get("pitchHand") or {}).get("code"),
            "primary_position": (person.get("primaryPosition") or {}).get("abbreviation"),
            "active": person.get("active"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    return out


def ensure_handedness_cache(
    player_ids: list[int],
    *,
    cache_path: Path = DEFAULT_HANDEDNESS_CACHE,
    batch_size: int = 100,
    timeout: int = 30,
    sleep: float = 0.05,
) -> dict[str, dict[str, Any]]:
    cache = _read_handedness_cache(cache_path)
    missing = [player_id for player_id in sorted(set(player_ids)) if str(player_id) not in cache]
    for batch in _chunks(missing, batch_size):
        cache.update(_fetch_people(batch, timeout=timeout))
        _write_handedness_cache(cache_path, cache)
        time.sleep(sleep)
    return cache


def enrich_handedness(rows: pd.DataFrame, people: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = rows.copy()
    batter_meta = out["player_id"].astype("Int64").astype(str).map(people)
    pitcher_meta = out["opposing_starter_id"].astype("Int64").astype(str).map(people)

    out["batter_bat_side"] = batter_meta.map(lambda row: _normalize_hand(_meta_field(row, "bat_side")))
    out["pitcher_throw_hand"] = pitcher_meta.map(lambda row: _normalize_hand(_meta_field(row, "pitch_hand")))
    out["batter_bats_left"] = out["batter_bat_side"].isin(["L", "S"]).astype(float)
    out["batter_bats_right"] = out["batter_bat_side"].isin(["R", "S"]).astype(float)
    out["batter_switch_hitter"] = (out["batter_bat_side"] == "S").astype(float)
    out["pitcher_throws_left"] = (out["pitcher_throw_hand"] == "L").astype(float)
    out["pitcher_throws_right"] = (out["pitcher_throw_hand"] == "R").astype(float)
    out["known_handedness_matchup"] = (
        (out["batter_bat_side"] != "U") & (out["pitcher_throw_hand"] != "U")
    ).astype(float)
    out["batter_platoon_advantage"] = [
        _batter_platoon_advantage(batter, pitcher)
        for batter, pitcher in zip(out["batter_bat_side"], out["pitcher_throw_hand"], strict=False)
    ]
    out["same_side_matchup"] = (
        (out["known_handedness_matchup"] == 1.0)
        & (out["batter_bat_side"] != "S")
        & (out["batter_bat_side"] == out["pitcher_throw_hand"])
    ).astype(float)
    return out


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
    return pd.read_csv(StringIO(response.text))


def _statcast_cache_covers_window(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    preseason_grace_days: int = 35,
) -> bool:
    if frame.empty or "game_date" not in frame.columns:
        return False
    dates = pd.to_datetime(frame["game_date"], errors="coerce").dropna()
    if dates.empty:
        return False
    latest_ok = dates.max().normalize() >= end.normalize()
    earliest_allowed = start.normalize() + pd.Timedelta(days=preseason_grace_days)
    earliest_ok = dates.min().normalize() <= earliest_allowed
    return bool(latest_ok and earliest_ok)


def fetch_statcast(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    cache: Path = DEFAULT_STATCAST_CACHE,
    refresh: bool = False,
    chunk_days: int = 7,
    timeout: int = 60,
    sleep: float = 0.25,
) -> pd.DataFrame:
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache)
        if _statcast_cache_covers_window(cached, start, end):
            return cached

    frames = []
    chunk_dir = cache.with_suffix("")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
        chunk_path = chunk_dir / f"statcast_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.csv"
        if chunk_path.exists() and not refresh:
            frame = pd.read_csv(chunk_path)
        else:
            frame = _fetch_statcast_chunk(chunk_start, chunk_end, timeout=timeout)
            frame.to_csv(chunk_path, index=False)
        if not frame.empty:
            frames.append(frame)
        time.sleep(sleep)

    if not frames:
        if cache.exists():
            return pd.read_csv(cache)
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
    missing_rows = left[left[row_id_col].isna()].copy()
    if not missing_rows.empty:
        merged_parts.append(missing_rows)
    known_rows = left[left[row_id_col].notna()].copy()
    for player_id, group in known_rows.groupby(row_id_col, sort=False):
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
    if not merged_parts:
        return left
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


def enrich_statcast(rows: pd.DataFrame, statcast: pd.DataFrame) -> pd.DataFrame:
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


def build_torch_candidate_features(
    candidates: pd.DataFrame,
    as_of: date,
    *,
    statcast_cache: Path = DEFAULT_STATCAST_CACHE,
    handedness_cache: Path = DEFAULT_HANDEDNESS_CACHE,
    refresh_statcast: bool = False,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    rows = candidates.copy()
    rows["game_date"] = pd.to_datetime(as_of)

    player_ids = pd.concat(
        [rows["player_id"], rows["opposing_starter_id"]],
        ignore_index=True,
    )
    ids = [int(value) for value in player_ids.dropna().unique()]
    people = ensure_handedness_cache(ids, cache_path=handedness_cache)
    rows = enrich_handedness(rows, people)

    start = pd.Timestamp(as_of) - pd.Timedelta(days=120)
    end = pd.Timestamp(as_of) - pd.Timedelta(days=1)
    statcast = fetch_statcast(
        start,
        end,
        cache=statcast_cache,
        refresh=refresh_statcast,
    )
    return enrich_statcast(rows, statcast)
