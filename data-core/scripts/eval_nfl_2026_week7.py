#!/usr/bin/env python3
"""Reconstruct and grade nfl-v2-live-20260906 for 2026 week 7 and season-to-date.

Uses the frozen live artifact plus nflverse schedules/PBP. This is the local
stand-in for the Tuesday 2026-10-13 published snapshot: v2 does not consume
injuries, and the feature store is leakage-safe (pregame row before the game
updates team state). Week-7 Sunday features therefore match the Tuesday freeze
except for SEA/DEN after TNF.

Outputs JSON under notebooks/cache/. Not a serving path.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.nfl_live import LIVE_MODEL_VERSION, load_live_artifact
from src.models.nfl_v2 import (
    FEATURE_COLUMNS,
    build_nfl_feature_store,
    calibration_table,
    expected_calibration_error,
    margin_metrics,
    no_vig_home_probability,
    probability_metrics,
)

CACHE_DIR = ROOT / "notebooks" / "cache"
OUTPUT_JSON = CACHE_DIR / "nfl_2026_week7_eval.json"
TEAM_GAMES_CACHE = CACHE_DIR / "nfl_v2_team_games_2020_2026.parquet"
WEEK7_START = pd.Timestamp("2026-10-15")
WEEK7_SUNDAY = pd.Timestamp("2026-10-18")
PUBLIC_SITE_SAMPLE = 92
PUBLIC_SITE_WINNER = (51, 41)
PUBLIC_SITE_SPREAD = (39, 53, 0)

# Public /models/results grades for 2026-10-18 (away-home scores, model-line ATS).
PUBLIC_SUNDAY_GRADES = {
    ("TEN", "IND"): {"away_score": 26, "home_score": 20, "spread": "loss", "winner": "loss"},
    ("PIT", "TB"): {"away_score": 26, "home_score": 19, "spread": "loss", "winner": "win"},
    ("CAR", "PHI"): {"away_score": 25, "home_score": 14, "spread": "loss", "winner": "win"},
    ("CHI", "ATL"): {"away_score": 31, "home_score": 23, "spread": "loss", "winner": "win"},
    ("NO", "NYG"): {"away_score": 34, "home_score": 27, "spread": "loss", "winner": "win"},
    ("BUF", "LV"): {"away_score": 33, "home_score": 30, "spread": "win", "winner": "win"},
    ("BAL", "CLE"): {"away_score": 10, "home_score": 13, "spread": "win", "winner": "loss"},
    ("ARI", "LAR"): {"away_score": 18, "home_score": 16, "spread": "loss", "winner": "loss"},
    ("NYJ", "NE"): {"away_score": 19, "home_score": 26, "spread": "loss", "winner": "win"},
    ("DAL", "GB"): {"away_score": 13, "home_score": 23, "spread": "win", "winner": "loss"},
    ("HOU", "JAX"): {"away_score": 34, "home_score": 31, "spread": "loss", "winner": "loss"},
    ("LAC", "KC"): {"away_score": 6, "home_score": 22, "spread": "win", "winner": "win"},
}
LIVE_SITE_MNF = {"away": "WAS", "home": "SF", "my_spread": -14.0, "home_win_prob": 0.782}


def _to_pandas(value) -> pd.DataFrame:
    if hasattr(value, "collect"):
        value = value.collect()
    return value.to_pandas() if hasattr(value, "to_pandas") else pd.DataFrame(value)


def american_to_implied(odds: object) -> float | None:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value == 0:
        return None
    return 100.0 / (value + 100.0) if value > 0 else (-value) / ((-value) + 100.0)


def american_profit(odds: float, won: bool) -> float:
    if not won:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def model_line_spread_result(actual_margin: float, my_spread: float) -> str:
    edge = actual_margin + my_spread
    if abs(edge) < 0.001:
        return "push"
    return "win" if edge > 0 else "loss"


def book_ats_result(actual_margin: float, my_spread: float, book_spread: float) -> str | None:
    if pd.isna(book_spread) or pd.isna(my_spread):
        return None
    if abs(float(my_spread) - float(book_spread)) < 0.001:
        return "no_bet"
    book_edge = actual_margin + float(book_spread)
    if abs(book_edge) < 0.001:
        return "push"
    home_covers = book_edge > 0
    model_wants_home = float(my_spread) < float(book_spread)
    picked_cover = home_covers if model_wants_home else not home_covers
    return "win" if picked_cover else "loss"


def aggregate_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp.copy()
    plays["play_type"] = plays["play_type"].astype(str)
    plays = plays[plays["play_type"].isin(["pass", "run"])]
    plays = plays[plays["posteam"].notna() & plays["defteam"].notna() & plays["epa"].notna()]
    if "success" in plays.columns:
        success = pd.to_numeric(plays["success"], errors="coerce")
    else:
        success = pd.Series(np.nan, index=plays.index)
    plays["success_flag"] = success.where(success.notna(), (plays["epa"] > 0).astype(float))
    yards = pd.to_numeric(plays.get("yards_gained"), errors="coerce").fillna(0.0)
    plays["explosive"] = (
        ((plays["play_type"] == "pass") & (yards >= 15)) | ((plays["play_type"] == "run") & (yards >= 10))
    ).astype(float)

    offense = (
        plays.groupby(["game_id", "posteam"], dropna=False)
        .agg(
            off_epa=("epa", "mean"),
            off_success=("success_flag", "mean"),
            off_explosive=("explosive", "mean"),
            offensive_plays=("epa", "size"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    defense = (
        plays.groupby(["game_id", "defteam"], dropna=False)
        .agg(
            def_epa_allowed=("epa", "mean"),
            def_success_allowed=("success_flag", "mean"),
            def_explosive_allowed=("explosive", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    qb_source = plays.copy()
    dropback = pd.to_numeric(qb_source.get("qb_dropback"), errors="coerce")
    passer_col = "passer_player_id" if "passer_player_id" in qb_source.columns else "passer_id"
    qb_source = qb_source[(dropback == 1) & qb_source[passer_col].notna()]
    if qb_source.empty:
        quarterbacks = pd.DataFrame(columns=["game_id", "team", "primary_qb_id"])
    else:
        counts = (
            qb_source.groupby(["game_id", "posteam", passer_col], dropna=False)
            .size()
            .reset_index(name="dropbacks")
            .sort_values(["game_id", "posteam", "dropbacks", passer_col], ascending=[True, True, False, True])
        )
        quarterbacks = (
            counts.drop_duplicates(["game_id", "posteam"])
            .rename(columns={"posteam": "team", passer_col: "primary_qb_id"})
            [["game_id", "team", "primary_qb_id"]]
        )

    merged = offense.merge(defense, on=["game_id", "team"], how="inner").merge(
        quarterbacks, on=["game_id", "team"], how="left"
    )
    return merged


def load_team_games(seasons: list[int]) -> pd.DataFrame:
    if TEAM_GAMES_CACHE.exists():
        cached = pd.read_parquet(TEAM_GAMES_CACHE)
        cached_seasons = set(cached["season"].dropna().astype(int).unique()) if "season" in cached.columns else set()
        if cached_seasons.issuperset(seasons) and len(cached) > 1000:
            return cached
    import nflreadpy as nfl

    frames = []
    for season in seasons:
        print(f"Loading PBP {season}...", flush=True)
        pbp = _to_pandas(nfl.load_pbp(seasons=season))
        if pbp.empty:
            continue
        team_games = aggregate_team_games(pbp)
        team_games["season"] = season
        frames.append(team_games)
        del pbp
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TEAM_GAMES_CACHE, index=False)
    return out


def espn_scoreboard(dates: str) -> list[dict[str, Any]]:
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    response = requests.get(url, params={"dates": dates, "limit": 100}, timeout=30)
    response.raise_for_status()
    events = []
    for event in response.json().get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in competition.get("competitors", [])}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        odds = (competition.get("odds") or [{}])[0]
        events.append(
            {
                "name": event.get("name"),
                "date": event.get("date"),
                "status": (event.get("status") or {}).get("type", {}).get("description"),
                "home_abbr": (home.get("team") or {}).get("abbreviation"),
                "away_abbr": (away.get("team") or {}).get("abbreviation"),
                "home_score": int(home["score"]) if home.get("score") not in (None, "") else None,
                "away_score": int(away["score"]) if away.get("score") not in (None, "") else None,
                "detail": (event.get("status") or {}).get("type", {}).get("detail"),
                "espn_spread": odds.get("spread"),
                "espn_home_spread": odds.get("homeTeamOdds", {}).get("spreadOdds") if isinstance(odds.get("homeTeamOdds"), dict) else None,
                "espn_details": odds.get("details"),
                "espn_over_under": odds.get("overUnder"),
            }
        )
    return events


def score_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["actual_margin"] = out["home_score"] - out["away_score"]
    out["predicted_margin"] = -out["my_spread"]
    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)
    out["model_winner"] = np.where(out["my_home_win_prob"] >= 0.5, out["home_team"], out["away_team"])
    out["actual_winner"] = np.where(out["home_win"] == 1, out["home_team"], out["away_team"])
    out["winner_result"] = np.where(out["model_winner"] == out["actual_winner"], "win", "loss")
    out["model_line_ats"] = [
        model_line_spread_result(margin, spread)
        for margin, spread in zip(out["actual_margin"], out["my_spread"])
    ]
    out["book_ats"] = [
        book_ats_result(margin, my_spread, book)
        for margin, my_spread, book in zip(out["actual_margin"], out["my_spread"], out["closing_spread"])
    ]
    out["margin_error"] = out["predicted_margin"] - out["actual_margin"]
    out["abs_margin_error"] = out["margin_error"].abs()
    out["close_game"] = out["actual_margin"].abs() <= 3
    out["market_home_prob"] = [
        no_vig_home_probability(home, away)
        for home, away in zip(out["home_moneyline"], out["away_moneyline"])
    ]
    out["market_favorite"] = np.where(out["market_home_prob"].fillna(0.5) >= 0.5, out["home_team"], out["away_team"])
    ml_units = []
    for row in out.itertuples(index=False):
        if pd.isna(row.home_moneyline) or pd.isna(row.away_moneyline):
            ml_units.append(np.nan)
            continue
        if row.my_home_win_prob >= 0.5:
            ml_units.append(american_profit(float(row.home_moneyline), bool(row.home_win)))
        else:
            ml_units.append(american_profit(float(row.away_moneyline), not bool(row.home_win)))
    out["ml_units"] = ml_units
    disagree_units = []
    for row in out.itertuples(index=False):
        if pd.isna(row.market_home_prob) or pd.isna(row.home_moneyline) or pd.isna(row.away_moneyline):
            disagree_units.append(np.nan)
            continue
        model_home = row.my_home_win_prob >= 0.5
        market_home = row.market_home_prob >= 0.5
        if model_home == market_home:
            disagree_units.append(np.nan)
            continue
        if model_home:
            disagree_units.append(american_profit(float(row.home_moneyline), bool(row.home_win)))
        else:
            disagree_units.append(american_profit(float(row.away_moneyline), not bool(row.home_win)))
    out["ml_disagree_units"] = disagree_units
    return out


def metrics_block(frame: pd.DataFrame) -> dict[str, Any]:
    graded = frame.dropna(subset=["home_score", "away_score", "my_spread", "my_home_win_prob"]).copy()
    if graded.empty:
        return {"rows": 0}
    probability = probability_metrics(graded["home_win"], graded["my_home_win_prob"])
    margin = margin_metrics(graded["actual_margin"], graded["predicted_margin"])
    book = graded.dropna(subset=["closing_spread"])
    book_margin = margin_metrics(book["actual_margin"], -book["closing_spread"]) if len(book) else {}
    market = graded.dropna(subset=["market_home_prob"])
    market_probability = (
        probability_metrics(market["home_win"], market["market_home_prob"]) if len(market) else {}
    )
    model_ats = graded["model_line_ats"].value_counts().to_dict()
    book_ats = graded["book_ats"].value_counts(dropna=False).to_dict()
    book_bets = graded[graded["book_ats"].isin(["win", "loss", "push"])]
    ats_wins = int((book_bets["book_ats"] == "win").sum())
    ats_losses = int((book_bets["book_ats"] == "loss").sum())
    ats_pushes = int((book_bets["book_ats"] == "push").sum())
    ats_units = ats_wins * (100.0 / 110.0) + ats_losses * -1.0
    ml = graded.dropna(subset=["ml_units"])
    disagree = graded.dropna(subset=["ml_disagree_units"])
    winner_wins = int((graded["winner_result"] == "win").sum())
    winner_losses = int((graded["winner_result"] == "loss").sum())
    return {
        "rows": int(len(graded)),
        "weeks": sorted(pd.to_numeric(graded["week"], errors="coerce").dropna().astype(int).unique().tolist()),
        "probability": probability,
        "margin": margin,
        "book_margin": book_margin,
        "market_probability": market_probability,
        "model_line_ats": {
            "wins": int(model_ats.get("win", 0)),
            "losses": int(model_ats.get("loss", 0)),
            "pushes": int(model_ats.get("push", 0)),
            "hit_rate": (model_ats.get("win", 0) / max(1, model_ats.get("win", 0) + model_ats.get("loss", 0))),
        },
        "book_ats": {
            "wins": ats_wins,
            "losses": ats_losses,
            "pushes": ats_pushes,
            "no_bet": int((graded["book_ats"] == "no_bet").sum()),
            "hit_rate": ats_wins / max(1, ats_wins + ats_losses),
            "flat_110_units": ats_units,
            "roi": ats_units / max(1, ats_wins + ats_losses),
        },
        "winner": {
            "wins": winner_wins,
            "losses": winner_losses,
            "hit_rate": winner_wins / max(1, winner_wins + winner_losses),
        },
        "moneyline": {
            "bets": int(len(ml)),
            "units": float(ml["ml_units"].sum()) if len(ml) else None,
            "roi": float(ml["ml_units"].mean()) if len(ml) else None,
        },
        "moneyline_disagreements": {
            "bets": int(len(disagree)),
            "units": float(disagree["ml_disagree_units"].sum()) if len(disagree) else None,
            "roi": float(disagree["ml_disagree_units"].mean()) if len(disagree) else None,
        },
        "close_games": int(graded["close_game"].sum()),
        "close_game_winner_hit_rate": float(
            (graded.loc[graded["close_game"], "winner_result"] == "win").mean()
        )
        if graded["close_game"].any()
        else None,
    }


def game_record(row: pd.Series) -> dict[str, Any]:
    return {
        "game_id": str(row["game_id"]),
        "season": int(row["season"]),
        "week": int(row["week"]) if pd.notna(row["week"]) else None,
        "game_date": pd.Timestamp(row["game_date"]).strftime("%Y-%m-%d"),
        "away_team": row["away_team"],
        "home_team": row["home_team"],
        "away_score": int(row["away_score"]) if pd.notna(row["away_score"]) else None,
        "home_score": int(row["home_score"]) if pd.notna(row["home_score"]) else None,
        "actual_margin": float(row["actual_margin"]) if pd.notna(row.get("actual_margin")) else None,
        "my_spread": float(row["my_spread"]) if pd.notna(row["my_spread"]) else None,
        "predicted_margin": float(row["predicted_margin"]) if pd.notna(row.get("predicted_margin")) else None,
        "my_home_win_prob": float(row["my_home_win_prob"]) if pd.notna(row["my_home_win_prob"]) else None,
        "closing_spread": float(row["closing_spread"]) if pd.notna(row.get("closing_spread")) else None,
        "home_moneyline": float(row["home_moneyline"]) if pd.notna(row.get("home_moneyline")) else None,
        "away_moneyline": float(row["away_moneyline"]) if pd.notna(row.get("away_moneyline")) else None,
        "market_home_prob": float(row["market_home_prob"]) if pd.notna(row.get("market_home_prob")) else None,
        "margin_error": float(row["margin_error"]) if pd.notna(row.get("margin_error")) else None,
        "abs_margin_error": float(row["abs_margin_error"]) if pd.notna(row.get("abs_margin_error")) else None,
        "model_winner": row.get("model_winner"),
        "actual_winner": row.get("actual_winner"),
        "winner_result": row.get("winner_result"),
        "model_line_ats": row.get("model_line_ats"),
        "book_ats": row.get("book_ats"),
        "ml_units": float(row["ml_units"]) if pd.notna(row.get("ml_units")) else None,
        "home_qb_name": row.get("home_qb_name"),
        "away_qb_name": row.get("away_qb_name"),
        "rest_diff": float(row["rest_diff"]) if pd.notna(row.get("rest_diff")) else None,
        "home_field": float(row["home_field"]) if pd.notna(row.get("home_field")) else None,
        "qb_stability_diff": float(row["qb_stability_diff"]) if pd.notna(row.get("qb_stability_diff")) else None,
        "qb_prior_epa_diff": float(row["qb_prior_epa_diff"]) if pd.notna(row.get("qb_prior_epa_diff")) else None,
        "games_before_diff": float(row["games_before_diff"]) if pd.notna(row.get("games_before_diff")) else None,
        "adj_off_epa_diff_5": float(row["adj_off_epa_diff_5"]) if pd.notna(row.get("adj_off_epa_diff_5")) else None,
        "win_diff_5": float(row["win_diff_5"]) if pd.notna(row.get("win_diff_5")) else None,
        "margin_diff_5": float(row["margin_diff_5"]) if pd.notna(row.get("margin_diff_5")) else None,
        "close_game": bool(row.get("close_game")) if pd.notna(row.get("close_game")) else None,
    }


def validate_against_public(week7: pd.DataFrame) -> dict[str, Any]:
    checks = []
    for (away, home), expected in PUBLIC_SUNDAY_GRADES.items():
        match = week7[(week7["away_team"] == away) & (week7["home_team"] == home)]
        if match.empty:
            # LAR vs LA alias
            alt_home = "LA" if home == "LAR" else home
            match = week7[(week7["away_team"] == away) & (week7["home_team"] == alt_home)]
        if match.empty:
            checks.append({"game": f"{away}@{home}", "ok": False, "reason": "missing"})
            continue
        row = match.iloc[0]
        score_ok = int(row["away_score"]) == expected["away_score"] and int(row["home_score"]) == expected["home_score"]
        spread_ok = row["model_line_ats"] == expected["spread"]
        winner_ok = row["winner_result"] == expected["winner"]
        checks.append(
            {
                "game": f"{away}@{home}",
                "ok": bool(score_ok and spread_ok and winner_ok),
                "score_ok": bool(score_ok),
                "spread_ok": bool(spread_ok),
                "winner_ok": bool(winner_ok),
                "reconstructed_spread": float(row["my_spread"]),
                "reconstructed_prob": float(row["my_home_win_prob"]),
            }
        )
    mnf = week7[(week7["away_team"] == "WAS") & (week7["home_team"] == "SF")]
    mnf_check = None
    if not mnf.empty:
        row = mnf.iloc[0]
        mnf_check = {
            "reconstructed_spread": float(row["my_spread"]),
            "reconstructed_prob": float(row["my_home_win_prob"]),
            "site_spread": LIVE_SITE_MNF["my_spread"],
            "site_prob": LIVE_SITE_MNF["home_win_prob"],
            "spread_match": abs(float(row["my_spread"]) - LIVE_SITE_MNF["my_spread"]) < 0.15,
            "prob_match": abs(float(row["my_home_win_prob"]) - LIVE_SITE_MNF["home_win_prob"]) < 0.005,
        }
    return {
        "sunday_grade_matches": checks,
        "sunday_all_match": all(item["ok"] for item in checks),
        "mnf_live_site": mnf_check,
    }


def main() -> None:
    import nflreadpy as nfl

    seasons = list(range(2020, 2027))
    print("Loading schedules...", flush=True)
    schedules = _to_pandas(nfl.load_schedules(seasons))
    schedules["league"] = "NFL"
    team_games = load_team_games(seasons)
    print(f"Schedules={len(schedules)} team-games={len(team_games)}", flush=True)

    artifact = load_live_artifact()
    print("Building feature store...", flush=True)
    features = build_nfl_feature_store(schedules, team_games, include_unplayed=True)
    live_2026 = features[(features["season"] == 2026) & (features["game_type"].astype(str).str.upper() == "REG")].copy()
    needed = live_2026[FEATURE_COLUMNS]
    if needed.isna().any().any():
        missing = needed.isna().sum()
        raise SystemExit(f"Missing live features: {missing[missing > 0].to_dict()}")

    probability = np.asarray(artifact["probability_model"].predict_probability(live_2026), dtype=float)
    margin = np.asarray(artifact["margin_model"].predict_margin(live_2026), dtype=float)
    live_2026["my_home_win_prob"] = probability
    live_2026["predicted_margin"] = margin
    live_2026["my_spread"] = -margin

    schedule_meta = schedules[
        [
            col
            for col in (
                "game_id",
                "home_qb_name",
                "away_qb_name",
                "home_qb_id",
                "away_qb_id",
                "spread_line",
                "total_line",
                "home_moneyline",
                "away_moneyline",
                "gametime",
                "weekday",
                "roof",
                "surface",
            )
            if col in schedules.columns
        ]
    ].copy()
    live_2026 = live_2026.merge(schedule_meta, on="game_id", how="left", suffixes=("", "_sched"))
    # nflverse `spread_line` is the away-team spread (negative = away favorite).
    # Sports Edge / this model use home-team spread (negative = home favorite).
    away_spread = pd.to_numeric(live_2026.get("spread_line"), errors="coerce")
    live_2026["closing_spread"] = -away_spread
    for column in ("home_moneyline", "away_moneyline", "closing_spread"):
        if column in live_2026.columns:
            live_2026[column] = pd.to_numeric(live_2026[column], errors="coerce")

    completed = live_2026[live_2026["home_score"].notna() & live_2026["away_score"].notna()].copy()
    completed = score_rows(completed)

    week7_mask = (
        (completed["week"] == 7)
        & (pd.to_datetime(completed["game_date"]) >= WEEK7_START)
        & (pd.to_datetime(completed["game_date"]) <= WEEK7_SUNDAY)
    )
    # Include TNF+Sunday only; MNF WAS@SF is still upcoming at report time.
    week7 = completed[week7_mask].copy()
    if week7.empty:
        week7 = completed[
            (pd.to_datetime(completed["game_date"]) >= WEEK7_START)
            & (pd.to_datetime(completed["game_date"]) <= WEEK7_SUNDAY)
        ].copy()

    upcoming = live_2026[live_2026["home_score"].isna() | live_2026["away_score"].isna()].copy()
    upcoming_week7 = upcoming[
        (pd.to_datetime(upcoming["game_date"]) >= WEEK7_START)
        & (pd.to_datetime(upcoming["game_date"]) <= pd.Timestamp("2026-10-19"))
    ].copy()

    print("Fetching ESPN scoreboards...", flush=True)
    espn = {}
    for dates in ("20261015", "20261018", "20261019"):
        try:
            espn[dates] = espn_scoreboard(dates)
        except Exception as exc:  # noqa: BLE001
            espn[dates] = {"error": str(exc)}

    public_validation = validate_against_public(
        pd.concat([week7, upcoming_week7], ignore_index=True) if not upcoming_week7.empty else week7
    )

    misses = week7[week7["winner_result"] == "loss"].copy()
    miss_records = [game_record(row) for _, row in misses.sort_values("abs_margin_error", ascending=False).iterrows()]

    s2d_through_week5 = completed[pd.to_numeric(completed["week"], errors="coerce") <= 5]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": LIVE_MODEL_VERSION,
        "sklearn_version": __import__("sklearn").__version__,
        "sources": {
            "predictions": "reconstructed from models/nfl_v2_live.joblib + nflverse PBP/schedules",
            "book_lines": "nflverse schedule moneylines plus home spread = -spread_line (spread_line is the away-team closing/consensus line, not the Tuesday Odds API snapshot)",
            "finals": "nflverse schedules, cross-checked against ESPN scoreboard and the public results page",
            "public_site": "https://sports-edge.drewboynton.com/models/results",
        },
        "caveats": [
            "Supabase/BigQuery credentials are not in this environment; published numeric forecasts are reconstructed from the frozen live artifact.",
            "Reconstruction is validated against the public site's winner/model-line ATS grades and the live WAS@SF board.",
            "v2 excludes injuries; expected QB is the mode of the last three PBP dropback QBs, not the posted starter.",
            "Daily Refresh regenerated 14 NFL predictions on 2026-10-13 19:55 UTC and skipped regeneration Wed-Sun (run_nfl_predictions=false).",
        ],
        "public_site_snapshot": {
            "updated": "2026-10-19",
            "v2_sample": PUBLIC_SITE_SAMPLE,
            "winner_wl": PUBLIC_SITE_WINNER,
            "spread_wlp": PUBLIC_SITE_SPREAD,
            "spread_ats_is_model_line": True,
        },
        "validation": public_validation,
        "week7_games": [game_record(row) for _, row in week7.sort_values(["game_date", "away_team"]).iterrows()],
        "upcoming_week7": [
            {
                "game_id": str(row.game_id),
                "game_date": pd.Timestamp(row.game_date).strftime("%Y-%m-%d"),
                "away_team": row.away_team,
                "home_team": row.home_team,
                "my_spread": float(row.my_spread),
                "my_home_win_prob": float(row.my_home_win_prob),
                "closing_spread": float(row.closing_spread) if pd.notna(row.closing_spread) else None,
            }
            for row in upcoming_week7.itertuples(index=False)
        ],
        "week7_metrics": metrics_block(week7),
        "season_to_date_metrics": metrics_block(completed),
        "through_week5_metrics": metrics_block(s2d_through_week5),
        "moneyline_misses": miss_records,
        "espn": espn,
        "feature_means_week7": {
            column: float(week7[column].mean()) if column in week7 else None
            for column in ("rest_diff", "home_field", "qb_stability_diff", "qb_prior_epa_diff", "games_before_diff")
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({
        "output": str(OUTPUT_JSON),
        "week7_games": int(len(week7)),
        "s2d_games": int(len(completed)),
        "public_sunday_match": public_validation["sunday_all_match"],
        "mnf": public_validation.get("mnf_live_site"),
        "week7_winner": report["week7_metrics"].get("winner"),
        "week7_margin": report["week7_metrics"].get("margin"),
        "s2d_margin": report["season_to_date_metrics"].get("margin"),
    }, indent=2))


if __name__ == "__main__":
    main()
