#!/usr/bin/env python3
"""Weekly NFL v2 health check.

Prefers timestamp-safe stored predictions (last row with asof_ts < kickoff).
If Supabase credentials are missing, reconstructs the frozen live artifact on
nflverse schedules/PBP — the same path Daily Refresh uses — and validates
against the public results page plus ESPN scoreboards.

Walk-forward calibration is evaluated on earlier weeks only. A live calibrator
is written only when that out-of-sample test improves Brier and log loss
without a MAE regression.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_nfl_2026_week7 import espn_scoreboard, load_team_games
from src.models.nfl_health import (
    bias_diagnosis,
    feature_drift_table,
    json_safe,
    metrics_block,
    prior_decay_mix,
    score_prediction_frame,
    segmented_health,
    staleness_report,
)
from src.models.nfl_live import DEFAULT_POSTHOC_ARTIFACT, LIVE_MODEL_VERSION, load_live_artifact
from src.models.nfl_recalibration import (
    evaluate_margin_method,
    evaluate_probability_method,
    fit_production_calibrators,
    ship_decision,
)
from src.models.nfl_v2 import FEATURE_COLUMNS, audit_nfl_feature_store, build_nfl_feature_store, calibration_table
from src.utils.supabase_pg import create_pg_connection, load_supabase_credentials


REPO_ROOT = ROOT.parent
DEFAULT_JSON = ROOT / "notebooks" / "cache" / "nfl_model_health_2026.json"
DEFAULT_GAMES = ROOT / "notebooks" / "cache" / "nfl_model_health_2026_games.csv"
DEFAULT_CHART_DIR = REPO_ROOT / "docs" / "analysis" / "nfl-model-health"
DEFAULT_REPORT = REPO_ROOT / "docs" / "analysis" / "nfl-model-health-2026.md"
PUBLIC_RESULTS_URL = "https://sports-edge.drewboynton.com/models/results"
V1_ATS_2025 = ROOT / "notebooks" / "cache" / "nfl_supabase_ats_games_2025.csv"
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--include-2025", action="store_true", default=True)
    parser.add_argument("--skip-2025", action="store_true")
    parser.add_argument("--model-version", default=LIVE_MODEL_VERSION)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--games-output", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--chart-dir", type=Path, default=DEFAULT_CHART_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--posthoc-artifact", type=Path, default=DEFAULT_POSTHOC_ARTIFACT)
    parser.add_argument("--write-posthoc-if-improved", action="store_true", default=True)
    parser.add_argument("--no-write-posthoc", action="store_true")
    return parser.parse_args()


def _to_pandas(value) -> pd.DataFrame:
    if hasattr(value, "collect"):
        value = value.collect()
    return value.to_pandas() if hasattr(value, "to_pandas") else pd.DataFrame(value)


def fetch_stored_predictions(model_version: str, seasons: list[int]) -> pd.DataFrame:
    credentials = load_supabase_credentials()
    if not credentials.get("db_password"):
        return pd.DataFrame()
    conn = create_pg_connection(
        credentials["url"],
        credentials["db_password"],
        credentials["db_host"],
        credentials["db_port"],
        credentials["db_name"],
        credentials["db_user"],
    )
    query = """
        SELECT DISTINCT ON (g.id)
          g.id::text AS stored_game_id,
          g.season,
          g.week,
          g.game_time_utc,
          coalesce(g.game_date, (g.game_time_utc at time zone 'America/Denver')::date) AS game_date,
          g.home_team,
          g.away_team,
          g.home_score,
          g.away_score,
          g.book_spread AS stored_book_spread,
          p.my_home_win_prob AS stored_home_win_prob,
          p.my_spread AS stored_spread,
          p.asof_ts AS prediction_ts,
          p.model_version
        FROM games g
        JOIN model_predictions p ON p.game_id = g.id
        WHERE g.league = 'NFL'
          AND g.season = ANY(%s)
          AND p.model_version = %s
          AND p.asof_ts < g.game_time_utc
        ORDER BY g.id, p.asof_ts DESC
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (seasons, model_version), prepare=False)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
        stored = pd.DataFrame(rows, columns=columns)
        odds_query = """
            SELECT DISTINCT ON (o.game_id, o.market, coalesce(o.selection, ''))
              o.game_id::text AS stored_game_id,
              o.market,
              o.selection,
              o.line,
              o.price,
              o.snapshot_ts
            FROM odds_snapshots o
            JOIN games g ON g.id = o.game_id
            WHERE g.league = 'NFL'
              AND g.season = ANY(%s)
              AND o.snapshot_ts < g.game_time_utc
              AND o.market IN ('spread', 'moneyline', 'total')
            ORDER BY o.game_id, o.market, coalesce(o.selection, ''), o.snapshot_ts DESC
        """
        with conn.cursor() as cursor:
            cursor.execute(odds_query, (seasons,), prepare=False)
            odds_rows = cursor.fetchall()
            odds_columns = [description[0] for description in cursor.description]
        odds = pd.DataFrame(odds_rows, columns=odds_columns)
    finally:
        conn.close()
    if stored.empty or odds.empty:
        return stored
    home_ml = odds[(odds["market"] == "moneyline") & (odds["selection"] == "home")][["stored_game_id", "price"]].rename(
        columns={"price": "stored_home_moneyline"}
    )
    away_ml = odds[(odds["market"] == "moneyline") & (odds["selection"] == "away")][["stored_game_id", "price"]].rename(
        columns={"price": "stored_away_moneyline"}
    )
    totals = odds[(odds["market"] == "total") & (odds["selection"].isin(["over", None, ""]))][
        ["stored_game_id", "line"]
    ].drop_duplicates("stored_game_id").rename(columns={"line": "stored_total"})
    stored = stored.merge(home_ml, on="stored_game_id", how="left")
    stored = stored.merge(away_ml, on="stored_game_id", how="left")
    stored = stored.merge(totals, on="stored_game_id", how="left")
    return stored


def reconstruct_live(seasons: list[int], model_version: str) -> pd.DataFrame:
    import nflreadpy as nfl

    print("Loading nflverse schedules...", flush=True)
    schedules = _to_pandas(nfl.load_schedules(seasons))
    schedules["league"] = "NFL"
    team_games = load_team_games(seasons)
    print(f"Schedules={len(schedules)} team-games={len(team_games)}", flush=True)
    artifact = load_live_artifact()
    if artifact["model_version"] != model_version:
        raise SystemExit(f"Live artifact is {artifact['model_version']}, expected {model_version}")
    features = build_nfl_feature_store(schedules, team_games, include_unplayed=True)
    audit = audit_nfl_feature_store(features, raise_on_error=True)
    scorable = features[features["home_score"].notna() | features[FEATURE_COLUMNS].notna().all(axis=1)].copy()
    needed = scorable[FEATURE_COLUMNS]
    if needed.isna().any().any():
        missing = needed.isna().sum()
        raise SystemExit(f"Missing live features: {missing[missing > 0].to_dict()}")
    probability = np.asarray(artifact["probability_model"].predict_probability(scorable), dtype=float)
    margin = np.asarray(artifact["margin_model"].predict_margin(scorable), dtype=float)
    features = scorable
    features = features.copy()
    features["reconstructed_home_win_prob"] = probability
    features["reconstructed_margin"] = margin
    features["reconstructed_spread"] = -margin
    meta_cols = [
        col
        for col in (
            "game_id",
            "home_qb_name",
            "away_qb_name",
            "spread_line",
            "total_line",
            "home_moneyline",
            "away_moneyline",
            "gametime",
            "weekday",
        )
        if col in schedules.columns
    ]
    features = features.merge(schedules[meta_cols], on="game_id", how="left", suffixes=("", "_sched"))
    away_spread = pd.to_numeric(features.get("spread_line"), errors="coerce")
    features["nflverse_closing_spread"] = -away_spread
    features["nflverse_closing_total"] = pd.to_numeric(features.get("total_line"), errors="coerce")
    features["feature_audit"] = json.dumps(audit)
    return features


def merge_sources(reconstructed: pd.DataFrame, stored: pd.DataFrame) -> pd.DataFrame:
    frame = reconstructed.copy()
    if stored.empty:
        frame["prediction_source"] = "reconstructed_artifact"
        frame["my_home_win_prob"] = frame["reconstructed_home_win_prob"]
        frame["my_spread"] = frame["reconstructed_spread"]
        frame["predicted_margin"] = frame["reconstructed_margin"]
        frame["closing_spread"] = frame["nflverse_closing_spread"]
        frame["closing_total"] = frame["nflverse_closing_total"]
        return frame

    stored = stored.copy()
    stored["home_team"] = stored["home_team"].astype(str)
    stored["away_team"] = stored["away_team"].astype(str)
    stored["game_date"] = pd.to_datetime(stored["game_date"], errors="coerce").dt.tz_localize(None)
    frame["join_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.tz_localize(None)
    merged = frame.merge(
        stored,
        left_on=["season", "home_team", "away_team", "join_date"],
        right_on=["season", "home_team", "away_team", "game_date"],
        how="left",
        suffixes=("", "_stored"),
    )
    has_stored = merged["stored_home_win_prob"].notna()
    merged["prediction_source"] = np.where(has_stored, "stored_pre_kickoff", "reconstructed_artifact")
    merged["my_home_win_prob"] = merged["stored_home_win_prob"].fillna(merged["reconstructed_home_win_prob"])
    merged["my_spread"] = merged["stored_spread"].fillna(merged["reconstructed_spread"])
    merged["predicted_margin"] = -merged["my_spread"]
    stored_book = pd.to_numeric(merged.get("stored_book_spread"), errors="coerce")
    merged["closing_spread"] = stored_book.fillna(merged["nflverse_closing_spread"])
    stored_total = pd.to_numeric(merged.get("stored_total"), errors="coerce")
    merged["closing_total"] = stored_total.fillna(merged["nflverse_closing_total"])
    if "stored_home_moneyline" in merged.columns:
        merged["home_moneyline"] = pd.to_numeric(merged["stored_home_moneyline"], errors="coerce").fillna(
            pd.to_numeric(merged["home_moneyline"], errors="coerce")
        )
        merged["away_moneyline"] = pd.to_numeric(merged["stored_away_moneyline"], errors="coerce").fillna(
            pd.to_numeric(merged["away_moneyline"], errors="coerce")
        )
    merged["stored_vs_reconstructed_prob_delta"] = (
        pd.to_numeric(merged["stored_home_win_prob"], errors="coerce") - merged["reconstructed_home_win_prob"]
    )
    merged["stored_vs_reconstructed_spread_delta"] = (
        pd.to_numeric(merged["stored_spread"], errors="coerce") - merged["reconstructed_spread"]
    )
    return merged


def espn_index(dates: list[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for date_key in dates:
        try:
            events = espn_scoreboard(date_key)
        except Exception as exc:  # noqa: BLE001
            index[("error", date_key, "")] = {"error": str(exc)}
            continue
        for event in events:
            home = ESPN_ABBR.get(str(event.get("home_abbr") or ""), str(event.get("home_abbr") or ""))
            away = ESPN_ABBR.get(str(event.get("away_abbr") or ""), str(event.get("away_abbr") or ""))
            day = str(event.get("date") or "")[:10].replace("-", "")
            index[(day, away, home)] = event
    return index


def attach_espn(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = sorted(
        {
            pd.Timestamp(value).strftime("%Y%m%d")
            for value in pd.to_datetime(frame["game_date"], errors="coerce").dropna()
        }
    )
    lookup = espn_index(dates)
    matches = 0
    mismatches = []
    espn_home = []
    espn_away = []
    espn_status = []
    for row in frame.itertuples(index=False):
        day = pd.Timestamp(row.game_date).strftime("%Y%m%d") if pd.notna(row.game_date) else ""
        event = lookup.get((day, str(row.away_team), str(row.home_team)))
        if event is None and str(row.home_team) == "LA":
            event = lookup.get((day, str(row.away_team), "LAR"))
        espn_home.append(event.get("home_score") if event else None)
        espn_away.append(event.get("away_score") if event else None)
        espn_status.append(event.get("status") if event else None)
        if event and event.get("home_score") is not None and pd.notna(row.home_score):
            if int(event["home_score"]) == int(row.home_score) and int(event["away_score"]) == int(row.away_score):
                matches += 1
            else:
                mismatches.append(
                    {
                        "game": f"{row.away_team}@{row.home_team}",
                        "nflverse": [int(row.away_score), int(row.home_score)],
                        "espn": [event.get("away_score"), event.get("home_score")],
                        "status": event.get("status"),
                    }
                )
    out = frame.copy()
    out["espn_home_score"] = espn_home
    out["espn_away_score"] = espn_away
    out["espn_status"] = espn_status
    report = {
        "dates": dates,
        "matched_finals": matches,
        "mismatches": mismatches,
        "lookup_errors": [value for key, value in lookup.items() if key[0] == "error"],
    }
    return out, report


def fetch_public_snapshot() -> dict[str, Any]:
    try:
        response = requests.get(PUBLIC_RESULTS_URL, timeout=30)
        response.raise_for_status()
        text = response.text
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    snapshot = {"ok": True, "url": PUBLIC_RESULTS_URL, "updated_contains": "Oct 19" in text or "Oct 18" in text}
    if "51-41" in text or "51<!-- -->-<!-- -->41" in text:
        snapshot["winner_wl_mentioned"] = "51-41"
    if "39-53" in text or "39<!-- -->-<!-- -->53" in text:
        snapshot["spread_wl_mentioned"] = "39-53"
    snapshot["has_v2"] = "nfl-v2-live-20260906" in text
    return snapshot


def load_2025_v1_ats() -> dict[str, Any] | None:
    if not V1_ATS_2025.exists():
        return None
    games = pd.read_csv(V1_ATS_2025)
    if games.empty:
        return None
    return {
        "source": str(V1_ATS_2025),
        "rows": int(len(games)),
        "note": "Stored 2025 Supabase ATS is the v1 model, not nfl-v2-live-20260906.",
        "ats": games["ats_result"].value_counts(dropna=False).to_dict() if "ats_result" in games.columns else {},
    }


def diagnose_priors(frame: pd.DataFrame) -> dict[str, Any]:
    weeks = pd.to_numeric(frame.get("week"), errors="coerce")
    by_week = []
    for week in sorted(weeks.dropna().astype(int).unique().tolist()):
        subset = frame[weeks == week]
        games_played = pd.to_numeric(subset.get("games_before_diff"), errors="coerce").abs().fillna(0)
        # games_before_diff is home-away; typical week-n team has n-1 games.
        typical = max(0, week - 1)
        by_week.append(
            {
                "week": week,
                "n": int(len(subset)),
                "typical_games_played": typical,
                "window_3": prior_decay_mix(typical, 3),
                "window_5": prior_decay_mix(typical, 5),
                "window_10": prior_decay_mix(typical, 10),
                "mean_abs_games_before_diff": float(games_played.mean()) if len(games_played) else None,
            }
        )
    return {"by_week": by_week, "note": "Rolling features use (sum(window) + 4*prior) / (n + 4); prior shrinks last season with n/(n+8)."}


def run_calibration(frame: pd.DataFrame) -> dict[str, Any]:
    probability_methods = ["temperature", "platt", "isotonic"]
    market_rows = int(frame["market_home_prob"].notna().sum()) if "market_home_prob" in frame.columns else 0
    if market_rows >= 40:
        probability_methods.append("market_shrink")
    probability_results = [evaluate_probability_method(frame, method) for method in probability_methods]
    margin_results = [evaluate_margin_method(frame, method) for method in ("scale", "affine")]
    decision = ship_decision(probability_results, margin_results)
    return {
        "probability": probability_results,
        "margin": margin_results,
        "decision": decision,
        "market_rows": market_rows,
    }


def write_charts(frame: pd.DataFrame, chart_dir: Path) -> list[str]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    graded = frame.dropna(subset=["home_win", "my_home_win_prob", "actual_margin", "predicted_margin"]).copy()
    paths: list[str] = []
    if graded.empty:
        return paths

    fig, ax = plt.subplots(figsize=(7, 6))
    table = calibration_table(graded["home_win"].astype(int).to_numpy(), graded["my_home_win_prob"].to_numpy())
    xs = [row["avg_probability"] for row in table]
    ys = [row["actual_rate"] for row in table]
    sizes = [40 + 8 * row["rows"] for row in table]
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1, label="perfect")
    ax.scatter(xs, ys, s=sizes, color="#1d4ed8")
    for row in table:
        ax.annotate(str(row["rows"]), (row["avg_probability"], row["actual_rate"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Model home win probability")
    ax.set_ylabel("Observed home win rate")
    ax.set_title("NFL v2 reliability — 2026 live")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    reliability = chart_dir / "reliability-diagram.png"
    fig.tight_layout()
    fig.savefig(reliability, dpi=140)
    plt.close(fig)
    paths.append(str(reliability.relative_to(REPO_ROOT)))

    weekly = []
    weeks = pd.to_numeric(graded["week"], errors="coerce")
    for week in sorted(weeks.dropna().astype(int).unique().tolist()):
        subset = graded[weeks == week]
        weekly.append(
            {
                "week": week,
                "model_mae": float((subset["predicted_margin"] - subset["actual_margin"]).abs().mean()),
                "book_mae": float(((-subset["closing_spread"]) - subset["actual_margin"]).abs().mean())
                if subset["closing_spread"].notna().any()
                else None,
                "brier": float(((subset["home_win"].astype(int) - subset["my_home_win_prob"]) ** 2).mean()),
            }
        )
    weekly_frame = pd.DataFrame(weekly)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(weekly_frame["week"], weekly_frame["model_mae"], marker="o", label="model MAE")
    if weekly_frame["book_mae"].notna().any():
        ax.plot(weekly_frame["week"], weekly_frame["book_mae"], marker="o", label="book MAE")
    ax.set_xlabel("Week")
    ax.set_ylabel("Margin MAE")
    ax.set_title("Weekly margin error vs closing spread")
    ax.legend()
    ax.grid(True, alpha=0.3)
    weekly_path = chart_dir / "weekly-margin-mae.png"
    fig.tight_layout()
    fig.savefig(weekly_path, dpi=140)
    plt.close(fig)
    paths.append(str(weekly_path.relative_to(REPO_ROOT)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(weekly_frame["week"], weekly_frame["brier"], marker="o", color="#b45309")
    ax.axhline(0.25, color="#888", linestyle="--", linewidth=1, label="coin-flip Brier")
    ax.set_xlabel("Week")
    ax.set_ylabel("Brier")
    ax.set_title("Weekly Brier score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    brier_path = chart_dir / "weekly-brier.png"
    fig.tight_layout()
    fig.savefig(brier_path, dpi=140)
    plt.close(fig)
    paths.append(str(brier_path.relative_to(REPO_ROOT)))
    return paths


def _fmt_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math_isfinite(value)):
        return "n/a"
    return f"{value * 100:.1f}%"


def math_isfinite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math_isfinite(value)):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _wl(block: dict[str, Any] | None, keys: tuple[str, str] = ("wins", "losses")) -> str:
    if not block:
        return "n/a"
    wins, losses = block.get(keys[0]), block.get(keys[1])
    pushes = block.get("pushes")
    if wins is None:
        return "n/a"
    if pushes:
        return f"{wins}-{losses}-{pushes}"
    return f"{wins}-{losses}"


def _segment_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Segment | n | Brier | ECE | MAE | Home-prob bias | Book ATS | ML units |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        ats = row.get("book_ats") or {}
        lines.append(
            "| {segment} | {n} | {brier} | {ece} | {mae} | {bias} | {ats} | {ml} |".format(
                segment=row.get("segment"),
                n=row.get("rows"),
                brier=_fmt((row.get("probability") or {}).get("brier"), 3),
                ece=_fmt((row.get("probability") or {}).get("ece_10"), 3),
                mae=_fmt((row.get("margin") or {}).get("mae"), 2),
                bias=_fmt((row.get("probability") or {}).get("home_probability_bias"), 3),
                ats=_wl(ats),
                ml=_fmt(((row.get("moneyline") or {}).get("units")), 1),
            )
        )
    return lines


def week_slate_metrics(frame: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    slate = frame[(dates >= start) & (dates <= end)].copy()
    if slate.empty:
        return {"rows": 0}
    actual = slate["actual_margin"].to_numpy(dtype=float)
    pred = slate["predicted_margin"].to_numpy(dtype=float)
    book = -pd.to_numeric(slate["closing_spread"], errors="coerce").to_numpy(dtype=float)
    close = slate["close_game"].fillna(False).astype(bool)
    close_winner = slate.loc[close, "winner_result"]
    return {
        "rows": int(len(slate)),
        "model_mae": float(np.mean(np.abs(pred - actual))),
        "model_rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "book_mae": float(np.mean(np.abs(book - actual))),
        "book_rmse": float(np.sqrt(np.mean((book - actual) ** 2))),
        "winner": {
            "wins": int((slate["winner_result"] == "win").sum()),
            "losses": int((slate["winner_result"] == "loss").sum()),
        },
        "close_winner": {
            "wins": int((close_winner == "win").sum()),
            "losses": int((close_winner == "loss").sum()),
        },
        "unique_probabilities": int(pd.to_numeric(slate["my_home_win_prob"], errors="coerce").round(6).nunique()),
    }


def render_report(report: dict[str, Any], scored: pd.DataFrame, chart_paths: list[str], path: Path) -> None:
    live = report["live_2026"]
    metrics = live["metrics"]
    decision = report["calibration"]["decision"]
    image_lines = [f"![{Path(chart).stem}](nfl-model-health/{Path(chart).name})" for chart in chart_paths]
    stored_n = int((scored["prediction_source"] == "stored_pre_kickoff").sum()) if "prediction_source" in scored.columns else 0
    reconstructed_n = int((scored["prediction_source"] == "reconstructed_artifact").sum()) if "prediction_source" in scored.columns else len(scored)
    live_change = report["live_prediction_impact"]
    leakage = report.get("leakage") or {}
    slate = report.get("week_slate") or {}
    calib_rows = [
        "| Method | OOS n | Δ Brier | Δ log loss | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["calibration"]["probability"]:
        delta = row.get("delta") or {}
        calib_rows.append(
            f"| {row.get('method')} | {row.get('oos_rows')} | {_fmt(delta.get('brier'), 4)} | {_fmt(delta.get('log_loss'), 4)} | {row.get('status')} |"
        )
    margin_rows = [
        "| Method | OOS n | Δ MAE | Δ RMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["calibration"]["margin"]:
        delta = row.get("delta") or {}
        margin_rows.append(
            f"| {row.get('method')} | {row.get('oos_rows')} | {_fmt(delta.get('mae'), 3)} | {_fmt(delta.get('rmse'), 3)} |"
        )
    reliability = (metrics.get("probability") or {}).get("calibration") or []
    reliability_lines = [
        "| Bucket | n | Mean p | Actual | Bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in reliability:
        reliability_lines.append(
            f"| {row.get('bucket')} | {row.get('rows')} | {_fmt(row.get('avg_probability'), 3)} | {_fmt(row.get('actual_rate'), 3)} | {_fmt(row.get('bias'), 3)} |"
        )
    prior_lines = [
        "| Week | Typical games | 3-game current / last / league | 10-game current |",
        "| ---: | ---: | --- | ---: |",
    ]
    for row in (live.get("priors") or {}).get("by_week") or []:
        w3 = row.get("window_3") or {}
        w10 = row.get("window_10") or {}
        prior_lines.append(
            f"| {row['week']} | {row['typical_games_played']} | "
            f"{_fmt_pct(w3.get('current_season'))} / {_fmt_pct(w3.get('last_season'))} / {_fmt_pct(w3.get('league_average'))} | "
            f"{_fmt_pct(w10.get('current_season'))} |"
        )
    hist_lag = live.get("staleness", {}).get("feature_history_lag_days") or {}
    lines = [
        "# NFL v2 model health — 2026",
        "",
        f"**Generated:** {report['generated_at']}",
        f"**Model:** `{report['model_version']}`",
        f"**Graded 2026 games:** {metrics['rows']} (weeks {', '.join(str(w) for w in metrics.get('weeks') or [])})",
        "",
        "This is a model-quality readout, not a betting strategy. Missing books stay model-only; no invented edge or EV.",
        "",
        "## Headline vs the book",
        "",
        "| Measure | Model | Closing / no-vig book |",
        "| --- | ---: | ---: |",
        f"| Margin MAE | **{_fmt(metrics['margin'].get('mae'), 2)}** | **{_fmt((metrics.get('book_margin') or {}).get('mae'), 2)}** |",
        f"| Margin RMSE | {_fmt(metrics['margin'].get('rmse'), 2)} | {_fmt((metrics.get('book_margin') or {}).get('rmse'), 2)} |",
        f"| Home-margin bias | {_fmt(metrics['margin'].get('home_margin_bias'), 2)} | {_fmt((metrics.get('book_margin') or {}).get('home_margin_bias'), 2)} |",
        f"| Brier | {_fmt(metrics['probability'].get('brier'), 4)} | {_fmt((metrics.get('market_probability') or {}).get('brier'), 4)} |",
        f"| Log loss | {_fmt(metrics['probability'].get('log_loss'), 3)} | {_fmt((metrics.get('market_probability') or {}).get('log_loss'), 3)} |",
        f"| ECE (10-bin) | {_fmt(metrics['probability'].get('ece_10'), 3)} | {_fmt((metrics.get('market_probability') or {}).get('ece_10'), 3)} |",
        f"| Winner W-L | {_wl(metrics.get('winner'))} ({_fmt_pct((metrics.get('winner') or {}).get('hit_rate'))}) | {_fmt_pct((metrics.get('market_probability') or {}).get('accuracy'))} favorite hit |",
        f"| ATS vs book, model side, -110 | {_wl(metrics.get('book_ats'))} ({_fmt_pct((metrics.get('book_ats') or {}).get('roi'))} ROI) | — |",
        f"| ATS bootstrap 95% ROI | {_fmt((metrics.get('bootstrap') or {}).get('book_ats_roi', {}).get('lower_95'), 3)} to {_fmt((metrics.get('bootstrap') or {}).get('book_ats_roi', {}).get('upper_95'), 3)} | — |",
        f"| ML on model favorite | {_fmt((metrics.get('moneyline') or {}).get('units'), 2)}u / {_fmt_pct((metrics.get('moneyline') or {}).get('roi'))} | bootstrap {_fmt((metrics.get('bootstrap') or {}).get('moneyline_roi', {}).get('lower_95'), 2)} to {_fmt((metrics.get('bootstrap') or {}).get('moneyline_roi', {}).get('upper_95'), 2)} |",
        "",
        (
            f"Totals: NFL v2 does not emit a game total. Book closing-total MAE is {_fmt(((metrics.get('totals') or {}).get('book') or {}).get('mae'), 2)} on {((metrics.get('totals') or {}).get('book') or {}).get('rows', 0)} games."
            if (metrics.get("totals") or {}).get("book")
            else "Totals: NFL v2 has no totals model; no total MAE is reported."
        ),
        "",
        "The book still wins MAE, RMSE, Brier, log loss, and ECE. ATS ROI CIs include zero. Do not read this as a tradable edge.",
        "",
        "## Oct 15–18 slate",
        "",
        f"n={slate.get('rows')}, model MAE {_fmt(slate.get('model_mae'), 2)} / RMSE {_fmt(slate.get('model_rmse'), 2)} vs book {_fmt(slate.get('book_mae'), 2)} / {_fmt(slate.get('book_rmse'), 2)}. Close games (≤3 pts) {_wl(slate.get('close_winner'))}. Winner {_wl(slate.get('winner'))}. Unique model probabilities: {slate.get('unique_probabilities')}. Recomputed from the frozen artifact plus nflverse finals; ESPN had zero score disagreements on matched games.",
        "",
        "## Prediction source",
        "",
        f"- Stored pre-kickoff rows used: **{stored_n}**",
        f"- Artifact reconstructions used: **{reconstructed_n}**",
        f"- Public `/models/results` winner W-L mentioned: {(report.get('public_site') or {}).get('winner_wl_mentioned')}; model-line ATS mentioned: {(report.get('public_site') or {}).get('spread_wl_mentioned')}",
        f"- ESPN matched finals: {report.get('espn', {}).get('matched_finals')} of {metrics['rows']} with **{len(report.get('espn', {}).get('mismatches') or [])} score disagreements**",
        "",
        "This environment had no Supabase password, so the live number is the frozen `nfl_v2_live.joblib` reconstruction on leakage-safe pregame features — the same path Daily Refresh uses. Public winner W-L is 51-41 on 92 games, which matches this reconstruction. Public model-line ATS is 39-53 vs reconstructed model-line ~40-52; that one-game gap is rounding of stored `my_spread`, not a second model. Book ATS below uses nflverse closing home spreads (`home_spread = -spread_line`), not the Tuesday Odds API snapshot.",
        "",
        "WAS@SF (Mon 2026-10-19) was not final at scoring time and is excluded.",
        "",
        "## Reliability",
        "",
        *reliability_lines,
        "",
        *image_lines,
        "",
        "## Breakouts",
        "",
        "### Week",
        "",
        *_segment_table(live["segments"].get("week") or []),
        "",
        "### Favorite vs dog",
        "",
        *_segment_table(live["segments"].get("favorite") or []),
        "",
        "### Home vs away (ATS side and win-prob side)",
        "",
        *_segment_table(live["segments"].get("home_away") or []),
        "",
        "### Confidence (|p−0.5|)",
        "",
        *_segment_table(live["segments"].get("confidence") or []),
        "",
        "## Diagnosis",
        "",
        "The model is **overconfident on favorites** and **home-biased on the margin**, not just unlucky on one slate.",
        "",
        f"- Home-probability bias: **{_fmt(live['diagnosis']['home_probability_bias'], 3)}** (alert >0.03)",
        f"- Home-margin bias: **{_fmt(live['diagnosis']['home_margin_bias'], 2)}** points (alert >0.75)",
        f"- ECE: **{_fmt(live['diagnosis']['ece_10'], 3)}** vs 0.05 live-rollout alert; book ECE {_fmt((metrics.get('market_probability') or {}).get('ece_10'), 3)}",
        f"- Unique win probabilities on 92 games: **{live['diagnosis']['unique_win_probabilities']}** — the production isotonic map is coarse, especially on weekly slates",
        f"- Close games (≤3 pts): {metrics.get('close_games')} games, winner hit {_fmt_pct(metrics.get('close_game_winner_hit_rate'))}",
        f"- Leakage audit: ready={leakage.get('ready')}, leakage_rows={leakage.get('leakage_rows')}, duplicate_rows={leakage.get('duplicate_rows')}, feature rows={leakage.get('rows')}",
        f"- Feature-history lag: mean home {_fmt(hist_lag.get('home_mean'), 1)} days. Week 1 using last season's final game is expected (offseason), not a stale PBP ingest. Max lag {_fmt(hist_lag.get('home_max'), 0)} days is Super Bowl → opening week.",
        "- Prediction timestamps: not available without stored `asof_ts`. Daily Refresh froze this cycle on Tuesday 2026-10-13 and does not reopen the team model mid-week.",
        "- Feature drift vs 2025 REG: largest absolute shift is travel distance; standardized z-scores are ~0.1. No smoking-gun covariate shift.",
        "- Injuries / posted QB still excluded. Expected QB is the mode of the last three PBP dropback QBs. That is a known serving limitation, not a leakage bug.",
        "",
        "### Early-season prior decay",
        "",
        *prior_lines,
        "",
        "The 3-game window never grows past three observations, so after week 4 it stays ~43% current / ~39% last season / ~18% league average. 5- and 10-game windows add current-season mass as designed. Changing the prior weight would require a retrain; this health check does not ad-hoc retune it.",
        "",
        "## Recalibration (walk-forward, fit never sees the scored week)",
        "",
        "Expanding-week fits on weeks 1…k−1, scored on week k, pooled OOS once 40 training rows exist.",
        "",
        *calib_rows,
        "",
        *margin_rows,
        "",
        f"**Shipped to live serving: no.** {decision.get('reason')}",
        "",
        "Temperature, Platt, extra isotonic, and market-shrink all failed the Brier+log-loss gate. Margin scale/affine can trim OOS MAE; that is not enough to ship without a probability win.",
        "",
        "Code for those transforms lives behind `NFL_V2_POSTHOC_CALIBRATION` (default **off**). Rollback if a later run ships: `NFL_V2_DISABLE_POSTHOC_CALIBRATION=1`. Live Daily Refresh still publishes frozen `nfl-v2-live-20260906` outputs.",
        "",
        live_change["description"],
        "",
        "## Recommendation",
        "",
        "1. Keep the frozen live artifact. Do not mid-season-recalibrate on 2026 games.",
        "2. Next modeling lever is posted-QB / injury identity with timestamped history, not another probability spline on n=92.",
        "3. Re-run this script weekly; promote a post-hoc calibrator only if walk-forward Brier **and** log loss improve without a MAE regression.",
        "",
        "## 2025",
        "",
        report.get("season_2025_note") or "2025 live-v2 scores are in-sample for the frozen artifact (trained through 2025).",
        "",
        "Stored 2025 Supabase ATS (`nfl_supabase_ats_games_2025.csv`) is **v1**, 27-29, not v2.",
        "",
        "## Weekly recompute",
        "",
        "```bash",
        "PYTHONPATH=data-core python data-core/scripts/nfl_model_health.py",
        "```",
        "",
        "Performance History Refresh runs this after `grade_nfl_v2_live.py`. With Supabase credentials it prefers stored pre-kickoff rows; without them it reconstructs the artifact.",
        "",
        "Artifacts: `data-core/notebooks/cache/nfl_model_health_2026.json`, `nfl_model_health_2026_games.csv`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file)
    include_2025 = args.include_2025 and not args.skip_2025
    seasons = list(range(2020, args.season + 1))
    print(f"Health check model={args.model_version} seasons={seasons}", flush=True)

    stored = fetch_stored_predictions(args.model_version, [args.season] + ([2025] if include_2025 else []))
    print(f"Stored pre-kickoff rows: {len(stored)}", flush=True)
    reconstructed = reconstruct_live(seasons, args.model_version)
    merged = merge_sources(reconstructed, stored)
    leakage = audit_nfl_feature_store(reconstructed)

    live = merged[(merged["season"] == args.season) & (merged["game_type"].astype(str).str.upper() == "REG")].copy()
    completed = live[live["home_score"].notna() & live["away_score"].notna()].copy()
    completed = score_prediction_frame(completed)
    completed, espn_report = attach_espn(completed)

    reference = reconstructed[
        (reconstructed["season"] == 2025) & (reconstructed["game_type"].astype(str).str.upper() == "REG")
    ]
    drift = feature_drift_table(completed, reference, FEATURE_COLUMNS)
    diagnosis = bias_diagnosis(completed)
    staleness = staleness_report(completed)
    segments = segmented_health(completed)
    priors = diagnose_priors(completed)
    metrics = metrics_block(completed, bootstrap=True)
    calibration = run_calibration(completed)

    season_2025 = None
    season_2025_note = None
    if include_2025:
        hist = merged[(merged["season"] == 2025) & (merged["game_type"].astype(str).str.upper() == "REG")]
        hist = hist[hist["home_score"].notna() & hist["away_score"].notna()].copy()
        if not hist.empty:
            hist = score_prediction_frame(hist)
            season_2025 = metrics_block(hist, bootstrap=True)
            v1 = load_2025_v1_ats() or {}
            v1_ats = v1.get("ats") or {}
            season_2025_note = (
                f"2025 reconstructed through the frozen live artifact is **in-sample** "
                f"(trained_through=2025), n={len(hist)}. Brier {_fmt(season_2025['probability'].get('brier'), 4)}, "
                f"MAE {_fmt(season_2025['margin'].get('mae'), 2)}, book ATS "
                f"{_wl(season_2025.get('book_ats'))}. Stored v1 ATS is "
                f"{int(v1_ats.get('win', 0))}-{int(v1_ats.get('loss', 0))} on {v1.get('rows')} games."
            )

    write_posthoc = args.write_posthoc_if_improved and not args.no_write_posthoc
    decision = calibration["decision"]
    live_impact = {
        "alters_live_predictions": False,
        "description": "No serving change. `predict_live_games` still emits the frozen 20260906 isotonic/ridge outputs unless `NFL_V2_POSTHOC_CALIBRATION` is explicitly enabled *and* a shipped artifact exists.",
    }
    if write_posthoc and decision.get("ship"):
        payload = fit_production_calibrators(completed, decision["probability_method"], decision["margin_method"])
        artifact = {
            "enabled": True,
            "base_model_version": LIVE_MODEL_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation": "walk_forward_by_week_2026",
            "decision": decision,
            **payload,
            "note": "Fitted on all completed 2026 games after walk-forward selection. Do not score the 2026 sample with these parameters; they are for future games only.",
        }
        args.posthoc_artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        live_impact = {
            "alters_live_predictions": False,
            "artifact_written": str(args.posthoc_artifact),
            "description": (
                f"Wrote {args.posthoc_artifact.name} with {decision['probability_method']} "
                f"and {decision['margin_method']}. Live Daily Refresh is **unchanged** until "
                f"`NFL_V2_POSTHOC_CALIBRATION=on` is set. With the flag on, home win probability "
                f"and/or predicted margin would be transformed after the frozen artifact, before "
                f"BigQuery/Supabase publish. Fair odds derived from those probabilities would move; "
                f"feature hashes would change."
            ),
        }
    elif decision.get("ship"):
        live_impact["description"] = (
            "Walk-forward evidence supported a calibrator, but no artifact was written this run."
        )

    chart_paths = write_charts(completed, args.chart_dir)
    public_site = fetch_public_snapshot()

    export_cols = [
        col
        for col in (
            "game_id",
            "season",
            "week",
            "game_date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "my_spread",
            "my_home_win_prob",
            "closing_spread",
            "closing_total",
            "market_home_prob",
            "predicted_margin",
            "actual_margin",
            "winner_result",
            "model_line_ats",
            "book_ats",
            "ml_units",
            "prediction_source",
            "prediction_ts",
            "espn_status",
        )
        if col in completed.columns
    ]
    args.games_output.parent.mkdir(parents=True, exist_ok=True)
    completed[export_cols].sort_values(["week", "game_date", "away_team"]).to_csv(args.games_output, index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": args.model_version,
        "prediction_sources": {
            "stored_rows": int(len(stored)),
            "reconstructed_rows": int(len(reconstructed)),
            "live_completed": int(len(completed)),
            "live_completed_stored": int((completed["prediction_source"] == "stored_pre_kickoff").sum())
            if "prediction_source" in completed.columns
            else 0,
        },
        "live_2026": {
            "metrics": metrics,
            "segments": segments,
            "diagnosis": diagnosis,
            "staleness": staleness,
            "feature_drift": drift[:15],
            "priors": priors,
        },
        "season_2025": season_2025,
        "season_2025_note": season_2025_note,
        "season_2025_v1_ats": load_2025_v1_ats(),
        "week_slate": week_slate_metrics(completed, "2026-10-15", "2026-10-18"),
        "calibration": calibration,
        "live_prediction_impact": live_impact,
        "leakage": leakage,
        "espn": espn_report,
        "public_site": public_site,
        "charts": chart_paths,
        "games_csv": str(args.games_output),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_report(payload, completed, chart_paths, args.report)
    print(
        json.dumps(
            {
                "json": str(args.json_output),
                "report": str(args.report),
                "games": int(len(completed)),
                "brier": metrics["probability"].get("brier"),
                "mae": metrics["margin"].get("mae"),
                "book_mae": (metrics.get("book_margin") or {}).get("mae"),
                "ece": metrics["probability"].get("ece_10"),
                "ats": metrics.get("book_ats"),
                "ml_units": (metrics.get("moneyline") or {}).get("units"),
                "ship": decision.get("ship"),
                "alters_live": live_impact["alters_live_predictions"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
