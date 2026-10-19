"""Reusable NFL v2 health-check metrics.

Scores timestamp-safe pre-kickoff predictions against final scores and, when
available, closing/book prices. Used by the weekly health script and unit tests.
This is model-quality evidence, not a betting strategy.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.nfl_experiment import bootstrap_mean_interval
from src.models.nfl_v2 import (
    calibration_table,
    margin_metrics,
    no_vig_home_probability,
    probability_metrics,
)


WIN_PROFIT_AT_MINUS_110 = 100.0 / 110.0
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261019

CONFIDENCE_BINS = (0.0, 0.05, 0.10, 0.15, 0.25, 1.01)
CONFIDENCE_LABELS = ("coin", "lean", "moderate", "strong", "heavy")

PRIOR_WEIGHT = 4.0
PRIOR_SEASON_PSEUDOCOUNT = 8.0


def american_implied_probability(odds: object) -> float | None:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value == 0:
        return None
    return 100.0 / (value + 100.0) if value > 0 else (-value) / ((-value) + 100.0)


def american_profit(odds: float, won: bool) -> float:
    if not won:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def model_line_spread_result(actual_margin: float, my_spread: float) -> str:
    edge = float(actual_margin) + float(my_spread)
    if abs(edge) < 1e-3:
        return "push"
    return "win" if edge > 0 else "loss"


def book_ats_result(actual_margin: float, my_spread: float, book_spread: float) -> str | None:
    """Grade the model side versus the book home spread at -110.

    ``my_spread < book_spread`` means the model likes the home team more than
    the book (more negative / less positive home spread), so we bet home.
    """
    if pd.isna(book_spread) or pd.isna(my_spread) or pd.isna(actual_margin):
        return None
    if abs(float(my_spread) - float(book_spread)) < 1e-3:
        return "no_bet"
    book_edge = float(actual_margin) + float(book_spread)
    if abs(book_edge) < 1e-3:
        return "push"
    home_covers = book_edge > 0
    model_wants_home = float(my_spread) < float(book_spread)
    return "win" if (home_covers if model_wants_home else not home_covers) else "loss"


def ats_units(result: str | None) -> float | None:
    if result == "win":
        return WIN_PROFIT_AT_MINUS_110
    if result == "loss":
        return -1.0
    if result == "push":
        return 0.0
    return None


def prior_decay_mix(
    games_played: int,
    window: int,
    *,
    prior_weight: float = PRIOR_WEIGHT,
    prior_season_games: int = 17,
    prior_season_pseudocount: float = PRIOR_SEASON_PSEUDOCOUNT,
) -> dict[str, float]:
    """Implied mix of current-season / last-season / league-average for one rolling feature."""
    observed = max(0, min(int(games_played), int(window)))
    current = observed / (observed + prior_weight)
    prior_mass = prior_weight / (observed + prior_weight)
    season_share = prior_season_games / (prior_season_games + prior_season_pseudocount)
    return {
        "window": int(window),
        "games_played": int(games_played),
        "observed_in_window": observed,
        "current_season": float(current),
        "last_season": float(prior_mass * season_share),
        "league_average": float(prior_mass * (1.0 - season_share)),
    }


def score_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add outcome, ATS, ML, and error columns. Does not drop rows."""
    out = frame.copy()
    for column in (
        "home_score",
        "away_score",
        "my_spread",
        "my_home_win_prob",
        "closing_spread",
        "home_moneyline",
        "away_moneyline",
        "closing_total",
        "predicted_total",
    ):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["actual_margin"] = out["home_score"] - out["away_score"]
    if "predicted_margin" not in out.columns or out["predicted_margin"].isna().all():
        out["predicted_margin"] = -out["my_spread"]
    else:
        out["predicted_margin"] = pd.to_numeric(out["predicted_margin"], errors="coerce")
        missing_margin = out["predicted_margin"].isna() & out["my_spread"].notna()
        out.loc[missing_margin, "predicted_margin"] = -out.loc[missing_margin, "my_spread"]

    out["home_win"] = (out["home_score"] > out["away_score"]).astype("Int64")
    out["model_home_favorite"] = out["my_home_win_prob"] >= 0.5
    out["model_winner"] = np.where(out["model_home_favorite"], out["home_team"], out["away_team"])
    out["actual_winner"] = np.where(out["home_win"] == 1, out["home_team"], out["away_team"])
    scored = out["home_score"].notna() & out["away_score"].notna() & out["my_home_win_prob"].notna()
    out["winner_result"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out.loc[scored, "winner_result"] = np.where(
        out.loc[scored, "model_winner"] == out.loc[scored, "actual_winner"], "win", "loss"
    )
    out["model_line_ats"] = [
        model_line_spread_result(margin, spread) if pd.notna(margin) and pd.notna(spread) else None
        for margin, spread in zip(out["actual_margin"], out["my_spread"])
    ]
    book_spread = out["closing_spread"] if "closing_spread" in out.columns else pd.Series(np.nan, index=out.index)
    out["book_ats"] = [
        book_ats_result(margin, my_spread, book)
        for margin, my_spread, book in zip(out["actual_margin"], out["my_spread"], book_spread)
    ]
    out["ats_units"] = [ats_units(result) for result in out["book_ats"]]
    out["model_bets_home"] = [
        (float(my_spread) < float(book)) if pd.notna(my_spread) and pd.notna(book) and abs(float(my_spread) - float(book)) >= 1e-3 else None
        for my_spread, book in zip(out["my_spread"], book_spread)
    ]
    out["margin_error"] = out["predicted_margin"] - out["actual_margin"]
    out["abs_margin_error"] = out["margin_error"].abs()
    if "closing_spread" in out.columns:
        out["book_predicted_margin"] = -out["closing_spread"]
        out["book_abs_margin_error"] = (out["book_predicted_margin"] - out["actual_margin"]).abs()
    if "closing_total" in out.columns:
        out["actual_total"] = out["home_score"] + out["away_score"]
        out["book_abs_total_error"] = (out["closing_total"] - out["actual_total"]).abs()
        if "predicted_total" in out.columns:
            out["abs_total_error"] = (out["predicted_total"] - out["actual_total"]).abs()
    out["close_game"] = out["actual_margin"].abs() <= 3
    out["market_home_prob"] = [
        no_vig_home_probability(home, away)
        for home, away in zip(out.get("home_moneyline", pd.Series(np.nan, index=out.index)), out.get("away_moneyline", pd.Series(np.nan, index=out.index)))
    ]
    out["model_confidence"] = (out["my_home_win_prob"] - 0.5).abs()
    out["confidence_bucket"] = pd.cut(
        out["model_confidence"],
        bins=list(CONFIDENCE_BINS),
        labels=list(CONFIDENCE_LABELS),
        include_lowest=True,
        right=False,
    )
    ml_units: list[float | None] = []
    for row in out.itertuples(index=False):
        home_ml = getattr(row, "home_moneyline", np.nan)
        away_ml = getattr(row, "away_moneyline", np.nan)
        prob = getattr(row, "my_home_win_prob", np.nan)
        home_win = getattr(row, "home_win", pd.NA)
        if pd.isna(home_ml) or pd.isna(away_ml) or pd.isna(prob) or pd.isna(home_win):
            ml_units.append(np.nan)
            continue
        if float(prob) >= 0.5:
            ml_units.append(american_profit(float(home_ml), bool(int(home_win) == 1)))
        else:
            ml_units.append(american_profit(float(away_ml), bool(int(home_win) == 0)))
    out["ml_units"] = ml_units
    return out


def _record_counts(values: Iterable[object], keys: Sequence[str] = ("win", "loss", "push")) -> dict[str, int]:
    series = pd.Series(list(values))
    return {key: int((series == key).sum()) for key in keys}


def _roi_block(results: pd.Series, units: pd.Series | None = None) -> dict[str, Any]:
    counts = _record_counts(results)
    wins, losses, pushes = counts["win"], counts["loss"], counts["push"]
    risked = wins + losses
    if units is None:
        net = wins * WIN_PROFIT_AT_MINUS_110 - losses
    else:
        net = float(pd.to_numeric(units, errors="coerce").sum())
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "bets": risked,
        "hit_rate": (wins / risked) if risked else None,
        "units": net if risked or pushes else None,
        "roi": (net / risked) if risked else None,
    }


def bootstrap_stat_interval(
    values: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | None]:
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return {"mean": None, "lower_95": None, "upper_95": None, "n": 0}
    interval = bootstrap_mean_interval(observed, samples=samples, seed=seed)
    interval["n"] = int(observed.size)
    return interval


def _ats_units_series(results: pd.Series) -> np.ndarray:
    mapped = []
    for result in results:
        if result == "win":
            mapped.append(WIN_PROFIT_AT_MINUS_110)
        elif result == "loss":
            mapped.append(-1.0)
        elif result == "push":
            mapped.append(0.0)
    return np.asarray(mapped, dtype=float)


def metrics_block(
    frame: pd.DataFrame,
    *,
    probability_column: str = "my_home_win_prob",
    predicted_margin_column: str = "predicted_margin",
    bootstrap: bool = True,
) -> dict[str, Any]:
    graded = frame.dropna(subset=["home_score", "away_score", "my_spread", probability_column]).copy()
    if graded.empty:
        return {"rows": 0}
    home_win = graded["home_win"].astype(int)
    probability = pd.to_numeric(graded[probability_column], errors="coerce")
    predicted_margin = pd.to_numeric(graded[predicted_margin_column], errors="coerce")
    actual_margin = pd.to_numeric(graded["actual_margin"], errors="coerce")

    probability_result = probability_metrics(home_win, probability)
    margin_result = margin_metrics(actual_margin, predicted_margin)

    book = graded.dropna(subset=["closing_spread"]) if "closing_spread" in graded.columns else graded.iloc[0:0]
    book_margin = (
        margin_metrics(book["actual_margin"], -pd.to_numeric(book["closing_spread"], errors="coerce"))
        if len(book)
        else {}
    )
    market = graded.dropna(subset=["market_home_prob"]) if "market_home_prob" in graded.columns else graded.iloc[0:0]
    market_probability = (
        probability_metrics(market["home_win"].astype(int), market["market_home_prob"]) if len(market) else {}
    )

    totals: dict[str, Any] = {"model": None, "book": None}
    if "abs_total_error" in graded.columns:
        total_err = pd.to_numeric(graded["abs_total_error"], errors="coerce").dropna()
        if len(total_err):
            totals["model"] = {"rows": int(len(total_err)), "mae": float(total_err.mean())}
    if "book_abs_total_error" in graded.columns:
        book_total = pd.to_numeric(graded["book_abs_total_error"], errors="coerce").dropna()
        if len(book_total):
            totals["book"] = {"rows": int(len(book_total)), "mae": float(book_total.mean())}

    model_line = _roi_block(graded["model_line_ats"])
    book_ats_rows = graded[graded["book_ats"].isin(["win", "loss", "push"])]
    book_ats = _roi_block(book_ats_rows["book_ats"], book_ats_rows["ats_units"] if "ats_units" in book_ats_rows else None)
    book_ats["no_bet"] = int((graded["book_ats"] == "no_bet").sum()) if "book_ats" in graded.columns else 0

    ml = graded.dropna(subset=["ml_units"]) if "ml_units" in graded.columns else graded.iloc[0:0]
    winner_wins = int((graded["winner_result"] == "win").sum())
    winner_losses = int((graded["winner_result"] == "loss").sum())

    out: dict[str, Any] = {
        "rows": int(len(graded)),
        "weeks": sorted(pd.to_numeric(graded.get("week"), errors="coerce").dropna().astype(int).unique().tolist()),
        "probability": probability_result,
        "margin": margin_result,
        "book_margin": book_margin,
        "market_probability": market_probability,
        "totals": totals,
        "model_line_ats": model_line,
        "book_ats": book_ats,
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
        "close_games": int(graded["close_game"].sum()) if "close_game" in graded.columns else None,
        "close_game_winner_hit_rate": (
            float((graded.loc[graded["close_game"], "winner_result"] == "win").mean())
            if "close_game" in graded.columns and graded["close_game"].any()
            else None
        ),
    }
    if bootstrap:
        out["bootstrap"] = {
            "book_ats_roi": bootstrap_stat_interval(
                _ats_units_series(book_ats_rows["book_ats"]) if len(book_ats_rows) else []
            ),
            "moneyline_roi": bootstrap_stat_interval(ml["ml_units"].to_numpy() if len(ml) else []),
            "brier": bootstrap_stat_interval((home_win.to_numpy() - probability.to_numpy()) ** 2),
            "abs_margin_error": bootstrap_stat_interval(graded["abs_margin_error"].to_numpy()),
        }
    return out


def _subset_metrics(frame: pd.DataFrame, mask: pd.Series, label: str) -> dict[str, Any] | None:
    subset = frame.loc[mask].copy()
    if subset.empty:
        return None
    block = metrics_block(subset, bootstrap=False)
    block["segment"] = label
    return block


def segmented_health(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    graded = frame.dropna(subset=["home_score", "away_score", "my_spread", "my_home_win_prob"]).copy()
    if graded.empty:
        return {"week": [], "favorite": [], "home_away": [], "confidence": []}

    week = pd.to_numeric(graded.get("week"), errors="coerce")
    week_rows = []
    for value in sorted(week.dropna().astype(int).unique().tolist()):
        row = _subset_metrics(graded, week == value, f"week_{value}")
        if row:
            week_rows.append(row)

    favorite_rows = []
    for label, mask in (
        ("model_favorite", graded["model_home_favorite"].fillna(False).astype(bool)),
        ("model_dog", ~graded["model_home_favorite"].fillna(False).astype(bool)),
        ("book_home_favorite", pd.to_numeric(graded.get("closing_spread"), errors="coerce") < 0),
        ("book_home_dog", pd.to_numeric(graded.get("closing_spread"), errors="coerce") > 0),
    ):
        row = _subset_metrics(graded, mask, label)
        if row:
            favorite_rows.append(row)

    home_away_rows = []
    bets_home = graded["model_bets_home"]
    for label, mask in (
        ("ats_bet_home", bets_home == True),  # noqa: E712
        ("ats_bet_away", bets_home == False),  # noqa: E712
        ("home_win_prob_side", graded["model_home_favorite"].fillna(False).astype(bool)),
        ("away_win_prob_side", ~graded["model_home_favorite"].fillna(False).astype(bool)),
    ):
        row = _subset_metrics(graded, mask.fillna(False), label)
        if row:
            home_away_rows.append(row)

    confidence_rows = []
    if "confidence_bucket" in graded.columns:
        for label in CONFIDENCE_LABELS:
            row = _subset_metrics(graded, graded["confidence_bucket"].astype(str) == label, f"confidence_{label}")
            if row:
                confidence_rows.append(row)

    return {
        "week": week_rows,
        "favorite": favorite_rows,
        "home_away": home_away_rows,
        "confidence": confidence_rows,
    }


def bias_diagnosis(frame: pd.DataFrame) -> dict[str, Any]:
    graded = frame.dropna(subset=["home_score", "away_score", "my_home_win_prob", "my_spread"]).copy()
    if graded.empty:
        return {"status": "no_games"}
    probability = probability_metrics(graded["home_win"].astype(int), graded["my_home_win_prob"])
    margin = margin_metrics(graded["actual_margin"], graded["predicted_margin"])
    unique_probs = int(pd.to_numeric(graded["my_home_win_prob"], errors="coerce").round(6).nunique())
    overconfident_bins = [
        row for row in probability.get("calibration", [])
        if row["rows"] >= 8 and abs(row["bias"]) >= 0.08 and row["avg_probability"] >= 0.6
    ]
    return {
        "home_probability_bias": probability["home_probability_bias"],
        "home_margin_bias": margin["home_margin_bias"],
        "ece_10": probability["ece_10"],
        "unique_win_probabilities": unique_probs,
        "favorite_overconfidence_bins": overconfident_bins,
        "mean_predicted_home_win": probability["avg_probability"],
        "actual_home_win_rate": probability["actual_rate"],
        "mean_model_confidence": float(graded["model_confidence"].mean()) if "model_confidence" in graded else None,
        "flags": {
            "home_probability_bias": abs(probability["home_probability_bias"]) > 0.03,
            "home_margin_bias": abs(margin["home_margin_bias"]) > 0.75,
            "ece_alert": probability["ece_10"] > 0.05,
            "isotonic_coarseness": unique_probs <= max(8, len(graded) // 10),
        },
    }


def feature_drift_table(
    live: pd.DataFrame,
    reference: pd.DataFrame,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        if column not in live.columns:
            continue
        live_values = pd.to_numeric(live[column], errors="coerce").dropna()
        ref_values = pd.to_numeric(reference[column], errors="coerce").dropna() if column in reference.columns else pd.Series(dtype=float)
        if live_values.empty:
            continue
        live_mean = float(live_values.mean())
        ref_mean = float(ref_values.mean()) if len(ref_values) else None
        live_std = float(live_values.std(ddof=0)) if len(live_values) > 1 else 0.0
        z = None if ref_mean is None or live_std == 0 else (live_mean - ref_mean) / (float(ref_values.std(ddof=0)) or 1.0)
        rows.append(
            {
                "feature": column,
                "live_mean": live_mean,
                "reference_mean": ref_mean,
                "live_std": live_std,
                "abs_mean_delta": None if ref_mean is None else abs(live_mean - ref_mean),
                "reference_z": z,
            }
        )
    rows.sort(key=lambda row: row["abs_mean_delta"] or 0.0, reverse=True)
    return rows


def staleness_report(frame: pd.DataFrame) -> dict[str, Any]:
    graded = frame.copy()
    hours_before_kickoff = None
    if "prediction_ts" in graded.columns and "game_time_utc" in graded.columns:
        pred = pd.to_datetime(graded["prediction_ts"], utc=True, errors="coerce")
        kick = pd.to_datetime(graded["game_time_utc"], utc=True, errors="coerce")
        delta = (kick - pred).dt.total_seconds() / 3600.0
        valid = delta.dropna()
        hours_before_kickoff = {
            "rows": int(len(valid)),
            "mean_hours": float(valid.mean()) if len(valid) else None,
            "min_hours": float(valid.min()) if len(valid) else None,
            "post_kickoff_rows": int((delta < 0).fillna(False).sum()),
        }
    history_lag_days = None
    if "home_history_max_date" in graded.columns and "game_date" in graded.columns:
        game_date = pd.to_datetime(graded["game_date"], errors="coerce")
        home_hist = pd.to_datetime(graded["home_history_max_date"], errors="coerce")
        away_hist = pd.to_datetime(graded.get("away_history_max_date"), errors="coerce")
        home_lag = (game_date - home_hist).dt.days
        away_lag = (game_date - away_hist).dt.days
        history_lag_days = {
            "home_mean": float(home_lag.mean()) if home_lag.notna().any() else None,
            "away_mean": float(away_lag.mean()) if away_lag.notna().any() else None,
            "home_max": float(home_lag.max()) if home_lag.notna().any() else None,
            "away_max": float(away_lag.max()) if away_lag.notna().any() else None,
        }
    return {
        "prediction_lead_time": hours_before_kickoff,
        "feature_history_lag_days": history_lag_days,
        "missing_book_spread": int(graded["closing_spread"].isna().sum()) if "closing_spread" in graded.columns else None,
        "missing_market_probability": int(graded["market_home_prob"].isna().sum()) if "market_home_prob" in graded.columns else None,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value
