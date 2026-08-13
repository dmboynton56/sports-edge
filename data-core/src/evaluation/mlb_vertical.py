"""Production-oriented evaluation for the MLB model vertical.

This module deliberately keeps model features and market prices separate.  The
model functions are responsible for time-split training/evaluation; this
orchestrator joins their held-out predictions to free public moneylines only
after the model has produced probabilities.  That makes odds useful for edge
and ROI analysis without introducing market leakage into training.

The output is a JSON-safe summary plus a long-form edge table.  The table has a
common schema across game probability markets (moneyline, run line, totals,
pitcher strikeouts) and player probabilities (batter home runs), so the web
surface and downstream jobs do not need market-specific parsing.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.data.odds_math import american_to_decimal, american_to_implied, remove_vig
from src.models.mlb_runline_model import train_and_evaluate_mlb_runline
from src.models.mlb_strikeouts_model import (
    reshape_starter_sides,
    train_and_evaluate_strikeouts,
)
from src.models.mlb_totals_model import train_and_evaluate_mlb_totals
from src.models.mlb_winner_model import (
    default_feature_columns,
    train_and_evaluate_mlb_winner,
)


MONEYLINE_SOURCE_PRIORITY: dict[str, int] = {
    "oddspapi": 1,
    "checkbestodds": 2,
    "fantasydata": 3,
    "legacy_free": 4,
    "unknown": 9,
}

COMMON_EDGE_COLUMNS = [
    "market",
    "market_type",
    "game_pk",
    "date",
    "entity_id",
    "entity_name",
    "team",
    "opponent",
    "side",
    "selection",
    "model_probability",
    "reference_probability",
    "market_probability",
    "price",
    "edge",
    "ev",
    "edge_signal",
    "odds_status",
    "odds_source",
    "join_quality",
    "actual",
    "won",
    "profit_units",
    "is_actionable",
]


def _read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or parquet table with a useful error for unsupported files."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    if table_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(table_path)
    if table_path.suffix.lower() == ".json":
        payload = json.loads(table_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return pd.DataFrame(payload["rows"])
        raise ValueError(f"Expected a row list in JSON table: {table_path}")
    return pd.read_csv(table_path)


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _date_string(value: object) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return str(parsed.date())


def _json_default(value: object) -> object:
    """Convert pandas/numpy scalars for artifact serialization."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (date,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default),
        encoding="utf-8",
    )


def profile_feature_store(
    features: pd.DataFrame,
    *,
    as_of_date: str | date | None = None,
) -> dict[str, Any]:
    """Profile the evaluation grain and identify quality/leakage blockers.

    The intended grain is one completed regular-season game per ``game_pk``.
    This check is intentionally independent of model training so a bad refresh
    cannot quietly produce an apparently strong backtest.
    """
    required = {
        "game_pk",
        "season",
        "game_date",
        "game_datetime",
        "home_win",
        "run_diff",
        "total_runs",
        "home_cover_15",
    }
    missing = sorted(required.difference(features.columns))
    frame = features.copy()
    date_values = pd.to_datetime(frame.get("game_date"), errors="coerce")
    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    game_pk = frame.get("game_pk", pd.Series(dtype="Int64"))
    numeric_targets = [column for column in ["home_win", "run_diff", "total_runs", "home_cover_15"] if column in frame]
    feature_columns: list[str] = []
    leakage_columns: list[str] = []
    if not missing:
        feature_columns = default_feature_columns(frame)
        outcome_keywords = {
            "home_win",
            "run_diff",
            "total_runs",
            "home_score",
            "away_score",
            "home_cover_15",
            "starter_ks_label",
        }
        leakage_columns = [column for column in feature_columns if column in outcome_keywords or column.startswith("actual_")]
    missingness = {
        column: float(frame[column].isna().mean())
        for column in sorted(required.union({"home_probable_pitcher_id", "away_probable_pitcher_id"}).intersection(frame.columns))
    }
    warnings: list[str] = []
    if missing:
        warnings.append(f"missing_required_columns:{','.join(missing)}")
    if not frame.empty and game_pk.duplicated().any():
        warnings.append("duplicate_game_pk")
    invalid_dates = int(date_values.isna().sum())
    if invalid_dates:
        warnings.append("invalid_game_date")
    future_rows = int((date_values > as_of).sum()) if date_values.notna().any() else 0
    if future_rows:
        warnings.append("future_dated_rows")
    if leakage_columns:
        warnings.append("feature_selector_contains_outcome_columns")
    # The v2 store currently uses observed game-time weather as a pregame
    # proxy. Keep it visible in the artifact instead of hiding the limitation.
    weather_proxy = sorted(set(frame.columns).intersection({"temp_f", "wind_mph", "wind_out", "wind_in", "wind_cross"}))
    notes = []
    if weather_proxy:
        notes.append("weather_fields_are_observed_game_time_values_in_v2_and_should_be_replaced_by_pregame_forecasts_for_live_scoring")

    seasons = sorted(pd.to_numeric(frame.get("season"), errors="coerce").dropna().astype(int).unique().tolist()) if "season" in frame else []
    return {
        "intended_grain": "one completed regular-season MLB game per game_pk",
        "rows": int(len(frame)),
        "unique_game_pk": int(game_pk.nunique(dropna=True)),
        "duplicate_game_pk_rows": int(game_pk.duplicated(keep=False).sum()),
        "min_game_date": _date_string(date_values.min()) if date_values.notna().any() else None,
        "max_game_date": _date_string(date_values.max()) if date_values.notna().any() else None,
        "as_of_date": str(as_of.date()),
        "future_dated_rows": future_rows,
        "invalid_game_date_rows": invalid_dates,
        "seasons": seasons,
        "missing_required_columns": missing,
        "missingness": missingness,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "leakage_columns": leakage_columns,
        "target_summary": {
            column: {
                "non_null": int(frame[column].notna().sum()),
                "mean": _finite_float(pd.to_numeric(frame[column], errors="coerce").mean()),
            }
            for column in numeric_targets
        },
        "weather_proxy_columns": weather_proxy,
        "warnings": warnings,
        "notes": notes,
        "status": "review" if warnings else "pass",
    }


def _infer_moneyline_source(frame: pd.DataFrame, path: Path) -> str:
    values = " ".join(str(value).lower() for value in frame.get("source", pd.Series(dtype=object)).dropna().head(100))
    haystack = f"{path.name.lower()} {values}"
    if "oddspapi" in haystack or "pinnacle" in haystack:
        return "oddspapi"
    if "checkbest" in haystack:
        return "checkbestodds"
    if "fantasydata" in haystack:
        return "fantasydata"
    if "free_moneyline" in haystack:
        return "legacy_free"
    return "unknown"


def _normalise_moneyline_frame(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one public odds export while retaining provenance fields."""
    source = _infer_moneyline_source(frame, path)
    rename = {
        "home_ml": "home_moneyline",
        "away_ml": "away_moneyline",
        "home_price": "home_moneyline",
        "away_price": "away_moneyline",
    }
    out = frame.rename(columns=rename).copy()
    required = {"game_pk", "home_moneyline", "away_moneyline"}
    missing = sorted(required.difference(out.columns))
    if missing:
        return pd.DataFrame(), {
            "path": str(path),
            "source": source,
            "input_rows": int(len(frame)),
            "valid_rows": 0,
            "missing_columns": missing,
            "status": "skipped",
        }
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["home_moneyline"] = pd.to_numeric(out["home_moneyline"], errors="coerce")
    out["away_moneyline"] = pd.to_numeric(out["away_moneyline"], errors="coerce")
    out = out.dropna(subset=["game_pk", "home_moneyline", "away_moneyline"]).copy()
    out["game_pk"] = out["game_pk"].astype(int)
    # Prices of zero are invalid American odds; retain no partial rows.
    out = out[(out["home_moneyline"] != 0) & (out["away_moneyline"] != 0)]
    out["source"] = source
    out["source_file"] = str(path)
    out["source_priority"] = MONEYLINE_SOURCE_PRIORITY[source]
    if "source_match_ts" in out:
        out["odds_timestamp"] = pd.to_datetime(out["source_match_ts"], errors="coerce", utc=True)
    elif "snapshot_ts" in out:
        out["odds_timestamp"] = pd.to_datetime(out["snapshot_ts"], errors="coerce", utc=True)
    else:
        out["odds_timestamp"] = pd.NaT
    if "hours_delta" in out:
        out["join_quality"] = "team_datetime_nearest"
    elif source == "fantasydata":
        out["join_quality"] = "date_team_final_score"
    elif source == "oddspapi":
        out["join_quality"] = "fixture_api"
    else:
        out["join_quality"] = "game_pk_export"
    keep = [
        "game_pk",
        "game_date",
        "home_team",
        "away_team",
        "home_moneyline",
        "away_moneyline",
        "source",
        "source_file",
        "source_priority",
        "odds_timestamp",
        "join_quality",
    ]
    if "hours_delta" in out:
        keep.append("hours_delta")
    if "source_url" in out:
        keep.append("source_url")
    if "source_href" in out:
        keep.append("source_href")
    keep = [column for column in keep if column in out.columns]
    return out[keep], {
        "path": str(path),
        "source": source,
        "input_rows": int(len(frame)),
        "valid_rows": int(len(out)),
        "duplicate_game_pk_rows": int(out["game_pk"].duplicated(keep=False).sum()),
        "missing_columns": [],
        "status": "loaded",
    }


def normalize_moneyline_sources(
    paths: Iterable[str | Path],
    *,
    target_game_pks: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, normalize, and prioritize free moneyline sources.

    Priority favors fixture/book-labelled records, then timestamped public
    comparison prices, then consensus fallbacks.  A source is never silently
    treated as a specific sportsbook.
    """
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            audits.append({"path": str(path), "status": "missing"})
            continue
        frame = _read_table(path)
        normalized, audit = _normalise_moneyline_frame(frame, path)
        audits.append(audit)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        empty = pd.DataFrame(
            columns=[
                "game_pk",
                "home_moneyline",
                "away_moneyline",
                "source",
                "join_quality",
            ]
        )
        return empty, {
            "sources": audits,
            "input_rows": 0,
            "normalized_rows": 0,
            "selected_rows": 0,
            "coverage": 0.0,
            "selected_by_source": {},
            "status": "missing",
        }
    combined = pd.concat(frames, ignore_index=True, sort=False)
    hours = combined["hours_delta"] if "hours_delta" in combined else pd.Series(np.nan, index=combined.index)
    combined["_hours_sort"] = pd.to_numeric(hours, errors="coerce").fillna(9999)
    combined = combined.sort_values(
        ["game_pk", "source_priority", "_hours_sort", "odds_timestamp"],
        na_position="last",
    )
    selected = combined.drop_duplicates("game_pk", keep="first").drop(columns=["_hours_sort"])
    target = set(int(value) for value in target_game_pks) if target_game_pks is not None else set()
    matched = int(selected["game_pk"].isin(target).sum()) if target else int(len(selected))
    selected_by_source = {
        str(source): int(count)
        for source, count in selected["source"].value_counts(dropna=False).items()
    }
    return selected.reset_index(drop=True), {
        "sources": audits,
        "input_rows": int(len(combined)),
        "normalized_rows": int(len(combined)),
        "selected_rows": int(len(selected)),
        "target_games": int(len(target)) if target else None,
        "matched_target_games": matched if target else None,
        "coverage": float(matched / len(target)) if target else None,
        "selected_by_source": selected_by_source,
        "status": "partial" if target and matched < len(target) else "loaded",
        "notes": [
            "Prices are joined after model scoring and are not model features.",
            "Public comparison/consensus rows are not equivalent to a named sportsbook quote.",
        ],
    }


def _moneyline_profit(won: object, price: object) -> float | None:
    if price is None or pd.isna(price):
        return None
    decimal = american_to_decimal(int(float(price)))
    if decimal <= 1:
        return None
    return float(decimal - 1.0) if bool(won) else -1.0


def attach_moneyline_edges(predictions: pd.DataFrame, odds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach no-vig probabilities, EV, selected side, and realized profit."""
    required = {"game_pk", "home_moneyline", "away_moneyline"}
    missing = sorted(required.difference(odds.columns))
    if missing:
        raise ValueError(f"Normalized odds missing columns: {missing}")
    out = predictions.merge(odds, on="game_pk", how="left", suffixes=("", "_odds"))
    home_price = pd.to_numeric(out["home_moneyline"], errors="coerce")
    away_price = pd.to_numeric(out["away_moneyline"], errors="coerce")
    home_raw = home_price.map(lambda value: american_to_implied(int(value)) if pd.notna(value) and value != 0 else np.nan)
    away_raw = away_price.map(lambda value: american_to_implied(int(value)) if pd.notna(value) and value != 0 else np.nan)
    overround = home_raw + away_raw
    out["market_home_probability"] = home_raw / overround
    out["market_away_probability"] = away_raw / overround
    out["home_decimal"] = home_price.map(lambda value: american_to_decimal(int(value)) if pd.notna(value) and value != 0 else np.nan)
    out["away_decimal"] = away_price.map(lambda value: american_to_decimal(int(value)) if pd.notna(value) and value != 0 else np.nan)
    out["home_edge"] = pd.to_numeric(out["home_probability"], errors="coerce") - out["market_home_probability"]
    out["away_edge"] = pd.to_numeric(out["away_probability"], errors="coerce") - out["market_away_probability"]
    out["home_ev"] = pd.to_numeric(out["home_probability"], errors="coerce") * out["home_decimal"] - 1.0
    out["away_ev"] = pd.to_numeric(out["away_probability"], errors="coerce") * out["away_decimal"] - 1.0
    valid = out["home_ev"].notna() & out["away_ev"].notna()
    out["pick_side"] = np.where(valid, np.where(out["home_ev"] >= out["away_ev"], "home", "away"), None)
    out["model_probability"] = np.where(out["pick_side"] == "home", out["home_probability"], out["away_probability"])
    out["market_probability"] = np.where(out["pick_side"] == "home", out["market_home_probability"], out["market_away_probability"])
    out["price"] = np.where(out["pick_side"] == "home", home_price, away_price)
    out["edge"] = out["model_probability"] - out["market_probability"]
    out["ev"] = np.where(out["pick_side"] == "home", out["home_ev"], out["away_ev"])
    if "home_win" in out:
        out["won"] = np.where(
            out["pick_side"].eq("home"), out["home_win"].eq(1), out["home_win"].eq(0)
        )
        out["profit_units"] = [
            _moneyline_profit(won, price) if pd.notna(price) else np.nan
            for won, price in zip(out["won"], out["price"])
        ]
    else:
        out["won"] = np.nan
        out["profit_units"] = np.nan
    out["odds_status"] = np.where(valid, "free_price_joined", "missing")
    out["edge_signal"] = np.where(out["edge"] > 0.02, "positive", np.where(out["edge"] < -0.02, "negative", "neutral"))
    out["is_actionable"] = valid & (out["ev"] > 0)
    matched = out.loc[valid]
    summary = {
        "prediction_rows": int(len(out)),
        "matched_rows": int(len(matched)),
        "coverage": float(len(matched) / len(out)) if len(out) else 0.0,
        "positive_ev_rows": int((matched["ev"] > 0).sum()) if len(matched) else 0,
        "flat_profit_units": float(matched["profit_units"].sum()) if matched["profit_units"].notna().any() else None,
        "flat_roi": float(matched["profit_units"].mean()) if matched["profit_units"].notna().any() else None,
        "source_counts": {
            str(source): int(count)
            for source, count in out.loc[valid, "source"].value_counts(dropna=False).items()
        },
    }
    return out, summary


def _empty_edge_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMON_EDGE_COLUMNS)


def _binary_edge_rows(
    frame: pd.DataFrame,
    *,
    market: str,
    market_type: str,
    selection: str,
    probability: Sequence[float],
    reference_probability: Sequence[float] | float | None,
    actual: Sequence[object] | None,
    game_pk: Sequence[object] | None = None,
    date_values: Sequence[object] | None = None,
    entity_id: Sequence[object] | None = None,
    entity_name: Sequence[object] | None = None,
    team: Sequence[object] | None = None,
    opponent: Sequence[object] | None = None,
    side: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Create a common statistical-edge frame without inventing odds."""
    count = len(probability)
    def values(value: Sequence[object] | object | None) -> list[object]:
        if value is None:
            return [None] * count
        if isinstance(value, (str, bytes)) or np.isscalar(value):
            return [value] * count
        return list(value)

    probs = np.asarray(probability, dtype=float)
    refs = (
        np.full(count, float(reference_probability), dtype=float)
        if reference_probability is not None and np.isscalar(reference_probability)
        else np.asarray(reference_probability, dtype=float) if reference_probability is not None else np.full(count, np.nan)
    )
    actual_values = np.asarray(actual, dtype=object) if actual is not None else np.full(count, np.nan, dtype=object)
    output = pd.DataFrame(
        {
            "market": market,
            "market_type": market_type,
            "game_pk": values(game_pk),
            "date": [_date_string(value) for value in values(date_values)],
            "entity_id": values(entity_id),
            "entity_name": values(entity_name),
            "team": values(team),
            "opponent": values(opponent),
            "side": values(side),
            "selection": selection,
            "model_probability": probs,
            "reference_probability": refs,
            "market_probability": np.nan,
            "price": np.nan,
            "edge": probs - refs,
            "ev": np.nan,
            "edge_signal": np.where(probs - refs > 0.02, "positive", np.where(probs - refs < -0.02, "negative", "neutral")),
            "odds_status": "not_available",
            "odds_source": None,
            "join_quality": None,
            "actual": actual_values,
        }
    )
    output["won"] = np.where(pd.isna(actual_values), np.nan, actual_values.astype(float).astype(bool))
    output["profit_units"] = np.nan
    output["is_actionable"] = False
    return output[COMMON_EDGE_COLUMNS]


def _moneyline_edge_rows(attached: pd.DataFrame) -> pd.DataFrame:
    if attached.empty:
        return _empty_edge_frame()
    selected = attached.copy()
    output = pd.DataFrame(
        {
            "market": "moneyline",
            "market_type": "market_odds",
            "game_pk": selected["game_pk"],
            "date": selected.get("date"),
            "entity_id": None,
            "entity_name": np.where(selected["pick_side"].eq("home"), selected.get("home_team"), selected.get("away_team")),
            "team": np.where(selected["pick_side"].eq("home"), selected.get("home_team"), selected.get("away_team")),
            "opponent": np.where(selected["pick_side"].eq("home"), selected.get("away_team"), selected.get("home_team")),
            "side": selected["pick_side"],
            "selection": np.where(selected["pick_side"].eq("home"), "home_win", "away_win"),
            "model_probability": selected["model_probability"],
            "reference_probability": np.nan,
            "market_probability": selected["market_probability"],
            "price": selected["price"],
            "edge": selected["edge"],
            "ev": selected["ev"],
            "edge_signal": selected["edge_signal"],
            "odds_status": selected["odds_status"],
            "odds_source": selected.get("source"),
            "join_quality": selected.get("join_quality"),
            "actual": selected.get("won"),
            "won": selected.get("won"),
            "profit_units": selected.get("profit_units"),
            "is_actionable": selected.get("is_actionable", False),
        }
    )
    return output[COMMON_EDGE_COLUMNS]


def _binary_metrics(y_true: Sequence[object], probabilities: Sequence[float]) -> dict[str, Any]:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce")
    p = pd.to_numeric(pd.Series(probabilities), errors="coerce")
    valid = y.notna() & p.notna()
    if not valid.any():
        return {"rows": 0}
    y_arr = y[valid].astype(int).to_numpy()
    p_arr = np.clip(p[valid].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    result: dict[str, Any] = {
        "rows": int(len(y_arr)),
        "positive_rate": float(y_arr.mean()),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, p_arr, labels=[0, 1])),
    }
    if len(np.unique(y_arr)) > 1:
        result["auc"] = float(roc_auc_score(y_arr, p_arr))
    return result


def _metric_gate(
    *,
    test_rows: int,
    model_brier: float | None = None,
    baseline_brier: float | None = None,
    minimum_rows: int = 500,
    extra_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    reasons = list(extra_reasons)
    if test_rows < minimum_rows:
        reasons.append(f"test_rows_below_{minimum_rows}")
    if model_brier is not None and baseline_brier is not None and model_brier > baseline_brier + 1e-9:
        reasons.append("model_brier_not_better_than_reference")
    status = "candidate" if not reasons else "review"
    if test_rows == 0:
        status = "blocked"
    return {"status": status, "reasons": reasons}


def _winner_predictions(features: pd.DataFrame, result: dict[str, Any], test_season: int) -> pd.DataFrame:
    test = features.loc[features["season"].astype(int).eq(int(test_season))].copy()
    test = test.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)
    probability = result["selected_model"].predict_proba(test[result["feature_columns"]])[:, 1]
    return pd.DataFrame(
        {
            "game_pk": test["game_pk"].astype(int),
            "date": test["game_date"].map(_date_string),
            "home_team": test.get("home_team"),
            "away_team": test.get("away_team"),
            "home_probability": probability,
            "away_probability": 1.0 - probability,
            "home_win": test.get("home_win"),
        }
    )


def _total_edge_rows(result: dict[str, Any]) -> pd.DataFrame:
    predictions = result["predictions"]
    rows: list[pd.DataFrame] = []
    for line, column in ((8.5, "p_over_8_5"), (9.5, "p_over_9_5")):
        key = str(line)
        reference = result["binary_heads"][key]["constant_base_rate"]["probability"]
        actual = pd.to_numeric(predictions["total_runs"], errors="coerce") > line
        rows.append(
            _binary_edge_rows(
                predictions,
                market="total",
                market_type="statistical_probability",
                selection=f"over_{str(line).replace('.', '_')}",
                probability=predictions[column],
                reference_probability=reference,
                actual=actual.astype(int),
                game_pk=predictions["game_pk"],
                date_values=predictions["date"],
                entity_name=predictions["home_team"].astype(str) + " vs " + predictions["away_team"].astype(str),
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else _empty_edge_frame()


def _runline_edge_rows(result: dict[str, Any]) -> pd.DataFrame:
    predictions = result["predictions"]
    reference = float(result["baseline"]["probability"])
    rows = []
    rows.append(
        _binary_edge_rows(
            predictions,
            market="run_line",
            market_type="statistical_probability",
            selection="home_minus_1_5",
            probability=predictions["p_home_cover_15"],
            reference_probability=reference,
            actual=predictions["home_cover_15"],
            game_pk=predictions["game_pk"],
            date_values=predictions["date"],
            entity_name=predictions["home_team"],
            team=predictions["home_team"],
            opponent=predictions["away_team"],
            side="home",
        )
    )
    rows.append(
        _binary_edge_rows(
            predictions,
            market="run_line",
            market_type="statistical_probability",
            selection="away_plus_1_5",
            probability=predictions["p_away_cover_plus_15"],
            reference_probability=1.0 - reference,
            actual=1.0 - pd.to_numeric(predictions["home_cover_15"], errors="coerce"),
            game_pk=predictions["game_pk"],
            date_values=predictions["date"],
            entity_name=predictions["away_team"],
            team=predictions["away_team"],
            opponent=predictions["home_team"],
            side="away",
        )
    )
    return pd.concat(rows, ignore_index=True)


def _strikeout_edge_rows(result: Any) -> pd.DataFrame:
    predictions = result.predictions
    calibration = result.metrics["test"]["threshold_calibration"]
    rows = []
    for threshold, column in (("5.5", "p_over_5_5"), ("6.5", "p_over_6_5")):
        calibration_key = f"over_{threshold.replace('.', '_')}"
        reference = calibration[calibration_key]["reference_base_rate"]
        actual = pd.to_numeric(predictions["actual"], errors="coerce") >= float(threshold) + 0.5
        rows.append(
            _binary_edge_rows(
                predictions,
                market="pitcher_strikeouts",
                market_type="statistical_probability",
                selection=f"over_{threshold.replace('.', '_')}",
                probability=predictions[column],
                reference_probability=reference,
                actual=actual.astype(int),
                game_pk=predictions["game_pk"],
                date_values=predictions["date"],
                entity_id=predictions["pitcher_id"],
                entity_name=predictions["pitcher_name"],
                side=predictions["side"],
            )
        )
    return pd.concat(rows, ignore_index=True)


def _evaluate_hr(
    *,
    predictions_path: str | Path | None,
    model_path: str | Path | None,
    training_rows_path: str | Path | None,
    metrics_path: str | Path | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the existing HR artifact on its held-out date range.

    The checked-in joblib is trained only on pre-test rows.  When the training
    row cache is available, inference is therefore performed only on rows at
    or after the recorded test start date.  A recent scored prediction file is
    also retained for edge output, but is reported separately because it may
    cover only a day or two.
    """
    payload: dict[str, Any] = {"status": "unavailable", "gaps": []}
    edge_rows = _empty_edge_frame()
    metrics_payload = {}
    if metrics_path and Path(metrics_path).exists():
        metrics_payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        payload["model_version"] = metrics_payload.get("model_version")
        payload["trained_test_metrics"] = metrics_payload.get("test")
        payload["training_window"] = metrics_payload.get("training_window")

    heldout = pd.DataFrame()
    if model_path and training_rows_path and Path(model_path).exists() and Path(training_rows_path).exists():
        try:
            import joblib

            artifact = joblib.load(model_path)
            rows = _read_table(training_rows_path)
            rows["game_date"] = pd.to_datetime(rows["game_date"], errors="coerce")
            feature_columns = list(artifact.get("feature_columns", metrics_payload.get("feature_columns", [])))
            test_start = (metrics_payload.get("training_window") or {}).get("test_start_date")
            if feature_columns and test_start:
                heldout = rows.loc[rows["game_date"] >= pd.Timestamp(test_start)].copy()
                heldout = heldout.dropna(subset=["actual_home_run"])
                missing = sorted(set(feature_columns).difference(heldout.columns))
                if missing:
                    raise ValueError(f"HR rows missing model features: {missing}")
                heldout["hr_probability"] = artifact["model"].predict_proba(heldout[feature_columns])[:, 1]
                heldout["hr_probability"] = heldout["hr_probability"].clip(1e-6, 1 - 1e-6)
                payload["heldout_test"] = _binary_metrics(heldout["actual_home_run"], heldout["hr_probability"])
                payload["heldout_test"]["baseline"] = _binary_metrics(heldout["actual_home_run"], heldout["baseline_probability"])
                payload["heldout_test"]["brier_lift_vs_baseline"] = payload["heldout_test"].get("baseline", {}).get("brier", np.nan) - payload["heldout_test"].get("brier", np.nan)
                payload["heldout_test"]["min_date"] = _date_string(heldout["game_date"].min())
                payload["heldout_test"]["max_date"] = _date_string(heldout["game_date"].max())
                payload["status"] = "evaluated"
        except Exception as exc:  # optional artifact should not hide core MLB results
            payload["gaps"].append(f"heldout_hr_inference_failed:{exc}")

    recent = pd.DataFrame()
    if predictions_path and Path(predictions_path).exists():
        recent = _read_table(predictions_path)
        recent["game_date"] = pd.to_datetime(recent.get("game_date"), errors="coerce")
        actual_col = "actual_home_run" if "actual_home_run" in recent else "actual_home_runs" if "actual_home_runs" in recent else None
        if actual_col and "hr_probability" in recent:
            recent = recent.dropna(subset=[actual_col, "hr_probability"]).copy()
            payload["recent_scored_predictions"] = _binary_metrics(recent[actual_col], recent["hr_probability"])
            payload["recent_scored_predictions"]["min_date"] = _date_string(recent["game_date"].min())
            payload["recent_scored_predictions"]["max_date"] = _date_string(recent["game_date"].max())
            if not recent.empty:
                baseline = recent.get("baseline_probability", pd.Series(np.nan, index=recent.index))
                edge_rows = _binary_edge_rows(
                    recent,
                    market="batter_home_runs",
                    market_type="statistical_probability",
                    selection="home_run",
                    probability=recent["hr_probability"],
                    reference_probability=baseline,
                    actual=recent[actual_col],
                    game_pk=recent.get("game_pk"),
                    date_values=recent["game_date"],
                    entity_id=recent.get("player_id"),
                    entity_name=recent.get("player_name"),
                    team=recent.get("team"),
                    opponent=recent.get("opponent"),
                )
        else:
            payload["gaps"].append("recent_hr_predictions_have_no_scored_outcome_or_probability")
    if payload["status"] == "unavailable":
        payload["gaps"].append("no_heldout_hr_artifact_or_scored_predictions")
    return payload, edge_rows


def evaluate_mlb_vertical(
    features: pd.DataFrame,
    *,
    odds_paths: Iterable[str | Path] = (),
    validation_season: int = 2025,
    test_season: int = 2026,
    as_of_date: str | date | None = None,
    hr_predictions_path: str | Path | None = None,
    hr_model_path: str | Path | None = None,
    hr_training_rows_path: str | Path | None = None,
    hr_metrics_path: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run all MLB market evaluations and return summary plus edge rows."""
    quality = profile_feature_store(features, as_of_date=as_of_date)
    odds, odds_audit = normalize_moneyline_sources(odds_paths, target_game_pks=features["game_pk"].astype(int).tolist())
    all_edges: list[pd.DataFrame] = []

    winner = train_and_evaluate_mlb_winner(features, validation_season=validation_season, test_season=test_season)
    winner_predictions = _winner_predictions(features, winner, test_season)
    winner_with_odds, moneyline_odds_summary = attach_moneyline_edges(winner_predictions, odds)
    moneyline_edges = _moneyline_edge_rows(winner_with_odds)
    all_edges.append(moneyline_edges)
    winner_brier = winner["selected_refit_test"].get("brier")
    winner_base_brier = winner["baseline"]["test"].get("brier")

    totals = train_and_evaluate_mlb_totals(features, validation_season=validation_season, test_season=test_season, odds=None)
    all_edges.append(_total_edge_rows(totals))
    total_model_brier = totals["binary_heads"]["8.5"]["model"].get("brier")
    total_base_brier = totals["binary_heads"]["8.5"]["constant_base_rate"].get("brier")

    runline = train_and_evaluate_mlb_runline(features, validation_season=validation_season, test_season=test_season)
    all_edges.append(_runline_edge_rows(runline))
    runline_model_brier = runline["selected_refit_test"].get("brier")
    runline_base_brier = runline["baseline"]["test"].get("brier")

    sides = reshape_starter_sides(features)
    strikeouts = train_and_evaluate_strikeouts(
        sides,
        train_start_season=int(min(pd.to_numeric(features["season"], errors="coerce").dropna())),
        train_end_season=validation_season - 1,
        validation_season=validation_season,
        test_season=test_season,
    )
    all_edges.append(_strikeout_edge_rows(strikeouts))

    hr_summary, hr_edges = _evaluate_hr(
        predictions_path=hr_predictions_path,
        model_path=hr_model_path,
        training_rows_path=hr_training_rows_path,
        metrics_path=hr_metrics_path,
    )
    all_edges.append(hr_edges)

    markets = {
        "moneyline": {
            "market_type": "binary_probability",
            "model": "mlb_winner",
            "model_version": "mlb-winner-time-split",
            "test_rows": int(len(winner_predictions)),
            "test_date_range": {"min": _date_string(winner_predictions["date"].min()), "max": _date_string(winner_predictions["date"].max())},
            "metrics": winner["selected_refit_test"],
            "reference": winner["baseline"]["test"],
            "model_lift": {"brier": winner_base_brier - winner_brier if winner_brier is not None and winner_base_brier is not None else None},
            "odds": moneyline_odds_summary,
            "quality_gate": _metric_gate(test_rows=len(winner_predictions), model_brier=winner_brier, baseline_brier=winner_base_brier),
        },
        "run_line": {
            "market_type": "binary_probability",
            "model": "mlb_runline",
            "model_version": "mlb-runline-time-split",
            "test_rows": int(len(runline["predictions"])),
            "metrics": runline["selected_refit_test"],
            "reference": runline["baseline"]["test"],
            "model_lift": {"brier": runline_base_brier - runline_model_brier if runline_model_brier is not None and runline_base_brier is not None else None},
            "odds": {"status": "not_available", "reason": "free moneyline sources do not include run-line prices"},
            "quality_gate": _metric_gate(test_rows=len(runline["predictions"]), model_brier=runline_model_brier, baseline_brier=runline_base_brier),
        },
        "total": {
            "market_type": "regression_plus_binary_probability",
            "model": "mlb_totals",
            "model_version": "mlb-totals-time-split",
            "test_rows": int(len(totals["predictions"])),
            "metrics": {"regression": totals["selected_refit_test"], "binary_heads": totals["binary_heads"]},
            "reference": totals["baselines"],
            "model_lift": {"over_8_5_brier": total_base_brier - total_model_brier if total_model_brier is not None and total_base_brier is not None else None},
            "odds": {"status": "not_available", "reason": "free moneyline sources do not include totals prices"},
            "quality_gate": _metric_gate(test_rows=len(totals["predictions"]), model_brier=total_model_brier, baseline_brier=total_base_brier),
        },
        "pitcher_strikeouts": {
            "market_type": "regression_plus_binary_probability",
            "model": "mlb_strikeouts",
            "model_version": "mlb-strikeouts-time-split",
            "test_rows": int(len(strikeouts.predictions)),
            "metrics": strikeouts.metrics,
            "reference": strikeouts.metrics["test"]["k9_expected_outs_baseline"],
            "odds": {"status": "not_available", "reason": "free moneyline sources do not include player prop prices"},
            "quality_gate": _metric_gate(
                test_rows=len(strikeouts.predictions),
                extra_reasons=[
                    reason
                    for reason in [
                        "probable_starter_mismatch_rows_are_excluded",
                        "observed_weather_proxy_in_feature_store" if quality["weather_proxy_columns"] else None,
                    ]
                    if reason
                ],
            ),
        },
        "batter_home_runs": {
            "market_type": "binary_probability",
            "model": "mlb_home_run",
            "model_version": hr_summary.get("model_version", "unknown"),
            "test_rows": int((hr_summary.get("heldout_test") or {}).get("rows", 0)),
            "metrics": hr_summary,
            "odds": {"status": "not_available", "reason": "no free historical player-prop price source configured"},
            "quality_gate": _metric_gate(
                test_rows=int((hr_summary.get("heldout_test") or {}).get("rows", 0)),
                model_brier=(hr_summary.get("heldout_test") or {}).get("brier"),
                baseline_brier=((hr_summary.get("heldout_test") or {}).get("baseline") or {}).get("brier"),
                extra_reasons=hr_summary.get("gaps", []),
            ),
        },
    }

    edges = pd.concat([frame for frame in all_edges if not frame.empty], ignore_index=True) if any(not frame.empty for frame in all_edges) else _empty_edge_frame()
    if not edges.empty:
        edges = edges[COMMON_EDGE_COLUMNS].copy()
        edges = edges.sort_values(["market", "date", "game_pk", "selection"], na_position="last").reset_index(drop=True)
    gaps = [
        "run-line, total, and player-prop odds are not present in the free moneyline adapters",
        "public comparison prices are consensus/best snapshots and do not identify a single sportsbook",
    ]
    if quality["warnings"]:
        gaps.extend(quality["warnings"])
    if hr_summary.get("gaps"):
        gaps.extend(hr_summary["gaps"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": quality["as_of_date"],
        "vertical": "MLB",
        "evaluation_contract": {
            "training_features_exclude_odds": True,
            "time_split": {"validation_season": int(validation_season), "test_season": int(test_season)},
            "probability_edges_without_price_are_statistical_reference_signals": True,
            "flat_profit_units_are_one_unit_per_selected_moneyline": True,
        },
        "data_quality": quality,
        "odds": odds_audit,
        "markets": markets,
        "edges": {
            "rows": int(len(edges)),
            "positive_statistical_signals": int(((edges["market_type"].eq("statistical_probability")) & edges["edge"].gt(0.02)).sum()) if not edges.empty else 0,
            "positive_ev_moneylines": int((edges["ev"].gt(0)).sum()) if not edges.empty else 0,
            "odds_joined_rows": int(edges["odds_status"].eq("free_price_joined").sum()) if not edges.empty else 0,
        },
        "gaps": gaps,
        "production_status": "candidate" if quality["status"] == "pass" else "review",
    }
    return summary, edges


def save_edges(path: str | Path, edges: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output, index=False)


__all__ = [
    "COMMON_EDGE_COLUMNS",
    "attach_moneyline_edges",
    "evaluate_mlb_vertical",
    "normalize_moneyline_sources",
    "profile_feature_store",
    "save_edges",
    "write_json",
]
