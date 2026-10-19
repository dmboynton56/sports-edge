"""Walk-forward NFL v2 bucket / price-rule backtest.

Analysis only. Does not write live artifacts or change serving.

Prices are nflverse closing spreads and moneylines. Book probabilities are
no-vig from the two moneylines. Spread bets use the home-perspective close
(`home_spread = -spread_line`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.models.nfl_experiment import (
    CalibratedProbabilityModel,
    ProbabilityCalibrator,
    candidate_specs,
    fit_candidate,
    fit_production_shadow_bundle,
)
from src.models.nfl_health import american_profit
from src.models.nfl_v2 import FEATURE_COLUMNS, no_vig_home_probability


BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261019
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "experiments" / "nfl_v2.yaml"
TOSS_UP_LOW = 0.40
TOSS_UP_HIGH = 0.60
EDGE_THRESHOLDS = (0.03, 0.05, 0.08)
MIN_SEASONS_FOR_STRATEGY = 3
MIN_SEASON_N_FOR_STRATEGY = 20
DEFAULT_MINUS_110 = -110.0

HOME_PROB_BUCKETS: tuple[tuple[str, float, float, bool, bool], ...] = (
    ("0-20", 0.00, 0.20, True, False),
    ("20-30", 0.20, 0.30, True, False),
    ("30-40", 0.30, 0.40, True, False),
    ("40-60", 0.40, 0.60, True, True),
    ("60-70", 0.60, 0.70, False, True),
    ("70-80", 0.70, 0.80, False, True),
    ("80+", 0.80, 1.01, False, True),
)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    question: str
    family: str
    bucket: str
    side: str
    market: str
    description: str


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def model_side(home_win_prob: object) -> str | None:
    probability = _finite(home_win_prob)
    if probability is None:
        return None
    if probability > 0.5:
        return "home"
    if probability < 0.5:
        return "away"
    return None


def book_favorite_side_spread(home_spread: object) -> str | None:
    spread = _finite(home_spread)
    if spread is None or abs(spread) < 1e-9:
        return None
    return "home" if spread < 0 else "away"


def book_favorite_side_moneyline(home_moneyline: object, away_moneyline: object) -> str | None:
    home = _finite(home_moneyline)
    away = _finite(away_moneyline)
    if home is None or away is None:
        return None
    if home == away:
        return None
    # More negative American price is the favorite.
    return "home" if home < away else "away"


def opposite_side(side: str | None) -> str | None:
    if side == "home":
        return "away"
    if side == "away":
        return "home"
    return None


def model_side_edge(model_home_p: object, book_home_p: object) -> float | None:
    """Model favorite probability minus no-vig book probability on that same side."""
    model = _finite(model_home_p)
    book = _finite(book_home_p)
    if model is None or book is None:
        return None
    if model >= 0.5:
        return model - book
    return book - model


def home_prob_bucket(home_win_prob: object) -> str | None:
    probability = _finite(home_win_prob)
    if probability is None:
        return None
    for name, low, high, left_closed, right_closed in HOME_PROB_BUCKETS:
        left = probability >= low if left_closed else probability > low
        right = probability <= high if right_closed else probability < high
        if left and right:
            return name
    return None


def in_toss_up_band(home_win_prob: object) -> bool:
    return home_prob_bucket(home_win_prob) == "40-60"


def grade_ats(actual_margin: object, home_spread: object, side: str) -> str | None:
    margin = _finite(actual_margin)
    spread = _finite(home_spread)
    if margin is None or spread is None or side not in {"home", "away"}:
        return None
    edge = margin + spread
    if abs(edge) < 1e-9:
        return "push"
    home_covers = edge > 0
    won = home_covers if side == "home" else not home_covers
    return "win" if won else "loss"


def grade_moneyline(home_score: object, away_score: object, side: str) -> str | None:
    home = _finite(home_score)
    away = _finite(away_score)
    if home is None or away is None or side not in {"home", "away"}:
        return None
    if home == away:
        return "push"
    home_won = home > away
    won = home_won if side == "home" else not home_won
    return "win" if won else "loss"


def units_from_result(result: str | None, price: object) -> float | None:
    if result is None:
        return None
    if result == "push":
        return 0.0
    american = _finite(price)
    if american is None or american == 0:
        return None
    return american_profit(american, result == "win")


def side_price(row: Mapping[str, Any], side: str, market: str) -> float | None:
    if side not in {"home", "away"}:
        return None
    if market == "ats_minus_110":
        return DEFAULT_MINUS_110
    if market == "ats_actual":
        column = "home_spread_odds" if side == "home" else "away_spread_odds"
        return _finite(row.get(column)) or DEFAULT_MINUS_110
    if market == "ml":
        column = "home_moneyline" if side == "home" else "away_moneyline"
        return _finite(row.get(column))
    raise ValueError(f"Unknown market: {market}")


def resolve_side(row: Mapping[str, Any], side_key: str, market: str) -> str | None:
    if side_key == "model":
        return model_side(row.get("my_home_win_prob"))
    if side_key == "book_fav":
        if market == "ml":
            return book_favorite_side_moneyline(row.get("home_moneyline"), row.get("away_moneyline"))
        return book_favorite_side_spread(row.get("home_spread"))
    if side_key == "book_dog":
        favorite = resolve_side(row, "book_fav", market)
        return opposite_side(favorite)
    if side_key in {"home", "away"}:
        return side_key
    raise ValueError(f"Unknown side: {side_key}")


def grade_bet(row: Mapping[str, Any], side: str, market: str) -> tuple[str | None, float | None]:
    price = side_price(row, side, market)
    if market == "ml":
        result = grade_moneyline(row.get("home_score"), row.get("away_score"), side)
    else:
        result = grade_ats(row.get("actual_margin"), row.get("home_spread"), side)
    return result, units_from_result(result, price)


def load_live_specs(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    selection = config["live_selection"]

    def _named(task: str, name: str) -> dict[str, Any]:
        matches = [spec for spec in candidate_specs(config, task) if spec["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one {task} candidate named {name!r}.")
        return matches[0]

    return {
        "config": config,
        "selection": selection,
        "probability_spec": _named("probability", selection["probability_candidate"]),
        "margin_spec": _named("margin", selection["margin_candidate"]),
        "calibration_method": str(selection["calibration_method"]),
    }


def train_walk_forward_bundle(
    train: pd.DataFrame,
    *,
    probability_spec: Mapping[str, Any],
    margin_spec: Mapping[str, Any],
    calibration_method: str = "isotonic",
) -> dict[str, Any]:
    """Fit the live v2 candidate on prior seasons only.

    Isotonic uses rolling out-of-fold seasons inside ``train``. A single prior
    season cannot support OOF calibration; that fold uses raw probabilities.
    """
    if train.empty:
        raise ValueError("Walk-forward train frame is empty.")
    seasons = sorted(int(value) for value in pd.to_numeric(train["season"], errors="coerce").dropna().unique())
    if len(seasons) >= 2:
        first_oof = int(seasons[1])
        bundle = fit_production_shadow_bundle(
            train,
            probability_spec,
            margin_spec,
            calibration_method,
            first_oof_season=first_oof,
        )
        return {
            **bundle,
            "calibration": calibration_method,
            "calibration_mode": "oof_seasons",
            "oof_seasons": bundle.get("oof_seasons"),
            "trained_through": int(train["season"].max()),
            "train_seasons": seasons,
            "train_rows": int(len(train)),
        }

    probability_model = fit_candidate(probability_spec, train, task="probability")
    return {
        "probability_model": CalibratedProbabilityModel(probability_model, ProbabilityCalibrator(method="none")),
        "margin_model": fit_candidate(margin_spec, train, task="margin"),
        "calibrator": ProbabilityCalibrator(method="none"),
        "calibration": "none",
        "calibration_mode": "single_prior_season_no_oof",
        "oof_seasons": [],
        "trained_through": int(train["season"].max()),
        "train_seasons": seasons,
        "train_rows": int(len(train)),
    }


def predict_bundle(frame: pd.DataFrame, bundle: Mapping[str, Any]) -> pd.DataFrame:
    needed = frame[FEATURE_COLUMNS]
    if needed.isna().any().any():
        missing = needed.isna().sum()
        raise ValueError(f"Missing features: {missing[missing > 0].to_dict()}")
    out = frame.copy()
    out["my_home_win_prob"] = np.asarray(bundle["probability_model"].predict_probability(out), dtype=float)
    out["predicted_margin"] = np.asarray(bundle["margin_model"].predict_margin(out), dtype=float)
    out["my_spread"] = -out["predicted_margin"]
    return out


def attach_schedule_prices(features: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    keep = [
        column
        for column in (
            "game_id",
            "spread_line",
            "home_spread_odds",
            "away_spread_odds",
            "home_moneyline",
            "away_moneyline",
            "total_line",
        )
        if column in schedules.columns
    ]
    prices = schedules[keep].copy()
    prices["game_id"] = prices["game_id"].astype(str)
    prices = prices.drop_duplicates("game_id")
    out = features.copy()
    out["game_id"] = out["game_id"].astype(str)
    out = out.merge(prices, on="game_id", how="left", suffixes=("", "_sched"))
    spread_line = pd.to_numeric(out.get("spread_line"), errors="coerce")
    out["home_spread"] = -spread_line
    out["actual_margin"] = pd.to_numeric(out.get("home_score"), errors="coerce") - pd.to_numeric(
        out.get("away_score"), errors="coerce"
    )
    out["market_home_prob"] = [
        no_vig_home_probability(home, away)
        for home, away in zip(out.get("home_moneyline"), out.get("away_moneyline"))
    ]
    return out


def annotate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "actual_margin" not in out.columns or pd.to_numeric(out.get("actual_margin"), errors="coerce").isna().all():
        out["actual_margin"] = pd.to_numeric(out.get("home_score"), errors="coerce") - pd.to_numeric(
            out.get("away_score"), errors="coerce"
        )
    out["home_prob_bucket"] = [home_prob_bucket(value) for value in out["my_home_win_prob"]]
    out["model_side"] = [model_side(value) for value in out["my_home_win_prob"]]
    out["book_fav_spread"] = [book_favorite_side_spread(value) for value in out["home_spread"]]
    out["book_fav_ml"] = [
        book_favorite_side_moneyline(home, away)
        for home, away in zip(out.get("home_moneyline"), out.get("away_moneyline"))
    ]
    out["model_minus_book"] = pd.to_numeric(out["my_home_win_prob"], errors="coerce") - pd.to_numeric(
        out["market_home_prob"], errors="coerce"
    )
    out["model_side_edge"] = [
        model_side_edge(model, book) for model, book in zip(out["my_home_win_prob"], out["market_home_prob"])
    ]
    out["toss_up_40_60"] = out["home_prob_bucket"].eq("40-60")
    return out


def define_rules() -> list[Rule]:
    rules: list[Rule] = []
    for side in ("model", "book_dog", "book_fav"):
        for market, market_label in (
            ("ats_minus_110", "ATS at -110"),
            ("ats_actual", "ATS at actual/close price"),
            ("ml", "moneyline at actual/close price"),
        ):
            rules.append(
                Rule(
                    rule_id=f"q1_toss_up_{side}_{market}",
                    question="q1",
                    family="toss_up_40_60",
                    bucket="40-60",
                    side=side,
                    market=market,
                    description=f"Toss-up 40-60: back {side.replace('_', ' ')} on {market_label}",
                )
            )

    bucket_specs = (
        ("under_40", "under 40", "q2"),
        ("over_60", "over 60", "q2"),
        ("0-20", "0-20 (80+ mirror)", "q2"),
        ("20-30", "20-30 (70-80 mirror)", "q2"),
        ("30-40", "30-40 (60-70 mirror)", "q2"),
        ("60-70", "60-70", "q2"),
        ("70-80", "70-80", "q2"),
        ("80+", "80+", "q2"),
    )
    for bucket, label, question in bucket_specs:
        for side in ("model", "book_fav"):
            for market, market_label in (
                ("ats_minus_110", "ATS at -110"),
                ("ats_actual", "ATS at actual/close price"),
                ("ml", "moneyline at actual/close price"),
            ):
                rules.append(
                    Rule(
                        rule_id=f"q2_{bucket}_{side}_{market}",
                        question=question,
                        family="raw_bucket",
                        bucket=bucket,
                        side=side,
                        market=market,
                        description=f"{label}: back {side.replace('_', ' ')} on {market_label}",
                    )
                )

    for threshold in EDGE_THRESHOLDS:
        points = int(round(threshold * 100))
        for market, market_label in (
            ("ats_minus_110", "ATS at -110"),
            ("ats_actual", "ATS at actual/close price"),
            ("ml", "moneyline at actual/close price"),
        ):
            rules.append(
                Rule(
                    rule_id=f"q2_edge_{points}_{market}",
                    question="q2",
                    family="model_minus_book",
                    bucket=f"edge>={points}",
                    side="model",
                    market=market,
                    description=(
                        f"Model-side edge vs no-vig book ≥ {points}pp: back model side on {market_label}"
                    ),
                )
            )
            rules.append(
                Rule(
                    rule_id=f"q2_edge_{points}_in_toss_up_{market}",
                    question="q1",
                    family="model_minus_book_in_toss_up",
                    bucket=f"40-60 & edge>={points}",
                    side="model",
                    market=market,
                    description=(
                        f"Toss-up 40-60 and model-side edge ≥ {points}pp: back model side on {market_label}"
                    ),
                )
            )
    return rules


def _bucket_mask(frame: pd.DataFrame, bucket: str) -> pd.Series:
    probability = pd.to_numeric(frame["my_home_win_prob"], errors="coerce")
    if bucket == "40-60":
        return frame["home_prob_bucket"].eq("40-60")
    if bucket == "under_40":
        return probability < 0.40
    if bucket == "over_60":
        return probability > 0.60
    if bucket.startswith("edge>="):
        points = int(bucket.split(">=")[1])
        edge = pd.to_numeric(frame["model_side_edge"], errors="coerce")
        return edge >= points / 100.0
    if " & edge>=" in bucket:
        raw_bucket, edge_part = bucket.split(" & edge>=", 1)
        return _bucket_mask(frame, raw_bucket) & _bucket_mask(frame, f"edge>={edge_part}")
    return frame["home_prob_bucket"].eq(bucket)


def select_rule_rows(frame: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    mask = _bucket_mask(frame, rule.bucket)
    return frame.loc[mask].copy()


def score_rule_rows(frame: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    rows = select_rule_rows(frame, rule)
    results: list[str | None] = []
    units: list[float | None] = []
    sides: list[str | None] = []
    for record in rows.to_dict("records"):
        side = resolve_side(record, rule.side, rule.market)
        sides.append(side)
        if side is None:
            results.append(None)
            units.append(None)
            continue
        result, profit = grade_bet(record, side, rule.market)
        results.append(result)
        units.append(profit)
    rows = rows.copy()
    rows["bet_side"] = sides
    rows["bet_result"] = results
    rows["bet_units"] = units
    return rows


def bootstrap_roi(
    units: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | None]:
    observed = np.asarray(units, dtype=float)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return {"mean": None, "lower_95": None, "upper_95": None, "p_value": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = observed[rng.integers(0, observed.size, observed.size)].mean()
    mean = float(observed.mean())
    if mean >= 0:
        count = float(np.sum(means <= 0.0))
    else:
        count = float(np.sum(means >= 0.0))
    p_value = min(1.0, 2.0 * (count + 1.0) / (samples + 1.0))
    return {
        "mean": mean,
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "p_value": float(p_value),
        "n": int(observed.size),
    }


def summarize_bets(rows: pd.DataFrame) -> dict[str, Any]:
    graded = rows[rows["bet_result"].isin(["win", "loss", "push"])].copy()
    units = pd.to_numeric(graded["bet_units"], errors="coerce")
    wins = int((graded["bet_result"] == "win").sum())
    losses = int((graded["bet_result"] == "loss").sum())
    pushes = int((graded["bet_result"] == "push").sum())
    risked = wins + losses
    net = float(units.sum()) if units.notna().any() else 0.0
    roi = (net / risked) if risked else None
    risked_units = units[graded["bet_result"].isin(["win", "loss"])]
    return {
        "n": int(len(graded)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "record": f"{wins}-{losses}-{pushes}",
        "units": net if (risked or pushes) else None,
        "roi": roi,
        "bootstrap": bootstrap_roi(risked_units.to_numpy(dtype=float) if len(risked_units) else []),
        "skipped": int(len(rows) - len(graded)),
    }


def summarize_rule(frame: pd.DataFrame, rule: Rule) -> dict[str, Any]:
    scored = score_rule_rows(frame, rule)
    summary = summarize_bets(scored)
    by_season = []
    if not scored.empty and "season" in scored.columns:
        for season, group in scored.groupby(scored["season"].astype(int), sort=True):
            block = summarize_bets(group)
            block["season"] = int(season)
            by_season.append(block)
    payload = asdict(rule)
    payload.update(summary)
    payload["by_season"] = by_season
    return payload


def season_holds(summary: Mapping[str, Any], *, min_seasons: int = MIN_SEASONS_FOR_STRATEGY, min_n: int = MIN_SEASON_N_FOR_STRATEGY) -> dict[str, Any]:
    usable = [row for row in summary.get("by_season") or [] if int(row.get("n") or 0) >= min_n and row.get("roi") is not None]
    if len(usable) < min_seasons:
        return {
            "holds": False,
            "reason": f"fewer than {min_seasons} seasons with n>={min_n}",
            "seasons_used": [row["season"] for row in usable],
            "positive_seasons": [],
            "negative_seasons": [],
        }
    positive = [row["season"] for row in usable if row["roi"] > 0]
    negative = [row["season"] for row in usable if row["roi"] < 0]
    sign = 1 if len(positive) >= min_seasons and len(negative) == 0 else -1 if len(negative) >= min_seasons and len(positive) == 0 else 0
    return {
        "holds": sign != 0,
        "reason": "same-sign ROI across qualifying seasons" if sign else "sign flips or mixed seasons",
        "seasons_used": [row["season"] for row in usable],
        "positive_seasons": positive,
        "negative_seasons": negative,
        "sign": sign,
    }


def ci_excludes_zero(summary: Mapping[str, Any]) -> bool:
    interval = summary.get("bootstrap") or {}
    lower, upper = interval.get("lower_95"), interval.get("upper_95")
    if lower is None or upper is None:
        return False
    return lower > 0 or upper < 0


def strategy_verdict(summary: Mapping[str, Any], *, n_tests: int, sample_kind: str) -> dict[str, Any]:
    interval = summary.get("bootstrap") or {}
    p_value = interval.get("p_value")
    adjusted = None if p_value is None else min(1.0, float(p_value) * max(1, int(n_tests)))
    holds = season_holds(summary)
    is_oos = sample_kind == "out_of_sample"
    qualifies = bool(
        is_oos
        and ci_excludes_zero(summary)
        and holds["holds"]
        and adjusted is not None
        and adjusted < 0.05
        and (summary.get("n") or 0) > 0
    )
    return {
        "qualifies_as_strategy": qualifies,
        "sample_kind": sample_kind,
        "ci_excludes_zero": ci_excludes_zero(summary),
        "bonferroni_n_tests": int(n_tests),
        "raw_p_value": p_value,
        "bonferroni_p_value": adjusted,
        "season_consistency": holds,
        "reason": (
            "Holds across seasons, 95% CI excludes 0, and Bonferroni-adjusted p < 0.05"
            if qualifies
            else "Does not meet the strategy bar (OOS + multi-season + CI excludes 0 + multiple-comparison)"
        ),
    }


def evaluate_rules(
    frame: pd.DataFrame,
    rules: Sequence[Rule] | None = None,
    *,
    sample_kind: str,
) -> list[dict[str, Any]]:
    rules = list(rules or define_rules())
    summaries = [summarize_rule(frame, rule) for rule in rules]
    n_tests = len(summaries)
    for summary in summaries:
        summary["verdict"] = strategy_verdict(summary, n_tests=n_tests, sample_kind=sample_kind)
        summary["sample_kind"] = sample_kind
    return summaries


def walk_forward_predictions(
    features: pd.DataFrame,
    *,
    test_seasons: Sequence[int],
    specs: Mapping[str, Any],
    predict_season: Callable[[pd.DataFrame, Mapping[str, Any], int], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Train on prior seasons and predict each test season.

    ``predict_season`` is an optional override used by tests. The default uses
    ``train_walk_forward_bundle`` plus ``predict_bundle``.
    """
    parts: list[pd.DataFrame] = []
    for season in test_seasons:
        train = features[pd.to_numeric(features["season"], errors="coerce") < int(season)].copy()
        test = features[pd.to_numeric(features["season"], errors="coerce") == int(season)].copy()
        if train.empty or test.empty:
            continue
        if predict_season is not None:
            predicted = predict_season(train, specs, int(season))
        else:
            bundle = train_walk_forward_bundle(
                train,
                probability_spec=specs["probability_spec"],
                margin_spec=specs["margin_spec"],
                calibration_method=str(specs["calibration_method"]),
            )
            predicted = predict_bundle(test, bundle)
            predicted["trained_through"] = bundle["trained_through"]
            predicted["calibration"] = bundle["calibration"]
            predicted["calibration_mode"] = bundle["calibration_mode"]
        predicted["prediction_source"] = "walk_forward_prior_seasons"
        predicted["sample_kind"] = "out_of_sample"
        predicted["test_season"] = int(season)
        parts.append(predicted)
    if not parts:
        return features.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def line_shopping_note(frame: pd.DataFrame) -> dict[str, Any]:
    """nflverse is one consensus close. We do not have multi-book historicals here."""
    spread_odds = frame[["home_spread_odds", "away_spread_odds"]].apply(pd.to_numeric, errors="coerce")
    return {
        "multi_book_rows": 0,
        "books_available": ["nflverse_pfr_consensus"],
        "can_measure_1pt5_disagreement": False,
        "spread_price_coverage": float(spread_odds.notna().all(axis=1).mean()) if len(frame) else 0.0,
        "moneyline_coverage": float(
            pd.to_numeric(frame.get("home_moneyline"), errors="coerce").notna().mean()
        )
        if len(frame)
        else 0.0,
        "note": (
            "Line shopping (±1.5 across books) is not measurable from these files. "
            "nflverse exposes one PFR/consensus close (`spread_line`, `home_spread_odds`, "
            "`away_spread_odds`, moneylines), not a book panel. Serving toss-up copy in "
            "#42 mentions books disagreeing at ±1.5; that is a live-book observation, not "
            "a historical series in this backtest."
        ),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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
