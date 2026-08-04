#!/usr/bin/env python3
"""Build the MLB v3 diagnosis artifacts from committed prediction caches only."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[5]
CACHE_DIR = REPO_ROOT / "data-core/notebooks/cache"
RUN_DIR = REPO_ROOT / ".cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics"
OUTPUT_JSON = CACHE_DIR / "mlb_ml_diagnosis_2026_ytd.json"
OUTPUT_REPORT = RUN_DIR / "artifacts/diagnosis-v3.md"
PREDICTION_FILES = {
    "2026_ytd": CACHE_DIR / "mlb_backtest_predictions_2026_ytd_free.csv",
    "2025": CACHE_DIR / "mlb_backtest_predictions_2025_free.csv",
}
METRICS_FILES = {
    "2026_ytd": CACHE_DIR / "mlb_backtest_metrics_2026_ytd_free.json",
    "2025": CACHE_DIR / "mlb_backtest_metrics_2025_free.json",
}
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 11)
EDGE_BUCKET_ORDER = ["0-2%", "2-4%", "4-6%", "6%+"]


def native(value: Any) -> Any:
    """Convert numpy/pandas scalars into strict JSON-compatible values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def wilson_interval(wins: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    rate = wins / n
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return [center - margin, center + margin]


def calibration(df: pd.DataFrame) -> tuple[list[dict[str, Any]], float]:
    work = df.copy()
    work["calibration_bin"] = pd.cut(
        work["home_win_prob"],
        CALIBRATION_EDGES,
        right=False,
        include_lowest=True,
    )
    rows: list[dict[str, Any]] = []
    weighted_error = 0.0
    for interval, group in work.groupby("calibration_bin", observed=False):
        n = len(group)
        mean_pred = group["home_win_prob"].mean() if n else None
        empirical = group["home_win"].mean() if n else None
        if n:
            weighted_error += n * abs(float(mean_pred) - float(empirical))
        rows.append(
            {
                "bin": f"{interval.left:.1f}-{interval.right:.1f}",
                "lower": native(interval.left),
                "upper": native(interval.right),
                "n": n,
                "mean_pred": native(mean_pred),
                "empirical_home_win_rate": native(empirical),
                "calibration_gap": native(
                    abs(float(mean_pred) - float(empirical)) if n else None
                ),
            }
        )
    return rows, weighted_error / len(work)


def model_metrics(df: pd.DataFrame, probability_col: str) -> dict[str, float]:
    y = df["home_win"].astype(int)
    probability = df[probability_col]
    return {
        "accuracy": float(accuracy_score(y, probability >= 0.5)),
        "brier": float(brier_score_loss(y, probability)),
        "auc": float(roc_auc_score(y, probability)),
    }


def summarize_bets(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    wins = int(df["pick_won"].sum())
    units = float(df["profit"].sum())
    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n if n else None,
        "win_rate_wilson_95": wilson_interval(wins, n),
        "units": units,
        "roi": units / n if n else None,
    }


def grouped_bets(
    df: pd.DataFrame,
    labels: pd.Series,
    order: list[Any] | None = None,
    label_formatter: Callable[[Any], str] = str,
) -> list[dict[str, Any]]:
    work = df.assign(_group=labels)
    groups = {key: group for key, group in work.groupby("_group", observed=False)}
    keys = order if order is not None else list(groups)
    output = []
    for key in keys:
        group = groups.get(key, work.iloc[0:0])
        output.append({"group": label_formatter(key), **summarize_bets(group)})
    return output


def edge_deciles(df: pd.DataFrame) -> list[dict[str, Any]]:
    work = df.copy()
    work["decile"] = pd.qcut(
        work["abs_edge"], 10, labels=False, duplicates="drop"
    ) + 1
    output = []
    for decile, group in work.groupby("decile"):
        output.append(
            {
                "decile": int(decile),
                "min_abs_edge": float(group["abs_edge"].min()),
                "max_abs_edge": float(group["abs_edge"].max()),
                **summarize_bets(group),
            }
        )
    return output


def error_buckets(df: pd.DataFrame) -> dict[str, Any]:
    confidence = np.maximum(df["home_win_prob"], 1 - df["home_win_prob"])
    confidence_edges = [0.5, 0.525, 0.55, 0.575, 0.60, 0.65, 1.0000001]
    confidence_labels = pd.cut(
        confidence, confidence_edges, right=False, include_lowest=True
    )
    confidence_rows = []
    for interval, group in df.assign(_band=confidence_labels).groupby(
        "_band", observed=False
    ):
        summary = summarize_bets(group)
        confidence_rows.append(
            {
                "band": f"{interval.left:.3f}-{interval.right:.3f}",
                **summary,
                "loss_rate": 1 - summary["win_rate"] if summary["n"] else None,
            }
        )

    home_prediction_rows = []
    home_bands = pd.cut(
        df["home_win_prob"], CALIBRATION_EDGES, right=False, include_lowest=True
    )
    for interval, group in df.assign(_band=home_bands).groupby(
        "_band", observed=False
    ):
        predicted_home = group["home_win_prob"] >= 0.5
        errors = predicted_home != group["home_win"].astype(bool)
        home_prediction_rows.append(
            {
                "band": f"{interval.left:.1f}-{interval.right:.1f}",
                "n": len(group),
                "classification_loss_rate": float(errors.mean()) if len(group) else None,
            }
        )

    picked_price_band = pd.cut(
        df["pick_price"],
        [-np.inf, -200, -110, 0, 150, np.inf],
        labels=[
            "heavy favorite (<= -200)",
            "favorite (-199 to -110)",
            "short favorite (-109 to -1)",
            "underdog (+1 to +150)",
            "long underdog (> +150)",
        ],
    )
    price_rows = grouped_bets(
        df,
        picked_price_band,
        order=list(picked_price_band.cat.categories),
    )
    for row in price_rows:
        row["loss_rate"] = 1 - row["win_rate"] if row["n"] else None

    market_pick_home = df["market_home_prob"] >= 0.5
    model_pick_home = df["home_win_prob"] >= 0.5
    disagreement = market_pick_home != model_pick_home
    cohorts = {
        "home_heavy_favorite_picks": (df["pick_side"] == "home")
        & (df["home_moneyline"] <= -200),
        "away_heavy_favorite_picks": (df["pick_side"] == "away")
        & (df["away_moneyline"] <= -200),
        "model_market_direction_disagreement": disagreement,
        "pick_confidence_0.575_to_0.600": confidence.between(
            0.575, 0.600, inclusive="left"
        ),
    }
    cohort_rows = []
    for name, mask in cohorts.items():
        group = df[mask]
        summary = summarize_bets(group)
        row = {
            "cohort": name,
            **summary,
            "loss_rate": 1 - summary["win_rate"] if summary["n"] else None,
        }
        if name == "model_market_direction_disagreement" and len(group):
            actual_home = group["home_win"].astype(bool)
            row["model_accuracy"] = float(
                (model_pick_home[mask] == actual_home).mean()
            )
            row["market_accuracy"] = float(
                (market_pick_home[mask] == actual_home).mean()
            )
        cohort_rows.append(row)
    return {
        "home_win_probability_bands": home_prediction_rows,
        "pick_confidence_bands": confidence_rows,
        "picked_price_bands": price_rows,
        "selected_cohorts": cohort_rows,
    }


def analyze(label: str, prediction_path: Path, metrics_path: Path) -> dict[str, Any]:
    df = pd.read_csv(prediction_path, parse_dates=["game_date"])
    odds = df.dropna(
        subset=["market_home_prob", "home_moneyline", "away_moneyline", "profit"]
    ).copy()
    committed = json.loads(metrics_path.read_text())
    calibration_bins, ece = calibration(df)
    model_all = model_metrics(df, "home_win_prob")
    model_same_rows = model_metrics(odds, "home_win_prob")
    market_same_rows = model_metrics(odds, "market_home_prob")
    model_pick_home = odds["home_win_prob"] >= 0.5
    market_pick_home = odds["market_home_prob"] >= 0.5
    disagreement = model_pick_home != market_pick_home

    favorite_type = pd.Series(
        np.select(
            [odds["pick_price"] < 0, odds["pick_price"] > 0],
            ["favorite", "underdog"],
            default="even",
        ),
        index=odds.index,
    )
    months = odds["game_date"].dt.strftime("%Y-%m")
    side_order = [side for side in ["home", "away"] if side in set(odds["pick_side"])]
    favorite_order = [
        value for value in ["favorite", "underdog", "even"] if value in set(favorite_type)
    ]
    month_order = sorted(months.unique())

    month_performance = []
    for month, group in df.groupby(df["game_date"].dt.strftime("%Y-%m")):
        month_performance.append(
            {
                "month": month,
                "n": len(group),
                **model_metrics(group, "home_win_prob"),
            }
        )

    edge_groups = grouped_bets(
        odds, odds["edge_bucket"], order=EDGE_BUCKET_ORDER
    )
    for row in edge_groups:
        row["edge_bucket"] = row.pop("group")

    flat = summarize_bets(odds)
    committed_flat_roi = committed["odds_summary"]["flat_roi"]
    analysis = {
        "label": label,
        "source": {
            "predictions": str(prediction_path.relative_to(REPO_ROOT)),
            "metrics": str(metrics_path.relative_to(REPO_ROOT)),
        },
        "row_count": len(df),
        "odds_row_count": len(odds),
        "date_range": [
            df["game_date"].min().date().isoformat(),
            df["game_date"].max().date().isoformat(),
        ],
        "calibration_bins": calibration_bins,
        "ece_10": ece,
        "pred_dist": {
            "p5": float(df["home_win_prob"].quantile(0.05)),
            "p50": float(df["home_win_prob"].quantile(0.50)),
            "p95": float(df["home_win_prob"].quantile(0.95)),
            "min": float(df["home_win_prob"].min()),
            "max": float(df["home_win_prob"].max()),
            "share_between_0_45_and_0_60": float(
                df["home_win_prob"].between(0.45, 0.60, inclusive="both").mean()
            ),
        },
        "market_comparison": {
            "n": len(odds),
            "probability_correlation": float(
                odds["home_win_prob"].corr(odds["market_home_prob"])
            ),
            "mean_absolute_probability_gap": float(
                (odds["home_win_prob"] - odds["market_home_prob"]).abs().mean()
            ),
            "direction_agreement_rate": float((model_pick_home == market_pick_home).mean()),
            "model_all_rows": model_all,
            "model_same_rows": model_same_rows,
            "market_same_rows": market_same_rows,
            "model_minus_market": {
                key: model_same_rows[key] - market_same_rows[key]
                for key in ["accuracy", "brier", "auc"]
            },
            "direction_disagreement": {
                "n": int(disagreement.sum()),
                "model_accuracy": float(
                    accuracy_score(
                        odds.loc[disagreement, "home_win"],
                        model_pick_home[disagreement],
                    )
                ),
                "market_accuracy": float(
                    accuracy_score(
                        odds.loc[disagreement, "home_win"],
                        market_pick_home[disagreement],
                    )
                ),
            },
        },
        "flat_roi": flat,
        "committed_flat_roi": committed_flat_roi,
        "flat_roi_difference": flat["roi"] - committed_flat_roi,
        "roi_splits": {
            "pick_side": grouped_bets(
                odds, odds["pick_side"], order=side_order
            ),
            "favorite_vs_underdog": grouped_bets(
                odds, favorite_type, order=favorite_order
            ),
            "month": grouped_bets(odds, months, order=month_order),
            "abs_edge_deciles": edge_deciles(odds),
        },
        "edge_buckets": edge_groups,
        "error_buckets": error_buckets(odds),
        "model_month_performance": month_performance,
    }
    return analysis


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def units(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def build_report(current: dict[str, Any], prior: dict[str, Any]) -> str:
    market = current["market_comparison"]
    prior_market = prior["market_comparison"]
    green = next(
        row for row in current["edge_buckets"] if row["edge_bucket"] == "0-2%"
    )
    green_prior = next(
        row for row in prior["edge_buckets"] if row["edge_bucket"] == "0-2%"
    )

    reliability_rows = [
        [
            row["bin"],
            row["n"],
            pct(row["mean_pred"]),
            pct(row["empirical_home_win_rate"]),
            pct(row["calibration_gap"]),
        ]
        for row in current["calibration_bins"]
    ]
    market_rows = []
    for label, analysis in [("2026 YTD", current), ("2025", prior)]:
        comparison = analysis["market_comparison"]
        market_rows.append(
            [
                label,
                comparison["n"],
                num(comparison["probability_correlation"]),
                pct(comparison["mean_absolute_probability_gap"]),
                pct(comparison["model_same_rows"]["accuracy"]),
                num(comparison["model_same_rows"]["brier"], 4),
                num(comparison["model_same_rows"]["auc"], 4),
                pct(comparison["market_same_rows"]["accuracy"]),
                num(comparison["market_same_rows"]["brier"], 4),
                num(comparison["market_same_rows"]["auc"], 4),
            ]
        )

    def bet_rows(rows: list[dict[str, Any]], key: str = "group") -> list[list[Any]]:
        return [
            [
                row[key],
                row["n"],
                row["wins"],
                pct(row["win_rate"]),
                units(row["units"]),
                pct(row["roi"]),
            ]
            for row in rows
        ]

    edge_rows = bet_rows(current["edge_buckets"], "edge_bucket")
    decile_rows = [
        [
            row["decile"],
            f"{pct(row['min_abs_edge'])}–{pct(row['max_abs_edge'])}",
            row["n"],
            row["wins"],
            units(row["units"]),
            pct(row["roi"]),
        ]
        for row in current["roi_splits"]["abs_edge_deciles"]
    ]
    confidence_rows = [
        [
            row["band"],
            row["n"],
            pct(row["loss_rate"]),
            units(row["units"]),
            pct(row["roi"]),
        ]
        for row in current["error_buckets"]["pick_confidence_bands"]
    ]
    price_rows = [
        [
            row["group"],
            row["n"],
            pct(row["loss_rate"]),
            units(row["units"]),
            pct(row["roi"]),
        ]
        for row in current["error_buckets"]["picked_price_bands"]
    ]
    month_model_rows = [
        [
            row["month"],
            row["n"],
            pct(row["accuracy"]),
            num(row["brier"], 4),
            num(row["auc"], 4),
        ]
        for row in current["model_month_performance"]
    ]
    disagreement_2026 = market["direction_disagreement"]
    disagreement_2025 = prior_market["direction_disagreement"]
    ci = green["win_rate_wilson_95"]
    prior_ci = green_prior["win_rate_wilson_95"]

    return f"""# MLB moneyline v3 underperformance diagnosis

## Technical summary

- **v3 adds little reliable discrimination beyond the free line.** On the same 673 games, the model's AUC was {market['model_same_rows']['auc']:.4f} versus {market['market_same_rows']['auc']:.4f} for the market. When their favored sides differed, v3 was correct {pct(disagreement_2026['model_accuracy'])} of the time versus {pct(disagreement_2026['market_accuracy'])} for the market; the same split persisted in 2025 ({pct(disagreement_2025['model_accuracy'])} versus {pct(disagreement_2025['market_accuracy'])}).
- **The −3.1% flat ROI is concentrated in April, favorite picks, and two edge regions.** April lost {units(current['roi_splits']['month'][0]['units'])} units, favorites lost {units(current['roi_splits']['favorite_vs_underdog'][0]['units'])}, and the 2–4% plus 6%+ buckets lost {units(sum(row['units'] for row in current['edge_buckets'] if row['edge_bucket'] in ['2-4%', '6%+']))} units. May and underdog picks were slightly positive.
- **Calibration is superficially good because probabilities are compressed.** ECE was {current['ece_10']:.4f}, but {pct(current['pred_dist']['share_between_0_45_and_0_60'])} of predictions sat between 0.45 and 0.60; p5/p50/p95 were {current['pred_dist']['p5']:.3f}/{current['pred_dist']['p50']:.3f}/{current['pred_dist']['p95']:.3f}. Low ECE therefore does not imply useful ranking or betting edges.
- **The green 0–2% edge bucket is not stable evidence.** It returned {pct(green['roi'])} on n={green['n']} in 2026, with a rough 95% Wilson interval of {pct(ci[0])}–{pct(ci[1])} for its {pct(green['win_rate'])} win rate. In 2025 the same bucket returned {pct(green_prior['roi'])} on n={green_prior['n']} (win-rate interval {pct(prior_ci[0])}–{pct(prior_ci[1])}).

## Reliability is good in aggregate but based on a narrow prediction range

The table uses ten fixed-width bins of the predicted home-win probability. Empty bins are retained so the bin counts reconcile exactly to all {current['row_count']} games. ECE is the row-count-weighted absolute gap between mean prediction and empirical home-win rate.

{markdown_table(['Home-win probability', 'n', 'Mean prediction', 'Observed home wins', 'Absolute gap'], reliability_rows)}

2026 YTD ECE is **{current['ece_10']:.4f}**. The apparent calibration is compatible with weak discrimination: only {pct(1-current['pred_dist']['share_between_0_45_and_0_60'])} of predictions fall outside 0.45–0.60. As a stability check, the 2025 p5/p50/p95 were {prior['pred_dist']['p5']:.3f}/{prior['pred_dist']['p50']:.3f}/{prior['pred_dist']['p95']:.3f}, with {pct(prior['pred_dist']['share_between_0_45_and_0_60'])} inside 0.45–0.60 and ECE {prior['ece_10']:.4f}.

## The free market ranks winners better when model and market disagree

Market probabilities are the no-vig home probabilities already stored in the committed free-line cache. Accuracy uses a 0.50 home/away threshold; Brier and AUC use the probabilities directly. The 2025 comparison is restricted to its {prior['odds_row_count']} non-null odds rows.

{markdown_table(['Window', 'n', 'Corr.', 'Mean abs. gap', 'Model acc.', 'Model Brier', 'Model AUC', 'Market acc.', 'Market Brier', 'Market AUC'], market_rows)}

In 2026 the model and market chose the same side {pct(market['direction_agreement_rate'])} of the time. On the {disagreement_2026['n']} disagreements, the market's accuracy advantage was {pct(disagreement_2026['market_accuracy']-disagreement_2026['model_accuracy'])}; in 2025 it was {pct(disagreement_2025['market_accuracy']-disagreement_2025['model_accuracy'])} across {disagreement_2025['n']} disagreements. v3's 2026 Brier was marginally better than the market by {abs(market['model_minus_market']['brier']):.4f}, but this coexists with worse accuracy and AUC and is consistent with conservative probability compression rather than superior ranking.

## The ROI loss sits mainly in April and favorite picks

Flat staking risks one unit per game and records American-odds profit on wins and −1 on losses. Re-derived 2026 profit is **{units(current['flat_roi']['units'])} units / {current['flat_roi']['n']} bets = {pct(current['flat_roi']['roi'])} ROI**, differing from the committed `flat_roi` by {current['flat_roi_difference']:.2e}.

### Pick side

{markdown_table(['Pick side', 'n', 'Wins', 'Win rate', 'Units', 'ROI'], bet_rows(current['roi_splits']['pick_side']))}

### Favorite versus underdog

Favorite/underdog status is based on the selected side's price: negative is favorite and positive is underdog.

{markdown_table(['Selected price class', 'n', 'Wins', 'Win rate', 'Units', 'ROI'], bet_rows(current['roi_splits']['favorite_vs_underdog']))}

### Month

{markdown_table(['Month', 'n', 'Wins', 'Win rate', 'Units', 'ROI'], bet_rows(current['roi_splits']['month']))}

April accounts for more than the full net loss because May offset it slightly. Model discrimination also changed sharply by month:

{markdown_table(['Month', 'n', 'Accuracy', 'Brier', 'AUC'], month_model_rows)}

The April-to-May reversal supports an early-season-state hypothesis, but two months are insufficient to identify a causal season-reset effect.

### Existing edge buckets

{markdown_table(['Absolute edge', 'n', 'Wins', 'Win rate', 'Units', 'ROI'], edge_rows)}

The 2–4% bucket lost {units(next(row['units'] for row in current['edge_buckets'] if row['edge_bucket']=='2-4%'))} units, the largest named-bucket loss. The 0–2% bucket's positive result is contradicted by 2025 and its win-rate confidence interval is wide relative to the edge being claimed.

### Absolute-edge deciles

Deciles are equal-count ranks computed separately within the 2026 odds sample; their edge ranges are descriptive and are not reusable thresholds.

{markdown_table(['Decile', 'Observed abs-edge range', 'n', 'Wins', 'Units', 'ROI'], decile_rows)}

Losses are not monotonic in claimed edge. Decile 5 (roughly 3.2%–4.1%) and decile 10 (>10.9%) lost {units(sum(row['units'] for row in current['roi_splits']['abs_edge_deciles'] if row['decile'] in [5,10]))} units combined, while decile 7 was positive. That jagged pattern is evidence against treating `abs_edge` as a calibrated staking signal.

## Error cohorts show a costly mid-confidence reversal

Pick confidence is `max(home_win_prob, 1-home_win_prob)`; loss rate is the fraction of model picks that lost.

{markdown_table(['Predicted pick probability', 'n', 'Loss rate', 'Units', 'ROI'], confidence_rows)}

The 0.575–0.600 confidence band was the clearest 2026 miss: it lost {units(next(row['units'] for row in current['error_buckets']['pick_confidence_bands'] if row['band']=='0.575-0.600'))} units at {pct(next(row['roi'] for row in current['error_buckets']['pick_confidence_bands'] if row['band']=='0.575-0.600'))} ROI. This was not stable in 2025, when the comparable band's loss rate was {pct(next(row['loss_rate'] for row in prior['error_buckets']['pick_confidence_bands'] if row['band']=='0.575-0.600'))}; it is a 2026 failure cohort, not a validated structural rule.

Selected-side price bands expose a second costly cohort:

{markdown_table(['Selected-side price', 'n', 'Loss rate', 'Units', 'ROI'], price_rows)}

Heavy favorites lost {units(current['error_buckets']['picked_price_bands'][0]['units'])} units on only {current['error_buckets']['picked_price_bands'][0]['n']} bets. Of those, home heavy-favorite picks were n={next(row['n'] for row in current['error_buckets']['selected_cohorts'] if row['cohort']=='home_heavy_favorite_picks')}, with {pct(next(row['loss_rate'] for row in current['error_buckets']['selected_cohorts'] if row['cohort']=='home_heavy_favorite_picks'))} losses and {pct(next(row['roi'] for row in current['error_buckets']['selected_cohorts'] if row['cohort']=='home_heavy_favorite_picks'))} ROI. The same broad heavy-favorite band was approximately flat in 2025 ({pct(prior['error_buckets']['picked_price_bands'][0]['roi'])}), so this too is a concentrated 2026 miss rather than stable proof of a bad cohort.

## Ranked hypotheses for weak AUC

1. **Feature ceiling, especially direct starter-quality signal — strongest hypothesis.** v3 loses the disagreement test to even the free market in both windows, while its feature manifest contains probable-starter history expressed mainly through team results and runs allowed/support, not starter K/9, ERA/FIP proxies, handedness, bullpen state, or confirmed lineup quality. That is consistent with omitted baseball-strength information, but the existing cache cannot isolate the contribution of any one missing feature.
2. **Season-reset states weaken April — supported, not proven.** April 2026 AUC was {current['model_month_performance'][0]['auc']:.4f} versus {current['model_month_performance'][1]['auc']:.4f} in May, and April generated {units(current['roi_splits']['month'][0]['units'])} units of loss. The feature process resets team/starter rolling state by season and drops the first five games rather than carrying prior-year strength forward. Month effects and a short two-month test can confound this pattern, so a carryover-state ablation is required.
3. **RF probability compression / limited separation — clearly observed, causal role uncertain.** The 0.45–0.60 concentration and low ECE show conservative outputs with few strong predictions. Compression alone does not mathematically lower AUC because AUC depends on ordering, but it signals weak separation and makes tiny market-relative edges sensitive to noise. Compare RF ranking with logistic/boosting models and calibrate only after model selection.
4. **Free odds are soft or stale — plausible for ROI, not an explanation for weak AUC.** Line quality can corrupt the size and profitability of `model_edge_home`, and the jagged decile ROI supports caution. However, AUC is computed from game outcomes without odds, and the free market itself had higher AUC in both windows. Closing/sharp lines are needed to judge true betting edge, not to explain v3's outcome-ranking ceiling.

## Scope and method

- Population: completed regular-season games in the committed v3 prediction caches; 2026 YTD spans April 1–May 21 ({current['row_count']} games), and the 2025 stability window spans April 1–September 28 ({prior['row_count']} model rows; {prior['odds_row_count']} odds rows).
- Calibration: ten fixed probability bins and standard count-weighted ECE. Empty bins remain visible and contribute zero weight.
- Market comparison: same-row model and no-vig market probabilities; no external odds lookup or rebuild.
- ROI: one unit staked on every model pick using stored `profit`; all splits are descriptive post hoc cuts, with no multiple-testing correction.
- Uncertainty: Wilson 95% intervals are rough binomial intervals for win rate, not odds-aware intervals for ROI. Tables are used instead of charts because the required cuts are small, exact audit partitions.

## Needs v2 store to answer

- Whether direct probable-starter form and quality (prior-start K/9, ERA/FIP proxies, handedness, and matchup differentials) explains the model–market disagreement gap.
- Whether weather, temperature, wind, and park-relative conditions identify systematic residuals, especially for run environment and derivative markets.
- Whether leakage-safe prior-season carryover or preseason priors repair the April AUC collapse versus a pure season reset.
- Whether bullpen workload/availability, confirmed lineups, injuries, travel, and umpire context explain the heavy-favorite and mid-confidence losses; these require v2 or additional enrichment if v2 does not include them.
- Whether the apparent edge survives against timestamped sharp closing prices; the free comparison line cannot answer this.

## Recommended next steps and further questions

Packet 08 should treat v3 as a probability baseline, not a betting signal. Test starter-quality and prior-season-carryover ablations first, report AUC/Brier changes on the identical 2026 rows, then repeat the market-disagreement and edge-decile checks. Do not promote the 0–2% bucket without out-of-sample replication and sharper closing odds. The open questions are whether improved baseball features close the stable disagreement gap and whether any residual market edge remains after line-quality and timestamp controls.
"""


def validate(result: dict[str, Any]) -> None:
    current = result["analysis_2026_ytd"]
    assert sum(row["n"] for row in current["calibration_bins"]) == current["row_count"]
    assert sum(row["n"] for row in current["edge_buckets"]) == current["odds_row_count"]
    for split in ["pick_side", "favorite_vs_underdog", "month", "abs_edge_deciles"]:
        assert sum(row["n"] for row in current["roi_splits"][split]) == current["odds_row_count"]
    assert abs(current["flat_roi_difference"]) < 1e-12
    serialized = json.dumps(result, allow_nan=False)
    assert serialized


def main() -> None:
    current = analyze("2026 YTD", PREDICTION_FILES["2026_ytd"], METRICS_FILES["2026_ytd"])
    prior = analyze("2025 stability check", PREDICTION_FILES["2025"], METRICS_FILES["2025"])
    result = {
        "artifact": "mlb_moneyline_v3_diagnosis",
        "generated_on": "2026-07-21",
        "calibration_bins": current["calibration_bins"],
        "market_comparison": current["market_comparison"],
        "roi_splits": current["roi_splits"],
        "edge_buckets": current["edge_buckets"],
        "pred_dist": current["pred_dist"],
        "analysis_2026_ytd": current,
        "stability_check_2025": prior,
    }
    validate(result)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    OUTPUT_REPORT.write_text(build_report(current, prior))
    print(f"wrote {OUTPUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUTPUT_REPORT.relative_to(REPO_ROOT)}")
    print(
        "validated: calibration/edge/split counts reconcile; "
        f"flat ROI={current['flat_roi']['roi']:.12f}; "
        f"committed={current['committed_flat_roi']:.12f}"
    )


if __name__ == "__main__":
    main()
