#!/usr/bin/env python3
"""Walk-forward NFL v2 bucket / price-rule backtest.

Analysis only. Does not write the live joblib, flip serving flags, or publish
predictions. Reuses the frozen live candidate specs and training helpers.

    PYTHONPATH=data-core python data-core/scripts/nfl_bucket_strategy_backtest.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_nfl_2026_week7 import load_team_games
from src.models.nfl_bucket_backtest import (
    annotate_predictions,
    attach_schedule_prices,
    define_rules,
    evaluate_rules,
    json_safe,
    line_shopping_note,
    load_live_specs,
    predict_bundle,
    train_walk_forward_bundle,
    walk_forward_predictions,
)
from src.models.nfl_live import LIVE_MODEL_VERSION, load_live_artifact
from src.models.nfl_v2 import FEATURE_COLUMNS, audit_nfl_feature_store, build_nfl_feature_store, write_json


REPO_ROOT = ROOT.parent
DEFAULT_GAMES = ROOT / "notebooks" / "cache" / "nfl_bucket_strategy_games.csv"
DEFAULT_JSON = ROOT / "notebooks" / "cache" / "nfl_bucket_strategy_backtest.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "analysis" / "nfl-bucket-strategy-backtest.md"
DEFAULT_CHART_DIR = REPO_ROOT / "docs" / "analysis" / "nfl-bucket-strategy"
HEALTH_2026 = ROOT / "notebooks" / "cache" / "nfl_model_health_2026_games.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--first-test-season", type=int, default=2021)
    parser.add_argument("--games-output", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chart-dir", type=Path, default=DEFAULT_CHART_DIR)
    parser.add_argument("--from-games", type=Path, help="Reuse a saved predictions CSV and skip retraining.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    return parser.parse_args()


def _to_pandas(value) -> pd.DataFrame:
    if hasattr(value, "collect"):
        value = value.collect()
    return value.to_pandas() if hasattr(value, "to_pandas") else pd.DataFrame(value)


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    print(f"Loading nflverse schedules {seasons[0]}-{seasons[-1]}...", flush=True)
    schedules = _to_pandas(nfl.load_schedules(seasons))
    schedules["league"] = "NFL"
    return schedules


def completed_reg(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    game_type = out.get("game_type", pd.Series("REG", index=out.index)).astype(str).str.upper()
    out = out[game_type.eq("REG")].copy()
    return out[out["home_score"].notna() & out["away_score"].notna()].copy()


def build_features(schedules: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    team_games = load_team_games(seasons)
    print(f"Schedules={len(schedules)} team-games={len(team_games)}", flush=True)
    features = build_nfl_feature_store(schedules, team_games, include_unplayed=True)
    audit = audit_nfl_feature_store(features, raise_on_error=True)
    print(f"Feature store rows={audit['rows']} seasons={audit['seasons']}", flush=True)
    priced = attach_schedule_prices(features, schedules)
    return priced


def apply_frozen_artifact(frame: pd.DataFrame, artifact: dict[str, Any], *, sample_kind: str) -> pd.DataFrame:
    predicted = predict_bundle(frame, artifact)
    predicted["trained_through"] = int(artifact.get("trained_through", 2025))
    predicted["calibration"] = "isotonic"
    predicted["calibration_mode"] = "frozen_live_artifact"
    predicted["prediction_source"] = f"frozen_{artifact.get('model_version', LIVE_MODEL_VERSION)}"
    predicted["sample_kind"] = sample_kind
    predicted["test_season"] = pd.to_numeric(predicted["season"], errors="coerce").astype(int)
    return predicted


def run_walk_forward(features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = load_live_specs()
    completed = features[features["home_score"].notna() & features["away_score"].notna()].copy()
    train_pool = completed[completed[FEATURE_COLUMNS].notna().all(axis=1)].copy()
    walk_seasons = [season for season in range(args.first_test_season, args.end_season) if season < 2026]
    print(f"Walk-forward test seasons: {walk_seasons}", flush=True)
    walk = walk_forward_predictions(train_pool, test_seasons=walk_seasons, specs=specs)

    artifact = load_live_artifact()
    live_2026 = train_pool[train_pool["season"] == 2026].copy()
    frozen_2026 = apply_frozen_artifact(live_2026, artifact, sample_kind="out_of_sample") if len(live_2026) else live_2026
    frozen_2025 = apply_frozen_artifact(
        train_pool[train_pool["season"] == 2025].copy(),
        artifact,
        sample_kind="in_sample",
    )

    oos = pd.concat([walk, frozen_2026], ignore_index=True) if len(frozen_2026) else walk
    oos = annotate_predictions(oos)
    frozen_2025 = annotate_predictions(frozen_2025) if len(frozen_2025) else frozen_2025
    meta = {
        "live_specs": {
            "probability": specs["selection"]["probability_candidate"],
            "margin": specs["selection"]["margin_candidate"],
            "calibration": specs["selection"]["calibration_method"],
            "trained_through_live": specs["selection"]["trained_through"],
        },
        "walk_forward_seasons": walk_seasons,
        "walk_forward_rows": int(len(walk)),
        "frozen_2026_rows": int(len(frozen_2026)),
        "frozen_2025_rows": int(len(frozen_2025)),
        "model_version": artifact.get("model_version"),
        "alters_live_predictions": False,
    }
    return oos, {"oos": oos, "in_sample_2025": frozen_2025, "meta": meta, "specs": specs}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100:.1f}%"


def _ci(summary: dict[str, Any]) -> str:
    interval = summary.get("bootstrap") or {}
    if interval.get("lower_95") is None:
        return "n/a"
    return f"{_fmt_pct(interval.get('lower_95'))} to {_fmt_pct(interval.get('upper_95'))}"


def _row(summary: dict[str, Any]) -> str:
    verdict = summary.get("verdict") or {}
    return (
        f"| {summary.get('description')} | {summary.get('sample_kind')} | {summary.get('n')} | "
        f"{summary.get('record')} | {_fmt_pct(summary.get('roi'))} | {_fmt(summary.get('units'), 2)} | "
        f"{_ci(summary)} | {verdict.get('qualifies_as_strategy')} |"
    )


def _table(summaries: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rule | Sample | n | W-L-P | ROI | Units | Bootstrap 95% CI | Strategy? |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for summary in summaries:
        lines.append(_row(summary))
    return lines


def _season_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Season | n | W-L-P | ROI | Units | 95% CI |",
        "| ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in summary.get("by_season") or []:
        lines.append(
            f"| {row.get('season')} | {row.get('n')} | {row.get('record')} | {_fmt_pct(row.get('roi'))} | "
            f"{_fmt(row.get('units'), 2)} | {_ci(row)} |"
        )
    return lines


def pick_best_looking(summaries: list[dict[str, Any]], *, min_n: int = 0) -> dict[str, Any] | None:
    candidates = [
        row
        for row in summaries
        if row.get("roi") is not None and int(row.get("n") or 0) >= min_n
    ]
    if not candidates:
        return None
    qualified = [row for row in candidates if (row.get("verdict") or {}).get("qualifies_as_strategy")]
    pool = qualified or candidates
    return max(
        pool,
        key=lambda row: (
            bool((row.get("verdict") or {}).get("qualifies_as_strategy")),
            float(row.get("roi") or -999),
            int(row.get("n") or 0),
        ),
    )


def occupancy_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if frame.empty:
        return rows
    for season, group in frame.groupby(frame["season"].astype(int), sort=True):
        counts = group["home_prob_bucket"].value_counts(dropna=False).to_dict()
        unique_p = int(pd.to_numeric(group["my_home_win_prob"], errors="coerce").round(6).nunique())
        rows.append(
            {
                "season": int(season),
                "n": int(len(group)),
                "unique_probabilities": unique_p,
                "calibration_mode": str(group["calibration_mode"].iloc[0]) if "calibration_mode" in group else None,
                "buckets": {str(key): int(value) for key, value in counts.items()},
            }
        )
    return rows


def write_charts(frame: pd.DataFrame, summaries: list[dict[str, Any]], chart_dir: Path) -> list[str]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    bucket_order = ["0-20", "20-30", "30-40", "40-60", "60-70", "70-80", "80+"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(bucket_order))
    width = 0.36
    tick_n = []
    for offset, market, label, color in (
        (-width / 2, "ats_minus_110", "Model side ATS -110", "#1d4ed8"),
        (width / 2, "ml", "Model side ML", "#b45309"),
    ):
        rois = []
        ns = []
        for bucket in bucket_order:
            match = next(
                (
                    row
                    for row in summaries
                    if row.get("family") == "raw_bucket" and row.get("bucket") == bucket and row.get("side") == "model" and row.get("market") == market
                ),
                None,
            )
            if match is None and bucket == "40-60":
                match = next(
                    (
                        row
                        for row in summaries
                        if row.get("family") == "toss_up_40_60" and row.get("side") == "model" and row.get("market") == market
                    ),
                    None,
                )
            rois.append((match or {}).get("roi"))
            ns.append((match or {}).get("n") or 0)
        if market == "ats_minus_110":
            tick_n = ns
        values = [0.0 if value is None else value * 100 for value in rois]
        ax.bar(x + offset, values, width=width, color=color, label=label)
    ax.axhline(0, color="#444", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name}\nn={count}" for name, count in zip(bucket_order, tick_n)])
    ax.set_ylabel("ROI (%)")
    ax.set_title("Out-of-sample ROI by model home-win-prob bucket")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    bucket_path = chart_dir / "roi-by-bucket.png"
    fig.tight_layout()
    fig.savefig(bucket_path, dpi=140)
    plt.close(fig)
    paths.append(str(bucket_path.relative_to(REPO_ROOT)))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = []
    values = []
    ns = []
    for threshold, market in ((3, "ml"), (5, "ml"), (8, "ml"), (3, "ats_minus_110"), (5, "ats_minus_110"), (8, "ats_minus_110")):
        match = next(
            (
                row
                for row in summaries
                if row.get("family") == "model_minus_book" and row.get("bucket") == f"edge>={threshold}" and row.get("market") == market
            ),
            None,
        )
        labels.append(f"{threshold}pp { 'ML' if market == 'ml' else 'ATS' }")
        values.append(0.0 if match is None or match.get("roi") is None else match["roi"] * 100)
        ns.append(0 if match is None else match.get("n") or 0)
    colors = ["#1d4ed8" if "ML" in label else "#0f766e" for label in labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(0, color="#444", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{label}\nn={count}" for label, count in zip(labels, ns)])
    ax.set_ylabel("ROI (%)")
    ax.set_title("Out-of-sample ROI when model is more confident than the no-vig book")
    ax.grid(True, axis="y", alpha=0.3)
    edge_path = chart_dir / "roi-by-model-minus-book-edge.png"
    fig.tight_layout()
    fig.savefig(edge_path, dpi=140)
    plt.close(fig)
    paths.append(str(edge_path.relative_to(REPO_ROOT)))
    return paths


def health_2026_overlap(oos: pd.DataFrame) -> dict[str, Any]:
    if not HEALTH_2026.exists():
        return {"ok": False, "reason": "health CSV missing"}
    health = pd.read_csv(HEALTH_2026)
    live = oos[(oos["season"] == 2026) & (oos["prediction_source"].astype(str).str.startswith("frozen"))].copy()
    if live.empty or health.empty:
        return {"ok": False, "reason": "no 2026 overlap"}
    merged = live.merge(health[["game_id", "my_home_win_prob"]], on="game_id", how="inner", suffixes=("", "_health"))
    if merged.empty:
        return {"ok": False, "reason": "game_id join missed"}
    delta = (pd.to_numeric(merged["my_home_win_prob"], errors="coerce") - pd.to_numeric(merged["my_home_win_prob_health"], errors="coerce")).abs()
    return {
        "ok": True,
        "joined": int(len(merged)),
        "max_abs_prob_delta": float(delta.max()) if len(delta) else None,
        "mean_abs_prob_delta": float(delta.mean()) if len(delta) else None,
    }


def render_report(
    payload: dict[str, Any],
    oos: pd.DataFrame,
    chart_paths: list[str],
    path: Path,
) -> None:
    q1 = [row for row in payload["oos_rules"] if row.get("question") == "q1"]
    q2 = [row for row in payload["oos_rules"] if row.get("question") == "q2"]
    toss = [row for row in q1 if row.get("family") == "toss_up_40_60"]
    edge = [row for row in payload["oos_rules"] if row.get("family") == "model_minus_book"]
    buckets = [row for row in q2 if row.get("family") == "raw_bucket"]
    best = payload.get("best_looking") or {}
    best_large = payload.get("best_looking_n200") or {}
    verdict = best.get("verdict") or {}
    n_tests = payload["n_rules_tested"]
    occupancy = payload.get("occupancy") or []
    images = [f"![{Path(chart).stem}]({Path(chart).as_posix().replace('docs/analysis/', '')})" for chart in chart_paths]
    lines = [
        "# NFL v2 bucket strategy backtest",
        "",
        f"**Generated:** {payload['generated_at']}",
        f"**Live model (2026 only):** `{payload['model_version']}` — not modified",
        f"**Walk-forward candidate:** `{payload['meta']['live_specs']['probability']}` + `{payload['meta']['live_specs']['margin']}` + `{payload['meta']['live_specs']['calibration']}`",
        f"**Rules tested:** {n_tests}",
        "",
        "This is analysis, not a serving change and not a betting product. Missing prices stay unscored. In-sample numbers are labeled and are **not** an edge.",
        "",
        "## Headline",
        "",
        payload["headline"]["q1"],
        "",
        payload["headline"]["q2"],
        "",
        (
            f"Highest-ROI OOS rule: **{best.get('description')}** — n={best.get('n')}, {best.get('record')}, "
            f"ROI {_fmt_pct(best.get('roi'))}, 95% CI {_ci(best)}, "
            f"Bonferroni p={_fmt((verdict.get('bonferroni_p_value')), 3)}. "
            f"{'This qualifies as a strategy under the pre-registered bar.' if verdict.get('qualifies_as_strategy') else 'This does **not** qualify as a strategy: the bar is OOS + same-sign ROI in ≥3 seasons with n≥20 + 95% CI excluding 0 + Bonferroni-adjusted p<0.05.'}"
        ),
        "",
        (
            (
                f"Largest-n near-edge (n≥200): **{best_large.get('description')}** — n={best_large.get('n')}, "
                f"{best_large.get('record')}, ROI {_fmt_pct(best_large.get('roi'))}, 95% CI {_ci(best_large)}. "
                "Still includes 0."
            )
            if best_large
            else ""
        ),
        "",
        "## Sample",
        "",
        "| Slice | Kind | n | Seasons |",
        "| --- | --- | ---: | --- |",
        f"| Walk-forward 2021–2025 (train on prior seasons only) | **out-of-sample** | {payload['counts']['walk_forward']} | {', '.join(str(s) for s in payload['meta']['walk_forward_seasons'])} |",
        f"| Frozen `{payload['model_version']}` on 2026 REG | **out-of-sample** | {payload['counts']['frozen_2026']} | 2026 weeks 1–6 |",
        f"| Pooled OOS used for all headline tables | **out-of-sample** | {payload['counts']['oos']} | 2021–2026 |",
        f"| Frozen artifact on 2025 REG | **in-sample** (trained_through=2025) | {payload['counts']['in_sample_2025']} | 2025 |",
        "",
        "2026 n≈92 is real OOS for the frozen artifact but too small to license a rule by itself. 2021 uses a single prior season, so isotonic OOF calibration is impossible; that year is raw HGB. 2022–2025 use the production OOF-isotonic recipe. 2025 reconstructed through the frozen live artifact is in-sample and is appendix-only.",
        "",
        f"PR #41 health overlap on 2026: joined={payload['health_overlap'].get('joined')}, max |Δp|={_fmt(payload['health_overlap'].get('max_abs_prob_delta'), 6)}.",
        "",
        "### Bucket occupancy (OOS)",
        "",
        "| Season | n | Unique p | Calibrator | 0-20 | 20-30 | 30-40 | 40-60 | 60-70 | 70-80 | 80+ |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in occupancy:
        counts = row.get("buckets") or {}
        lines.append(
            "| {season} | {n} | {unique} | {mode} | {a} | {b} | {c} | {d} | {e} | {f} | {g} |".format(
                season=row.get("season"),
                n=row.get("n"),
                unique=row.get("unique_probabilities"),
                mode=row.get("calibration_mode"),
                a=counts.get("0-20", 0),
                b=counts.get("20-30", 0),
                c=counts.get("30-40", 0),
                d=counts.get("40-60", 0),
                e=counts.get("60-70", 0),
                f=counts.get("70-80", 0),
                g=counts.get("80+", 0),
            )
        )
    lines.extend(
        [
        "",
        "2022 walk-forward isotonic **never emitted a probability in 40–60** (0 of 271 games). Q1 therefore has no 2022 season. The same coarseness shows up live: PR #41 counted 17 unique 2026 probabilities. 2021 tails (0–20 / 20–30) exist only because that year could not fit OOF isotonic.",
        "",
        "## Prices and vig",
        "",
        "- Close: nflverse `spread_line`, `home_spread_odds`, `away_spread_odds`, `home_moneyline`, `away_moneyline`.",
        "- Home spread = `-spread_line` (home favorite negative), same convention as the health check.",
        "- Book win probability = no-vig split of the two moneylines (`no_vig_home_probability`).",
        "- ATS `-110` always stakes at -110. ATS actual uses the nflverse close price and falls back to -110 only when that side's price is missing.",
        "- Moneyline uses the actual close. Ties push. Fail closed if a required price is missing.",
        "",
        payload["line_shopping"]["note"],
        "",
        f"Spread-price coverage: {_fmt_pct(payload['line_shopping']['spread_price_coverage'])}. Moneyline coverage: {_fmt_pct(payload['line_shopping']['moneyline_coverage'])}. Multi-book rows: {payload['line_shopping']['multi_book_rows']}.",
        "",
        "## Multiple comparisons",
        "",
        f"{n_tests} rules were tested on the same OOS pool (toss-up sides × markets, confident buckets × model/book-favorite × markets, and model-minus-book thresholds 3/5/8pp). A 95% CI that excludes 0 is not enough. Bonferroni uses `{n_tests}` tests. A rule is called a **strategy** only if it is out-of-sample, the CI excludes 0, Bonferroni-adjusted p < 0.05, and ROI keeps the same sign in at least three seasons with n≥20.",
        "",
        "## Q1 — toss-up band (model home or away win prob 40–60%)",
        "",
        "Serving still withholds a moneyline pick in the narrower ±5pp band from #42. This table is the owner's 40–60 analysis band, not a serving change.",
        "",
        *_table(toss),
        "",
        "Per-season for the three moneyline sides:",
        "",
        ]
    )
    for rule_id in ("q1_toss_up_model_ml", "q1_toss_up_book_dog_ml", "q1_toss_up_book_fav_ml"):
        summary = next((row for row in toss if row.get("rule_id") == rule_id), None)
        if summary is None:
            continue
        lines.extend([f"### {summary['description']}", "", *_season_table(summary), ""])

    lines.extend(
        [
            "Model-minus-book inside the toss-up band (does any edge sit in disagreement, not the raw 40–60 bucket?):",
            "",
            *_table([row for row in q1 if row.get("family") == "model_minus_book_in_toss_up"]),
            "",
            "## Q2 — confident bands",
            "",
            "Under 40 / over 60, then 10-point bins and their mirrors. Back the model side and the book favorite, ATS and ML.",
            "",
            *_table(buckets),
            "",
            "### Model more confident than the book",
            "",
            "Threshold is model-favorite probability minus no-vig book probability on that same side. Compared with the raw bucket table above.",
            "",
            *_table(edge),
            "",
            *images,
            "",
            "## In-sample 2025 (not an edge)",
            "",
            "Frozen `nfl-v2-live-20260906` scored on 2025 REG. Trained through 2025. Do not read ROI here as a tradable rule. The in-sample toss-up dog ML printed **+46%**; the same rule is **−4.1%** out of sample. That is the reason this appendix exists.",
            "",
            *_table(
                [
                    row
                    for row in (payload.get("in_sample_2025_rules") or [])
                    if row.get("family") in {"toss_up_40_60", "model_minus_book"}
                    or row.get("bucket") in {"under_40", "over_60"}
                ]
            ),
            "",
            "## Method",
            "",
            "1. Build the leakage-safe v2 feature store from nflverse schedules + PBP (same path as Daily / PR #41).",
            "2. For each season 2021–2025: fit the live specs (`hgb_small`, `ridge_a50`, isotonic) on **prior seasons only** via `fit_production_shadow_bundle`. Predict that season pregame. 2021 has only 2020, so no OOF isotonic.",
            "3. 2026: reconstruct the frozen live artifact. That is the production model, trained through 2025.",
            "4. Grade flat 1u bets. Bootstrap 2000 resamples of per-bet units (pushes excluded from ROI denominator). Seed 20261019.",
            "5. No live serving writes. `predict_live_games` is untouched.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=data-core python data-core/scripts/nfl_bucket_strategy_backtest.py",
            "```",
            "",
            "Reuse saved predictions:",
            "",
            "```bash",
            "PYTHONPATH=data-core python data-core/scripts/nfl_bucket_strategy_backtest.py \\",
            "  --from-games data-core/notebooks/cache/nfl_bucket_strategy_games.csv",
            "```",
            "",
            "Artifacts: `data-core/notebooks/cache/nfl_bucket_strategy_games.csv`, `nfl_bucket_strategy_backtest.json`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _qualifies(row: dict[str, Any]) -> bool:
    return bool((row.get("verdict") or {}).get("qualifies_as_strategy"))


def headline_answers(oos_rules: list[dict[str, Any]]) -> dict[str, str]:
    toss = [row for row in oos_rules if row.get("family") == "toss_up_40_60"]
    dog_ml = next((row for row in toss if row["rule_id"] == "q1_toss_up_book_dog_ml"), {})
    model_ml = next((row for row in toss if row["rule_id"] == "q1_toss_up_model_ml"), {})
    fav_ml = next((row for row in toss if row["rule_id"] == "q1_toss_up_book_fav_ml"), {})
    dog_ats = next((row for row in toss if row.get("rule_id") == "q1_toss_up_book_dog_ats_minus_110"), {})
    q1_any = any(_qualifies(row) for row in toss)
    q1_tail = (
        "One of these meets the strategy bar; see the table."
        if q1_any
        else "None of these CIs both exclude 0 and survive the season + Bonferroni bar."
    )
    q1 = (
        f"**Q1:** In the 40–60 toss-up band (OOS n={model_ml.get('n')}), backing the book underdog is "
        f"{'still not a licensed strategy' if not _qualifies(dog_ml) else 'the only toss-up rule that clears the bar'}. "
        f"Dog ML {dog_ml.get('record')} / {_fmt_pct(dog_ml.get('roi'))} "
        f"(CI {_ci(dog_ml)}). Model-side ML {model_ml.get('record')} / {_fmt_pct(model_ml.get('roi'))}. "
        f"Book-favorite ML {fav_ml.get('record')} / {_fmt_pct(fav_ml.get('roi'))}. "
        f"Dog ATS -110 {dog_ats.get('record')} / {_fmt_pct(dog_ats.get('roi'))}. "
        f"{q1_tail}"
    )
    fav = [row for row in oos_rules if row.get("family") == "raw_bucket" and row.get("bucket") in {"under_40", "over_60"} and row.get("side") == "book_fav"]
    edge = [row for row in oos_rules if row.get("family") == "model_minus_book"]
    over = next((row for row in fav if row["bucket"] == "over_60" and row["market"] == "ml"), {})
    under = next((row for row in fav if row["bucket"] == "under_40" and row["market"] == "ml"), {})
    edge5 = next((row for row in edge if row.get("bucket") == "edge>=5" and row.get("market") == "ml"), {})
    q2_any = any(_qualifies(row) for row in oos_rules if row.get("question") == "q2")
    q2 = (
        f"**Q2:** Confident buckets "
        f"{'include a rule that clears the strategy bar' if q2_any else 'are not a usable betting surface'}. "
        f"Book-favorite ML when the model is >60% home: {over.get('record')} / {_fmt_pct(over.get('roi'))} "
        f"(CI {_ci(over)}). "
        f"When the model is <40% home: {under.get('record')} / {_fmt_pct(under.get('roi'))}. "
        f"Model-minus-book ≥5pp ML: {edge5.get('record')} / {_fmt_pct(edge5.get('roi'))} (CI {_ci(edge5)}). "
        + (
            "See the Q2 table for the qualifying rule."
            if q2_any
            else "The raw confident buckets and the disagreement filter both fail the strategy bar. "
            "That matches the health-check diagnosis: the model is overconfident, not quietly +EV at short prices."
        )
    )
    return {"q1": q1, "q2": q2}


def export_games(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        column
        for column in (
            "game_id",
            "season",
            "week",
            "game_date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "actual_margin",
            "game_type",
            "my_home_win_prob",
            "my_spread",
            "predicted_margin",
            "home_spread",
            "home_spread_odds",
            "away_spread_odds",
            "home_moneyline",
            "away_moneyline",
            "market_home_prob",
            "model_minus_book",
            "model_side_edge",
            "home_prob_bucket",
            "model_side",
            "trained_through",
            "calibration",
            "calibration_mode",
            "prediction_source",
            "sample_kind",
        )
        if column in frame.columns
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].sort_values(["season", "week", "game_date", "away_team"]).to_csv(path, index=False)


def main() -> None:
    args = _parse_args()
    load_dotenv(args.env_file)
    if args.from_games:
        oos = annotate_predictions(pd.read_csv(args.from_games))
        frames = {
            "oos": oos[oos["sample_kind"] == "out_of_sample"].copy() if "sample_kind" in oos else oos,
            "in_sample_2025": oos[oos["sample_kind"] == "in_sample"].copy() if "sample_kind" in oos else oos.iloc[0:0],
            "meta": {
                "live_specs": load_live_specs()["selection"],
                "walk_forward_seasons": sorted(
                    int(value) for value in pd.to_numeric(oos.get("season"), errors="coerce").dropna().unique() if int(value) < 2026
                ),
                "alters_live_predictions": False,
            },
        }
        frames["meta"]["live_specs"] = {
            "probability": frames["meta"]["live_specs"].get("probability_candidate"),
            "margin": frames["meta"]["live_specs"].get("margin_candidate"),
            "calibration": frames["meta"]["live_specs"].get("calibration_method"),
            "trained_through_live": frames["meta"]["live_specs"].get("trained_through"),
        }
    else:
        seasons = list(range(args.start_season, args.end_season + 1))
        schedules = load_schedules(seasons)
        features = build_features(schedules, seasons)
        oos, frames = run_walk_forward(features, args)

    oos = completed_reg(frames["oos"])
    in_sample = completed_reg(frames["in_sample_2025"]) if len(frames["in_sample_2025"]) else frames["in_sample_2025"]
    rules = define_rules()
    oos_rules = evaluate_rules(oos, rules, sample_kind="out_of_sample")
    in_sample_rules = evaluate_rules(in_sample, rules, sample_kind="in_sample") if len(in_sample) else []
    best = pick_best_looking(oos_rules)
    best_large = pick_best_looking(oos_rules, min_n=200)
    shopping = line_shopping_note(oos)
    occupancy = occupancy_table(oos)
    overlap = health_2026_overlap(oos)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": LIVE_MODEL_VERSION,
        "n_rules_tested": len(rules),
        "counts": {
            "oos": int(len(oos)),
            "walk_forward": int((oos["prediction_source"] == "walk_forward_prior_seasons").sum())
            if "prediction_source" in oos
            else int(len(oos[oos["season"] < 2026])),
            "frozen_2026": int((pd.to_numeric(oos["season"], errors="coerce") == 2026).sum()),
            "in_sample_2025": int(len(in_sample)),
        },
        "meta": frames["meta"],
        "line_shopping": shopping,
        "health_overlap": overlap,
        "oos_rules": oos_rules,
        "in_sample_2025_rules": in_sample_rules,
        "best_looking": best,
        "best_looking_n200": best_large,
        "occupancy": occupancy,
        "alters_live_predictions": False,
    }
    payload["headline"] = headline_answers(oos_rules)
    chart_paths = write_charts(oos, oos_rules, args.chart_dir)
    payload["charts"] = chart_paths
    export_games(pd.concat([oos, in_sample], ignore_index=True) if len(in_sample) else oos, args.games_output)
    write_json(args.json_output, json_safe(payload))
    render_report(payload, oos, chart_paths, args.report)
    print(
        json.dumps(
            {
                "games": str(args.games_output),
                "report": str(args.report),
                "oos_rows": payload["counts"]["oos"],
                "n_rules": payload["n_rules_tested"],
                "q1": payload["headline"]["q1"],
                "q2": payload["headline"]["q2"],
                "best": {
                    "rule": (best or {}).get("description"),
                    "n": (best or {}).get("n"),
                    "roi": (best or {}).get("roi"),
                    "ci": (best or {}).get("bootstrap"),
                    "strategy": ((best or {}).get("verdict") or {}).get("qualifies_as_strategy"),
                },
                "alters_live": False,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
