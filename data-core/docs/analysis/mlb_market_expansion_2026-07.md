# MLB Market Expansion Analysis

Generated: 2026-07-21

## 1. Scope & TL;DR

This run rebuilt completed regular-season MLB data through 2026-07-20, trained on 2021-2024, selected models on 2025, refit through 2025, and tested on 2026 YTD. It is a research package only: no model was wired into production and no result below is a betting recommendation.

**Moneyline:** v4 shipped as a reproducible research artifact, but the result is negative. On the identical 1,428-game test window, full v2 scored Brier **0.247468** and AUC **0.549654**, versus **0.247423** and **0.550626** for the reconstructed v1-feature arm. Flat ROI on the shared 673-game free-line subset declined from **-3.08%** to **-5.86%**. Source: `notebooks/cache/mlb_ml_ablation_2026_ytd.json`.

**Totals:** the ridge research model shipped. On 1,428 test games it scored **3.5364 MAE** and **4.4925 RMSE**; removing the combined weather-and-park group worsened MAE by **0.0547 runs**. ROI not measurable — no odds source. Source: `notebooks/cache/mlb_totals_metrics_2026_ytd.json`.

**Run line:** the random-forest home -1.5 classifier research model shipped. Its test Brier was **0.22830**, better than the constant pre-test cover-rate baseline's **0.23019**, with AUC **0.55135**. ROI not measurable — no odds source. Source: `notebooks/cache/mlb_runline_metrics_2026_ytd.json`.

**Starter strikeouts:** feasibility passed and the Poisson-loss histogram-gradient-boosting research model shipped. Clean probable-starter coverage was **97.79%** (2,793/2,856 test sides), and test MAE was **1.7960**, versus **1.8673** for the K/9 x expected-outs baseline. ROI not measurable — no odds source. Source: `notebooks/cache/mlb_strikeouts_metrics_2026_ytd.json`.

## 2. Why v3 underperformed

The v3 diagnosis covers 673 completed games from 2026-04-01 through 2026-05-21. Its aggregate calibration looked acceptable (ECE **0.0120**), but **85.0%** of predictions were between 0.45 and 0.60 and the p5/p50/p95 were 0.456/0.528/0.631. This compression explains why low ECE did not translate into useful separation. On the same games, v3 AUC was **0.5431** versus the free market's **0.5501**. When their favored sides differed, v3 was correct **46.9%** of the time and the market **53.1%**. Source for this section: `notebooks/cache/mlb_ml_diagnosis_2026_ytd.json` and the run artifact `artifacts/diagnosis-v3.md`.

### v3 reliability

| Home-win probability | n | Mean prediction | Observed home wins | Absolute gap |
| --- | ---: | ---: | ---: | ---: |
| 0.0-0.3 | 0 | — | — | — |
| 0.3-0.4 | 2 | 39.6% | 0.0% | 39.6% |
| 0.4-0.5 | 183 | 47.2% | 47.5% | 0.3% |
| 0.5-0.6 | 412 | 54.1% | 52.7% | 1.5% |
| 0.6-0.7 | 76 | 62.8% | 61.8% | 1.0% |
| 0.7-1.0 | 0 | — | — | — |

### v3 ROI autopsy

Flat staking lost **21.09 units** over 673 bets for **-3.1% ROI**. The loss was concentrated rather than monotonic in model edge: April lost **22.47 units**, favorite picks lost **24.41 units**, and the 2-4% plus 6%+ edge buckets lost **24.22 units** together. May and underdog picks were slightly positive. The apparently green 0-2% bucket did not replicate in 2025, where it returned **-6.1%** on 616 bets.

| Absolute edge | n | Units | ROI |
| --- | ---: | ---: | ---: |
| 0-2% | 168 | +6.20 | +3.7% |
| 2-4% | 159 | -13.73 | -8.6% |
| 4-6% | 107 | -3.08 | -2.9% |
| 6%+ | 239 | -10.49 | -4.4% |

The clearest error cohort was the 0.575-0.600 predicted-pick-probability band: 72 bets, **52.8% loss rate**, and **-19.5% ROI**. Heavy favorites at -200 or shorter lost **9.20 units** on 42 bets (**-21.9% ROI**). Neither cohort was stable in 2025, so these are post-hoc failure descriptions, not reusable betting rules.

### Hypotheses tested by the v4 ablation

| Ranked v3 hypothesis | v4 evidence on identical 1,428 rows | Verdict |
| --- | --- | --- |
| Direct starter-quality omissions were the main ceiling | Adding the starter group changed Brier by **+0.000147**, AUC by **-0.002759**, and free-line ROI by **-4.04 percentage points** versus v1 features | Killed for this implementation/window; the added rolling starter group hurt rather than helped |
| Season-reset state caused the April weakness | v2 carried starter and team histories across seasons, but no isolated carryover-only arm was run; full v2 changed Brier **+0.000045** and AUC **-0.000973** | Not confirmed; full-v2 evidence argues against a material net repair, but does not isolate carryover |
| RF outputs were too compressed | Full-v2 p5/p95 widened to **0.4452/0.6331**, yet Brier/AUC did not improve | Compression was observed, but widening alone did not repair ranking |
| Soft/stale free odds distorted ROI | Odds covered only **47.13%** (673/1,428) and v4 ROI worsened despite nearly unchanged probability metrics | Still plausible for ROI; not an explanation for outcome AUC |

Weather alone was nearly neutral for moneyline Brier (**+0.000014**), improved AUC by **0.001220**, and reduced ROI by **1.68 percentage points** versus the v1 arm. The prescribed moneyline arms did not isolate the cross-season run-environment group, so no standalone causal claim is possible for that group. Source: `notebooks/cache/mlb_ml_ablation_2026_ytd.json`.

## 3. Data upgrades

The raw rebuild added temperature, condition, wind speed/direction, starter strikeouts, and team batting/pitching totals from MLB boxscore payloads. Venue metadata supplies roof type and elevation where MLB exposes them. The v2 store then adds cross-season probable-starter rolling K/9, BB/9, ERA proxy, pitches, outs, workload/history, rest and handedness; cross-season 15-game team offense, defense, total-runs and batting-K state; a combined expected-total feature; weather/wind buckets; and market labels excluded from model features.

**Weather columns are observed game-time boxscore records used as pregame forecast proxies.** They are not archived forecasts available before first pitch. Consequently, the totals weather/park ablation is useful research evidence but is not sufficient for production promotion; a prediction-time forecast source must replace these values and be re-evaluated.

| Audit item | Coverage / null rate |
| --- | ---: |
| Raw games and boxscores | 13,653 / 13,653; 100.00% coverage; 0 error rows |
| Raw date/season coverage | 2021-2026; 2026 has 1,505 rows through 2026-07-20 |
| Raw `temp_f` null rate | 0.0000% |
| Raw `weather_condition` null rate | 0.0000% |
| Raw `wind_mph` / `wind_dir` null rate | 0.0073% each |
| Raw team batting/pitching total null rates | 0.0000% for hits, HR, walks, batting Ks, and pitching Ks |
| v2 feature rows | 13,182 from 2021-04-06 through 2026-07-20 |
| Starter history on both sides | 94.6063% |
| Known probable starter matches actual starter | 99.8368% |
| v2 `temp_f` null rate | 0.0000% |
| v2 `wind_mph` null rate | 0.0076% |
| v2 `elevation` null rate | 0.1897% |

Sources: `notebooks/cache/mlb_boxscores_2021_2026_audit.json` and `notebooks/cache/mlb_feature_store_v2_2021_2026_audit.json`.

## 4. Per-market results

Every model used 9,404 rows from 2021-2024 for selection training, 2,350 2025 validation game rows, and 1,428 2026 test games, except strikeouts, which reshaped games to starter sides (18,203 train, 4,599 validation, and 2,793 clean test sides). The artifacts report validation selection metrics and final refit test metrics, not in-sample training scores.

### Moneyline v4

Random forest was selected by 2025 validation Brier. Its selection-stage validation Brier/log loss/AUC were **0.245255 / 0.683462 / 0.561652**. After refitting on 11,754 rows through 2025, the 2026 metrics were:

| Identical-row arm | Features | Brier | Log loss | AUC | ECE | Flat ROI (673 odds rows) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 baseline | 59 | 0.247423 | 0.687934 | 0.550626 | 0.012184 | -3.08% |
| v1 + starter | 89 | 0.247571 | 0.688189 | 0.547868 | 0.012333 | -7.12% |
| v1 + weather/park | 67 | 0.247437 | 0.687975 | 0.551846 | 0.012689 | -4.77% |
| full v2 / v4 | 110 | 0.247468 | 0.687978 | 0.549654 | 0.012746 | -5.86% |
| full v2 minus v1 | +51 | **+0.000045** | +0.000043 | **-0.000973** | +0.000562 | **-2.78 pp** |

The free-line cache covers 673 of 1,428 test games. Those are comparison/consensus lines rather than timestamped sharp closing prices, so even the measured moneyline ROI is a soft-line result.

| v4 absolute edge | Games | Units | ROI |
| --- | ---: | ---: | ---: |
| 0-2% | 178 | -4.86 | -2.73% |
| 2-4% | 165 | -21.37 | -12.95% |
| 4-6% | 113 | +0.10 | +0.09% |
| 6%+ | 217 | -13.33 | -6.14% |

Sources: `notebooks/cache/mlb_ml_ablation_2026_ytd.json`, `notebooks/cache/mlb_backtest_metrics_2026_ytd_v4_free.json`, and `models/mlb_winner_model_v4_metrics.json`.

### Totals

Ridge won selection with 2025 validation MAE **3.6025** and RMSE **4.5479**. After refit through 2025, test MAE/RMSE were **3.5364 / 4.4925**.

| Test point model / baseline | MAE | RMSE |
| --- | ---: | ---: |
| Selected ridge refit | 3.5364 | 4.4925 |
| Constant 2021-2024 train mean | 3.5885 | 4.5594 |
| Trailing 30-day league mean | 3.6245 | 4.5722 |
| Venue rolling mean | 3.6503 | 4.6395 |

| Binary head | Model Brier | Model log loss | Model AUC | Model ECE | Constant Brier | Constant AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Over 8.5 | 0.2469 | 0.6868 | 0.5739 | 0.0365 | 0.2501 | 0.5000 |
| Over 9.5 | 0.2378 | 0.6684 | 0.5813 | 0.0447 | 0.2414 | 0.5000 |

Removing the combined weather-and-park group worsened MAE from **3.5364** to **3.5911** (ablation minus full **+0.0547**) and reduced over-8.5 AUC from **0.5739** to **0.5238** (full minus ablation **+0.0501**). This is a grouped observational ablation, not weather-only causal attribution. ROI not measurable — no odds source. Source: `notebooks/cache/mlb_totals_metrics_2026_ytd.json`.

### Run line

Random forest won classifier selection with 2025 validation Brier/log loss/AUC of **0.22706 / 0.64631 / 0.55393**. The refit test result and base-rate comparison were:

| Test head / baseline | Brier | Log loss | AUC | ECE | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected home -1.5 classifier | 0.22830 | 0.64886 | 0.55135 | 0.02415 | 64.57% |
| Constant pre-test cover rate | 0.23019 | 0.65298 | 0.50000 | 0.00021 | 64.08% |

The selected margin regressor had test MAE **3.62657**, but its empirical-residual cover head was worse than the classifier (Brier **0.23331**, ECE **0.06683**). The classifier's probability correlated **0.94475** with v4 home-win probability, indicating limited independence from moneyline. ROI not measurable — no odds source. Source: `notebooks/cache/mlb_runline_metrics_2026_ytd.json`.

### Starter strikeouts

The conservative Poisson-loss histogram-gradient-boosting model won 2025 selection with MAE/RMSE **1.8135 / 2.2708**, versus **1.9276 / 2.4222** for the K/9 x expected-outs baseline. Test results were:

| Test point model / baseline | MAE | RMSE |
| --- | ---: | ---: |
| Selected model | 1.7960 | 2.2482 |
| K/9 x expected-outs baseline | 1.8673 | 2.3640 |

| Threshold | Model Brier | Reference base-rate Brier | Empirical hit rate | Mean predicted probability |
| --- | ---: | ---: | ---: | ---: |
| K >= 6 (over 5.5) | 0.2074 | 0.2319 | 36.52% | 38.18% |
| K >= 7 (over 6.5) | 0.1638 | 0.1811 | 23.74% | 24.76% |

The 2026 probable-starter mismatch discarded fraction was **0.35%**; clean labeled coverage was **97.79%**. Threshold probabilities assume a Poisson distribution and may miss overdispersion. ROI not measurable — no odds source. Source: `notebooks/cache/mlb_strikeouts_metrics_2026_ytd.json`.

## 5. Gap register

- **Sharp closing lines:** The Odds API historical endpoint remains blocked on the configured free plan (`HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`; see `docs/analysis/mlb_performance_2026-05-21.md` section “Historical Odds Status”). Available moneylines are free comparison/consensus data, cover only 47.13% of the v4 test window, and should be treated as soft rather than true closing prices.
- **Derivative-market prices:** there is no historical totals, run-line/spread, or starter-K odds source. Therefore ROI is deliberately absent for those three markets.
- **Pregame context:** confirmed lineups, umpire assignments, injuries and travel are not hydrated. Bullpen fatigue also remains unused despite leakage-safe `bullpen_*` columns already existing in the boxscore layer.
- **Weather timing:** observed game-time weather is not an archived pregame forecast. Replacing it with forecast snapshots is mandatory before using the totals lift operationally.
- **Ablation resolution:** moneyline isolates starter and weather/park additions, but not cross-season run environment alone. Totals removes weather and park as one group, so it cannot attribute the lift solely to weather.

## 6. What to try next (ranked)

1. **Acquire timestamped historical closing lines.** Expected payoff: highest, because it unlocks honest ROI, closing-line-value tests, and calibration against a sharper prior for all four markets. Cost: high and likely paid; requires normalized joins for moneyline, total, spread and pitcher props.
2. **Replace observed weather with archived or prediction-time forecasts.** Expected payoff: high for totals because the grouped ablation improved MAE by 0.0547 and over-8.5 AUC by 0.0501, but current timing is not deployable. Cost: medium; map venue coordinates, snapshot forecasts before first pitch, reproduce wind/roof semantics, and retrain.
3. **Model bullpen fatigue from existing `bullpen_*` columns.** Expected payoff: medium for moneyline/run line and possibly totals; direct starter additions were negative, so the next baseball-strength hypothesis should target late-game availability instead of adding more starter variants. Cost: medium; create prior-game workload/availability rolls and leakage tests.
4. **Add an out-of-fold calibration layer after model selection.** Expected payoff: medium for binary heads, especially totals (model ECE exceeded its constant baseline) and the rejected run-line residual head. Cost: low to medium; compare isotonic/Platt calibration on held-out seasons without tuning on 2026. Calibration cannot repair weak AUC, so it follows feature/data work rather than replacing it.
5. **Hydrate confirmed lineups, injuries and batting-order quality.** Expected payoff: medium; v3 repeatedly lost model-versus-market disagreement tests, while starter-only enrichment did not close the gap. Cost: high because availability timestamps, player identity, and missing-data behavior must be made reproducible.
6. **Isolate carryover and run-environment effects.** Expected payoff: uncertain but diagnostic; v3's April collapse motivated cross-season state, yet the v4 full arm was negative and did not isolate it. Cost: low; add carryover-only and run-environment-only arms on identical rows before further feature expansion.

## 7. Reproducible commands

Run from the repository root. The first two commands call the public MLB Stats API; the remaining commands use local artifacts.

```bash
PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/backfill_mlb_raw.py \
  --start-season 2021 --end-season 2026 --refresh-games \
  --games-cache data-core/notebooks/cache/mlb_games_2021_2026.parquet \
  --boxscores-cache data-core/notebooks/cache/mlb_boxscores_2021_2026.parquet \
  --fetch-boxscores

PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/fetch_mlb_venue_meta.py \
  --output data-core/notebooks/cache/mlb_venue_meta.json

PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/build_mlb_feature_store.py --version v2

PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_winner_model.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet \
  --validation-season 2025 --test-season 2026 --model-version v4 \
  --output-model data-core/models/mlb_winner_model_v4.pkl

PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/backtest_mlb_winners.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet \
  --validation-season 2025 --test-season 2026 \
  --odds-path data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv \
  --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd_v4_free.csv \
  --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd_v4_free.json

PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/ablate_mlb_winner_features.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet \
  --validation-season 2025 --test-season 2026 \
  --odds-path data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv \
  --output data-core/notebooks/cache/mlb_ml_ablation_2026_ytd.json

PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_totals_model.py

PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_runline_model.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet \
  --validation-season 2025 --test-season 2026 \
  --metrics-output data-core/notebooks/cache/mlb_runline_metrics_2026_ytd.json \
  --predictions-output data-core/notebooks/cache/mlb_runline_predictions_2026_ytd.csv \
  --output-model data-core/models/mlb_runline_model_v1.pkl

PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_strikeouts_model.py
```

## 8. Decision

| Market | Recommendation | Reason |
| --- | --- | --- |
| Moneyline v4 | **Research-grade; do not promote** | Controlled full-v2 result was worse than v1 on Brier, AUC, and free-line ROI |
| Totals v1 | **Research-grade; not yet a ship candidate** | Beats the constant baseline and weather/park ablation is encouraging, but observed-weather timing and missing totals odds block promotion |
| Run line v1 | **Research-grade; not yet a ship candidate** | Small Brier lift over base rate, weak AUC, high moneyline correlation, and no spread prices |
| Starter Ks v1 | **Research-grade; not yet a ship candidate** | Feasible and better than its point baseline, but Poisson calibration needs stress testing and no K prices exist |

The portfolio contract remains unchanged: these artifacts are offline research outputs only. No production workflow, prediction table schema, or daily-refresh wiring was changed in this run.
