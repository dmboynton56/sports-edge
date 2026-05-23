# Prediction and ROI History

Generated: 2026-05-22

Consolidated machine-readable history: `notebooks/cache/performance_history.json`; markdown summary: `docs/PERFORMANCE_HISTORY.md`.

## Conventions

Spread values are home-team perspective. Example: `book_spread = -5.5` means the home team is favored by 5.5.

For Supabase ATS notebooks:

```text
actual_margin = home_score - away_score
cover_margin = actual_margin + my_spread
win  = cover_margin > 0
push = abs(cover_margin) < 0.001
loss = cover_margin < 0
```

Flat ROI at -110:

```text
risked_games = wins + losses
net_units = wins * (100 / 110) - losses
roi = net_units / risked_games
```

## Supabase Season 2025 Results

| League | Graded range | Latest prediction | Record | ATS pct | Net units | ROI | Missing book spread in graded rows |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| NBA | 2026-04-20 to 2026-05-21 | 2026-05-21T16:44:33Z | 31-33-0 | 48.4% | -4.82 | -7.5% | 9 |
| NFL | 2025-10-23 to 2025-11-17 | 2025-11-18T01:07:59Z | 27-29-0 | 48.2% | -4.45 | -8.0% | 0 |

These are latest-prediction-per-game results from Supabase `games` joined to `model_predictions`, matching `nba_ats_roi.ipynb` and `nfl_ats_roi.ipynb`.

Machine-readable exports:

- `notebooks/cache/nba_supabase_ats_summary_2025.json`
- `notebooks/cache/nba_supabase_ats_games_2025.csv`
- `notebooks/cache/nfl_supabase_ats_summary_2025.json`
- `notebooks/cache/nfl_supabase_ats_games_2025.csv`

## NFL BigQuery Model Backtest

Command used:

```bash
PYTHONPATH=data-core python3 data-core/scripts/export_nfl_backtest_history.py \
  --project learned-pier-478122-p7 \
  --season 2025 \
  --model-version v1
```

Result: 285 completed games from 2025-09-04 through 2026-02-08, with 285 v1 predictions generated from `raw_schedules` and `raw_pbp`.

| Metric | Value |
| --- | ---: |
| Accuracy | 53.7% |
| Brier | 0.2648 |
| Log loss | 0.7315 |
| AUC | 0.5790 |
| ECE | 0.1451 |
| Spread MAE | 11.15 |
| Spread RMSE | 13.96 |

This is not an ATS/ROI result because no NFL BigQuery spread-odds source is documented. The Supabase ATS sample remains the only odds-backed NFL result in this repo.

## NBA BigQuery Backtest

Benchmark command:

```bash
cd data-core
PYTHONWARNINGS=ignore python3 scripts/backtest_nba_spread.py \
  --project learned-pier-478122-p7 \
  --season 2025 \
  --start-date 2025-10-22 \
  --model-version v3 \
  --output-csv notebooks/cache/nba_backtest_2025_v3.csv
```

Result: 1,175 completed games from 2025-10-22 through 2026-05-21, 1,175 predictions generated, 717 games joined with book odds from 767 raw odds rows.

Default strategy (`edge_threshold=1.0`, `min_confidence=0.0`): 591 bets, 301 wins, 50.9% accuracy, -0.5% ROI.

Threshold sweep from the script:

| Edge threshold | Min confidence | Bets | ROI |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.0 | 659 | -0.8% |
| 0.5 | 0.2 | 437 | 2.0% |
| 0.5 | 0.4 | 112 | 8.3% |
| 1.0 | 0.0 | 591 | -0.5% |
| 1.0 | 0.2 | 390 | 2.3% |
| 1.0 | 0.4 | 94 | 6.3% |
| 1.5 | 0.0 | 537 | -0.7% |
| 1.5 | 0.2 | 350 | 2.2% |
| 1.5 | 0.4 | 85 | 1.5% |
| 2.0 | 0.0 | 481 | 0.7% |
| 2.0 | 0.2 | 313 | 2.5% |
| 2.0 | 0.4 | 76 | 5.9% |

Machine-readable exports:

- `notebooks/cache/nba_backtest_2025_v3.csv`
- `notebooks/cache/nba_backtest_2025_v3_metrics.json`

## MLB Winner Backtest

Command used:

```bash
PYTHONPATH=data-core python3 data-core/scripts/backtest_mlb_winners.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_2021_2025.parquet \
  --validation-season 2024 \
  --test-season 2025 \
  --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2025.csv \
  --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2025.json
```

Benchmark result: 2,350 tested 2025 regular-season games. Selected model was random forest from the 2024 validation split, refit on 2021-2024 and tested on 2025.

| Metric | MLB v2 | Home-rate baseline |
| --- | ---: | ---: |
| Accuracy | 53.8% | 54.1% |
| Brier | 0.2461 | 0.2484 |
| Log loss | 0.6852 | 0.6900 |
| AUC | 0.5542 | 0.5000 |
| ECE | 0.0255 | 0.0110 |

Current command:

```bash
PYTHONPATH=data-core python3 data-core/scripts/backtest_mlb_winners.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_2021_2026.parquet \
  --validation-season 2025 \
  --test-season 2026 \
  --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd.csv \
  --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd.json
```

Current v3 result: 673 tested 2026 YTD regular-season games through 2026-05-21. Selected model was random forest from the 2025 validation split, refit on 2021-2025 and tested on 2026 YTD.

| Metric | MLB v3 | Home-rate baseline |
| --- | ---: | ---: |
| Accuracy | 53.8% | 52.2% |
| Brier | 0.2478 | 0.2497 |
| Log loss | 0.6888 | 0.6925 |
| AUC | 0.5431 | 0.5000 |
| ECE | 0.0120 | 0.0110 |

Odds-backed ROI is now partially available via OddsPapi (69 matched 2026 YTD games, +3.4% flat ROI):

```bash
PYTHONPATH=data-core python3 data-core/scripts/backtest_mlb_winners.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_2021_2026.parquet \
  --validation-season 2025 \
  --test-season 2026 \
  --odds-path data-core/notebooks/cache/mlb_oddspapi_moneylines_2026_ytd.csv \
  --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd.csv \
  --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd.json
```

Result: `odds_rows=69`, flat ROI +3.4%, flat units +2.38. Coverage is limited to the recent OddsPapi historical window (May 2026) on the free tier.

Historical odds probe (Odds API — blocked):

```bash
PYTHONPATH=data-core python3 data-core/scripts/backfill_historical_moneylines.py \
  --sport MLB \
  --games-path data-core/notebooks/cache/mlb_games_2021_2026.parquet \
  --start-date 2025-09-28 \
  --end-date 2025-09-28 \
  --limit-dates 1
```

Result: The configured Odds API key returned `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`. The audit is stored at `notebooks/cache/mlb_moneylines_historical_audit.json`. This blocks real MLB sportsbook ROI until a paid historical endpoint or equivalent historical CSV export is available. When access exists, `backfill_historical_moneylines.py` writes `game_pk`, `home_moneyline`, and `away_moneyline`, so its output can be passed directly to `backtest_mlb_winners.py --odds-path`.

## CBB Tournament CV

Command used:

```bash
PYTHONPATH=data-core python3 data-core/scripts/export_cbb_cv_history.py
```

Result: 9 expanding-window folds from validation years 2016-2025, excluding 2020. The cache contains 2,002 mirrored matchup rows and no 2026 tournament labels.

| Model | Mean log loss | Mean Brier | Mean AUC | Mean accuracy |
| --- | ---: | ---: | ---: | ---: |
| LGBM | 0.5813 | 0.2003 | 0.7474 | 67.7% |
| XGB | 0.5751 | 0.1979 | 0.7581 | 68.3% |
| Meta | 0.5952 | 0.2043 | 0.7564 | 68.8% |

There is no CBB sportsbook odds or ROI history in the repo. The raw Kaggle MMLM files are also absent, so the cache is measurable but not fully rebuildable from source here.
