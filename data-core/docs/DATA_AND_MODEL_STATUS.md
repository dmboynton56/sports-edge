# Data and Model Status

Generated: 2026-05-22. Service inventory spot-checked from the terminal on
2026-05-23.

This inventory is read-only except for the PGA ESPN supplement refresh noted below. BigQuery project: `learned-pier-478122-p7`.

Current consolidated performance hub: `docs/PERFORMANCE_HISTORY.md`, generated from `scripts/export_performance_history.py` and `notebooks/cache/performance_history.json`.

Live 2026-05-23 check: BigQuery `sports_edge_curated.feature_snapshots` had
93,424 rows and `sports_edge_curated.model_predictions` had 972 rows. Latest
NBA `v3` prediction timestamp was 2026-05-23T15:04:38Z; latest NFL `v1`
prediction timestamp was 2026-02-09T14:37:51Z. Supabase serving tables had 586
`games`, 711 `model_predictions`, and 98 `odds_snapshots` rows.

## BigQuery Inventory

Season filter: `season = 2025` for NBA 2025-26 and NFL 2025.

| Area | League | Rows | Min date | Max date | Notes |
| --- | --- | ---: | --- | --- | --- |
| `sports_edge_raw.raw_schedules` | NBA | 1,266 | 2025-10-02 | 2026-05-22 | 1,246 completed games |
| `sports_edge_raw.raw_schedules` | NFL | 285 | 2025-09-04 | 2026-02-08 | 285 completed games |
| `sports_edge_raw.raw_nba_game_logs` | NBA | 2,516 | 2025-10-02 | 2026-05-21 | Loaded by NBA backtest |
| `sports_edge_raw.raw_pbp` | NFL | 97,349 | 2025-09-04 | 2026-02-08 | NFL play-by-play |
| `sports_edge_raw.raw_nba_odds` | NBA | 4,192 | 2024-10-22 | 2026-02-12 | 2,096 spread lines across 2,096 games; stale after Feb. 12 |
| `sports_edge_curated.feature_snapshots` | NBA | 42,905 | 2025-09-04 | 2026-05-22 | `feature_version=v1`; 42,821 null `book_spread` rows |
| `sports_edge_curated.feature_snapshots` | NFL | 42,905 | 2025-09-04 | 2026-05-22 | Same row count/date range as NBA; likely cross-league contamination |
| `sports_edge_curated.model_predictions` | NBA v3 | 455 | 2026-01-30 | 2026-05-21 | Latest `prediction_ts=2026-05-21T16:44:33Z`; 302 null `book_spread` |
| `sports_edge_curated.model_predictions` | NFL v1 | 15 | 2025-11-16 | 2026-02-08 | Latest `prediction_ts=2026-02-09T14:37:51Z`; all null `book_spread` |
| `sports_edge_curated.model_predictions` | NFL v2 | 363 | 2025-09-04 | 2026-01-04 | Latest `prediction_ts=2026-01-14T20:53:07Z`; all null `book_spread` |

NFL full-season model-vs-results export: `notebooks/cache/nfl_backtest_2025_v1_metrics.json` covers 285 completed games through 2026-02-08. v1 accuracy 53.7%, Brier 0.2648, log loss 0.7315, AUC 0.5790, spread MAE 11.15. No NFL BigQuery odds rows were available for ROI.

## Supabase Inventory

| Metric | NBA | NFL |
| --- | ---: | ---: |
| Latest active `asof_ts` | 2026-05-21T16:44:33Z (`v3`) | 2026-02-08T14:11:19Z (`v1`) |
| Rows by model | v3: 497 | v1: 51, v2: 129, v3: 32 |
| Current `book_spread` window (-3/+14 days) | 3 games, 0 missing | 0 games |
| Season games in Supabase | 451 | 133 |
| Season games missing `book_spread` | 284 | 38 |
| Graded games with latest prediction | 64 | 56 |
| Graded games missing `book_spread` | 9 | 0 |
| ATS | 31-33-0 | 27-29-0 |
| ATS pct | 48.4% | 48.2% |
| Flat ROI at -110 | -7.5% | -8.0% |

Supabase ATS uses the notebook convention: `actual_margin = home_score - away_score`, `cover_margin = actual_margin + my_spread`, win if `cover_margin > 0`, loss if `< 0`, push if approximately zero.

## PGA Inventory

| Artifact | Rows | Events | Max start | Notes |
| --- | ---: | ---: | --- | --- |
| `src/data/archive/pga_results_2001-2025.tsv` | 148,883 | 1,142 | 2025-12-11 | Primary archive has no 2026 rows |
| `src/data/archive/pga_results_espn_supplement.tsv` before refresh | 46,532 | 385 | 2026-04-02 | Stopped before Masters |
| `src/data/archive/pga_results_espn_supplement.tsv` after refresh | 47,597 | 396 | 2026-05-14 | Added 1,073 lines net; now includes Masters and PGA Championship |
| Rebuilt `notebooks/cache/pga_feature_store_event_level.csv` | 175,743 | 1,374 | 2026-05-14 | Gitignored local cache |
| Existing committed parquet cache | 41,182 | n/a | 2022-08-11 | Stale and should not be used as current status |

Latest 2026 events now present: Masters Tournament (2026-04-09), RBC Heritage and LIV Golf Mexico City (2026-04-16), Volvo China Open and Zurich Classic (2026-04-23), Cadillac Championship and Turkish Airlines Open (2026-04-30), Estrella Damm Catalunya Championship, LIV Golf Virginia, Truist Championship (2026-05-07), PGA Championship (2026-05-14).

Known ESPN gaps from the refresh: The Sentry had no board; same-week collision skipped Puerto Rico Open and ONEflight Myrtle Beach Classic; Crown Australian Open also collided with Nedbank Golf Challenge. The fetch coverage report still showed `Joaquin Niemann` missing under exact-name matching.

## CBB Inventory

| Artifact | Status |
| --- | --- |
| `data-core/data/cbb/` | Not present |
| Kaggle MMLM raw files | Not present in repo |
| `notebooks/cache/cbb_team_stats.csv` | 5,651 rows; seasons 2010-2026 |
| `notebooks/cache/cbb_matchup_feature_store.csv` | 2,002 rows; seasons 2010-2025 |
| `notebooks/cache/cbb_expanding_cv_2016_2025.json` | 9 folds; XGBoost mean log loss 0.575, Brier 0.198, AUC 0.758 |
| 2026 tournament labels | Not present in matchup feature store |

## MLB Inventory

| Artifact | Status |
| --- | --- |
| Public MLB Stats API import | 12,898 completed regular-season games fetched for 2021-2026 YTD |
| Local cache | `notebooks/cache/mlb_games_2021_2026.parquet`; gitignored by `*.parquet` |
| Feature store | `notebooks/cache/mlb_feature_store_2021_2026.parquet`; 12,427 rows after requiring both teams to have at least five prior same-season games |
| Feature audit | `notebooks/cache/mlb_feature_store_2021_2026_audit.json`; 6 missing home probable pitchers, 7 missing away probable pitchers |
| Boxscore enrichment cache | `notebooks/cache/mlb_boxscores_2021_2025.parquet`; 25 rows fetched as pipeline smoke test, full 12,898-game crawl still pending |
| Max imported game date | 2026-05-21 |
| Model artifact | retained committed `models/mlb_winner_model_v3.pkl` plus `models/mlb_winner_model_v3_metrics.json` |
| Test window | 2026 YTD, 673 feature rows |
| Ship-candidate result | Random forest selected on 2025 validation; 2026 YTD refit test accuracy 53.8%, Brier 0.2478, log loss 0.6888, AUC 0.5431 |
| Backtest outputs | `notebooks/cache/mlb_backtest_predictions_2025.csv`, `notebooks/cache/mlb_backtest_metrics_2025.json`, `notebooks/cache/mlb_backtest_predictions_2026_ytd.csv`, and `notebooks/cache/mlb_backtest_metrics_2026_ytd.json`; no odds rows yet |
| Historical odds probe | `notebooks/cache/mlb_moneylines_historical_audit.json`; `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN` from The Odds API |
| Serving smoke test | `predict_mlb_winners.py --date 2026-05-21 --include-final --model-path models/mlb_winner_model_v3.pkl` scored 7 games with probable pitchers and wrote `notebooks/cache/mlb_predictions_2026-05-21_v3.csv` |
| MLB notebooks | `mlb_data_audit.ipynb`, `mlb_model_training_backtest.ipynb`, `mlb_moneyline_roi.ipynb`, `sports_model_performance_hub.ipynb` |

## Gap Table

| Sport | Gap | Severity | Fix |
| --- | --- | --- | --- |
| NBA | `raw_nba_odds` maxes at 2026-02-12 while schedules/predictions continue into May | High | OddsPapi tail patch added May 2026 spreads; continue backfill for Feb–Apr gap |
| NBA | Supabase season has historical games missing `book_spread` | High | Same as above; current Odds API window is clean but cannot fill the older season gap |
| NBA/NFL | `feature_snapshots` has identical NBA/NFL counts and NFL rows through 2026-05-22 | High | Audit `build_feature_snapshots.py` output and delete/rebuild contaminated rows only with approval |
| NFL | Supabase has v1/v2/v3 rows while daily CI now emits v1 | Medium | Keep v1 active; archive or filter stale v2/v3 rows in docs/API if not needed |
| NFL | No NFL `book_spread` source in BigQuery path | Medium | Add/restore NFL spread source before using BQ edge analysis |
| PGA | ESPN gaps for no-board and same-week events | Medium | Add DataGolf or another source for skipped PGA/LIV/DP World events |
| PGA | Committed parquet cache is stale to 2022-08-11 | Medium | Prefer rebuilt CSV; regenerate parquet only if consumers still require it |
| PGA | Pre-Masters prediction name matching only joined 39/91 actual Masters rows exactly | Medium | Add player ID/name normalization before event post-mortems |
| CBB | Raw Kaggle MMLM files absent | High | Restore Kaggle raw data under ignored local `data-core/data/cbb/`, rebuild team stats and matchup feature store |
| CBB | Major-underdog probabilities are too high after de-duplicating mirrored rows | Medium | Calibrate upset tails or add seed-prior constraints before using bracket upset probabilities |
| MLB | Full boxscore enrichment is only smoke-tested for 25/12,898 games | Medium | Run `backfill_mlb_raw.py --fetch-boxscores` without `--limit-boxscores`; then rebuild feature store |
| MLB | v3 winner model has rolling team, probable starter, and venue features, but no bullpen availability, lineup, injury, weather, or market features | Medium | Add the missing pregame inputs incrementally and require held-out Brier/log-loss lift |
| MLB | OddsPapi moneyline archive covers recent 2026 window only (69 games) | Medium | Resume `backfill_oddspapi_moneylines.py` across quota windows for full-season ROI |
| NBA | `raw_nba_odds` Feb–Apr 2026 gap remains after OddsPapi May tail patch (+22 rows) | High | Continue OddsPapi spread backfill as historical coverage/quota allows |
