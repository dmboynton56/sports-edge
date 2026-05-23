# PGA Performance - 2026-05-21

Feature store: rebuilt local `notebooks/cache/pga_feature_store_event_level.csv`, 175,743 rows, test split 2025-01-02 through 2026-05-14.

## Test Metrics

Regression target: `target_sg_per_round`.

| Model | N | RMSE | MAE | R2 | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| lgbm | 15,172 | 2.406 | 1.488 | 0.123 | 0.391 |
| rf | 15,172 | 2.410 | 1.489 | 0.120 | 0.389 |
| ridge | 15,172 | 2.429 | 1.504 | 0.106 | 0.381 |
| xgb | 15,172 | 2.574 | 1.653 | -0.004 | 0.388 |

Best calibration by classification target:

| Target | Best model | Base rate | Log loss | Brier | AUC | ECE | Capture@actualK |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| made cut | meta_ensemble | 0.595 | 0.579 | 0.200 | 0.734 | 0.024 | 0.743 |
| top 10 | meta_ensemble | 0.153 | 0.378 | 0.113 | 0.730 | 0.018 | 0.328 |
| top 20 | meta_ensemble | 0.246 | 0.487 | 0.157 | 0.730 | 0.015 | 0.448 |
| win | lr | 0.0058 | 0.033 | 0.0058 | 0.756 | 0.0007 | 0.125 |

Ranking metrics:

| Target | Best ranking model | Primary metrics |
| --- | --- | --- |
| win | lgbm | hit@1 27.3%, hit@5 33.0%, hit@10 40.9% |
| top 10 | lgbm | recall@10 22.1%, precision@10 34.3% |
| top 20 | meta_ensemble | recall@20 33.9%, precision@20 46.2% |

## Masters 2026 Post-Mortem

Pre-event file: `notebooks/cache/masters_2026_predictions.csv`, `as_of=2026-04-09`, latest result at prediction time `2026-04-02`.

Actual Masters rows: 91. Exact name join to pre-event predictions: 39 players, so the event metrics below are directional until player normalization is fixed.

| Metric | Value |
| --- | ---: |
| Spearman predicted SG/R vs actual SG/R, matched players | 0.350 |
| Spearman predicted SG/R vs finish rank, matched players | 0.356 |
| Winner rank by simulated win percent | 9 |
| Winner rank by calibrated win probability | 6 |
| Top 10 predicted by sim win -> actual top 10 | 5/10 |
| Top 20 predicted by sim win -> actual top 20 | 14/20 |

Notable misses: Jon Rahm was ranked first by simulated win probability and finished T38; Scottie Scheffler was ranked fourth and finished second; Rory McIlroy was ranked ninth by simulated win probability and won.

## Weaknesses

- ESPN ingest still misses no-board and same-week collision events.
- Player name normalization is not strong enough for event-level post-mortems.
- Win model calibration and ranking disagree; LR is best by log loss but LGBM is best by hit@K.
- The refreshed store materially changes training distribution; existing v2 models should be retrained only inside a controlled experiment.
- The committed parquet cache is stale and should not be used for current PGA analysis.
