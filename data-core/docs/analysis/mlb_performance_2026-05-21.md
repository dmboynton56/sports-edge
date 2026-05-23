# MLB Performance

Generated: 2026-05-22

## Scope

This is the first MLB winner-model ship candidate. It imports completed regular-season games from the public MLB Stats API, builds rolling team/probable-starter/venue features using only games played before first pitch, and predicts whether the home team wins. The current active artifact is v3, trained through 2025 and evaluated on completed 2026 games through 2026-05-21.

Command:

```bash
PYTHONPATH=data-core python3 data-core/scripts/train_mlb_winner_model.py \
  --start-season 2021 \
  --end-season 2026 \
  --validation-season 2025 \
  --test-season 2026 \
  --cache-path data-core/notebooks/cache/mlb_games_2021_2026.parquet \
  --model-version v3 \
  --output-model data-core/models/mlb_winner_model_v3.pkl
```

## Data

| Item | Value |
| --- | ---: |
| Imported completed games | 12,898 |
| Feature rows after five-prior-game filter | 12,427 |
| Boxscore enrichment rows | 25 smoke-test rows |
| Date range after feature filter | 2021-04-06 to 2026-05-21 |
| Train seasons | 2021-2024 |
| Validation season | 2025 |
| Test season | 2026 YTD |
| Train rows | 9,404 |
| Validation rows | 2,350 |
| Test rows | 673 |
| Overall home win rate | 53.19% |

## Metrics

The selected model is random forest because it had the best 2025 validation Brier score. The final saved model was refit on 2021-2025 and tested on 2026 YTD.

| Model | 2025 validation Brier | 2025 validation log loss | 2026 YTD accuracy | 2026 YTD Brier | 2026 YTD log loss | 2026 YTD AUC | 2026 YTD ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic | 0.2473 | 0.6926 | 54.68% | 0.2513 | 0.7052 | 0.5506 | 0.0262 |
| HistGradientBoosting | 0.2525 | 0.6990 | 54.09% | 0.2499 | 0.6936 | 0.5535 | 0.0345 |
| Random forest | 0.2461 | 0.6852 | 54.09% | 0.2477 | 0.6885 | 0.5462 | 0.0092 |
| Selected random forest refit | n/a | n/a | 53.79% | 0.2478 | 0.6888 | 0.5431 | 0.0120 |
| Home-rate baseline | n/a | n/a | 52.15% | 0.2497 | 0.6925 | 0.5000 | 0.0110 |

## Serving Smoke Test

Command:

```bash
PYTHONPATH=data-core python3 data-core/scripts/predict_mlb_winners.py \
  --date 2026-05-21 \
  --model-path data-core/models/mlb_winner_model_v3.pkl \
  --include-final \
  --output-csv data-core/notebooks/cache/mlb_predictions_2026-05-21_v3.csv
```

Result: scored 7 games and returned home/away win probabilities with probable pitchers. The review CSV is `notebooks/cache/mlb_predictions_2026-05-21_v3.csv`. This proves the artifact can be loaded and used for a dated slate outside the training script.

## Weaknesses

- The edge over the home-rate baseline is real but small: 0.0018 Brier and 0.0037 log-loss improvement on 2026 YTD.
- Probable starter identity/history and venue context are included, but not actual starter line stats, bullpen availability, lineup, injury, weather, umpire, or travel features.
- No moneyline odds are imported, so ROI and market-relative edge are not measured.
- Accuracy beats the 2026 YTD home-rate baseline, but probability quality remains the promotion criterion.
- The feature store resets by season and discards each team's first five games, which is simple but ignores prior-year strength carryover.

## Current Decision

MLB v3 is a ship candidate for a labeled probability surface because import, training, evaluation, artifact persistence, and dated scoring all run locally. It is not a betting recommendation model until moneyline odds and ROI by edge bucket are measured.

## Reproducible Artifacts

- Raw backfill: `scripts/backfill_mlb_raw.py`
- Feature-store build: `scripts/build_mlb_feature_store.py`
- Season backtest: `scripts/backtest_mlb_winners.py`
- Historical moneyline backfill/probe: `scripts/backfill_historical_moneylines.py`
- Data audit notebook: `notebooks/mlb_data_audit.ipynb`
- Training/backtest notebook: `notebooks/mlb_model_training_backtest.ipynb`
- Moneyline ROI notebook: `notebooks/mlb_moneyline_roi.ipynb`

## Historical Odds Status

`scripts/backfill_historical_moneylines.py` successfully reached The Odds API historical endpoint, but the configured key returned `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`. The blocker is data access, not model/backtest plumbing. Once a historical odds source is available, rerun the moneyline backfill and then pass the resulting file to `backtest_mlb_winners.py --odds-path`.
