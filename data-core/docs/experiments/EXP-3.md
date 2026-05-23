# EXP-3 MLB Starter/Venue Ship Candidate

Date: 2026-05-22

## Hypothesis

Adding probable starter rolling history and venue context should improve the MLB winner model enough to make it usable as a probability-display ship candidate, even if it is not ready for betting recommendations.

## Setup

- Source: public MLB Stats API schedule endpoint with `team`, `probablePitcher`, and `venue` hydration.
- Seasons imported: 2021-2026 YTD regular season.
- Features added over v1: probable starter known flag, prior starts, team win rate in starter games, runs allowed/support per start, recent starter-game run differential, starter rest, prior venue home win rate, and prior venue total runs.
- Split: train 2021-2024, validate 2025, test 2026 YTD.
- Selection metric: 2025 validation Brier.
- Serving smoke test: `predict_mlb_winners.py --date 2026-05-21 --include-final`.

## Result

| Candidate | 2025 validation Brier | 2026 YTD test Brier | 2026 YTD test log loss | 2026 YTD test AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.2473 | 0.2513 | 0.7052 | 0.5506 |
| HistGradientBoosting | 0.2525 | 0.2499 | 0.6936 | 0.5535 |
| Random forest | 0.2461 | 0.2477 | 0.6885 | 0.5462 |
| Selected random forest refit | n/a | 0.2478 | 0.6888 | 0.5431 |
| Home-rate baseline | n/a | 0.2497 | 0.6925 | 0.5000 |

The dated prediction CLI scored 7 games for 2026-05-21 from the saved v3 artifact and wrote `notebooks/cache/mlb_predictions_2026-05-21_v3.csv`.

## Decision

Ship candidate for probability display only. The software path is usable: import, feature build, training, metrics sidecar, saved model, season backtest, and slate scoring all work. Do not ship as betting advice until moneyline odds and ROI are measured.

## Promotion Gate

Before exposing MLB v3 in production:

1. Label output as model probability, not betting edge.
2. Add moneyline ingestion before showing EV or recommended bets.
3. Re-evaluate after adding bullpen, lineup, weather, and actual starter line stats.
