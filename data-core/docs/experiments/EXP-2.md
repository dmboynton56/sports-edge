# EXP-2 MLB Results-Only Winner Baseline

Date: 2026-05-21

## Hypothesis

A leakage-safe rolling team form model should beat a constant home-win baseline for MLB game winner prediction, but it will probably be too weak to promote without pitchers and odds.

## Setup

- Source: public MLB Stats API schedule/results.
- Seasons imported: 2021-2025 regular season.
- Features: same-season rolling win rate, run differential, runs for/against, venue split, last-10 form, rest, doubleheader flag, month/day-of-week.
- Split: train 2021-2023, validate 2024, test 2025.
- Selection metric: 2024 validation Brier.

## Result

| Candidate | 2024 validation Brier | 2025 test Brier | 2025 test log loss | 2025 test AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.2451 | 0.2466 | 0.6864 | 0.5529 |
| HistGradientBoosting | 0.2511 | 0.2533 | 0.7010 | 0.5372 |
| Random forest | 0.2454 | 0.2459 | 0.6849 | 0.5576 |
| Selected logistic refit | n/a | 0.2465 | 0.6861 | 0.5537 |
| Home-rate baseline | n/a | 0.2484 | 0.6900 | 0.5000 |

## Decision

No ship. The results-only MLB baseline is a useful scaffold and beats the constant baseline modestly, but the lift is not strong enough for serving or ROI claims.

## Next

Add starting pitcher features, bullpen usage, park/weather, and moneyline odds keyed by `game_pk` before retesting. Promotion should require improved Brier/log loss and positive moneyline ROI by edge bucket on a held-out season.
