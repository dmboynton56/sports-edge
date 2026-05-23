# EXP-1: NBA Confidence Filter for Spread Edges

Date: 2026-05-21

## Hypothesis

The NBA v3 spread model has useful edge signal only when the model confidence is high enough. Adding a confidence floor should improve flat ROI versus betting every edge.

## Data

`scripts/backtest_nba_spread.py`, season 2025, full-season range 2025-10-22 through 2026-05-20, model version `v3`.

Completed games: 1,174. Games joined with book odds: 717.

## Results

| Rule | Bets | ROI |
| --- | ---: | ---: |
| abs(edge) > 1.0, confidence >= 0.0 | 591 | -0.5% |
| abs(edge) > 1.0, confidence >= 0.2 | 390 | 2.3% |
| abs(edge) > 1.0, confidence >= 0.4 | 94 | 6.3% |
| abs(edge) > 0.5, confidence >= 0.4 | 112 | 8.3% |

## Decision

No ship yet.

The confidence filter improves this backtest, but the high-confidence sample is small and odds coverage is incomplete after 2026-02-12. Treat this as a candidate rule for a locked-forward validation, not as a production betting filter.

## Next Measurement

Backfill historical NBA playoff odds, rerun the exact same sweep, and compare against a no-confidence baseline on the same complete odds set.
