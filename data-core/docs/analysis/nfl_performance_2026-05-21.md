# NFL Performance - 2026-05-22

## BigQuery Model Backtest

Command:

```bash
PYTHONPATH=data-core python3 data-core/scripts/export_nfl_backtest_history.py \
  --project learned-pier-478122-p7 \
  --season 2025 \
  --model-version v1
```

Result: 285 completed NFL games from 2025-09-04 through 2026-02-08. The exporter generated one prediction per game using the saved v1 production artifact, BigQuery `raw_schedules`, and BigQuery `raw_pbp`.

| Metric | Value |
| --- | ---: |
| Accuracy | 53.7% |
| Brier | 0.2648 |
| Log loss | 0.7315 |
| AUC | 0.5790 |
| ECE | 0.1451 |
| Spread MAE | 11.15 |
| Spread RMSE | 13.96 |
| Average predicted home win | 64.5% |
| Actual home win rate | 53.3% |

Machine-readable exports:

- `notebooks/cache/nfl_backtest_2025_v1.csv`
- `notebooks/cache/nfl_backtest_2025_v1_metrics.json`

## Supabase ATS

Latest-prediction-per-game, season 2025: 56 graded games from 2025-10-23 through 2025-11-17. Record: 27-29-0, 48.2% ATS, -4.45 units, -8.0% flat ROI at -110. No graded rows in this subset are missing `book_spread`.

Edge buckets:

| Abs edge bucket | Games | Record | ROI |
| --- | ---: | --- | ---: |
| <1 | 13 | 7-6-0 | 2.8% |
| 1-2 | 10 | 3-7-0 | -42.7% |
| 2-3 | 6 | 3-3-0 | -4.5% |
| 3-5 | 16 | 8-8-0 | -4.5% |
| 5+ | 11 | 6-5-0 | 4.1% |

## Inventory Notes

BigQuery raw schedules and play-by-play are current through 2026-02-08. Daily CI emits NFL `v1`, while Supabase contains v1, v2, and v3 rows. BigQuery `model_predictions` contains NFL v1 rows for playoff/Super Bowl dates and v2 rows for regular season dates.

## Weaknesses

- No NFL BigQuery odds source is documented in the current pipeline path.
- The full-season BigQuery model backtest is not odds-backed; it measures prediction quality only.
- The v1 win-probability output is overconfident toward home teams: average predicted home win 64.5% vs actual 53.3%.
- Supabase has stale model-version skew; latest-row semantics can be fragile if old versions have newer timestamps.
- Feature snapshot table shows likely cross-league contamination and must be fixed before model iteration.
- The available Supabase ATS window is only 56 graded games, not a full-season backtest.
- The current daily workflow runs NFL daily even though the sport is weekly; odds movement handling should be explicit.
