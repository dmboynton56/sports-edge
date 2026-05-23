# NBA Performance - 2026-05-21

## BigQuery Backtest

Command:

```bash
cd data-core
PYTHONWARNINGS=ignore python3 scripts/backtest_nba_spread.py \
  --project learned-pier-478122-p7 \
  --season 2025 \
  --start-date 2025-10-22 \
  --model-version v3 \
  --output-csv notebooks/cache/nba_backtest_2025_v3.csv
```

Range: 2025-10-22 through 2026-05-21. Completed games: 1,175. Predictions generated: 1,175. Joined with book odds: 717.

Default strategy (`edge_threshold=1.0`, `min_confidence=0.0`): 591 bets, 301 wins, 50.9% accuracy, -0.5% ROI.

Best cells in the printed sweep were confidence-filtered:

| Edge threshold | Min confidence | Bets | ROI |
| ---: | ---: | ---: | ---: |
| 0.5 | 0.4 | 112 | 8.3% |
| 1.0 | 0.4 | 94 | 6.3% |
| 2.0 | 0.4 | 76 | 5.9% |
| 2.0 | 0.2 | 313 | 2.5% |

## Supabase ATS

Latest-prediction-per-game, season 2025: 64 graded games from 2026-04-20 through 2026-05-21. Record: 31-33-0, 48.4% ATS, -4.82 units, -7.5% flat ROI at -110. Nine graded games are missing `book_spread`.

Edge buckets using rows with book lines:

| Abs edge bucket | Games | Record | ROI |
| --- | ---: | --- | ---: |
| <1 | 4 | 3-1-0 | 43.2% |
| 1-2 | 6 | 2-4-0 | -36.4% |
| 2-3 | 2 | 1-1-0 | -4.5% |
| 3-5 | 13 | 4-9-0 | -41.3% |
| 5+ | 30 | 18-12-0 | 14.5% |

## Weaknesses

- BigQuery raw odds stop at 2026-02-12, so only 717 of 1,175 completed games joined to odds.
- Supabase season has 284/451 NBA rows missing `book_spread`.
- The backtest uses local sklearn artifacts with version warnings; pinning the training/runtime version would reduce reproducibility risk.
- Confidence filtering looks promising but needs out-of-sample validation before shipping a betting rule.
- Feature snapshot league contamination must be audited before retraining or promoting new versions.
