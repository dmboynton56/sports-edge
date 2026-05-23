# Free Historical Odds Audit

Generated: 2026-05-23

Goal: exhaust free routes before buying a historical odds API.

## Current Result

Free route status:

- NFL 2025 spreads are covered without a paid odds API.
- NBA 2026 tail spreads are covered without a paid odds API, except four All-Star/exhibition rows.
- MLB 2025 moneylines are covered well enough for ROI backtesting without a paid odds API.
- MLB 2026 YTD moneylines are covered without a paid odds API.
- OddsPapi free remains useful but quota-limited and not dependable for the full backfill.

Current free coverage from repo-generated audits:

| Sport/source | Target rows | Free matched rows | Coverage |
| --- | ---: | ---: | ---: |
| NFL 2025 via nflverse | 285 | 285 | 100.0% |
| NBA 2026 tail via FantasyData | 357 | 353 | 98.9% |
| MLB 2025 via CheckBestOdds + MLB 2026 via FantasyData | 3,180 | 3,137 | 98.6% |

Combined current free coverage: 3,775 of 3,822 target rows, or 98.8%. This clears the original "free/cheap routes cover at least 80 percent" gate. Do not buy a historical odds API for ROI backtesting right now.

## NFL: Pass

Source: `nflreadpy.load_schedules([2025])`, backed by nflverse schedule data.

Generated:

- `data-core/scripts/export_nfl_nflverse_spreads.py`
- `data-core/notebooks/cache/nfl_nflverse_spreads_2025.csv`
- `data-core/notebooks/cache/nfl_nflverse_spreads_2025_audit.json`

Evidence:

- Requested games: 285
- Exported spread rows: 285
- Missing games: 0
- Null home-perspective line rows: 0
- Null home spread price rows: 0

Important convention: nflverse `spread_line` is positive when the home team is favored, while this repo expects home perspective where a home favorite is negative. The export writes `line = -spread_line`.

Limitation: this is `nflverse_pfr`, not a named sportsbook or Pinnacle. It is good enough for a first NFL ROI pass, but not ideal for sharp-book CLV.

## NBA: Pass For ROI Tail

Source: public FantasyData team odds pages, for example `https://fantasydata.com/nba/cleveland-cavaliers-odds`.

Generated:

- `data-core/scripts/export_nba_fantasydata_spreads.py`
- `data-core/notebooks/cache/nba_fantasydata_spreads_2026_tail.csv`
- `data-core/notebooks/cache/nba_fantasydata_spreads_2026_tail_audit.json`

Evidence:

- Target games from `nba_backtest_2025_v3.csv`, 2026-02-13 through 2026-05-21: 357
- Exported spread rows: 353
- Missing rows: 4
- Missing rows are All-Star/exhibition-style teams: `STARS`, `WORLD`, `STRIPES`

Limitation: this is FantasyData consensus, not Pinnacle. It is public HTML, not a supported bulk/API feed.

## MLB: Pass For ROI

Source: CheckBestOdds MLB historical pages, `https://checkbestodds.com/baseball-odds/historical-odds-usa-mlb/{year}`.

Generated:

- `data-core/scripts/export_mlb_checkbestodds_moneylines.py`
- `data-core/scripts/export_mlb_fantasydata_moneylines.py`
- `data-core/notebooks/cache/mlb_checkbestodds_moneylines_2025_2026.csv`
- `data-core/notebooks/cache/mlb_checkbestodds_moneylines_2025_2026_audit.json`
- `data-core/notebooks/cache/mlb_fantasydata_moneylines_2026_ytd.csv`
- `data-core/notebooks/cache/mlb_fantasydata_moneylines_2026_ytd_audit.json`
- `data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv`
- `data-core/notebooks/cache/mlb_free_moneylines_2025_2026_audit.json`
- `data-core/notebooks/cache/mlb_backtest_metrics_2025_checkbestodds.json`
- `data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd_checkbestodds.json`
- `data-core/notebooks/cache/mlb_backtest_metrics_2025_free.json`
- `data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd_free.json`

Evidence:

- 2025 regular-season target: 2,430 games; matched: 2,387, 98.2%.
- 2026 YTD target through 2026-05-21 from FantasyData: 750 games; matched: 750, 100.0%.
- Combined free MLB target: 3,180 games; matched: 3,137, 98.6%.

Backtest verification:

```bash
PYTHONPATH=data-core python3 data-core/scripts/backtest_mlb_winners.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_2021_2026.parquet \
  --validation-season 2024 \
  --test-season 2025 \
  --odds-path data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv \
  --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2025_free.csv \
  --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2025_free.json
```

Result: `odds_rows=2309`, flat ROI `-8.0%`.

```bash
PYTHONPATH=data-core python3 data-core/scripts/backtest_mlb_winners.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_2021_2026.parquet \
  --validation-season 2025 \
  --test-season 2026 \
  --odds-path data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv \
  --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd_free.csv \
  --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd_free.json
```

Result: `odds_rows=673`, flat ROI `-3.1%`.

Limitation: these are public HTML consensus/odds-comparison sources, not named sharp books and not supported APIs. CheckBestOdds decimal prices are converted to American moneylines; FantasyData MLB prices are already American.

## Kaggle/Public CSV: Fail For Required Gaps

Downloaded and inspected these public Kaggle datasets with `kagglehub`:

1. `oliviersportsdata/us-sports-master-historical-closing-odds`
2. `ehallmar/nba-historical-stats-and-betting-data`
3. `thedevastator/uncovering-hidden-trends-in-nba-betting-lines-20`

Findings:

- `oliviersportsdata/us-sports-master-historical-closing-odds` only downloads 50-row sample CSVs per sport. Its Kaggle description says the full dataset is off-platform on Gumroad. It is also moneyline-only for NBA/NFL/MLB samples, so it does not satisfy NBA/NFL spread requirements.
- `ehallmar/nba-historical-stats-and-betting-data` has real NBA spread rows, including Pinnacle Sports, but joins to games only from 2006-11-01 through 2018-06-08. It cannot fill the 2026-02-13 to 2026-05-21 NBA gap.
- `thedevastator/uncovering-hidden-trends-in-nba-betting-lines-20` has NBA spread CSVs only from 2022-10-18 through 2022-12-11 in the downloaded files. It cannot fill the 2026 NBA gap.

These are useful references for schema ideas, but CheckBestOdds and FantasyData are stronger free sources for the current repo gaps.

## ParlayAPI Free Tier: Not Needed

ParlayAPI exposes unauthenticated historical coverage stats, but historical price endpoints require an API key. Its public pricing page says the free tier has a 48-hour historical window, so it is not a solution for 2025 or early-2026 backfills.

Coverage stats checked from `https://parlay-api.com/v1/historical/stats`:

- `baseball_mlb`: earliest 2010-04-04, latest 2021-11-02.
- `basketball_nba`: earliest 2007-10-30, latest 2023-01-16.
- `americanfootball_nfl`: earliest 1999-09-12, latest 2025-02-09.

Those stats do not cover the remaining 2026 MLB/NBA needs, and the free tier window is too shallow anyway.

## OddsPapi Free: Partial/Quota-Limited

Existing repo evidence:

- NBA OddsPapi free tail cache: 22 spread rows from late April/May 2026.
- MLB OddsPapi free cache: 69 moneyline rows from May 2026.
- NFL OddsPapi free cache: 0 rows for the 2025 archive; prior audit notes quota/rate limit exhaustion.

Additional probe on 10 older 2026-02 NBA games hit `OddsPapiRateLimited: OddsPapi rate limit retries exhausted` before market catalog loading. No additional free rows were produced.

Verdict: continue using OddsPapi free opportunistically after quota resets, but do not plan the full ROI backfill around it.

## Remaining Uncovered Gaps

| Sport | Required gap | Free status |
| --- | ---: | --- |
| NFL | 2025 full season, 285 games | Covered by nflverse export |
| NBA | 2026-02-13 through 2026-05-21, 357 rows in local backtest file | Covered for 353 rows by FantasyData; four missing exhibition rows |
| MLB | 2025 full regular season, 2,430 rows in local schedule cache | Covered for 2,387 rows by CheckBestOdds |
| MLB | 2026 YTD through 2026-05-21, 750 rows in local schedule cache | Covered for 750 rows by FantasyData |

## Next Free Checks

Before buying an API, the only free/cheap checks left that may still be worth doing are:

1. Optionally load `nba_fantasydata_spreads_2026_tail.csv` into BigQuery `raw_nba_odds` after reviewing source quality.
2. Optionally add a tiny parity check against existing Pinnacle/Kaggle NBA rows before 2026-02-12 to quantify FantasyData line differences.
3. Optionally search for the 43 missing 2025 MLB rows, but they are not blocking ROI backtests.

Paid data is justified only if the product specifically requires supported API terms, sharp-book/Pinnacle provenance, or exact 100% historical coverage. It is not justified for ROI backtesting coverage.

## Source Links

- nflverse schedules dictionary: https://nflreadr.nflverse.com/articles/dictionary_schedules.html
- nflreadpy: https://github.com/nflverse/nflreadpy
- Kaggle US Sports Master sample/full-dataset description: https://www.kaggle.com/datasets/oliviersportsdata/us-sports-master-historical-closing-odds
- Kaggle NBA historical stats and betting data: https://www.kaggle.com/datasets/ehallmar/nba-historical-stats-and-betting-data
- Kaggle NBA betting lines dataset: https://www.kaggle.com/datasets/thedevastator/uncovering-hidden-trends-in-nba-betting-lines-20
- CheckBestOdds MLB historical odds: https://checkbestodds.com/baseball-odds/historical-odds-usa-mlb
- FantasyData NBA odds: https://fantasydata.com/nba/odds
- FantasyData MLB odds: https://fantasydata.com/mlb/odds
- Scottfree free sample page: https://shop.scottfreellc.com/shop/p/historical-odds-sample-data
