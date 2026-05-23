# Data Sources

Generated: 2026-05-22

| Source | Sports | Cadence | Keys | Current failure modes |
| --- | --- | --- | --- | --- |
| BigQuery `sports_edge_raw.raw_schedules` | NBA, NFL | Daily via `.github/workflows/daily-refresh.yml` | `league`, `season`, `game_id`, `game_date`, teams | Feature snapshot table appears cross-contaminated; do not delete/rebuild without approval |
| BigQuery `sports_edge_raw.raw_nba_game_logs` | NBA | Daily via `scripts/backfill_nba_raw.py` | `season`, `game_id`, `team`, `game_date` | Present through 2026-05-21 |
| BigQuery `sports_edge_raw.raw_pbp` | NFL | Daily via `scripts/backfill_nfl_raw.py --replace` | `season`, `game_id`, `play_id`, `game_date` | Present through Super Bowl date 2026-02-08 |
| BigQuery `sports_edge_raw.raw_nba_odds` | NBA | Current/prediction-time fetch plus historical CSV backfills | `season`, `game_id`, `market`, `book` | OddsPapi tail patch added 22 spread rows for May 2026; Feb–Apr 2026 gap remains |
| OddsPapi v4 historical API | MLB, NBA, NFL | Manual via `scripts/backfill_oddspapi_*.py` | `fixtureId`, team+date join, Pinnacle closing lines | Free tier: ~250 req/mo; historical coverage limited to recent fixtures (~May 2026+) |
| The Odds API | NBA, NFL | Daily current serving window | commence time, home/away team, bookmaker, spread outcome | Current-only endpoint does not repair old missing season rows |
| Supabase `games`, `model_predictions`, `odds_snapshots` | NBA, NFL | Daily BQ sync plus odds/scores repair | `games.id`, game date/team tuple, `model_predictions.game_id` | NBA season rows missing `book_spread`; NFL has stale model-version rows |
| ESPN golf scoreboard/calendar | PGA, LIV, DP World | Manual via `scripts/fetch_espn_pga_results.py --season 2026` | `season`, `start`, `tournament`, `name` | No-board events and same-week collisions skip some events |
| PGA archive TSV | PGA | Manual archive updates | `season`, `start`, `tournament`, `name` | Primary file stops at 2025-12-11; supplement carries 2026 |
| Kaggle March Machine Learning Mania | CBB | Manual local restore | season, team IDs, seeds, tournament slots/results | Raw files are not present in repo; caches exist only as derived artifacts |
| MLB Stats API schedule endpoint | MLB | Manual via `scripts/backfill_mlb_raw.py`; serving smoke test via `scripts/predict_mlb_winners.py` | `game_pk`, `season`, team IDs, `game_date`, probable pitcher IDs, venue ID | No confirmed lineup, odds, weather, or injuries yet |
| MLB Stats API boxscore endpoint | MLB | Manual optional enrichment via `scripts/backfill_mlb_raw.py --fetch-boxscores` | `game_pk`, actual starter IDs, pitcher lines, bullpen usage | Only 25 rows fetched as a smoke test; full historical crawl pending |
| The Odds API historical endpoint | MLB, NBA, NFL | Manual via `scripts/backfill_historical_moneylines.py` | sport key, snapshot timestamp, event/team names, matched `game_pk`, bookmaker, moneyline | Superseded for historical work by OddsPapi v4; Odds API key still returns `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN` |

## Refresh Commands

OddsPapi validation and backfill:

```bash
PYTHONPATH=data-core python3 data-core/scripts/validate_oddspapi_spike.py
PYTHONPATH=data-core python3 data-core/scripts/backfill_oddspapi_moneylines.py --start-date 2026-05-15 --end-date 2026-05-21
PYTHONPATH=data-core python3 data-core/scripts/backfill_oddspapi_nba_spreads.py --limit-games 50 --load-bq --project learned-pier-478122-p7
PYTHONPATH=data-core python3 data-core/scripts/backfill_oddspapi_nfl_spreads.py --limit-games 285 --resume
```

Performance hub export:

```bash
python3 data-core/scripts/export_performance_history.py
```

Supabase ATS/ROI export:

```bash
PYTHONPATH=data-core python3 data-core/scripts/export_supabase_ats_history.py \
  --season 2025 \
  --leagues NBA NFL \
  --output-dir data-core/notebooks/cache
```

NFL full-season model backtest:

```bash
PYTHONPATH=data-core python3 data-core/scripts/export_nfl_backtest_history.py \
  --project learned-pier-478122-p7 \
  --season 2025 \
  --model-version v1 \
  --output-csv data-core/notebooks/cache/nfl_backtest_2025_v1.csv \
  --metrics-output data-core/notebooks/cache/nfl_backtest_2025_v1_metrics.json
```

PGA:

```bash
cd data-core
python3 scripts/fetch_espn_pga_results.py --season 2026
PYTHONPATH=. python3 -m src.data.build_pga_feature_store
```

Supabase validation:

```bash
cd data-core
python3 scripts/validate_supabase_sync.py --strict
```

NBA backtest:

```bash
cd data-core
PYTHONWARNINGS=ignore PYTHONPATH=. python3 scripts/export_nba_backtest_history.py \
  --project learned-pier-478122-p7 \
  --season 2025 \
  --start-date 2025-10-22 \
  --model-version v3 \
  --output-csv notebooks/cache/nba_backtest_2025_v3.csv \
  --metrics-output notebooks/cache/nba_backtest_2025_v3_metrics.json
```

CBB CV without overwriting saved models:

```bash
PYTHONPATH=data-core python3 data-core/scripts/export_cbb_cv_history.py \
  --feature-store data-core/notebooks/cache/cbb_matchup_feature_store.csv \
  --models-dir /tmp/sports-edge-cbb-cv-models \
  --folds-output data-core/notebooks/cache/cbb_expanding_cv_2016_2025.csv \
  --summary-output data-core/notebooks/cache/cbb_expanding_cv_2016_2025.json
```

MLB import, train, and test:

```bash
PYTHONPATH=data-core python3 data-core/scripts/backfill_mlb_raw.py \
  --start-season 2021 \
  --end-season 2026 \
  --games-cache data-core/notebooks/cache/mlb_games_2021_2026.parquet \
  --refresh-games

PYTHONPATH=data-core python3 data-core/scripts/build_mlb_feature_store.py \
  --games-cache data-core/notebooks/cache/mlb_games_2021_2026.parquet \
  --output data-core/notebooks/cache/mlb_feature_store_2021_2026.parquet \
  --audit-output data-core/notebooks/cache/mlb_feature_store_2021_2026_audit.json

PYTHONPATH=data-core python3 data-core/scripts/backtest_mlb_winners.py \
  --features-path data-core/notebooks/cache/mlb_feature_store_2021_2026.parquet \
  --validation-season 2025 \
  --test-season 2026 \
  --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd.csv \
  --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd.json

PYTHONPATH=data-core python3 data-core/scripts/train_mlb_winner_model.py \
  --start-season 2021 \
  --end-season 2026 \
  --validation-season 2025 \
  --test-season 2026 \
  --cache-path data-core/notebooks/cache/mlb_games_2021_2026.parquet \
  --model-version v3 \
  --output-model data-core/models/mlb_winner_model_v3.pkl
```

MLB score a date with the saved artifact:

```bash
PYTHONPATH=data-core python3 data-core/scripts/predict_mlb_winners.py \
  --date 2026-05-21 \
  --model-path data-core/models/mlb_winner_model_v3.pkl \
  --include-final \
  --output-csv data-core/notebooks/cache/mlb_predictions_2026-05-21_v3.csv
```

MLB optional boxscore enrichment:

```bash
PYTHONPATH=data-core python3 data-core/scripts/backfill_mlb_raw.py \
  --start-season 2021 \
  --end-season 2026 \
  --games-cache data-core/notebooks/cache/mlb_games_2021_2026.parquet \
  --boxscores-cache data-core/notebooks/cache/mlb_boxscores_2021_2026.parquet \
  --fetch-boxscores
```

Historical moneyline probe/backfill:

```bash
PYTHONPATH=data-core python3 data-core/scripts/backfill_historical_moneylines.py \
  --sport MLB \
  --games-path data-core/notebooks/cache/mlb_games_2021_2026.parquet \
  --start-date 2025-09-28 \
  --end-date 2025-09-28 \
  --limit-dates 1 \
  --output data-core/notebooks/cache/mlb_moneylines_historical.csv \
  --audit-output data-core/notebooks/cache/mlb_moneylines_historical_audit.json
```
