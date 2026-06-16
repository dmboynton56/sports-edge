# World Cup Prediction Pipeline

Generated: 2026-06-13
Last live sync: 2026-06-16

## Scope

World Cup support is a Sports Edge extension, not a separate project. The
production shape should remain:

```text
raw World Cup sources
  -> BigQuery sports_edge_raw
  -> curated team ratings, fixtures, features, simulations
  -> Python prediction + Monte Carlo simulation
  -> Supabase world_cup_* serving tables
  -> portfolio Sports Edge / World Cup tab
```

## GPU Decision

Do not use the GPU for v0/v1. The immediate problem is tabular team strength,
calibration, and tournament simulation. CPU-based Python, numpy, pandas, and
scikit-learn-style models are enough for:

- group-stage match probabilities;
- 1X2 home/draw/away probabilities;
- projected goals;
- 50k to 200k Monte Carlo tournament simulations;
- group-rank and champion probabilities.

Use GPU only for a later research version that learns from player embeddings,
club-level event data, tracking data, or deep sequence/event models. That is a
v2 research path, not the first production surface.

## Data Sources

Primary source responsibilities:

- FIFA: official fixtures, kickoff times, venues, groups, final scores, and
  FIFA rankings.
- World Football Elo: independent national-team strength rating.
- Odds source: World Cup futures and match odds, when available, as a market
  prior and not as the target label.
- Player/squad sources: squad lists, club minutes, goals/xG, assists/xA,
  goalkeeper quality, and injury availability.
- Historical World Cup results: time-decayed tournament experience and program
  strength only; do not let old trophies dominate current roster strength.

## BigQuery Tables

Planned warehouse/source-of-truth tables:

- `sports_edge_raw.raw_wc_fixtures`
- `sports_edge_raw.raw_wc_results`
- `sports_edge_raw.raw_fifa_rankings`
- `sports_edge_raw.raw_world_elo`
- `sports_edge_raw.raw_wc_squads`
- `sports_edge_raw.raw_wc_player_form`
- `sports_edge_raw.raw_wc_odds`
- `sports_edge_curated.wc_team_ratings`
- `sports_edge_curated.wc_feature_snapshots`
- `sports_edge_curated.wc_match_predictions`
- `sports_edge_curated.wc_team_probabilities`
- `sports_edge_curated.wc_simulation_runs`

Supabase serving tables are created by `sql/014_world_cup_serving_tables.sql`
and mirrored in the portfolio migration
`supabase/migrations/009_world_cup_serving_tables.sql`.
The live Supabase migration was applied and verified on 2026-06-15.

## Implemented v0

Implemented files:

- `src/models/world_cup.py`: transparent team-rating model and World Cup-style
  tournament simulator.
- `src/data/world_cup_sources.py`: source normalizers for ESPN fixtures/results,
  World Football Elo, FIFA rankings, recent form, World Cup history,
  player/star strength, and market/futures priors.
- `scripts/build_world_cup_inputs.py`: builds the model-ready teams and fixtures
  CSVs from repeatable source extracts, can fetch ESPN fixtures/results and
  World Football Elo, can derive unresolved Round-of-32 slots, and can load the
  normalized outputs into BigQuery.
- `scripts/predict_world_cup.py`: reads teams/fixtures CSVs and writes a
  portfolio-shaped prediction JSON.
- `scripts/sync_world_cup_to_supabase.py`: upserts the JSON payload into
  `world_cup_matches`, `world_cup_match_predictions`, and
  `world_cup_team_probabilities`.
- `scripts/refresh_world_cup.py`: end-to-end scheduled/manual runner that
  builds inputs, generates the prediction payload, appends BigQuery warehouse
  rows, and syncs Supabase on normal runs.
- `sql/bigquery_world_cup_tables.sql`: BigQuery raw and curated table DDL for
  the warehouse side of the pipeline.

The v0 model blends Elo, FIFA rank, recent form, goal-difference form,
tournament experience, star-player score, host boost, and market rating when
those columns are supplied.

## Local Commands

Create a prediction payload:

Build model input CSVs from source extracts:

```bash
PYTHONPATH=data-core python3 data-core/scripts/build_world_cup_inputs.py \
  --fixtures-csv data-core/notebooks/cache/world_cup_fixtures_seed.csv \
  --teams-csv data-core/notebooks/cache/world_cup_teams_seed.csv \
  --fifa-rankings-csv data-core/notebooks/cache/fifa_rankings.csv \
  --world-elo-csv data-core/notebooks/cache/world_football_elo.csv \
  --recent-results-csv data-core/notebooks/cache/international_results.csv \
  --world-cup-history-csv data-core/notebooks/cache/world_cup_history.csv \
  --player-form-csv data-core/notebooks/cache/world_cup_player_form.csv \
  --market-odds-csv data-core/notebooks/cache/world_cup_futures_odds.csv
```

Or fetch the no-key ESPN fixture/result feed and World Football Elo snapshot:

```bash
PYTHONPATH=data-core python3 data-core/scripts/build_world_cup_inputs.py \
  --fetch-espn --start-date 2026-06-11 --end-date 2026-07-19 \
  --fetch-world-elo \
  --teams-csv data-core/notebooks/cache/world_cup_teams_seed.csv \
  --output-round-of-32-slots-json data-core/notebooks/cache/world_cup_round_of_32_slots.json
```

```bash
PYTHONPATH=data-core python3 data-core/scripts/predict_world_cup.py \
  --teams-csv data-core/notebooks/cache/world_cup_team_ratings.csv \
  --fixtures-csv data-core/notebooks/cache/world_cup_fixtures.csv \
  --round-of-32-slots-json data-core/notebooks/cache/world_cup_round_of_32_slots.json \
  --model-version world-cup-v0 \
  --n-sims 100000 \
  --seed 20260613 \
  --output-json data-core/notebooks/cache/world_cup_predictions_v0.json
```

Dry-run Supabase serving sync:

```bash
PYTHONPATH=data-core python3 data-core/scripts/sync_world_cup_to_supabase.py \
  --input-json data-core/notebooks/cache/world_cup_predictions_v0.json \
  --dry-run
```

Write to Supabase after the migration is applied:

```bash
PYTHONPATH=data-core python3 data-core/scripts/sync_world_cup_to_supabase.py \
  --input-json data-core/notebooks/cache/world_cup_predictions_v0.json
```

Run the scheduled path locally:

```bash
PYTHONPATH=data-core python3 data-core/scripts/refresh_world_cup.py \
  --season 2026 \
  --start-date 2026-06-11 \
  --end-date 2026-07-19 \
  --model-version world-cup-v0-live \
  --n-sims 50000 \
  --project "$GCP_PROJECT_ID" \
  --write-bigquery \
  --sync-supabase
```

The dedicated `.github/workflows/world-cup-refresh.yml` GitHub Actions workflow
gates this command with `scripts/plan_daily_refresh.py` during the June/July
World Cup window. It runs at 03:00, 13:00, 17:00, and 21:00 UTC so the
portfolio can pick up same-day final results and refreshed next-match
probabilities. Dry-run workflow executions still build the payload and report
would-write row counts without writing BigQuery or Supabase.

## Portfolio

The portfolio `/api/sports-edges` route now includes `worldCup` in the same
payload as NFL/NBA/MLB. `SportsEdgeCard` renders a `World Cup` tab with:

- champion probabilities;
- upcoming match 1X2 probabilities;
- projected goals;
- group winner probabilities.

Verified live serving state on 2026-06-16:

- model version: `world-cup-v0-live`;
- simulations: 50,000;
- bracket source: `configured_round_of_32_slots`;
- Supabase rows: 72 enriched match predictions and 48 team probability rows;
- portfolio API: 72 matches, 48 teams, and 12 group-rank sections.

## Remaining Production Work

1. Build official source fetchers for FIFA fixtures/results/rankings.
2. Add World Cup odds ingestion once the configured odds provider exposes the
   market consistently.
3. Add player/squad feature ingestion and injury availability.
4. Store evaluation/backtest rows for historical World Cup and international
   match validation before promoting claims beyond a candidate model.
