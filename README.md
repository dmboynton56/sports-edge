# Sports Edge

![Build Status](https://github.com/dmboynton56/sports-edge/actions/workflows/test.yml/badge.svg)

Sports Edge is a sports analytics and prediction pipeline displayed on the
portfolio. BigQuery is the warehouse/source of truth for raw data, features,
model predictions, and historical analysis. Supabase is the serving cache that
the portfolio and dashboard read.

The public story is still a work in progress: NBA daily predictions are the
most active spread surface, NFL has completed 2025 season coverage, MLB v3 is
promoted as a probability-only display, and PGA/CBB research remains useful but
is not all promoted to recruiter-facing production claims.

## Current Production Flow

The daily workflow is `.github/workflows/daily-refresh.yml` and runs at
13:00 UTC.

```text
raw league data + odds
  -> BigQuery sports_edge_raw
  -> feature snapshots in sports_edge_curated
  -> Python model refreshes
  -> BigQuery sports_edge_curated.model_predictions
  -> scripts/sync_bq_to_supabase.py
  -> Supabase games/model_predictions/odds_snapshots
  -> portfolio /api/sports-edges and project chat
```

### Portfolio chat

The personal portfolio exposes this project via scoped chat (`scope=sports-edge`)
on `/projects/sports-edge`. Answers use BigQuery warehouse SQL, canned metrics,
and RAG over `personal-portfolio/docs/project-knowledge/sports-edge/`. After
changing methodology or metrics docs, rebuild portfolio `rag_embeddings.json`
(see `plans/04-rag-knowledge-strategy.md`).

Current scheduled model versions:

- NBA: `refresh_nba --model-version v3`
- NFL: `refresh_nfl --model-version v1`
- MLB: `refresh_mlb --model-version v3` for home-win probability display only

## Repository Layout

```text
data-core/
  src/                 production Python package
    data/              league/API fetchers and loaders
    features/          feature builders
    models/            model code
    pipeline/          NBA/NFL/MLB refresh entry points
    utils/             shared DB/env helpers
  scripts/             backfills, syncs, validation, exports
  models/              intentionally versioned model artifacts
  notebooks/           research notebooks and selected cached evidence
  docs/                model status, data sources, freshness, and plans
  sql/                 BigQuery/Supabase helper SQL
web/
  app/, components/, lib/
                      standalone sports dashboard surfaces
ios/
  SportsEdgeApp/       early iOS app shell
```

The root-level `models/` directory was removed because it duplicated older v1
artifacts. Active workflow commands run from `data-core`, so production model
artifacts belong in `data-core/models`.

## Local Setup

```bash
cd data-core
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt google-cloud-bigquery nflreadpy supabase
```

Required environment variables:

- `GCP_PROJECT_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_DB_HOST`
- `SUPABASE_DB_NAME`
- `SUPABASE_DB_PORT`
- `SUPABASE_DB_USER`
- `SUPABASE_DB_PASSWORD` or `supabaseDBpass`
- `ODDS_API_KEY`

## Manual Operations

Run the same commands from `data-core`, matching GitHub Actions:

```bash
python scripts/backfill_nba_raw.py --project "$GCP_PROJECT_ID" --seasons 2025 --schedule-start 2026-05-22 --schedule-window-days 3
python scripts/build_feature_snapshots.py --project "$GCP_PROJECT_ID" --league NBA --seasons 2025
python -m src.pipeline.refresh_nba --project "$GCP_PROJECT_ID" --model-version v3
python scripts/sync_bq_to_supabase.py --project "$GCP_PROJECT_ID" --league NBA --append
python scripts/sync_final_scores.py --project "$GCP_PROJECT_ID" --lookback-days 4 --lookahead-days 1
python scripts/validate_supabase_sync.py --strict
```

NFL uses the same pattern with `--league NFL` and `refresh_nfl --model-version
v1`. MLB uses `python -m src.pipeline.refresh_mlb --project "$GCP_PROJECT_ID"
--model-version v3` followed by `sync_bq_to_supabase.py --league MLB --append`.
MLB rows intentionally set `predicted_spread`/`book_spread` to null; the
portfolio displays only model home-win probabilities, probable pitchers, final
scores, and winner-hit status.

## Data Contracts

BigQuery datasets currently used by the portfolio story:

- `sports_edge_raw`
- `sports_edge_curated`
- `sports_edge_results`
- `sports_edge_models`

Supabase serving tables:

- `games`
- `model_predictions`
- `odds_snapshots`
- `features`
- `model_runs`
- `games_today_enriched` view

The portfolio is a read-only consumer of those Supabase tables. Sports Edge
owns writes for games, odds, predictions, and final scores.

Last terminal verification: 2026-05-23. BigQuery project
`learned-pier-478122-p7` exposed the four datasets above; curated warehouse
tables included 93,424 `feature_snapshots` rows and 972 `model_predictions`
rows. NBA `v3` had 464 current prediction rows with latest
`prediction_ts=2026-05-23T15:04:38Z`; NFL `v1` remained at 7 rows with latest
`prediction_ts=2026-02-09T14:37:51Z`. Supabase public serving counts were
586 `games`, 711 `model_predictions`, and 98 `odds_snapshots`.

MLB serving was added after that verification. Before claiming the live surface
is fully populated, confirm the next daily workflow writes MLB rows through
BigQuery and Supabase.

## Tests

```bash
cd data-core
pytest
```

## Documentation State

Important current docs:

- `data-core/docs/DATA_AND_MODEL_STATUS.md`
- `data-core/docs/MODEL_VERSIONS.md`
- `data-core/docs/DATA_SOURCES.md`
- `data-core/docs/PREDICTION_AND_ROI_HISTORY.md`
- `data-core/docs/PGA_REFRESH_PIPELINE.md`

Planning docs for future work are kept under `data-core/docs/`. Root-level,
machine-specific execution plans were removed to keep the repo focused.

## Future Work

- Promote a clear model registry so code, docs, and portfolio claims all point
  at the same active artifact per league.
- Separate reproducible evidence artifacts from notebook scratch caches.
- Finish a unified cross-league performance board for NBA/NFL/PGA/CBB.
- Add moneyline odds and ROI only after MLB v3 is evaluated against a stable
  market source.
