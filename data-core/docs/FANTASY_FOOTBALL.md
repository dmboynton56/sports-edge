# Fantasy Football Projection Surface

Sports Edge's fantasy surface is a public half-PPR-first projection board with
custom scoring, a local browser-only snake-draft assistant, and a weekly lineup
optimizer. It does not connect to private ESPN/Yahoo/Sleeper leagues or store
user rosters in Supabase.

## Data and refresh

- Original component projections use nflverse player stats, player metadata,
  schedules, and current roster information.
- Sleeper's public NFL player directory supplies daily roster status, team,
  depth-chart, and injury context. The refresh makes one request per run and
  attributes the source in the artifact.
- FantasyPros is used only for the separately labeled ADP/market signal. The
  API key is read from `FANTASYPROS_API_KEY`, sent in the `x-api-key` header by
  the refresh job, and never shipped to the browser.
- `scripts/generate_fantasy_projections.py` writes
  `web/public/data/fantasy_projections.json` and model metrics under
  `data-core/models/`.
- `scripts/sync_fantasy_projections_to_supabase.py` publishes rows after
  `sql/020_fantasy_serving_tables.sql` is applied.

The FantasyPros adapter supports the documented v2 inputs for players,
consensus rankings/ADP, projections, player points, news, and injuries. The
projection model does not use FantasyPros projections as training labels; ADP
is a draft-market context field only.

## Local validation

```bash
cd data-core
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_fantasy_scoring.py tests/unit/test_fantasypros.py -q
PYTHONPATH=. .venv/bin/python scripts/generate_fantasy_projections.py --season 2026 --history-seasons 2023 2024 2025
PYTHONPATH=. .venv/bin/python scripts/validate_public_json.py ../web/public/data/fantasy_projections.json
```

The artifact includes season and per-week rows, projected games, median/floor/
ceiling points, stat components, confidence, deterministic explanations, model
metrics, data-source gaps, and ADP provenance.
