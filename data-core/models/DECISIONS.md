# Sports Edge — model registry decisions (retroactive log)

This file satisfies [plans/sports-edge/03-model-versioning.md](plans/sports-edge/03-model-versioning.md): a human-readable trace of which **NBA/NFL** artifacts are treated as production and why.

| Date | League | Version / artifact | Decision |
|------|--------|--------------------|----------|
| — | NBA | v3 (LightGBM stack) | Default for `refresh_nba` in daily pipeline until superseded. |
| — | NFL | v1 | Default for `refresh_nfl` in daily pipeline until superseded. |

Update this table when you promote a new artifact, archive an old one in `data-core/models/archive/`, or change the BigQuery `model_registry` / naming convention.
