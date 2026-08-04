# Codex done — task 03

## Summary

- Added the v2 MLB market feature builder as a chronological single pass over completed games. It reuses the v1 row/state machinery, emits each row before updates, and adds cross-season actual-starter histories keyed by the pregame probable pitcher, cross-season 15-game team run/K histories, local handedness, weather/park fields, and market labels.
- Extended the feature-store CLI with `--version v2` / `--v2`, v2 cache defaults, venue metadata loading, and an audit sidecar with null rates, coverage, and an observed-weather caveat.
- Hardened `default_feature_columns` against all new labels and raw postgame boxscore fields while retaining leakage-safe rolling pitches/outs features.
- Built `mlb_feature_store_v2_2021_2026.parquet`: 13,182 rows, six seasons, 94.61% starter history on both sides, and 99.84% probable-to-actual match coverage among known comparisons.
- The exact 51-column modeling delta and ablation groups are listed in `03-done.md`.

## Files touched

- `data-core/src/features/mlb_market_features.py`
- `data-core/src/features/mlb_features.py`
- `data-core/src/models/mlb_winner_model.py` (`default_feature_columns` only)
- `data-core/scripts/build_mlb_feature_store.py`
- `data-core/tests/unit/test_mlb_market_features.py`
- `data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet`
- `data-core/notebooks/cache/mlb_feature_store_v2_2021_2026_audit.json`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/STATUS.json`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/03-done.md`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-03.md`

## Verification

- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest data-core/tests/unit/test_mlb_market_features.py data-core/tests/unit/test_mlb_features.py -q` — 6 passed.
- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest data-core/tests/unit -k mlb` — 48 passed, 88 deselected; six pre-existing sklearn pickle-version warnings.
- `PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/build_mlb_feature_store.py --version v2` — wrote 13,182 rows spanning 2021-04-06 through 2026-07-20; row count exactly matches the v1 builder with the same five-prior-game filter.
- `train_and_evaluate_mlb_winner(v2, validation_season=2025, test_season=2026)` — passed end to end with 9,404 train, 2,350 validation, and 1,428 test rows; selected random forest, test Brier 0.24747 and AUC 0.54965.
- Schema checks — 13,182 unique `game_pk` values; all six labels absent from the 110 selected model columns; no infinite selected values; syntax compilation and `git diff --check` passed.

## Residual risks / follow-ups

- Weather is the observed boxscore record, not an archived pregame forecast. This intentional proxy is documented in the module and audit and should remain explicit in downstream analysis.
- Handedness is nullable and limited to the existing local player cache; no network enrichment was performed.
- MLB venue metadata leaves elevation null for about 0.19% of feature rows, and wind speed is null for one row; downstream model pipelines already median-impute numeric features.
