# Codex done — task 04

## Summary

- Added optional prebuilt-feature loading to the MLB winner training CLI; without `--features-path`, its existing fetch/cache/build path is unchanged.
- Added a reproducible four-arm moneyline ablation using the existing selection/refit and free-moneyline ROI logic.
- Trained and saved v4, its metrics sidecar, 1,428 backtest predictions, backtest metrics, and the ablation artifact.
- Full v2 selected random forest and scored Brier 0.247468, log loss 0.687978, AUC 0.549654, accuracy 54.27%, and ECE 0.012746. Free-line ROI was −5.86% on 673 joined games (47.13% coverage).
- The controlled full-v2 versus v1 verdict is in `04-done.md`; the published v3 673-game figures remain context only.

## Files touched

- `data-core/scripts/train_mlb_winner_model.py`
- `data-core/scripts/ablate_mlb_winner_features.py`
- `data-core/models/mlb_winner_model_v4.pkl`
- `data-core/models/mlb_winner_model_v4_metrics.json`
- `data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd_v4_free.json`
- `data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd_v4_free.csv`
- `data-core/notebooks/cache/mlb_ml_ablation_2026_ytd.json`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/STATUS.json`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/04-done.md`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-04.md`

## Verification

- `PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/backtest_mlb_winners.py --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet --validation-season 2025 --test-season 2026 --odds-path data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd_v4_free.csv --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd_v4_free.json` — passed; 1,428 test rows and 673 odds rows.
- `PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_winner_model.py --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet --validation-season 2025 --test-season 2026 --model-version v4 --output-model data-core/models/mlb_winner_model_v4.pkl` — passed; saved v4 pickle and metrics sidecar with 110 model features.
- `PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/ablate_mlb_winner_features.py --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet --validation-season 2025 --test-season 2026 --odds-path data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv --output data-core/notebooks/cache/mlb_ml_ablation_2026_ytd.json` — passed; four arms with identical 1,428-row tests and 673-row odds joins.
- Artifact invariant check — passed: full/v1 feature counts 110/59, the 51-column v2 delta is unique and complete, arm-selected columns exactly match 59/89/67/110 definitions, all test counts match, odds coverage is 673/1,428, and the pickle identifies model version v4.
- `PYTHONPATH=data-core data-core/.venv/bin/python -m py_compile data-core/scripts/train_mlb_winner_model.py data-core/scripts/ablate_mlb_winner_features.py` — passed.
- `git diff --check -- data-core/scripts/train_mlb_winner_model.py data-core/scripts/ablate_mlb_winner_features.py` — passed.

## Residual risks / follow-ups

- The free odds cache ends on 2026-05-21, so ROI covers 47.13% of the test games rather than the full mid-July outcome window.
- Weather fields are observed game-time records used as pregame forecast proxies, as documented by task 03.
- The prescribed arms isolate starter and weather additions, but not the cross-season run-environment group alone; its independent effect remains unmeasured.
