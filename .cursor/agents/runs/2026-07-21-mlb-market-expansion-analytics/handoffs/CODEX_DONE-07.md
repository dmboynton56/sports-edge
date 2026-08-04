# Codex done — task 07

## Summary

- Passed the feasibility gate with 97.79% clean 2026 side coverage.
- Added a side-level probable-starter reshaper and leakage-aware strikeout regression pipeline.
- Selected among three gradient-boosting candidates on 2025, refit through 2025, and evaluated on 2026 YTD.
- Added the required K/9 × trailing expected-outs baseline and Poisson-tail calibration for 5.5 and 6.5 thresholds.
- Generated real metrics and prediction artifacts with no ROI claim.

## Files touched

- `data-core/src/models/mlb_strikeouts_model.py`
- `data-core/scripts/train_mlb_strikeouts_model.py`
- `data-core/notebooks/cache/mlb_strikeouts_metrics_2026_ytd.json`
- `data-core/notebooks/cache/mlb_strikeouts_predictions_2026_ytd.csv`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/07-done.md`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/STATUS.json`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-07.md`

## Verification

- `PYTHONPATH=data-core data-core/.venv/bin/python -m py_compile data-core/src/models/mlb_strikeouts_model.py data-core/scripts/train_mlb_strikeouts_model.py` — passed.
- `PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_strikeouts_model.py` — wrote 2,793 test predictions; model MAE/RMSE 1.7960/2.2482 vs baseline 1.8673/2.3640.
- Artifact assertion script — passed requested columns, row count, probability bounds/order, model-vs-baseline metrics, threshold Brier comparisons, and `roi: null` checks.
- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest -q data-core/tests/unit/test_mlb_market_features.py` — 5 passed in 1.32s.
- `git diff --check -- data-core/src/models/mlb_strikeouts_model.py data-core/scripts/train_mlb_strikeouts_model.py` — passed.
- Ruff was not installed in the project virtual environment, so no Ruff check was run.

## Residual risks / follow-ups

- Threshold probabilities assume a Poisson distribution around each point estimate; pitcher strikeout outcomes may be overdispersed.
- The weather fields inherit the feature store's documented forecast-proxy limitation.
- Strikeout ROI cannot be evaluated until a historical strikeout odds source is available.
