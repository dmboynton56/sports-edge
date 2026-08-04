# Codex done — task 06

## Summary

- Implemented season-split MLB home -1.5 classifier selection (LR/HGB/RF), a margin regression head (Ridge/HGB/RF), empirical-residual cover probabilities, calibration tables, base-rate comparison, mirrored away +1.5 sanity checks, and optional moneyline consistency correlation.
- Ran the real 2021-2024 train / 2025 validation / 2026 YTD test workflow. Random forest won classifier selection and beat the constant baseline on test Brier: 0.22830 vs 0.23019. The random-forest margin head achieved MAE 3.62657 but its cover Brier was worse at 0.23331.
- Wrote 1,428 complete 2026 prediction rows and explicitly reported `roi: null` / `no run-line odds source`.

## Files touched

- `data-core/src/models/mlb_runline_model.py`
- `data-core/scripts/train_mlb_runline_model.py`
- `data-core/tests/unit/test_mlb_runline_model.py`
- `data-core/notebooks/cache/mlb_runline_metrics_2026_ytd.json`
- `data-core/notebooks/cache/mlb_runline_predictions_2026_ytd.csv`
- `data-core/models/mlb_runline_model_v1.pkl`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/06-done.md`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-06.md`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/STATUS.json` (task 06 fields and notes only)

## Verification

- `PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_runline_model.py --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet --validation-season 2025 --test-season 2026 --metrics-output data-core/notebooks/cache/mlb_runline_metrics_2026_ytd.json --predictions-output data-core/notebooks/cache/mlb_runline_predictions_2026_ytd.csv --output-model data-core/models/mlb_runline_model_v1.pkl` — succeeded; wrote metrics, predictions, and model.
- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest -q data-core/tests/unit/test_mlb_runline_model.py data-core/tests/unit/test_mlb_market_features.py` — 9 passed.
- Artifact assertions — JSON and pickle reload successfully; 1,428 test/prediction rows; no CSV nulls; calibration counts total 1,428; mirrored probability max error below 1e-12; all 1,428 rows joined to moneyline predictions with Pearson correlation 0.94475.
- `git diff --check -- data-core/src/models/mlb_runline_model.py data-core/scripts/train_mlb_runline_model.py data-core/tests/unit/test_mlb_runline_model.py` — clean.

## Residual risks / follow-ups

- Probability quality is only modestly above the base-rate baseline (AUC 0.55135), so this remains a research model rather than a production betting signal.
- ROI cannot be evaluated until a historical run-line price source is available; `--odds-path` is intentionally a future-source stub and does not claim ROI.
- The margin residual head is under-confident on home covers (average 0.2973 vs actual 0.3592) and should not replace the classifier without improved residual calibration.
