# Codex done — task 05

## Summary

- Added a leakage-aware total-runs training module with ridge, histogram gradient boosting, and random forest candidates selected on 2025 validation MAE, followed by a refit on all rows through 2025 and evaluation on 2026 YTD.
- Added Gaussian validation-residual over-probability heads for 8.5 and 9.5 with Brier, log-loss, AUC, and ECE comparisons against constant base rates.
- Added the three required test-row baselines, weather/park ablation, weather direction correlations, optional supported-line odds ROI evaluation, JSON-safe artifact output, and a CLI.
- The real run selected ridge. Test MAE was 3.5364, beating all baselines (3.5885–3.6503). Weather/park removal worsened MAE by 0.0547 and reduced over-8.5 AUC by 0.0501.
- Metrics explicitly report `"roi": null` and `"reason": "no totals odds source"`.

## Files touched

- `data-core/src/models/mlb_totals_model.py`
- `data-core/scripts/train_mlb_totals_model.py`
- `data-core/tests/unit/test_mlb_totals_model.py`
- `data-core/notebooks/cache/mlb_totals_metrics_2026_ytd.json`
- `data-core/notebooks/cache/mlb_totals_predictions_2026_ytd.csv`
- `data-core/models/mlb_totals_model_v1.pkl`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/05-done.md`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-05.md`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/STATUS.json`

## Verification

- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest data-core/tests/unit/test_mlb_totals_model.py -q` — 3 passed.
- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest data-core/tests/unit -q` — 143 passed; six pre-existing sklearn pickle-version warnings.
- `PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_totals_model.py` — completed on 13,182 real feature rows; wrote 1,428 test predictions, metrics JSON, and model pickle.
- `data-core/.venv/bin/python -m compileall -q ...` — passed for the model, CLI, and test files.
- Artifact integrity checks confirmed strict JSON, exact required prediction columns, finite probabilities, a refit cutoff before 2026, and a loadable pickle.

## Residual risks / follow-ups

- The weather ablation combines weather and venue/park variables, so its measured lift should not be described as weather-only causal attribution.
- Weather values are observed game-time conditions used as pregame forecast proxies; a production scorer needs archived or live pregame forecasts with the same schema.
- Gaussian residual probabilities improve Brier/log-loss over the constant base rates but have worse ECE; calibration should be revisited with more held-out seasons before production use.
- The odds evaluator supports joined 8.5/9.5 lines with `over_price` and `under_price`, but no local totals odds source was available to exercise real ROI.

