# Packet 07 verdict — starter strikeouts model

## Verdict

Pass. The feasibility gate cleared and the real 2026 YTD evaluation artifacts were generated.

- Clean probable-starter coverage: 2,793 / 2,856 sides (97.79%; gate: 70%).
- Probable mismatch discarded fraction: 0.35% for 2026 and 0.22% across all seasons.
- Selected model: conservative `HistGradientBoostingRegressor` with Poisson loss.
- Test MAE: 1.7960 model vs 1.8673 K/9 × expected-outs baseline.
- Test RMSE: 2.2482 model vs 2.3640 baseline.
- P(K ≥ 6) Brier: 0.2074 vs 0.2319 reference base-rate Brier.
- P(K ≥ 7) Brier: 0.1638 vs 0.1811 reference base-rate Brier.
- ROI is explicitly `null`: no strikeout odds source.

## Verification

- `PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/train_mlb_strikeouts_model.py`
- Artifact schema, row-count, probability bounds/order, baseline comparisons, calibration comparisons, and no-ROI assertions passed.
- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest -q data-core/tests/unit/test_mlb_market_features.py` — 5 passed.
