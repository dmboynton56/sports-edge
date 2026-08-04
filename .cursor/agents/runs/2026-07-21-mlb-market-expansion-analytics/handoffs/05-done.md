# Task 05 complete

Canonical worker handoff: `CODEX_DONE-05.md`.

## Verdict

Weather/park context improved totals accuracy on the 1,428-game 2026 YTD test set. The selected ridge model produced 3.5364 MAE and 4.4925 RMSE. Removing the combined weather/park group worsened MAE to 3.5911, a **+0.0547-run degradation**, and reduced over-8.5 AUC from 0.5739 to 0.5238, a **0.0501 AUC loss**.

The directional checks were sensible: predicted total correlated +0.5895 with temperature and +0.1807 with `wind_out - wind_in`. This is an observational feature ablation, not a causal estimate, and the weather fields are observed game-time records used as forecast proxies.

## Test results

- Ridge test MAE 3.5364 beat the constant train mean (3.5885), trailing 30-day league mean (3.6245), and venue-only baseline (3.6503).
- Over 8.5: Brier 0.2469, log loss 0.6868, AUC 0.5739, ECE 0.0365; constant-rate Brier 0.2501 and AUC 0.5000.
- Over 9.5: Brier 0.2378, log loss 0.6684, AUC 0.5813, ECE 0.0447; constant-rate Brier 0.2414 and AUC 0.5000.
- O/U probabilities are derived from the selected regression mean plus Gaussian residual sigma estimated on the held-out 2025 validation season.
- ROI is null with reason `no totals odds source`; no totals ROI is claimed.

