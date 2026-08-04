# Packet 06 — Run line (home −1.5) model

**Actor:** Codex · **Depends on:** 03 · **Parallel with:** 04, 05, 07 (file-disjoint) · **Feeds:** 08
**No commit/push.** Repo root: `/home/dmboynton/projects/sports-edge`. Python: `data-core/.venv/bin/python`, `PYTHONPATH=data-core`.

## Why

Second new market. MLB run line is effectively fixed at ±1.5, so `home_cover_15` (label in the v2 store, = `run_diff >= 2`) is a stable classification target even with **no run-line odds on disk** — probability quality only, `"roi": null, "reason": "no run-line odds source"`.

## Files you own

- `data-core/src/models/mlb_runline_model.py` (new)
- `data-core/scripts/train_mlb_runline_model.py` (new)
- Artifacts: `data-core/notebooks/cache/mlb_runline_metrics_2026_ytd.json`, `mlb_runline_predictions_2026_ytd.csv`, optional `data-core/models/mlb_runline_model_v1.pkl`
- Optional test file if cheap: `data-core/tests/unit/test_mlb_runline_model.py`
- Reads only: `notebooks/cache/mlb_feature_store_v2_2021_2026.parquet`. Do not edit shared modules (owned by 03/04).

## Design

Same conventions/template as the winner model (`train_and_evaluate_mlb_winner`), same split as packets 04/05 (train 2021–2024, val 2025, test 2026 YTD, refit ≤2025), features via `default_feature_columns` from `src.models.mlb_winner_model`.

1. **Classifier head:** P(home covers −1.5), i.e. P(`run_diff` ≥ 2). Candidates LR / HGB / RF, val-Brier selection. Metrics: acc/Brier/log-loss/AUC/ECE vs the constant base-rate baseline (historically ~0.42–0.46 — compute, don't assume). Also report the mirrored P(away covers +1.5) sanity (= 1 − p, plus push impossibility since ±1.5 has no push).
2. **Margin regression head:** predict `run_diff`; derive P(cover) from margin + residual distribution as a second candidate; report MAE and which head calibrates better.
3. **Consistency check vs moneyline:** correlation between P(home covers −1.5) and the same-row model home-win probability if packet 04's predictions CSV exists at run time (`mlb_backtest_predictions_2026_ytd_v4_free.csv`); skip gracefully if not yet written — do not create a hard dependency.
4. Predictions CSV: game_pk, date, teams, p_home_cover_15, predicted_margin, home_cover_15 actual. Metrics JSON: candidates val/test, base-rate baseline, calibration table, `"roi": null` note. CLI mirrors packet 05 (`--features-path`, seasons, outputs, optional `--odds-path` stub for future spread prices).

## Acceptance

- Metrics JSON + predictions CSV from a real run; classifier compared against base rate (beat it or document the miss); ECE/calibration table included.
- No ROI claimed; no edits outside owned files; `handoffs/06-done.md` verdict; STATUS.json updated.
