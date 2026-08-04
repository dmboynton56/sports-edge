# Packet 05 — Total runs (over/under) model

**Actor:** Codex · **Depends on:** 03 · **Parallel with:** 04, 06, 07 (file-disjoint) · **Feeds:** 08
**No commit/push.** Repo root: `/home/dmboynton/projects/sports-edge`. Python: `data-core/.venv/bin/python`, `PYTHONPATH=data-core`.

## Why

First new market. Weather (temp/wind) + park + starter quality + run environment are exactly the totals signal set; this is where the new features should show up most clearly. **No totals odds exist locally** — deliver probability/point-estimate quality with honest framing (`"roi": null, "reason": "no totals odds source"`), ROI only if an odds file is ever supplied.

## Files you own

- `data-core/src/models/mlb_totals_model.py` (new)
- `data-core/scripts/train_mlb_totals_model.py` (new)
- `data-core/tests/unit/test_mlb_totals_model.py` (new)
- Artifacts: `data-core/notebooks/cache/mlb_totals_metrics_2026_ytd.json`, `mlb_totals_predictions_2026_ytd.csv`, optional `data-core/models/mlb_totals_model_v1.pkl`
- Reads only: `notebooks/cache/mlb_feature_store_v2_2021_2026.parquet`. Do not edit shared modules (`mlb_winner_model.py`, `src/features/*` — owned by 03/04).

## Design

Mirror the winner-model conventions (`train_and_evaluate_mlb_winner` in `src/models/mlb_winner_model.py` is the template: sorted time split, candidate dict, val-selection, ≤2025 refit, metrics dict, JSON-safe artifact saver):

1. **Regression head:** predict `total_runs`. Candidates: Ridge/linear pipeline, HistGradientBoostingRegressor, RandomForestRegressor (median-imputer pipelines). Select on validation MAE. Feature columns: `default_feature_columns(frame)` from `src.models.mlb_winner_model` (03 guarantees label exclusion) — do not roll your own selector.
2. **Binary O/U heads:** P(total > 8.5) and P(total > 9.5) classifiers (or derive from regression + residual sigma — pick one, justify in metrics JSON). Report Brier/log-loss/AUC/ECE vs the constant base rate.
3. **Baselines (required):** (a) league rolling mean total (trailing 30 days), (b) `venue_total_runs_per_game` column alone, (c) constant train-mean. Report MAE/RMSE for all on the same test rows.
4. **Weather value check:** rerun the selected regressor minus the weather/park group → ΔMAE and ΔAUC(8.5) attributable to weather; include correlation of predicted total with `temp_f` and `wind_out`−`wind_in` as a sanity direction check (warm + wind out ⇒ higher totals).
5. **Split:** train 2021–2024, validation 2025, test 2026 YTD, refit ≤2025 — identical to packet 04.

Predictions CSV: game_pk, date, teams, predicted_total, p_over_8_5, p_over_9_5, actual total_runs. Metrics JSON: candidates val/test, baselines, weather ablation, `"roi": null` note. CLI: argparse with `--features-path`, `--validation-season`, `--test-season`, `--odds-path` (optional stub: if provided with `total_line`/over/under prices, compute ROI; otherwise null), `--metrics-output`, `--predictions-output`, `--output-model`.

## Tests (no network, synthetic frame)

- Label exclusion: model never receives `total_runs`/`home_score`/`away_score` as features (assert on the resolved feature list).
- Time split: no test-season rows in train (assert on seasons).
- O/U labeling arithmetic at the threshold (total 8 vs 9 for 8.5 line; pushes handled for integer lines if you support them — 8.5/9.5 have none).

## Acceptance

- Metrics JSON + predictions CSV written from a real run; regressor beats all three naive baselines on test MAE **or** the negative result is documented with the weather-ablation numbers.
- Classifier heads report Brier/AUC/ECE vs base rate; no ROI claimed anywhere.
- Tests green; `handoffs/05-done.md` verdict (does weather move totals accuracy? by how much?); STATUS.json updated.
