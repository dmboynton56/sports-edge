# Model Versions

Generated: 2026-05-22

Cross-sport measured performance is summarized in `docs/PERFORMANCE_HISTORY.md`.

## Active Daily Pipeline

| League | Daily command | Active model version | Notes |
| --- | --- | --- | --- |
| NBA | `python -m src.pipeline.refresh_nba --project "$PROJECT_ID" --model-version v3` | `v3` | Daily workflow writes v3 predictions |
| NFL | `python -m src.pipeline.refresh_nfl --project "$PROJECT_ID" --model-version v1` | `v1` | Supabase also contains stale v2/v3 rows |
| PGA | Manual `train_models_v2` / prediction scripts | v2 saved artifacts | Do not retrain until refreshed-store evaluation justifies it |
| CBB | Manual matchup trainer | saved ensemble artifacts | CV run used temp model dir to avoid overwriting artifacts |
| MLB | Manual `scripts/train_mlb_winner_model.py`; score via `scripts/predict_mlb_winners.py` | `v3` ship candidate | Not in daily workflow; trained through 2025 and evaluated on 2026 YTD |

## Local Artifacts

NBA artifacts in `data-core/models`: `spread_model_nba_v1.pkl` through `v4`, `win_prob_model_nba_v1.pkl` through `v4`, `meta_ensemble_nba_v2.pkl` through `v4`, medians and link functions. Daily CI uses NBA `v3`, not all local versions.

NFL artifacts in `data-core/models`: `spread_model_nfl_v1.pkl`, `spread_model_nfl_v2.pkl`, `win_prob_model_nfl_v1.pkl`, `win_prob_model_nfl_v2.pkl`, link functions and medians. Daily CI uses NFL `v1`. Full-season 2025 model-vs-results export for v1: 285 games, Brier 0.2648, log loss 0.7315, AUC 0.5790, spread MAE 11.15; no BigQuery odds-backed ROI.

PGA artifacts in `data-core/models/saved`: v2 SG models (`lgbm_sg_model_v2.joblib`, `xgb_sg_model_v2.joblib`, `meta_ensemble_sg_v2.joblib`, `pytorch_tabular_v2.pth`) plus target classifiers and v2 target meta-ensembles.

CBB artifacts in `data-core/models/saved`: `lgbm_cbb_matchup.joblib`, `xgb_cbb_matchup.joblib`, `meta_cbb_matchup.joblib`, `cbb_scaler.joblib`. Latest non-destructive CV export used `/tmp/sports-edge-cbb-cv-models` and found XGBoost strongest on 2016-2025 mean log loss: 0.575 vs LGBM 0.581 and meta 0.595.

MLB artifacts in `data-core/models`: `mlb_winner_model_v1.pkl`, `mlb_winner_model_v1_metrics.json`, `mlb_winner_model_v2.pkl`, `mlb_winner_model_v2_metrics.json`, `mlb_winner_model_v3.pkl`, and `mlb_winner_model_v3_metrics.json`. The v3 model is a random forest selected by 2025 validation Brier and refit on 2021-2025 before testing on 2026 YTD. It is a ship candidate for probability display, not for betting/ROI recommendations.

## Promotion Criteria

Promote a new version only when:

1. Backtest or CV metrics improve on the target objective.
2. ATS/ROI or ranking lift is measured against a fixed baseline and date range.
3. The source data window and model artifact names are documented here.
4. The daily workflow and serving sync agree on the promoted version.
5. Old serving rows are filtered or archived so stale versions do not become latest predictions by accident.

No NBA/NFL retrain was promoted in this pass. PGA retrain remains pending because the refreshed feature store changed materially and needs a controlled v2/v3 experiment first.

MLB v3 may be promoted only to a clearly labeled probability surface after a serving dry run because it now has a prediction CLI and reproducible artifact. It should not be promoted to betting/ROI surfaces until it is compared against moneyline markets. Its 2026 YTD test edge over the home-rate baseline is modest: Brier 0.2478 vs 0.2497 and log loss 0.6888 vs 0.6925.
