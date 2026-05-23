# Model Versions

Generated: 2026-05-22

Cross-sport measured performance is summarized in `docs/PERFORMANCE_HISTORY.md`.

## Active Daily Pipeline

| League | Daily command | Active model version | Notes |
| --- | --- | --- | --- |
| NBA | `python -m src.pipeline.refresh_nba --project "$PROJECT_ID" --model-version v3` | `v3` | Daily workflow writes v3 predictions |
| NFL | `python -m src.pipeline.refresh_nfl --project "$PROJECT_ID" --model-version v1` | `v1` | Supabase also contains stale v2/v3 rows |
| PGA | Manual `train_models_v2` / prediction scripts | local regenerated artifacts | Not promoted to daily workflow |
| CBB | Manual matchup trainer | local regenerated artifacts | CV run used temp model dir to avoid overwriting artifacts |
| MLB | Manual `scripts/train_mlb_winner_model.py`; score via `scripts/predict_mlb_winners.py` | `v3` ship candidate | Not in daily workflow; trained through 2025 and evaluated on 2026 YTD |

## Committed Runtime Artifacts

NBA artifacts retained in `data-core/models`: `win_prob_model_nba_v3.pkl`, `spread_model_nba_v3.pkl`, `link_function_nba_v3.pkl`, `feature_medians_nba_v3.pkl`, and `meta_ensemble_nba_v3.pkl`. Daily CI uses NBA `v3`.

NFL artifacts retained in `data-core/models`: `win_prob_model_nfl_v1.pkl`, `spread_model_nfl_v1.pkl`, `link_function_nfl_v1.pkl`, and `feature_medians_nfl_v1.pkl`. Daily CI uses NFL `v1`. Full-season 2025 model-vs-results export for v1: 285 games, Brier 0.2648, log loss 0.7315, AUC 0.5790, spread MAE 11.15; no BigQuery odds-backed ROI.

MLB artifacts retained in `data-core/models`: `mlb_winner_model_v3.pkl` and `mlb_winner_model_v3_metrics.json`. The v3 model is a random forest selected by 2025 validation Brier and refit on 2021-2025 before testing on 2026 YTD. It is a ship candidate for probability display, not for betting/ROI recommendations.

See `data-core/models/README.md` for the retained artifact registry.

## Regenerable Research Artifacts

PGA and CBB trainer outputs are no longer committed. They regenerate into
ignored `data-core/models/saved/` or a caller-supplied temp directory when
running the research trainers locally.

Latest CBB non-destructive CV export used `/tmp/sports-edge-cbb-cv-models` and found XGBoost strongest on 2016-2025 mean log loss: 0.575 vs LGBM 0.581 and meta 0.595.

Removed historical artifacts include NBA `v1`, `v2`, and `v4`; NFL `v2`; MLB `v1` and `v2`; and unpromoted TD scorer files. Recreate them from the relevant trainer only when a documented comparison justifies promotion.

## Promotion Criteria

Promote a new version only when:

1. Backtest or CV metrics improve on the target objective.
2. ATS/ROI or ranking lift is measured against a fixed baseline and date range.
3. The source data window and model artifact names are documented here.
4. The daily workflow and serving sync agree on the promoted version.
5. Old serving rows are filtered or archived so stale versions do not become latest predictions by accident.

No NBA/NFL retrain was promoted in this pass. PGA retrain remains pending because the refreshed feature store changed materially and needs a controlled v2/v3 experiment first.

MLB v3 may be promoted only to a clearly labeled probability surface after a serving dry run because it now has a prediction CLI and reproducible artifact. It should not be promoted to betting/ROI surfaces until it is compared against moneyline markets. Its 2026 YTD test edge over the home-rate baseline is modest: Brier 0.2478 vs 0.2497 and log loss 0.6888 vs 0.6925.
