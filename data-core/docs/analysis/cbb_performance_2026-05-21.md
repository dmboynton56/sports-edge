# CBB Performance - 2026-05-22

Cache input: `notebooks/cache/cbb_matchup_feature_store.csv`, 2,002 rows, seasons 2010-2025. Raw Kaggle MMLM files are absent from the repo, and 2026 tournament labels are not in the matchup feature store.

The expanding-window CV was run with `models_dir=/tmp/sports-edge-cbb-cv-models` to avoid overwriting saved production artifacts.

Command:

```bash
PYTHONPATH=data-core python3 data-core/scripts/export_cbb_cv_history.py
```

Machine-readable exports:

- `notebooks/cache/cbb_expanding_cv_2016_2025.csv`
- `notebooks/cache/cbb_expanding_cv_2016_2025.json`

## Expanding-Window CV

| Validation year | LGBM log loss | XGB log loss | Meta log loss | XGB Brier | XGB AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2016 | 0.613 | 0.595 | 0.620 | 0.203 | 0.756 |
| 2017 | 0.531 | 0.523 | 0.533 | 0.177 | 0.815 |
| 2018 | 0.614 | 0.609 | 0.618 | 0.210 | 0.735 |
| 2019 | 0.507 | 0.517 | 0.549 | 0.174 | 0.818 |
| 2021 | 0.625 | 0.630 | 0.643 | 0.221 | 0.691 |
| 2022 | 0.659 | 0.630 | 0.654 | 0.221 | 0.687 |
| 2023 | 0.631 | 0.624 | 0.643 | 0.217 | 0.720 |
| 2024 | 0.598 | 0.569 | 0.593 | 0.197 | 0.760 |
| 2025 | 0.454 | 0.479 | 0.505 | 0.162 | 0.841 |
| Mean | 0.581 | 0.575 | 0.595 | 0.198 | 0.758 |

XGBoost has the best mean log loss, Brier, ECE, and AUC. The current stacker is not beating the base models in this CV run.

## Upset Calibration

The upset calibration helper now de-duplicates mirrored matchup rows before counting physical tournament games.

| Matchup | Games | Avg underdog probability | Actual upsets | Actual upset rate |
| --- | ---: | ---: | ---: | ---: |
| 1v16 | 36 | 25.7% | 2 | 5.6% |
| 2v15 | 36 | 26.5% | 4 | 11.1% |
| 3v14 | 37 | 31.1% | 3 | 8.1% |

The model remains materially overconfident in major underdogs, but the counts are now physically plausible.

## Weaknesses

- Raw Kaggle data is missing, so caches cannot be independently rebuilt from source.
- 2026 team stats exist in cache, but 2026 tournament matchup labels do not.
- Major-underdog probabilities are too high after de-duplicating mirrored rows.
- The meta stacker underperforms the XGBoost base model on mean log loss.
- `/cbb` remains a seed-sim experience rather than a fully sourced 2026 tournament model.
