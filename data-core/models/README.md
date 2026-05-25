# Model Artifact Registry

This directory contains only artifacts that are intentionally retained for the
current Sports Edge story. Regenerable research outputs should stay local and
ignored unless they are promoted through `data-core/docs/MODEL_VERSIONS.md`.

## Committed Runtime Artifacts

| Surface | Files | Status |
| --- | --- | --- |
| NBA daily pipeline | `win_prob_model_nba_v3.pkl`, `spread_model_nba_v3.pkl`, `link_function_nba_v3.pkl`, `feature_medians_nba_v3.pkl`, `meta_ensemble_nba_v3.pkl` | Active daily workflow version |
| NFL daily pipeline | `win_prob_model_nfl_v1.pkl`, `spread_model_nfl_v1.pkl`, `link_function_nfl_v1.pkl`, `feature_medians_nfl_v1.pkl` | Active daily workflow version |
| MLB probability display | `mlb_winner_model_v3.pkl`, `mlb_winner_model_v3_metrics.json` | Active probability-only daily workflow version, not a betting surface |

## Ignored Or Removed Artifacts

- NBA `v1`, `v2`, and `v4` artifacts were older or unpromoted versions.
- NFL `v2` artifacts were superseded by the active `v1` workflow contract.
- MLB `v1` and `v2` artifacts were superseded by the retained `v3` candidate.
- TD scorer artifacts are not part of the current daily or portfolio surface.
- PGA and CBB research artifacts are generated into ignored
  `data-core/models/saved/` when those trainers are run locally.

Promotion rule: commit a new artifact only when its source window, metrics,
daily/serving version tag, and portfolio claim are all documented.
