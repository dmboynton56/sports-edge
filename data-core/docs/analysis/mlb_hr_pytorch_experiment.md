# MLB HR PyTorch Experiment

Generated: 2026-06-18

## Current Baseline

Source artifact: `data-core/models/mlb_hr_model_v1_metrics.json`.

| Model | Split | Rows | Positive rate | Brier | Log loss | AUC | Top 10 hit rate | Top 25 hit rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlb-hr-v1` random forest | Test, starting 2026-05-15 | 8,185 | 11.63% | 0.1017 | 0.3548 | 0.6048 | 18.75% | 18.75% |
| League-rate baseline | Same rows | 8,185 | 11.63% | 0.1024 | 0.3575 | n/a | n/a | n/a |
| `mlb-hr-torch-v1` raw PyTorch | Same rows | 8,185 | 11.63% | 0.1019 | 0.3550 | 0.5966 | 16.56% | 17.25% |
| PyTorch + heuristic blend | Same rows | 8,185 | 11.63% | 0.1015 | 0.3535 | 0.6057 | 17.81% | 18.25% |
| Handedness PyTorch raw | Same rows | 8,185 | 11.63% | 0.1020 | 0.3556 | 0.5852 | 18.75% | 17.88% |
| Handedness PyTorch + heuristic blend | Same rows | 8,185 | 11.63% | 0.1016 | 0.3538 | 0.5998 | 21.25% | 18.63% |
| Statcast PyTorch raw | Same rows | 8,185 | 11.63% | 0.1019 | 0.3553 | 0.5993 | 20.31% | 18.38% |
| Statcast PyTorch + heuristic blend | Same rows | 8,185 | 11.63% | 0.1016 | 0.3536 | 0.6081 | 18.75% | 19.88% |

The current model is leakage-aware and directionally useful, but the lift over
the league-rate baseline is modest. That makes this a good candidate for a
controlled neural experiment rather than a blind model swap.

The first raw neural model is comparable to the random forest but not better by
itself. The stronger result is a validation-chosen blend: 84% PyTorch and 16%
heuristic probability. That blend improves held-out Brier, log loss, and AUC
versus the current random forest, but still trails the current top-10/top-25 hit
rates. This is a good research signal, not a promotion-ready betting edge yet.

Handedness enrichment used MLB Stats API player metadata for 768 unique players
and produced 100% known batter/pitcher handedness coverage in this training set.
The handedness model did not improve calibration, but its blend raised top-10
hit rate to 21.25%. That suggests platoon context may be more useful for ranking
HR candidates than for raw probability calibration unless paired with richer
pitch-level and contact-quality features.

Statcast enrichment used 379,038 Baseball Savant pitch-level rows from
2026-03-01 through 2026-06-15. Feature readiness was 99.1% overall and 99.4%
in the held-out test window. The Statcast blend produced the best AUC so far
(0.6081) and the best top-25 hit rate (19.88%), while the handedness blend
remained best for top-10 hit rate.

## Daily Outcome Loop

New script: `data-core/scripts/evaluate_mlb_home_run_predictions.py`.

The current daily prediction CSV was joined to completed MLB boxscores for
2026-06-16:

| Rows | Positive rate | Brier | Log loss | AUC | Top 10 hit rate | Top 25 hit rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 110 | 20.00% | 0.1642 | 0.5158 | 0.5594 | 10.00% | 20.00% |

This is a one-slate live-board check, not a model promotion sample. It confirms
the outcome join works and gives us the rolling feedback loop needed to monitor
candidate-ranking quality after games complete.

## PyTorch Design

New script: `data-core/scripts/train_mlb_home_run_torch_model.py`.

The experiment uses the same training rows as the random forest and preserves
the same time split. The PyTorch model combines:

- continuous lagged batter, pitcher, venue, lineup, baseline, and heuristic features;
- batter, opposing starter, team, and opponent embeddings;
- a compact MLP trained with binary cross-entropy;
- same-split comparison against `mlb-hr-v1`, heuristic probability, and baseline probability.

Output artifacts:

- `data-core/models/mlb_hr_torch_model_v1.pt`
- `data-core/models/mlb_hr_torch_model_v1_metrics.json`
- `data-core/models/mlb_hr_torch_handed_model_v1.pt`
- `data-core/models/mlb_hr_torch_handed_model_v1_metrics.json`
- `data-core/models/mlb_hr_torch_statcast_model_v1.pt`
- `data-core/models/mlb_hr_torch_statcast_model_v1_metrics.json`
- `web/public/data/mlb_hr_experiment.json`

## Commands

```bash
cd data-core
python3 -m pip install -r requirements.txt

python3 scripts/train_mlb_home_run_torch_model.py \
  --dataset notebooks/cache/mlb_home_run_training_rows.csv \
  --device auto \
  --epochs 35

python3 scripts/enrich_mlb_hr_training_rows.py
python3 scripts/enrich_mlb_hr_statcast_features.py \
  --start-date 2026-03-01 \
  --end-date 2026-06-15

python3 scripts/train_mlb_home_run_torch_model.py \
  --dataset notebooks/cache/mlb_home_run_training_rows_enriched.csv \
  --model-version mlb-hr-torch-handed-v1 \
  --model-out models/mlb_hr_torch_handed_model_v1.pt \
  --metrics-out models/mlb_hr_torch_handed_model_v1_metrics.json \
  --device auto \
  --epochs 45

python3 scripts/train_mlb_home_run_torch_model.py \
  --dataset notebooks/cache/mlb_home_run_training_rows_statcast.csv \
  --model-version mlb-hr-torch-statcast-v1 \
  --model-out models/mlb_hr_torch_statcast_model_v1.pt \
  --metrics-out models/mlb_hr_torch_statcast_model_v1_metrics.json \
  --device auto \
  --epochs 45

python3 scripts/evaluate_mlb_home_run_predictions.py
python3 scripts/export_mlb_hr_experiment_summary.py
```

The local RTX 5070 is visible through `nvidia-smi`; `data-core/.venv` has a
CUDA-enabled PyTorch build and was used for these runs.

## Data Expansion

The next feature lift should come from richer matchup data, not just a deeper
network:

- [Baseball Savant Statcast CSV docs](https://baseballsavant.mlb.com/csv-docs):
  pitch-level pitch type, velocity, launch angle, exit velocity, batted-ball
  events, and outcomes.
- [pybaseball](https://github.com/jldbc/pybaseball): Python access to Baseball
  Savant, FanGraphs, and Baseball Reference for reproducible backfills.
- MLB Stats API: current repo source for schedule, probable pitcher, lineup,
  and boxscore-derived outcomes.

## Publish Gate

Use the website insight page as a draft until all of these are true:

1. Continue accumulating daily outcome joins across multiple slates.
2. Add sportsbook HR prices and compare model probabilities to implied prices.
3. The blend beats `mlb-hr-v1` on both probability metrics and top-K ranking,
   or the post clearly explains the ranking tradeoff.
4. The post clearly labels the model as candidate unless odds-backed ROI exists.
