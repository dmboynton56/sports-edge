# NFL v2 experiment workflow

NFL v2 is the canonical NFL live feed under an explicitly approved monitored
rollout. NFL v1 remains available only as a research comparison and rollback
artifact. This activation is not recorded as having passed the formal
promotion gates; graded 2026 predictions remain the next untouched evidence.

## Build the immutable dataset

```bash
PYTHONPATH=data-core data-core/.venv/bin/python \
  data-core/scripts/build_nfl_v2_dataset.py \
  --project learned-pier-478122-p7
```

The command exports a content-addressed Parquet file and manifest under
`data-core/data/curated/`. It refuses to overwrite an existing export. The
manifest records source and data fingerprints, extraction time, schemas,
coverage, row counts, source timestamps, and excluded feature-family gates.

## Run or reproduce an experiment

The initial configuration is
`data-core/config/experiments/nfl_v2.yaml`. Its 2025 locked test is marked
consumed. The runner now refuses to reopen it by default. Exact reproduction
requires the explicit `--allow-consumed-locked-test` flag and must not be used
as new tuning evidence.

Every run creates an immutable directory under
`data-core/artifacts/experiments/nfl_v2/` with the frozen config, fingerprints,
row identities, candidate/ablation/calibration/locked metrics, predictions,
models, environment, market status, and promotion decision.

New iterations must register one hypothesis and change family, add 2025 as a
rolling-origin validation fold, and treat graded 2026 predictions as the next
untouched evidence. Negative runs stay recorded; do not add filters selected
from the consumed 2025 results.

## Market track

Aggregate nflverse closing lines are benchmark-only. The residual-log-odds
market model remains disabled unless `market_home_prob`, `market_spread`,
`market_as_of_ts`, and `kickoff_ts` have at least 90% joint coverage and every
market timestamp precedes kickoff. Enabled market comparisons report paired
bootstrap intervals and probability CLV when a closing probability exists.

## Live artifact and evaluation

The frozen live selection is recorded under `live_selection` in the experiment
config. Rebuild its artifact only from the content-addressed 2020–2025 dataset:

```bash
PYTHONPATH=data-core data-core/.venv/bin/python \
  data-core/scripts/train_nfl_v2_live.py \
  --dataset data-core/data/curated/nfl_v2_2020_2025_c975f402_4eb6118c.parquet
```

The daily refresh publishes `nfl-v2-live-20260906`. It fails closed when the
artifact contract changes, completed-game PBP coverage falls below 99%, target
features are missing, or an output is invalid. Completed or already-started
games are excluded so later refreshes cannot overwrite pregame evidence.

```bash
PYTHONPATH=data-core data-core/.venv/bin/python \
  data-core/scripts/evaluate_nfl_shadow.py \
  --predictions /path/to/graded-shadow.csv \
  --output /path/to/immutable-shadow-evaluation.json
```

Promotion requires both 64 graded games and four distinct weeks, complete data
fingerprints, prediction timestamps before kickoff, passing freshness and
integrity flags, and the configured probability and margin bias/calibration
gates.

The weekly performance-history workflow runs `grade_nfl_v2_live.py` against
the Supabase serving rows and records probability calibration, home bias,
margin error, and early/late-season segments. A live rollout is not itself a
claim that these gates passed.

## Curated partition repair

Build each target partition into a custom staging table with
`build_feature_snapshots.py --destination-table ...`. Custom destinations clone
the canonical schema. `promote_feature_snapshot_partitions.py` is read-only by
default; `--apply` is available only after its exact partition, grain, and raw
schedule-lineage audit passes. Application creates a dated backup and replaces
only NBA 2025, NFL 2025, and NFL 2026 in one transaction.
