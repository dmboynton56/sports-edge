# Codex implementation pass

You are **Codex** (`gpt-5.6-sol`), the implementer/tester. Read `.cursor/agents/codex-worker.md` and `.cursor/agents/PROTOCOL.md`.

- Repo: `/home/dmboynton/projects/sports-edge`
- Run: `2026-07-21-mlb-market-expansion-analytics`
- Run dir: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics`
- Task id: `04`
- Task packet: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/tasks/04-moneyline-v4.md`

## Job

Execute the attached task packet exactly. Prefer concrete code + tests over prose.

When finished, write `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-04.md` with summary, files touched, verification commands/results, and residual risks.

Do not commit or push. Do not set `goal_done`. Do not expand scope beyond the packet.

## Attached task packet

# Packet 04 — Moneyline v4: retrain on v2 store, ablation, ROI vs v3

**Actor:** Codex · **Depends on:** 03 · **Parallel with:** 05, 06, 07 (file-disjoint) · **Feeds:** 08
**No commit/push.** Repo root: `/home/dmboynton/projects/sports-edge`. Python: `data-core/.venv/bin/python`, `PYTHONPATH=data-core`.

## Why

Answer BRIEF's core question with one controlled experiment: does the expanded feature set (starter rolling stats, weather, cross-season carryover) move Brier/AUC/ROI on 2026, and which feature group is responsible? v3 baseline (673-game window ending 2026-05-21): Brier 0.2478, log loss 0.6888, AUC 0.5431, acc 53.79%, flat ROI −3.1% on free lines.

## Files you own

- `data-core/scripts/train_mlb_winner_model.py` (add optional `--features-path` to train from a prebuilt store instead of refetching/rebuilding v1 features; default behavior unchanged)
- `data-core/scripts/ablate_mlb_winner_features.py` (new)
- Artifacts: `data-core/models/mlb_winner_model_v4.pkl` (+ auto `_metrics.json`), `data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd_v4_free.json`, `mlb_backtest_predictions_2026_ytd_v4_free.csv`, `mlb_ml_ablation_2026_ytd.json`
- Uses read-only: `scripts/backtest_mlb_winners.py` (CLI already supports `--features-path/--odds-path/--validation-season/--test-season/--predictions-output/--metrics-output`), v2 store, `notebooks/cache/mlb_free_moneylines_2025_2026.csv`. Touch nothing else — **do not edit** `mlb_winner_model.py`, `backtest_mlb_winners.py`, or any `src/features` file (owned by 03).

## Work

1. **Train/test v4** on `notebooks/cache/mlb_feature_store_v2_2021_2026.parquet` — split: validation 2025, test 2026 (train 2021–2024 for selection; `train_and_evaluate_mlb_winner` handles selection + ≤2025 refit already):
   ```bash
   PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/backtest_mlb_winners.py \
     --features-path data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet \
     --validation-season 2025 --test-season 2026 \
     --odds-path data-core/notebooks/cache/mlb_free_moneylines_2025_2026.csv \
     --predictions-output data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd_v4_free.csv \
     --metrics-output data-core/notebooks/cache/mlb_backtest_metrics_2026_ytd_v4_free.json
   ```
   Then save the pickle via `train_mlb_winner_model.py --features-path ... --model-version v4 --output-model data-core/models/mlb_winner_model_v4.pkl` (same split args).
2. **Ablation** (`ablate_mlb_winner_features.py`): identical split/pipeline, arms = feature-column subsets of the v2 frame:
   - `v1_baseline` — exactly the v1 column set (reconstruct: columns produced by the v1 builder; 03's handoff note lists which columns are new — everything else is v1),
   - `v1_plus_starter` — + starter rolling/handedness group,
   - `v1_plus_weather` — + weather/park group,
   - `full_v2` — everything `default_feature_columns` returns.
   For each arm output val + test Brier/log-loss/AUC/acc/ECE **and** flat ROI/edge buckets on the same free-odds join, all on identical test rows; write `mlb_ml_ablation_2026_ytd.json` with row counts and the column list per arm. This is the apples-to-apples Δ: **compare v4 to the `v1_baseline` arm on this window**, not to v3's published 673-game numbers (report those as context only).
3. **Sanity checks:** test rows ≈ all completed 2026 games post-filter (expect ~1.3–1.5k in mid-July); odds join coverage reported (free CSV covers 2025–2026; expect high but not full coverage); prediction distribution p5/p95 noted (is v4 less compressed than v3?).

## Acceptance

- Metrics + predictions + ablation artifacts written; ablation arms share identical test row counts.
- A clear one-paragraph verdict in `handoffs/04-done.md`: ΔBrier/ΔAUC/ΔROI of full_v2 vs v1_baseline with which group drove it — improvement or honest negative both acceptable; unmeasured claims are not.
- Model pickle + metrics sidecar saved; default (no `--features-path`) behavior of the train script unchanged.
- STATUS.json updated.
