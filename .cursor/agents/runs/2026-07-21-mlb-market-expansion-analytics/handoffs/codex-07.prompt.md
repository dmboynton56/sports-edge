# Codex implementation pass

You are **Codex** (`gpt-5.6-sol`), the implementer/tester. Read `.cursor/agents/codex-worker.md` and `.cursor/agents/PROTOCOL.md`.

- Repo: `/home/dmboynton/projects/sports-edge`
- Run: `2026-07-21-mlb-market-expansion-analytics`
- Run dir: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics`
- Task id: `07`
- Task packet: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/tasks/07-strikeouts-model.md`

## Job

Execute the attached task packet exactly. Prefer concrete code + tests over prose.

When finished, write `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-07.md` with summary, files touched, verification commands/results, and residual risks.

Do not commit or push. Do not set `goal_done`. Do not expand scope beyond the packet.

## Attached task packet

# Packet 07 — Starter strikeouts model (stretch, feasibility-gated)

**Actor:** Codex · **Depends on:** 03 · **Parallel with:** 04, 05, 06 (file-disjoint) · **Feeds:** 08 · **Optional:** if the label-quality gate below fails, write the feasibility note and stop — that is a valid completion.
**No commit/push.** Repo root: `/home/dmboynton/projects/sports-edge`. Python: `data-core/.venv/bin/python`, `PYTHONPATH=data-core`.

## Why

BRIEF stretch goal: expected strikeouts for probable starters. Labels (`home/away_starter_ks_label`) and the key features (starter `k9_last5`, `pitches_last3_avg`, opponent `team_bat_k_pg_15`, weather/park) are already in the v2 store. No K odds exist locally → probability/point quality only, `"roi": null, "reason": "no strikeout odds source"`.

## Files you own

- `data-core/src/models/mlb_strikeouts_model.py` (new)
- `data-core/scripts/train_mlb_strikeouts_model.py` (new)
- Artifacts: `data-core/notebooks/cache/mlb_strikeouts_metrics_2026_ytd.json`, `mlb_strikeouts_predictions_2026_ytd.csv`
- Reads only: `notebooks/cache/mlb_feature_store_v2_2021_2026.parquet`. Do not edit shared modules.

## Feasibility gate (do first)

Reshape check on the v2 store: rows where `probable_matches_actual` is true per side. If **< 70%** of 2026 sides have a clean probable==actual match with non-null K label and starter history, write `handoffs/07-infeasible.md` explaining coverage numbers and stop.

## Design

1. **Reshape to one row per (game, side):** starter id, own rolling features (k9_last5, bb9_last5, era_proxy, pitches, rest, handedness), opponent batting K-rate (`team_bat_k_pg_15` of the opposing team), park/weather, `is_home`; label = that side's `_starter_ks_label`. **Train/eval only on probable==actual rows**; report the discarded fraction.
2. **Model:** regression on Ks (Poisson-appropriate: HistGradientBoostingRegressor with `loss="poisson"` if available in the installed sklearn, else squared error — record which). Candidates + a required baseline: pitcher's own trailing K/9 × expected outs (their `outs_per_start_last5`). Split identical to other packets (train 2021–2024, val 2025, test 2026 YTD, refit ≤2025).
3. **Metrics:** test MAE/RMSE vs baseline; distribution calibration for the common prop thresholds — P(K ≥ 5.5), P(K ≥ 6.5) via Poisson tail around the point estimate, with empirical hit rates (Brier vs base rate).
4. Predictions CSV: game_pk, date, side, pitcher name/id, expected_ks, p_over_5_5, p_over_6_5, actual. Metrics JSON includes the probable==actual coverage and the `"roi": null` note. CLI mirrors packets 05/06.

## Acceptance

- Either: metrics JSON + predictions CSV from a real run with model-vs-baseline MAE and threshold calibration, **or** the documented infeasibility note with coverage numbers. Both pass.
- Discarded-row fraction (probable≠actual) reported; no ROI claimed; `handoffs/07-done.md` verdict; STATUS.json updated.
