# Codex implementation pass

You are **Codex** (`gpt-5.6-sol`), the implementer/tester. Read `.cursor/agents/codex-worker.md` and `.cursor/agents/PROTOCOL.md`.

- Repo: `/home/dmboynton/projects/sports-edge`
- Run: `2026-07-21-mlb-market-expansion-analytics`
- Run dir: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics`
- Task id: `02`
- Task packet: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/tasks/02-diagnosis-v3.md`

## Job

Execute the attached task packet exactly. Prefer concrete code + tests over prose.

When finished, write `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-02.md` with summary, files touched, verification commands/results, and residual risks.

Do not commit or push. Do not set `goal_done`. Do not expand scope beyond the packet.

## Attached task packet

# Packet 02 — Diagnose why MLB moneyline v3 underperforms (existing artifacts only)

**Actor:** Codex · **Depends on:** none (uses committed CSV/JSON caches, **no network, no rebuild**) · **Parallel with:** 01 · **Feeds:** 08
**No commit/push.** Repo root: `/home/dmboynton/projects/sports-edge`. Python: `data-core/.venv/bin/python`.

## Why

BRIEF deliverable #1 needs an autopsy of v3 (Brier 0.2478, AUC 0.5431, flat ROI −3.1%). The inputs already exist in git-tracked caches — this runs independently of packet 01's long backfill.

## Inputs (read-only)

- `data-core/notebooks/cache/mlb_backtest_predictions_2026_ytd_free.csv` — per-game 2026 predictions with `home_win_prob`, `pick_side`, `pick_won`, `home_moneyline`, `away_moneyline`, `market_home_prob`, `model_edge_home`, `profit`, `edge_bucket`.
- `mlb_backtest_metrics_2026_ytd_free.json`, `mlb_backtest_predictions_2025*.csv/json` (same shapes, 2025 window), `models/mlb_winner_model_v3_metrics.json`.
- Context docs: `data-core/docs/analysis/mlb_performance_2026-05-21.md`, `data-core/docs/PERFORMANCE_HISTORY.md`.

## Files you own

- `data-core/notebooks/cache/mlb_ml_diagnosis_2026_ytd.json` (new artifact)
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/artifacts/diagnosis-v3.md` (new)
- A throwaway or run-dir analysis script is fine; if you want it in-repo, name it `data-core/scripts/analyze_mlb_ml_errors.py` (new file, nobody else owns it). Touch nothing else.

## Work

Compute, for 2026 YTD (and 2025 as a stability check where the file allows):

1. **Calibration:** reliability table in 10 bins of `home_win_prob` (n, mean pred, empirical home-win rate), ECE; note the prediction distribution (v3 RF is known to hug 0.45–0.60 — quantify: p5/p50/p95 of `home_win_prob`).
2. **Model vs market:** correlation and mean absolute gap between `home_win_prob` and `market_home_prob`; accuracy/Brier of the *market* probabilities on the same rows vs the model — i.e., is v3 adding anything beyond the free line, or is it a noisier copy of it?
3. **ROI autopsy:** re-derive flat ROI and edge-bucket table; ROI split by pick side (home vs away), by favorite vs underdog picks, by month, and by `abs_edge` deciles. Identify where the −3.1% concentrates and whether the green 0–2% bucket is signal or small-n noise (report n and a rough binomial CI).
4. **Error buckets:** loss rate by predicted-prob band; biggest systematic miss cohorts you can extract from the columns available (e.g., heavy-favorite home losses). Don't invent features you don't have — say what can't be diagnosed without the v2 store (weather, starters) and leave it to packet 08.
5. **Verdict paragraph:** rank hypotheses for weak AUC (feature ceiling: no starter-quality signal; season-reset states discard April information; free odds are soft/stale; RF compressing probabilities) with the evidence each has.

Write the numbers into `mlb_ml_diagnosis_2026_ytd.json` (structured: `calibration_bins`, `market_comparison`, `roi_splits`, `edge_buckets`, `pred_dist`) and the narrative into `artifacts/diagnosis-v3.md` (tables + verdict; this becomes the diagnosis section of the final doc).

## Acceptance

- JSON artifact exists and is self-consistent (bucket n's sum to row count; ROI matches the committed `flat_roi` within rounding).
- `diagnosis-v3.md` has: reliability table, model-vs-market comparison, ROI split tables, ranked hypotheses, and an explicit list of "needs v2 store to answer".
- No network calls; no modification to any existing file. STATUS.json updated + `handoffs/02-done.md` note.
