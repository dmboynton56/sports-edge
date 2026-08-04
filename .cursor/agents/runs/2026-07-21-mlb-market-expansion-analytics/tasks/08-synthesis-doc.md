# Packet 08 — Synthesis doc + optional dashboard insight

**Actor:** Codex · **Depends on:** 02, 04, 05, 06 (and 07 if it produced metrics) · **Final packet.**
**No commit/push.** Repo root: `/home/dmboynton/projects/sports-edge`. Python: `data-core/.venv/bin/python`.

## Why

BRIEF deliverable #1: one honest, reproducible report tying together the v3 diagnosis and every new market's real train/test numbers, plus the ranked next-steps list.

## Files you own

- `data-core/docs/analysis/mlb_market_expansion_2026-07.md` (new — the main deliverable)
- `data-core/docs/PERFORMANCE_HISTORY.md` (append rows only)
- Optional (skip if time-boxed; doc alone is acceptable per BRIEF): one insight under `web/app/insights/` following existing post conventions there, surfacing new-market metrics **read from the artifact JSONs** — no hardcoded stats.

## Inputs (all read-only; every quoted number must trace to one of these)

- Diagnosis: `notebooks/cache/mlb_ml_diagnosis_2026_ytd.json`, run-dir `artifacts/diagnosis-v3.md` (packet 02)
- Moneyline: `mlb_backtest_metrics_2026_ytd_v4_free.json`, `mlb_ml_ablation_2026_ytd.json`, `models/mlb_winner_model_v4_metrics.json` (packet 04)
- Totals: `mlb_totals_metrics_2026_ytd.json` (05) · Run line: `mlb_runline_metrics_2026_ytd.json` (06) · Ks: `mlb_strikeouts_metrics_2026_ytd.json` or `handoffs/07-infeasible.md` (07)
- Data audits: `mlb_boxscores_2021_2026_audit.json`, `mlb_feature_store_v2_2021_2026_audit.json`
- v3 context: `docs/analysis/mlb_performance_2026-05-21.md`, `docs/PERFORMANCE_HISTORY.md`

## Doc structure (follow the style of `mlb_performance_2026-05-21.md`: scope → data → metrics tables → weaknesses → decision → reproducible commands)

1. **Scope & TL;DR** — one paragraph per market: shipped/negative/infeasible, headline metric.
2. **Why v3 underperformed** — condensed from packet 02: calibration, model-vs-market, ROI autopsy, ranked hypotheses, now cross-referenced against the packet-04 ablation (which hypotheses did the new features confirm or kill?).
3. **Data upgrades** — weather from MLB boxscore payloads (with the observed-vs-forecast leakage caveat, verbatim honest), cross-season starter rolling stats, run environment, venue meta; coverage/null-rate table from the audits.
4. **Per-market results** — moneyline v4 (v1-baseline-arm comparison on identical rows, ΔBrier/ΔAUC/ΔROI + edge buckets), totals (vs three naive baselines, weather ablation Δ), run line (vs base rate), strikeouts (or infeasibility note). ROI **only** for moneyline; other markets state `ROI not measurable — no odds source` explicitly.
5. **Gap register** — sharp/closing lines unavailable (free-plan blocker per `mlb_performance_2026-05-21.md` §Historical Odds Status), no totals/spread/K odds, no lineup/umpire/injury/travel features, weather is observed not forecast.
6. **What to try next (ranked)** — justified by the ablation/diagnosis evidence, e.g.: historical closing-line source; forecast-based weather at prediction time; bullpen fatigue from existing `bullpen_*` columns; calibration layer; lineup hydration. Each with expected payoff + cost.
7. **Reproducible commands** — the exact backfill/build/train/backtest commands (copy from packet handoffs), all `PYTHONPATH=data-core data-core/.venv/bin/python ...`.
8. **Decision** — promotion recommendation per market (research-grade vs ship-candidate), consistent with the portfolio contract (no production wiring in this run).

**Verification step (required):** before finishing, re-open each metrics JSON and diff every number quoted in the doc against it; fix mismatches. Append one summary row per market to `PERFORMANCE_HISTORY.md` matching its existing format.

## Acceptance

- Doc exists at the path above with all eight sections; every metric traceable to an artifact; caveats present (observed weather, no odds for new markets, free-line softness).
- PERFORMANCE_HISTORY.md appended without altering existing rows.
- Optional insight only if trivially cheap and artifact-driven; otherwise explicitly skipped in the handoff note.
- `handoffs/08-done.md` with the TL;DR; STATUS.json: mark run `goal_done` if run-level ACCEPTANCE.md hard gates all hold (check them one by one, list any misses instead of flipping the flag).
