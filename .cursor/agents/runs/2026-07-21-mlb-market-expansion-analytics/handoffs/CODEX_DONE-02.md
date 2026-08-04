# Codex done — task 02

## Summary

- Diagnosed MLB winner-model v3 from committed 2026 YTD and 2025 free-line caches without network access or rebuilding data.
- Generated a structured JSON artifact with calibration, model-versus-market, ROI splits, edge buckets, error cohorts, and a 2025 stability check.
- Generated the Packet 08-ready technical narrative with exact tables, ranked hypotheses, uncertainty, and an explicit v2-store question list.

## Files touched

- `data-core/notebooks/cache/mlb_ml_diagnosis_2026_ytd.json` (new)
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/artifacts/diagnosis-v3.md` (new)
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/artifacts/analyze_diagnosis.py` (new, offline reproducibility script)
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/STATUS.json` (Packet 02 status only)
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-02.md` (new)
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/02-done.md` (new compatibility note required by packet acceptance)

## Verification

- `data-core/.venv/bin/python .cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/artifacts/analyze_diagnosis.py` — passed; wrote both artifacts and asserted all calibration/edge/split counts reconcile. Re-derived ROI `-0.031341652507` equals committed ROI `-0.031341652507`.
- `data-core/.venv/bin/python -m json.tool data-core/notebooks/cache/mlb_ml_diagnosis_2026_ytd.json >/dev/null` — passed strict JSON parsing.
- Independent artifact assertions — passed: 673 calibration rows, 673 edge rows, 673 rows in every ROI partition, finite JSON values, and flat-ROI difference below `1e-12`.
- Markdown acceptance check — passed: reliability, model-versus-market, ROI splits, edge deciles, error cohorts, ranked hypotheses, and `Needs v2 store to answer` are present.

## Residual risks / follow-ups

- 2026 YTD covers only April 1–May 21, so month and post-hoc cohort results are descriptive and unstable.
- Wilson intervals quantify win-rate uncertainty, not odds-aware ROI uncertainty.
- Existing artifacts cannot isolate starter/weather/season-carryover effects or validate edges against timestamped sharp closing lines; those are explicitly handed to downstream packets.
- Concurrent Packet 01 changes in MLB fetcher/backfill/test files were present and left untouched.
