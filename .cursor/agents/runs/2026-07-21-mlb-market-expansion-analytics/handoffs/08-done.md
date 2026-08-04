# Packet 08 done — synthesis

The synthesis report is at `data-core/docs/analysis/mlb_market_expansion_2026-07.md`. Moneyline v4 is a controlled negative versus v1 features on identical rows; totals, run line, and starter strikeouts beat their stated baselines but remain research-grade because derivative-market odds are absent, and totals additionally relies on observed game-time weather as a forecast proxy. Four artifact-backed rows were appended to `data-core/docs/PERFORMANCE_HISTORY.md`. The optional dashboard insight was skipped.

## Run-level hard-gate audit

1. **Data rebuilt end to end — holds.** Games/boxscores contain 13,653 matched rows through 2026-07-20, with 100% coverage, zero error rows, required weather/starter/team fields, and an audit JSON.
2. **Feature store v2 leakage-safe — holds.** The 13,182-row store contains required starter/weather/run-environment/label columns; label exclusion assertions and `pytest -k mlb` passed.
3. **Moneyline v4 honestly compared — holds.** Four ablation arms share 1,428 test rows and 673 odds joins; full v2 is reported as a negative result with Brier/AUC/ROI deltas and edge buckets.
4. **New market metrics — holds.** Totals, run line, and strikeouts have test metrics and predictions; required naive baselines are present, and ROI is null with a no-odds reason.
5. **Synthesis document — holds.** All eight requested sections, v3 diagnosis tables, artifact metrics, observed-weather caveat, gap register, ranked next steps, commands, and decisions are present.
6. **Document metrics match artifacts — holds.** A verification script reopened all nine input JSONs and asserted rendered metrics and derived deltas against the report.

No hard-gate misses were found. `goal_done` remains `false` and was not changed, per the worker protocol and explicit task instruction; Fable owns the run-level completion decision.
