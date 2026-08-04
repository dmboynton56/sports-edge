# Codex done — task 08

## Summary

- Added the eight-section MLB market-expansion synthesis with an artifact-traceable v3 diagnosis, controlled moneyline ablation, totals/run-line/strikeout results, data coverage, caveats, gap register, ranked next steps, reproducible commands, and promotion decisions.
- Appended one summary row per market to performance history without changing an existing row.
- Audited all run-level hard gates and moved the run to Fable review without changing `goal_done`.
- Skipped the optional dashboard insight; the packet explicitly permits the document-only deliverable.

## Files touched

- `data-core/docs/analysis/mlb_market_expansion_2026-07.md` (new)
- `data-core/docs/PERFORMANCE_HISTORY.md` (four appended table rows)
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/STATUS.json`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/08-done.md` (new)
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-08.md` (new)

## Verification

- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest data-core/tests/unit -k mlb -q` — 55 passed, 88 deselected; six pre-existing sklearn pickle-version warnings.
- Artifact-to-document assertion script — reopened all nine input JSONs; checked rendered diagnosis, moneyline, totals, run-line, strikeout, coverage, null-rate, ROI-null, and derived-delta values; passed.
- Hard-gate artifact/schema script — required parquet/CSV artifacts exist, required v2 columns are present, and labels are absent from `default_feature_columns`; passed.
- Append-only history assertion — every pre-existing line remains and exactly four lines were added; passed.
- `git diff --check -- data-core/docs/analysis/mlb_market_expansion_2026-07.md data-core/docs/PERFORMANCE_HISTORY.md` — passed.

## Residual risks / follow-ups

- Free moneylines cover only 673/1,428 test games and are soft comparison/consensus lines, not timestamped sharp closers.
- Totals, run-line, and strikeout ROI cannot be measured without historical market-specific odds.
- Weather lift uses observed game-time records as pregame forecast proxies and must be replicated with prediction-time forecasts.
- The moneyline ablation does not isolate cross-season run environment, and the totals ablation combines weather with park context.
- No production wiring or dashboard insight was added; all models remain research-grade pending Fable review.
