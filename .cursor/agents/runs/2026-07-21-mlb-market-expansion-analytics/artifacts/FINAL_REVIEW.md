# FINAL REVIEW — MLB market expansion analytics

Reviewer: Fable · 2026-07-21 · Verdict: **PASS — goal done**

All six hard gates verified independently (re-ran tests, reopened parquets/JSONs, cross-checked doc numbers against artifacts). Worker done-notes were accurate; no discrepancies found.

## Hard-gate verification

1. **Data rebuilt** — PASS. `mlb_games_2021_2026.parquet` + `mlb_boxscores_2021_2026.parquet` present (13,653 boxscore rows, 2021→2026-07-20). Boxscores carry `temp_f`, `wind_mph`, `wind_dir`, starter/team strikeouts, team batting/pitching totals. `mlb_boxscores_2021_2026_audit.json` has `null_rate_per_new_column`, per-season rows, coverage 1.0, zero error rows.
2. **Feature store v2 leakage-safe** — PASS. 13,182 rows, seasons 2021–2026. All six labels present (`home_win`, `run_diff`, `total_runs`, `home_cover_15`, `home/away_starter_ks_label`). `default_feature_columns` returns 110 columns with zero label/postgame leakage (checked by name and by keyword scan). Cross-season starter rolling features (`*_starter_k9_last5`, `bb9_last5`, `era_proxy_last5`, rest days) and weather columns (`temp_f`, `wind_mph`, `wind_out/in/cross`, `is_dome_or_closed`, `elevation`) present. Tests prove prior-start-only histories, cross-season carry, label exclusion, dome/wind parsing. **Full suite re-run: 55 passed, 88 deselected.**
3. **Moneyline v4 honestly compared** — PASS. Ablation JSON has four arms (v1_baseline / +starter / +weather / full_v2) with `identical_test_rows: true` on 1,428 games; ROI + 4-row edge-bucket table on the 673-game free-line join. Clean **negative result**, correctly attributed: full v2 Brier 0.247468 vs v1-arm 0.247423, AUC 0.549654 vs 0.550626, ROI −5.86% vs −3.08%. BRIEF explicitly accepts a negative result with ablation attribution.
4. **New markets with real metrics** — PASS (both, plus stretch). Totals: ridge, test MAE 3.5364 beating all three baselines (3.5885–3.6503), binary heads at 8.5/9.5 with Brier/ECE vs constant base rate; `"roi": null` + reason. Run line: RF, test Brier 0.22830 vs 0.23019 base-rate baseline, ECE 0.0241, calibration table sums to 1,428; `roi: null` + reason. Strikeouts (soft gate): MAE 1.7960 vs 1.8673 baseline, 97.79% clean coverage (2,793/2,856), mismatch handling documented; `roi: null`.
5. **Synthesis doc** — PASS. `docs/analysis/mlb_market_expansion_2026-07.md` (223 lines, 8 sections): v3 reliability + ROI autopsy from packet 02, per-market train/val/test tables, gap register naming sharp closing lines / derivative odds / lineups-umpires-injuries / weather timing / ablation resolution, ranked 6-item next-steps list, reproducible commands, promotion decision. Weather-as-observed caveat appears in doc, module, audit sidecar, and STATUS notes.
6. **Doc numbers match artifacts** — PASS. Spot-checked every headline figure (v4 Brier/AUC/ROI per arm, totals MAE/RMSE/ablation deltas, run-line Brier/AUC/ECE, Ks MAE/coverage) against the reopened JSONs; all reconcile, including the −5.86%/−3.08% ROI pair and the +0.0547 MAE / +0.0501 AUC totals ablation deltas.

## Soft gates

- Strikeout market: **shipped** with clean-coverage filter and mismatch note.
- Pickles + metrics sidecars: **shipped** (v4 winner, totals v1, runline v1).
- `PERFORMANCE_HISTORY.md`: **appended** exactly four rows, append-only verified via git diff; numbers match artifacts.
- Dashboard insight: **skipped** — explicitly permitted by BRIEF/packet 08.

## Process constraints

- No commit/push (working tree only, verified via git status). No changes outside MLB analytics paths + run dir. No schema changes to portfolio tables. No paid API usage; free MLB Stats API only. No secrets in artifacts.

## Notes for dispatcher

- Headline honest outcome: **moneyline v4 is a negative result** (starter/weather additions did not beat the v1 feature set on identical rows); run line and totals both beat their naive baselines; no derivative-market ROI is claimable without odds sources.
- Top follow-ups (from doc §6): acquire timestamped closing lines; replace observed weather with pregame forecasts before operational use of the totals lift; bullpen-fatigue features next.
