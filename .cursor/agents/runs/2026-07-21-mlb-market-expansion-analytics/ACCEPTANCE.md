# ACCEPTANCE — MLB market expansion analytics

Overall run is DONE when every item below holds. Per-packet gates live in each `tasks/NN-*.md`; this file is the run-level bar.

## Hard gates

1. **Data rebuilt end-to-end.** `data-core/notebooks/cache/mlb_games_2021_2026.parquet` and `mlb_boxscores_2021_2026.parquet` exist, cover 2021 through the latest completed 2026 games, and the boxscore rows carry parsed `temp_f`, `wind_mph`, `wind_dir`, starter strikeouts, and team batting/pitching totals with an audit JSON reporting null rates per column.
2. **Feature store v2 is leakage-safe.** `mlb_feature_store_v2_2021_2026.parquet` exists with: cross-season probable-starter rolling features (K/9, ERA proxy, BB/9 over last-N prior starts, rest days), weather columns, run-environment columns, and labels `home_win`, `run_diff`, `total_runs`, `home_cover_15`, `home_starter_ks_label`/`away_starter_ks_label`. `default_feature_columns` on this frame contains **no** label column and no same-game postgame column, and `data-core/tests/unit/test_mlb_market_features.py` proves (a) a pitcher's features before start N exclude start N, (b) label/postgame exclusion, (c) wind/weather text parsing including dome/missing variants. All new unit tests pass via `data-core/.venv/bin/python -m pytest data-core/tests/unit -k mlb`.
3. **Moneyline v4 trained and honestly compared.** Metrics JSON + predictions CSV for test-2026-YTD exist; the ablation JSON contains at minimum arms {v1 features, +pitcher, +weather/full} evaluated on identical test rows, so ΔBrier/ΔAUC vs the v1 feature set is stated on one window; free-moneyline flat ROI + edge-bucket table computed for 2026. An improvement **or** a clean negative result with ablation attribution both pass — an unmeasured claim fails.
4. **At least one new market shipped with real train/test metrics** (totals or run line; both preferred): metrics JSON + predictions CSV in `data-core/notebooks/cache/`, including a naive-baseline comparison (league/venue rolling mean for totals; base cover rate for run line) and calibration/ECE for any classifier head. No ROI reported for markets without odds — instead an explicit `"roi": null, "reason": "no odds source"` style field.
5. **Synthesis doc exists** at `data-core/docs/analysis/mlb_market_expansion_2026-07.md` containing: v3 diagnosis (calibration table, error buckets, edge-bucket ROI autopsy from packet 02), every trained market's train/val/test metrics pulled from the artifact JSONs (no hand-typed numbers that disagree with artifacts), the weather-as-observed leakage caveat, data-gap register (sharp closing lines, totals/spread odds, umpire/lineups), and a ranked "what to try next" list.
6. **Metrics in docs match artifacts.** Spot-check: every number quoted in the synthesis doc for Brier/AUC/ROI must appear in (or be derivable from) a JSON artifact in `notebooks/cache/` or `models/`.

## Soft gates (nice-to-have, do not block)

- Strikeout market (packet 07) metrics with probable==actual starter filter and mismatch-rate note.
- `models/mlb_winner_model_v4.pkl` (+ totals/runline pickles) saved with metrics sidecars.
- Insights/Performance dashboard hook reading only from artifacts.
- `docs/PERFORMANCE_HISTORY.md` appended with v4 + new-market rows.

## Process constraints

- **No `git commit` / `git push`.** Working tree changes + cache artifacts only.
- No schema changes to `games` / `model_predictions` / `odds_snapshots`; no paid API calls; no secrets in artifacts.
- Scripts follow existing CLI conventions (`argparse`, `PYTHONPATH=data-core`, defaults pointing at `data-core/notebooks/cache/...`).
- Every packet updates `STATUS.json` (`active_tasks`/`completed_tasks`) and drops a short completion note in `handoffs/`.
