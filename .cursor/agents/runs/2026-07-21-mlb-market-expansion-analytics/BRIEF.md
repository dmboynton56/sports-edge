# BRIEF

## Goal

Expand the MLB market palette beyond home-win moneyline: train/test leakage-aware models for **moneyline**, **run line (spread)**, **total runs (over/under)**, and explore **strikeout / pitcher-matchup** markets. Deliver a reproducible analytics + modeling package that (1) diagnoses why current MLB moneyline results are weak (AUC ~0.54, ROI −3.1%), (2) adds engineered features — especially **weather (temp/wind)** for totals and **probable-pitcher matchups** for ML/spread/Ks — and (3) reports honest train/test metrics + ROI-where-odds-exist plus a clear "what to try next" doc. No commit/push.

## Context

Repo: `/home/dmboynton/projects/sports-edge` (work primarily under `data-core/`; optional thin dashboard insight under `web/app/insights/` only if a packet explicitly asks).

### Current MLB surface (known baseline)

- Active winner model: `mlb_winner_model_v3` — RF, train 2021–2025 / test 2026 YTD.
- Metrics (`data-core/docs/PERFORMANCE_HISTORY.md`, `docs/analysis/mlb_performance_2026-05-21.md`, cache `mlb_backtest_metrics_2026_ytd_free.json`):
  - Brier 0.2478, log loss 0.6888, AUC **0.5431**, flat ROI **−3.1%** on free historical moneylines.
  - Tiny lift over home-rate baseline; only the 0–2% edge bucket was green.
- Feature stack today (`src/features/mlb_features.py`, `scripts/build_mlb_feature_store.py`, `src/models/mlb_winner_model.py`):
  - Rolling team form + **probable pitcher IDs** / history + venue.
  - Explicitly **missing** per prior analysis: actual starter line stats, bullpen, lineup, injury, **weather**, umpire, travel.
- Player market already exists for **HR** (`predict_mlb_home_runs.py`, Statcast blend) — do not rebuild HR; may reuse Statcast/pitcher enrichment patterns.
- Odds: free historical moneylines via `export_mlb_*_moneylines.py`; sharp closing lines are a known gap. Spreads/totals/K odds may be sparse — if missing, still train outcome models and measure probability quality; ROI only where odds join cleanly.

### Useful existing entry points

- Train/eval: `scripts/train_mlb_winner_model.py`, `scripts/backtest_mlb_winners.py`
- Features: `src/features/mlb_features.py`, `scripts/build_mlb_feature_store.py`
- Predict: `scripts/predict_mlb_winners.py`
- Games cache: `notebooks/cache/mlb_games_2021_2026.parquet` (and similar)
- HR feature patterns (Statcast, pitcher): `src/models/mlb_hr_statcast_features.py`

### Weather / pitcher data

Prefer sources already reachable without new paid APIs if possible:
- MLB Stats API game/boxscore (probable pitchers, line scores, Ks already in boxscores).
- Weather: check whether game payloads already carry temp/wind/condition; else use a free historical weather join keyed by venue+date (Open-Meteo / similar) with clear attribution. Document gaps honestly.

## Must deliver

1. **Diagnosis report** — `data-core/docs/analysis/mlb_market_expansion_2026-07.md` (or similar dated path): why ML underperforms, error buckets, calibration, edge-bucket ROI, feature ablation notes, recommended upgrades.
2. **Expanded feature store** — leakage-safe columns for:
   - probable starter rolling K/9, ERA/FIP proxies from prior starts, handedness if cheap;
   - weather (temp, wind speed/direction or park-relative wind factor) for totals;
   - team offense/defense run environment for totals & run line.
3. **Train/test models** (time-based split; prefer train≤2025 / test=2026 YTD, same spirit as v3):
   - Moneyline (home win) — improved vs v3 baseline, report ΔBrier/AUC/ROI.
   - Run line / spread (home covers −1.5 or model-predicted margin vs line when odds exist).
   - Total runs (over/under or expected total regression + binary OU if line available).
   - Optional stretch: starter strikeouts (expected Ks or over/under) if labels join cleanly from boxscores.
4. **Artifacts** — metrics JSON + predictions CSV under `data-core/notebooks/cache/` (and optional `models/` pickles if training succeeds). Scripts under `data-core/scripts/` matching existing CLI patterns; unit tests for leakage / slate-date / feature join where cheap.
5. **Dashboard/insight hook (optional, last packet)** — one Insights post or Performance MLB section that surfaces the new market metrics from artifacts (no hardcoded stats). Skip if time-boxed; doc alone is acceptable.

## Out of scope

- Committing or pushing.
- Changing `games` / `model_predictions` / `odds_snapshots` schemas (portfolio contract).
- Production workflow promotion / daily-refresh wiring (research package first; note follow-ups).
- Rebuilding MLB HR pipeline.
- Paid Odds API spend beyond existing secrets/patterns unless already cached.

## Constraints

- Leakage-aware: features only from info available before first pitch.
- Prefer extending existing MLB modules over new parallel stacks.
- No secrets in artifacts; no drive-by refactors outside MLB analytics paths.
- Parallelize Codex packets only when file-disjoint.
- Do not declare success without running train/test and writing real metrics.

## Notes for Fable

- Prefer Codex (`gpt-5.6-sol`) for implementation packets.
- Keep Fable for plan + review gates only.
- First explore: `mlb_features.py`, Stats API fields for weather, boxscore K/R columns, existing moneyline join in `backtest_mlb_winners.py`.
- Packetize as: (1) data/feature inventory + weather join, (2) moneyline upgrade train/test, (3) totals model, (4) run-line/spread model, (5) K/pitcher market if feasible, (6) synthesis doc + optional insight.
- If weather is unavailable cheaply, document blocker and still ship pitcher-enriched ML/totals from boxscore-derived env features.
- Target acceptance: improved ML metrics vs v3 on same 2026 test window **or** clear negative result with ablation; plus at least one new market (totals or run line) with train/test metrics committed to the analysis doc.
