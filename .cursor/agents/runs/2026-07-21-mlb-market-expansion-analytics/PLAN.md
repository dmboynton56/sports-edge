# PLAN — MLB market expansion analytics

Run: `2026-07-21-mlb-market-expansion-analytics` · Repo: `/home/dmboynton/projects/sports-edge` · No commit/push anywhere in this run.

## Verified ground truth (from exploration, 2026-07-21)

1. **Local data caches are missing.** All `mlb_games_*.parquet` / `mlb_boxscores_*.parquet` / feature-store parquets are gitignored and not present in `data-core/notebooks/cache/` (only CSV/JSON artifacts survive). Everything downstream requires a fresh backfill via `data-core/scripts/backfill_mlb_raw.py` (MLB Stats API, free, ~12.9k games for 2021–2026).
2. **Weather needs no new API.** The MLB boxscore endpoint (`/api/v1/game/{pk}/boxscore`) `info` list carries `Weather` ("66 degrees, Partly Cloudy.") and `Wind` ("9 mph, L To R.") — verified live against game_pk 778545. The same payload already yields actual starter IDs and starter strikeouts (K-market labels). Venue elevation/roofType come from `/api/v1/venues?hydrate=location,fieldInfo` (one call). The live-feed endpoint has the same weather if ever needed.
3. **Current stack:** `src/models/mlb_winner_model.py` holds the v1 feature builder (season-scoped team/pitcher/venue rolling states) + `train_and_evaluate_mlb_winner` (LR / HGB / RF candidates, val-Brier selection, refit ≤ test season). `src/features/mlb_features.py` is a thin re-export. `scripts/backtest_mlb_winners.py` reads a feature-store parquet, trains, and computes flat ROI + edge buckets from a moneyline CSV. `scripts/train_mlb_winner_model.py` builds features from the games cache directly (no feature-store input yet) and saves the pickle.
4. **v3 baseline to beat (test = 2026 YTD through 2026-05-21, 673 games):** Brier 0.2478, log loss 0.6888, AUC 0.5431, acc 53.79%, ECE 0.0120; flat ROI −3.1% on free moneylines; only the 0–2% edge bucket was green. Home-rate baseline Brier 0.2497. Known missing features: starter line stats, bullpen, lineup, injury, weather, umpire, travel; pitcher/team states reset each season.
5. **Odds coverage:** `notebooks/cache/mlb_free_moneylines_2025_2026.csv` (game_pk, home/away_moneyline) → ML ROI measurable on 2026. **No totals / run-line / strikeout odds exist locally** → those markets report probability quality only; documented as such, no fake ROI.
6. **Env:** interpreter `data-core/.venv/bin/python` (pandas, sklearn, requests present). Run scripts from repo root with `PYTHONPATH=data-core`. `mlb_player_handedness_cache.json` from the HR pipeline is reusable for starter handedness. Existing test conventions: `data-core/tests/unit/test_mlb_*.py`, no network in unit tests.

## Strategy

Two waves of work after a parallel foundation pair:

- **Wave 0 (parallel):** `01` rebuilds raw data 2021–2026 and extends the boxscore fetcher with weather + team batting/pitching totals (long-running network job); `02` independently diagnoses v3 from the *existing* committed backtest CSV/JSON artifacts (calibration, error buckets, edge-bucket ROI autopsy). File-disjoint.
- **Gate:** `03` builds feature store v2 — cross-season pitcher rolling stats (K/9, ERA proxy, BB/9, rest), weather columns, run-environment columns, and all market labels (`total_runs`, `home_cover_15`, starter-K labels) — plus leakage guards and unit tests. Every model packet consumes only its parquet output.
- **Wave 1 (parallel, file-disjoint by construction — each owns its own new module/script/artifacts):** `04` moneyline v4 + ablation + ROI; `05` totals; `06` run line; `07` strikeouts (stretch, feasibility-gated).
- **Wave 2:** `08` synthesis doc (BRIEF deliverable #1) + optional dashboard insight.

### File ownership (conflict map)

| Packet | Owns (writes) |
|---|---|
| 01 | `src/data/mlb_boxscore_fetcher.py`, `scripts/backfill_mlb_raw.py`, `scripts/fetch_mlb_venue_meta.py` (new), `tests/unit/test_mlb_boxscore_parsing.py` (new), caches |
| 02 | run-dir `artifacts/diagnosis-v3.md`, `notebooks/cache/mlb_ml_diagnosis_2026_ytd.json` (new artifact only) |
| 03 | `src/features/mlb_market_features.py` (new), `src/features/mlb_features.py`, `src/models/mlb_winner_model.py` (exclusion list only), `scripts/build_mlb_feature_store.py`, `tests/unit/test_mlb_market_features.py` (new), feature-store parquet |
| 04 | `scripts/train_mlb_winner_model.py`, `scripts/ablate_mlb_winner_features.py` (new), v4 model/metrics/prediction artifacts |
| 05 | `src/models/mlb_totals_model.py` (new), `scripts/train_mlb_totals_model.py` (new), `tests/unit/test_mlb_totals_model.py` (new), totals artifacts |
| 06 | `src/models/mlb_runline_model.py` (new), `scripts/train_mlb_runline_model.py` (new), runline artifacts |
| 07 | `src/models/mlb_strikeouts_model.py` (new), `scripts/train_mlb_strikeouts_model.py` (new), K artifacts |
| 08 | `docs/analysis/mlb_market_expansion_2026-07.md` (new), optional `web/app/insights/` post, `docs/PERFORMANCE_HISTORY.md` append |

All paths relative to `data-core/` unless prefixed. `backtest_mlb_winners.py` is used read-only by 04 (its CLI already supports `--features-path`/`--odds-path`).

### Dependency graph

```
01 ──┐                ┌── 04 (moneyline v4) ──┐
     ├── 03 (store) ──┼── 05 (totals) ────────┼── 08 (synthesis)
02 ──┼────────────────┼── 06 (run line) ──────┤
     (02 feeds 08     └── 07 (Ks, stretch) ───┘
      directly)
```

### Split discipline (all model packets)

Time-based, same spirit as v3: candidate selection trains 2021–2024, validates 2025, tests 2026 YTD; selected model refit on ≤2025 before scoring 2026. Test rows are whatever completed 2026 games exist at backfill time (~mid-July, roughly 1,400+ games — a *larger* window than v3's 673). For the apples-to-apples ΔBrier/ΔAUC claim, packet 04's ablation includes a "v1 feature set" arm evaluated on the identical new test rows, so deltas never compare across different windows.

### Leakage rules (enforced in 03's tests, restated in every packet)

- Features use only information available before first pitch: prior-game rolling states, probable pitchers, venue statics, scheduled first-pitch time.
- Same-game boxscore quantities (actual starter line, bullpen usage, team batting totals, attendance, game duration) are **labels/audit columns only**, excluded from `default_feature_columns` by name-list or postgame-keyword guard.
- Weather is the observed game-time record, used as a proxy for a pregame forecast — a small optimistic bias for totals; must be caveated in the synthesis doc, never silently.
- Strikeout labels only count when actual starter == probable starter; mismatch rate reported.

### Risks / fallbacks

- **Backfill duration:** ~12.9k boxscore requests at ~0.2–0.4 s each ≈ 45–90 min. Run in background; packet 01 includes a 200-game smoke first. If MLB API throttles, raise sleep and resume (fetcher already skips cached game_pks).
- **Weather text variants:** domes report "0 mph, None" or roof-closed conditions; parser must degrade to nulls + `is_dome/roof_closed` flag, never crash (unit-tested on a fixture list).
- **No totals/spread odds:** already accepted — probability-quality metrics + explicit "ROI not measurable" note. If a totals line source materializes later, scripts accept an optional `--odds-path` stub.
- **Negative result on ML v4:** acceptable per BRIEF if the ablation cleanly attributes it; 08 must then say so plainly.
