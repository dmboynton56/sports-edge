# Sports-Edge: Data Exploration & Modeling Plan

> Goal: Build a lightweight, defensible model to compute **our spread** and **home win probability** for NFL/NBA games, refresh it intraday, and benchmark it against sportsbook lines.

---

## 1) Data Sources (MVP)
- **NFL**: `nflreadpy` (or `nfl_data_py` if preferred). Season schedule, team-week aggregates, rolling EPA/success rate where available.
- **NBA**: `nba_api` (team game logs, advanced team box scores).
- **Odds/Lines**: The Odds API (spreads, totals, moneylines). Alternatives: SportsDataIO, Sportradar.
- **Reference Strength** (optional): Elo/SRS from Basketball-Reference (NBA), nflfastR-style EPA-derived strength (NFL).

> We’ll cache raw pulls to `/data/raw/{league}/YYYY-MM-DD/` and curated features to `/data/curated/{league}/YYYY-MM-DD/` for reproducibility.

---

## 2) Notebook-First EDA (Jupyter)
Create one notebook per league under `/notebooks`:
- `nfl_eda.ipynb`
- `nba_eda.ipynb`

### What we’ll inspect
- Distributions: margins, totals, pace (NBA), EPA (NFL).
- **Rest & schedule** features: back-to-backs, 3-in-4, travel distance (Haversine), days rest.
- **Form** windows: rolling (last 3 / 5 / 10) team net rating (NBA) or EPA/success rate (NFL).
- **Correlations** with outcome margin and win%.
- **Leak checks**: ensure no post-game info leaks into pre-game features.

### Quick visuals
- Feature importance via permutation on a simple model.
- Calibration curves for win%.
- Residuals (model margin − closing spread).

---

## 3) Feature Contract (v1)
We’ll lock a minimal, pre-game **feature contract** used by both training & daily scoring:

```
game_id, league, season, game_time_utc,
home_team, away_team,
rest_home, rest_away, b2b_home, b2b_away,
travel_km_home_last, travel_km_away_last,
form_home_net_rating_3, form_away_net_rating_3,     # NBA
form_home_epa_off_3, form_away_epa_off_3,           # NFL
form_home_epa_def_3, form_away_epa_def_3,           # NFL
opp_strength_home_season, opp_strength_away_season
```

Keep this **stable**; add columns only with migration. Store as a JSON in `features` table for flexibility.

---

## 4) Modeling Approach (MVP → v2)
**Targets**
- `home_win ~ logistic(features)` → home win probability (0–1)
- `home_margin ~ regression(features)` → our point spread (home favored = positive)

**Models**
- Start simple: LogisticRegression / Ridge / LightGBM (or XGBoost) for margin.
- v2: Add hierarchical shrinkage or team random effects to stabilize small samples.

**Linking spread ↔ win%**
- Convert `home_margin` to win% using a logistic mapping calibrated on historical game margins.
- Keep the link function under version control (documented).

**Training windows**
- Rolling 2–3 seasons, with early-season priors (blend season priors with latest form).

**Validation**
- Time-based split (train until date T, test on T+).
- Metrics: Brier score & log loss (win%), MAE vs spread (margin), **BTCL%** (Beat The Closing Line).

---

## 5) Data Model in Supabase
Tables (MVP):
- `games(id, league, season, game_time_utc, home_team, away_team)`
- `odds_snapshots(id, game_id, book, market, line, price, snapshot_ts)`
- `model_predictions(id, game_id, model_name, model_version, my_spread, my_home_win_prob, asof_ts)`
- (optional) `features(id, game_id, feature_json, asof_ts)`
- (optional) `model_runs(id, started_at, finished_at, league, rows_written, success, error_text)`

Views:
- `games_today_enriched` → latest odds + latest model per game for same-day display.

---

## 6) Refresh Cadence
- 08:00 America/Denver pre-load (schedule & priors).
- Every 15 minutes until tipoff/kickoff: refresh odds + recompute predictions.
- Final snapshot at event start.

---

## 7) Risks & Mitigations
- **Upstream rate limits** → precompute via cron, cache to DB, UI only SELECTs.
- **Schema drift** → feature contract + SQL migrations.
- **Early-season volatility** → blend team priors / shrinkage.
- **Injuries/uncertainty** → add an optional “what-if” endpoint in v2.
