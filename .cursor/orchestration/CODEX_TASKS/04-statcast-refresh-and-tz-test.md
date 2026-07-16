# Task 04 — Statcast trailing-window refresh (P2) + timezone boundary test (P4)

Repo: `/home/dmboynton/projects/sports-edge`. Work only under `data-core/` and `.github/workflows/daily-refresh.yml`. Do NOT commit or push.
Independent of tasks 01/02/03/05 — parallel-safe.

## Goal

Two remaining hardening-plan items:

1. **P2:** The daily-refresh Statcast preload calls `fetch_statcast(..., refresh=False)` (`daily-refresh.yml` ~line 180), so the cache is validated but never updated — stale Savant data degrades the blend until the health gate trips. Refresh the trailing window in CI.
2. **P4:** Prediction-side `game_date` and the Denver-anchored Supabase serving views (`sql/015`, `sql/017`) must agree at the day boundary. Add a test that fails on skew; fix the write side if it disagrees.

## Context to read first

- `.github/workflows/daily-refresh.yml` — the "Statcast preload" inline-python step (~lines 160–190) and its env (`MLB_HR_STATCAST_DEADLINE_SECONDS`, `MLB_HR_STATCAST_MIN_*`)
- `data-core/src/models/mlb_hr_statcast_features.py` — `fetch_statcast` signature, `DEFAULT_STATCAST_CACHE`, chunk/caching + retry behavior
- `data-core/scripts/predict_mlb_home_runs.py` — where slate `game_date` is computed/written
- `data-core/scripts/plan_daily_refresh.py` — how "today" is anchored
- `data-core/sql/015_player_market_serving_tables.sql` and `sql/017_player_market_health_results.sql` — date filters/conversions (`America/Denver`)
- `data-core/tests/unit/test_mlb_hr_statcast_features.py`, `test_plan_daily_refresh.py`, `test_mlb_hr_probability_board.py` — existing test patterns

## Exact changes

### A. Trailing-window Statcast refresh

1. In the daily-refresh preload step, split the fetch: the season-to-`(today - N)` range keeps `refresh=False` (cache-only), and the trailing `N` days (default `N=10`, env `MLB_HR_STATCAST_REFRESH_DAYS`, declared in the workflow `env:` block alongside the other `MLB_HR_*` vars) use `refresh=True`. If `fetch_statcast` already supports date-range arguments, use them; if the refresh granularity is per-chunk, refresh only chunks overlapping the trailing window — read the implementation first and pick the minimal change. If a helper in `mlb_hr_statcast_features.py` makes this cleaner than inline workflow python (e.g. a `preload_statcast_cache(refresh_days=...)` function called from the workflow), prefer the helper + a unit test.
2. Keep the existing soft-fail contract: on any exception the step prints the existing `WARNING: Statcast preload failed...` message and continues — the downstream `validate_mlb_hr_statcast_health.py` gate stays the hard enforcement point. Keep the existing deadline behavior so a Savant outage cannot hang the job.

### B. Timezone boundary test

1. Trace where `game_date` for the HR slate is produced (predict script and/or plan script). Determine the convention (UTC vs `America/Denver`).
2. The serving views convert/anchor with `America/Denver` (see `game_prediction_results` in 017 and the `*_latest`/today filters in 015). If the write side uses UTC "today", fix it to compute the slate date in `America/Denver` (minimal, localized change; keep any CLI `--date` override behavior intact).
3. Add a unit test (e.g. in `tests/unit/test_plan_daily_refresh.py` or a new `test_slate_date_convention.py`) that freezes time at **2026-07-15 23:30 America/Denver == 2026-07-16 05:30 UTC** (use the repo's existing time-mocking approach; `freezegun` only if already in requirements — otherwise monkeypatch the clock/`datetime` seam) and asserts the computed slate date is `2026-07-15` (the Denver date). Add the mirror case 00:30 Denver.

## Constraints

- Discord notification steps and the `sports-edge-refresh` repository dispatch block in `daily-refresh.yml` must remain byte-identical.
- Do not change the health-gate thresholds or `validate_mlb_hr_statcast_health.py`.
- No new dependencies unless already in `data-core/requirements.txt`.
- Small diff: workflow step + (optionally) one helper in `mlb_hr_statcast_features.py` + prediction-date fix if needed + tests.
- No commits/pushes.

## Done definition

- Preload refreshes only the trailing window; older cache untouched; failure path still WARNs and continues.
- Boundary test exists and passes; if a skew was found and fixed, the diff to the write side is called out in your summary.

## Verification

```bash
cd data-core
PYTHONPATH=. pytest tests/unit/test_mlb_hr_statcast_features.py tests/unit/test_plan_daily_refresh.py -q
PYTHONPATH=. pytest tests/ -q
# Workflow YAML sanity:
python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('../.github/workflows/daily-refresh.yml').read_text()); print('yaml ok')"
```
