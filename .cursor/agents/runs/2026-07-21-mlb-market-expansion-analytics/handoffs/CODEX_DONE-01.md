# Codex done — task 01

## Summary
- Extended MLB boxscore summaries with tolerant weather, wind, first-pitch, attendance, duration, and team batting/pitching totals while preserving existing starter and bullpen fields.
- Added a one-request MLB venue metadata fetcher.
- Rebuilt completed regular-season game and boxscore caches for 2021 through 2026 YTD: 13,653 games and 13,653 boxscores, with 100% game-pk coverage and zero final error rows.
- Wrote the boxscore audit JSON. Temperature null rate is 0%; wind speed/direction null rate is 0.0073% on non-error rows.

## Files touched
- `data-core/src/data/mlb_boxscore_fetcher.py`
- `data-core/scripts/backfill_mlb_raw.py`
- `data-core/scripts/fetch_mlb_venue_meta.py`
- `data-core/tests/unit/test_mlb_boxscore_parsing.py`
- `data-core/notebooks/cache/mlb_games_2021_2026.parquet`
- `data-core/notebooks/cache/mlb_boxscores_2021_2026.parquet`
- `data-core/notebooks/cache/mlb_boxscores_2021_2026_audit.json`
- `data-core/notebooks/cache/mlb_venue_meta.json`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/STATUS.json`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-01.md`
- `.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/01-done.md`

## Verification
- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest data-core/tests/unit/test_mlb_boxscore_parsing.py -q` — 11 passed.
- `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest data-core/tests/unit/test_mlb_features.py -q` — 1 passed.
- Live parse of game 778545 — temperature 66, condition `Partly Cloudy.`, wind 9 / `L To R`, attendance 45,568, duration `3:05.`.
- 200-row smoke backfill — zero errors; temperature/wind null rates 0%.
- Full backfill and audit — 13,653 unique games, 13,653 unique boxscores, coverage 1.0, zero error rows. Per-season rows: 2021 2,429; 2022 2,430; 2023 2,430; 2024 2,429; 2025 2,430; 2026 YTD 1,505. Dates span 2021-04-01 through 2026-07-20.
- Venue fetch — 1,652 venue IDs written with the required keys; MLB supplied hydrated elevation/azimuth values for 38 venue records.

## Residual risks / follow-ups
- MLB omits wind for game 662189 and omits first pitch/duration for games 632924 and 663023; parsers correctly retain nulls. Attendance is absent for 0.72% of non-error rows.
- The venues endpoint includes historical venues and leaves elevation/azimuth null for most of them; all records are retained exactly so downstream joins can select relevant venue IDs.
- `ruff` is not installed in `data-core/.venv`; syntax compilation and the requested tests passed.
