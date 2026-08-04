# Packet 01 — Rebuild MLB raw caches 2021–2026 with weather + team totals

**Actor:** Codex · **Depends on:** none · **Parallel with:** 02 · **Blocks:** 03
**No commit/push.** Repo root: `/home/dmboynton/projects/sports-edge`. Python: `data-core/.venv/bin/python`. Run scripts from repo root with `PYTHONPATH=data-core`.

## Why

All MLB parquet caches are gitignored and missing locally — nothing downstream can run until raw data is rebuilt. While rebuilding, extend the boxscore fetcher so one request per game also captures weather (totals models), team batting/pitching totals (K rates, run environment), and first-pitch/day-night — verified present in the payload: the boxscore `info` list contains `{"label": "Weather", "value": "66 degrees, Partly Cloudy."}` and `{"label": "Wind", "value": "9 mph, L To R."}` (checked live on game_pk 778545).

## Files you own

- `data-core/src/data/mlb_boxscore_fetcher.py` (extend)
- `data-core/scripts/backfill_mlb_raw.py` (defaults/flags only if needed — CLI already takes `--start-season/--end-season/--games-cache/--boxscores-cache/--fetch-boxscores/--limit-boxscores`)
- `data-core/scripts/fetch_mlb_venue_meta.py` (new)
- `data-core/tests/unit/test_mlb_boxscore_parsing.py` (new)
- Cache outputs listed below. Touch nothing else.

## Work

1. **Extend `mlb_boxscore_fetcher.py`.** Keep every existing output column (`{home,away}_actual_starter_id`, `_starter_strikeouts`, `_bullpen_*`, …) byte-compatible. Add per game:
   - From `info` list: `temp_f` (int from "NN degrees, …"), `weather_condition` (text after the comma), `wind_mph` (int), `wind_dir` (text after "mph, ", trailing period stripped — values like `L To R`, `R To L`, `Out To CF`, `In From LF`, `None`, `Varies`, `Calm`; these are already park-relative), `first_pitch` (text), `attendance` (int, commas stripped), `game_duration` (text). All tolerant: missing label → NA, never raise. Domes typically report `0 mph, None` — preserve as-is, downstream derives flags.
   - From `teams.{home,away}.teamStats.batting`: `{prefix}_team_strikeouts`, `_team_walks`, `_team_hits`, `_team_home_runs` (batting side). From `teamStats.pitching`: `{prefix}_team_pitching_strikeouts`. Use the same `_int_stat`-style tolerant coercion.
   - Put the `info`-parsing in small pure functions (e.g. `_parse_weather(info: list) -> dict`) so unit tests need no network.
2. **New `scripts/fetch_mlb_venue_meta.py`:** one GET to `https://statsapi.mlb.com/api/v1/venues?hydrate=location,fieldInfo`; write `data-core/notebooks/cache/mlb_venue_meta.json` mapping venue_id → `{name, elevation, roofType, azimuthAngle, city, state}`. (Verified fields exist: `location.elevation`, `fieldInfo.roofType`, `location.azimuthAngle`.)
3. **Unit tests** (`tests/unit/test_mlb_boxscore_parsing.py`, no network): weather/wind parsing on fixtures — normal ("66 degrees, Partly Cloudy." / "9 mph, L To R."), dome ("72 degrees, Roof Closed." / "0 mph, None."), missing labels, malformed values → NA not exception; attendance comma handling; team-stats coercion of `".---"`/None.
4. **Run the backfill** (background job; total ~45–90 min for ~12.9k boxscores at the default 0.05 s sleep — if you see 429s, bump `sleep_seconds` and rerun, it resumes from cached game_pks):
   ```bash
   cd /home/dmboynton/projects/sports-edge
   PYTHONPATH=data-core data-core/.venv/bin/python data-core/scripts/backfill_mlb_raw.py \
     --start-season 2021 --end-season 2026 --refresh-games \
     --games-cache data-core/notebooks/cache/mlb_games_2021_2026.parquet \
     --boxscores-cache data-core/notebooks/cache/mlb_boxscores_2021_2026.parquet \
     --fetch-boxscores --limit-boxscores 200   # smoke first; inspect, then rerun without the limit
   ```
   Smoke-check the 200 rows (null rates of new columns), then run the full fetch without `--limit-boxscores`.
5. **Write an audit JSON** `data-core/notebooks/cache/mlb_boxscores_2021_2026_audit.json`: row counts per season, null rate per new column, count of `boxscore_error` rows, distinct `wind_dir` values seen. A tiny inline script or a `--audit-output` flag on the backfill is fine.

## Acceptance

- `mlb_games_2021_2026.parquet` has ~12.9k+ completed regular-season games, 2021-04 → latest completed 2026 date (run date 2026-07-21).
- `mlb_boxscores_2021_2026.parquet` covers ≥ 99% of those game_pks; `temp_f`/`wind_mph` null rate < 5% on non-error rows; distinct `wind_dir` values enumerated in audit.
- `pytest data-core/tests/unit/test_mlb_boxscore_parsing.py` green; existing `test_mlb_features.py` still green (fetcher stays backward-compatible).
- Audit JSON written; completion note in `handoffs/01-done.md` with row counts + any anomalies; STATUS.json updated.
