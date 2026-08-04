# Codex implementation pass

You are **Codex** (`gpt-5.6-sol`), the implementer/tester. Read `.cursor/agents/codex-worker.md` and `.cursor/agents/PROTOCOL.md`.

- Repo: `/home/dmboynton/projects/sports-edge`
- Run: `2026-07-21-mlb-market-expansion-analytics`
- Run dir: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics`
- Task id: `03`
- Task packet: `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/tasks/03-feature-store-v2.md`

## Job

Execute the attached task packet exactly. Prefer concrete code + tests over prose.

When finished, write `/home/dmboynton/projects/sports-edge/.cursor/agents/runs/2026-07-21-mlb-market-expansion-analytics/handoffs/CODEX_DONE-03.md` with summary, files touched, verification commands/results, and residual risks.

Do not commit or push. Do not set `goal_done`. Do not expand scope beyond the packet.

## Attached task packet

# Packet 03 — MLB feature store v2: starter rolling stats, weather, run environment, market labels

**Actor:** Codex · **Depends on:** 01 (needs `mlb_games_2021_2026.parquet`, `mlb_boxscores_2021_2026.parquet`, `mlb_venue_meta.json`) · **Blocks:** 04, 05, 06, 07
**No commit/push.** Repo root: `/home/dmboynton/projects/sports-edge`. Python: `data-core/.venv/bin/python`, `PYTHONPATH=data-core`.

## Why

v3's ceiling is its feature set: season-reset team/pitcher states, no starter line stats, no weather, no run-environment view. This packet builds the single parquet every model packet (04–07) consumes.

## Files you own

- `data-core/src/features/mlb_market_features.py` (new — the v2 builder)
- `data-core/src/features/mlb_features.py` (add re-exports)
- `data-core/src/models/mlb_winner_model.py` (**only** the `default_feature_columns` exclusion sets — nothing else)
- `data-core/scripts/build_mlb_feature_store.py` (add v2 path/flags)
- `data-core/tests/unit/test_mlb_market_features.py` (new)
- Output: `data-core/notebooks/cache/mlb_feature_store_v2_2021_2026.parquet` + `_audit.json`. Touch nothing else.

## Design

`build_mlb_market_features(games, boxscores, venue_meta) -> DataFrame`, one row per completed game. Start from the v1 row (reuse/import the existing `_feature_row_for_game` state machinery from `src.models.mlb_winner_model` rather than duplicating it — chronological single pass, update states only *after* emitting the row). Add:

1. **Cross-season starter rolling features** (fixes the season-reset weakness). Maintain `pitcher_id → deque(last 10 prior starts)` built from **actual-starter** boxscore lines (`{prefix}_actual_starter_id`, `_starter_outs/_strikeouts/_walks/_earned_runs/_pitches`), keyed at feature time by the **probable** pitcher id (pregame-known). Per side emit: `k9_last5`, `bb9_last5`, `era_proxy_last5` (ER×27/outs), `pitches_last3_avg`, `outs_per_start_last5`, `starts_last365`, `days_since_last_start`, `career_starts_prior`; sensible priors when empty (league-ish defaults, e.g. K9 8.2, ERA 4.3) + `_has_history` flag; diffs home−away. Handedness: join throw hand from `notebooks/cache/mlb_player_handedness_cache.json` where present (`home/away_starter_throws_r`, nullable) — do not fetch anything new.
2. **Weather / park:** from same-game boxscore: `temp_f`, `wind_mph`, `wind_dir` one-hot/bucketed (`wind_out`, `wind_in`, `wind_cross` from Out To/In From/L To R|R To L), `is_dome_or_closed` (wind "None"/0 + roofType from venue meta), `is_day_game` (first-pitch hour < 17 local — from `game_datetime` if first_pitch text is messy), venue `elevation`. **Leakage caveat:** weather is the observed game-time record standing in for a pregame forecast — acceptable, must be flagged in audit + doc, and these columns must be listed in the audit as "observed-weather".
3. **Run environment (totals/run line support):** rolling per team (last 15 games, cross-season like pitchers): `runs_scored_pg_15`, `runs_allowed_pg_15`, `team_total_pg_15`; `combined_expected_total` = home_scored+away_allowed avg etc.; venue rolling `venue_total_runs_pg` already exists in v1 — keep. Team batting K-rate rolling (`team_bat_k_pg_15` from packet 01's `{prefix}_team_strikeouts`) for the K market.
4. **Labels (never features):** `home_win`, `run_diff`, `total_runs` (= home+away score), `home_cover_15` (= `run_diff >= 2`), `home_starter_ks_label`/`away_starter_ks_label` (actual-starter Ks), `home_probable_matches_actual`/`away_...` (bool).

**Exclusion guard** (edit in `mlb_winner_model.py`): add to `exclude` the new labels (`total_runs`, `home_cover_15`, `home_starter_ks_label`, `away_starter_ks_label`, `home_probable_matches_actual`, `away_probable_matches_actual`) and postgame passthroughs (`attendance`, `game_duration`, `first_pitch`, `weather_condition`, `wind_dir`, team-total raw columns). Keep the existing `postgame_keywords` guard; extend keywords with `_team_strikeouts`, `_team_walks`, `_team_hits`, `_team_home_runs`, `_team_pitching_strikeouts`.

**Build script:** extend `build_mlb_feature_store.py` with `--version v2` (or `--v2`) using defaults `--games-cache .../mlb_games_2021_2026.parquet --boxscores-cache .../mlb_boxscores_2021_2026.parquet --venue-meta .../mlb_venue_meta.json --output .../mlb_feature_store_v2_2021_2026.parquet`. Audit JSON: rows, seasons, null rates of key new columns, % rows with starter history both sides, % probable==actual, observed-weather caveat string.

## Tests (`test_mlb_market_features.py`, no network — build tiny synthetic games/boxscores frames)

1. **Pitcher leakage:** a pitcher with starts on D1, D2 — feature row for D2 uses only D1's line; row for D1 shows priors/no-history.
2. **Cross-season carryover:** start in Sept 2024 informs April 2025 features (unlike v1).
3. **Exclusion guard:** `default_feature_columns` on a v2 frame contains none of the label/postgame columns (assert explicitly per name).
4. **Weather derivation:** dome row → `is_dome_or_closed`, wind buckets correct for "Out To CF"/"In From LF"/"L To R"/"None".
5. **Total/cover labels:** `total_runs`, `home_cover_15` arithmetic (run_diff 2 → cover; 1 → not).

## Acceptance

- v2 parquet built for 2021–2026 with audit; row count ≈ v1 store (~12.4k after the 5-prior-game filter); starter-history coverage reported.
- All new tests green plus existing `test_mlb_features.py` (v1 path untouched): `PYTHONPATH=data-core data-core/.venv/bin/python -m pytest data-core/tests/unit -k mlb`.
- `train_and_evaluate_mlb_winner(v2_frame)` runs end-to-end (smoke, e.g. validation 2025 / test 2026) — proves 04 can consume it unchanged.
- STATUS.json updated + `handoffs/03-done.md` (include the exact feature-column list added, for 04's ablation groups).
