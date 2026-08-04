# Task 03 complete

Canonical worker handoff: `CODEX_DONE-03.md`.

## Exact v2 modeling columns added

These are the 51 columns returned by `default_feature_columns(v2)` that are not returned for the v1 builder output. Packet 04 can use these groups directly for ablation.

### Starter rolling / handedness (30)

- `home_starter_k9_last5`
- `home_starter_bb9_last5`
- `home_starter_era_proxy_last5`
- `home_starter_pitches_last3_avg`
- `home_starter_outs_per_start_last5`
- `home_starter_starts_last365`
- `home_starter_days_since_last_start`
- `home_starter_career_starts_prior`
- `home_starter_has_history`
- `home_starter_throws_r`
- `away_starter_k9_last5`
- `away_starter_bb9_last5`
- `away_starter_era_proxy_last5`
- `away_starter_pitches_last3_avg`
- `away_starter_outs_per_start_last5`
- `away_starter_starts_last365`
- `away_starter_days_since_last_start`
- `away_starter_career_starts_prior`
- `away_starter_has_history`
- `away_starter_throws_r`
- `starter_k9_last5_diff`
- `starter_bb9_last5_diff`
- `starter_era_proxy_last5_diff`
- `starter_pitches_last3_avg_diff`
- `starter_outs_per_start_last5_diff`
- `starter_starts_last365_diff`
- `starter_days_since_last_start_diff`
- `starter_career_starts_prior_diff`
- `starter_has_history_diff`
- `starter_throws_r_diff`

### Weather / park (8)

- `temp_f`
- `wind_mph`
- `wind_out`
- `wind_in`
- `wind_cross`
- `is_dome_or_closed`
- `is_day_game`
- `elevation`

`temp_f`, wind, and dome/closed fields use observed game-time records as pregame forecast proxies; the audit flags this caveat.

### Cross-season team run environment (13)

- `home_runs_scored_pg_15`
- `home_runs_allowed_pg_15`
- `home_team_total_pg_15`
- `home_team_bat_k_pg_15`
- `away_runs_scored_pg_15`
- `away_runs_allowed_pg_15`
- `away_team_total_pg_15`
- `away_team_bat_k_pg_15`
- `runs_scored_pg_15_diff`
- `runs_allowed_pg_15_diff`
- `team_total_pg_15_diff`
- `team_bat_k_pg_15_diff`
- `combined_expected_total`

The six new non-feature labels are `total_runs`, `home_cover_15`, `home_starter_ks_label`, `away_starter_ks_label`, `home_probable_matches_actual`, and `away_probable_matches_actual`.
