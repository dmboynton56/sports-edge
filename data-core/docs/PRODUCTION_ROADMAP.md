# Sports Edge Production Roadmap

Generated: 2026-06-02

## Current Biggest Gap

The largest gap is not another standalone model feature. It is the closed-loop
production system around the models: reproducible backtests, calibration checks,
strategy ROI gates, injury-aware feature adjustments, and daily serving-data
validation.

Until those pieces are wired together, a live prediction can be directionally
interesting but still hard to trust, explain, or promote.

## North Star

By the end of this push, every promoted model version should have:

- A stored evaluation run with calibration metrics, sample size, train/test
  dates, artifact references, and approval status.
- Strategy-level backtest rows for the markets it serves.
- Daily serving validation for schedule count, prediction count, final scores,
  duplicate rows, and stale rows.
- Injury-aware inputs for NFL and NBA, with base and adjusted feature deltas
  preserved for audit.
- A portfolio/dashboard surface that can show model quality and data freshness,
  not just the latest picks.

## Week 1: Contracts And Evidence

### 1. Evaluation Storage

Status: in progress.

- Store model evaluation runs in Supabase.
- Store strategy backtest results linked to evaluation runs.
- Keep evaluation identities unique by league, model name, model version, and
  evaluation name.
- Keep strategy rows unique by evaluation run and strategy id.

Acceptance checks:

- Supabase contains current NBA, NFL, MLB, PGA, and CBB evaluation rows.
- Strategy rows exist for NBA, NFL, and MLB where odds evidence exists.
- `validate_supabase_sync.py --strict` passes.

### 2. MLB Serving Reliability

Status: in progress.

- Persist `game_date` separately from UTC timestamp for serving-day queries.
- Dedupe game rows by league, season, matchup, and serving time.
- Purge stale MLB predictions by serving date instead of UTC date.
- Keep the website query path aligned with Supabase `games_today_enriched`.

Acceptance checks:

- June 1, 2026 MLB serving date shows 9 games.
- June 2, 2026 MLB serving date shows 15 games.
- No duplicate recent game groups.
- Every MLB game in the current serving window has a prediction.

### 3. Injury Data Contract

Status: started.

- Add `player_availability_reports` for source-level game availability.
- Add `player_impact_estimates` for model-ready player deltas.
- Keep raw source payloads in `raw_record`.
- Store explicit `metric_name`, `player_value`, `replacement_value`,
  `usage_share`, and `team_delta`.
- Sync normalized CSV/JSON/JSONL rows with
  `scripts/sync_injury_reports_to_supabase.py`.

Acceptance checks:

- Injury tables exist with public read policies and stable unique indexes.
- Impact estimates can be joined by `game_id` or by `game_date + team`.
- Feature snapshots preserve injury deltas and injured-player counts.
- A normalized injury row can create an availability report, an impact estimate,
  or both.

## Week 2: Injury-Aware Prediction And Strategy Testing

### 4. NFL Injury Modeling

Status: started.

- Estimate player EPA impact from play-by-play.
- Use QB passer EPA/play as the first high-confidence path.
- Extend to rusher/receiver impacts when sample size is meaningful.
- Apply offensive player absences to existing `form_*_epa_off_*` features.
- Preserve `home_injury_epa_delta` and `away_injury_epa_delta`.
- Load Supabase `player_impact_estimates` into weekly NFL predictions with
  `scripts/predict_week.py --injury-aware`.

Acceptance checks:

- A known QB outage can generate a deterministic team EPA delta.
- Base and injury-adjusted predictions can be compared for the same game.
- Injury-adjusted EPA differentials are recomputed after feature adjustment.
- Weekly prediction runs can filter injury impacts by optional
  `--injury-model-version`.

### 5. NBA Injury Modeling

Status: started.

- Estimate player rating impact from normalized player impact inputs.
- Weight rating impact by minutes share.
- Apply missing-player deltas to existing `form_*_net_rating_*` features.
- Preserve `home_injury_net_rating_delta` and
  `away_injury_net_rating_delta`.

Acceptance checks:

- A player with +8 net rating, -2 replacement rating, and 32 percent minutes
  share produces a -3.2 team net-rating delta.
- Base and injury-adjusted NBA predictions can be compared for the same game.
- Net-rating differentials are recomputed after feature adjustment.

### 6. Strategy Backtesting

Status: needs next implementation pass.

- Define a strategy registry instead of one-off scripts.
- Backtest moneyline and spread strategies against consistent odds inputs.
- Track no-bet decisions, edge thresholds, confidence thresholds, units, ROI,
  hit rate, and calibration by bucket.
- Compare injury-aware predictions against base predictions.

Acceptance checks:

- Each strategy has a stable id and parameter set.
- Backtests write reproducible rows to `strategy_backtest_results`.
- Model promotion requires passing data-quality and strategy gates.

### 7. Dashboard Polish

Status: partially started.

- Show performance history and threshold tables from public artifacts.
- Add model evaluation and strategy backtest tables from Supabase.
- Add data freshness and validation status.
- Mark predictions as base or injury-adjusted when injury data exists.

Acceptance checks:

- Dashboard shows current model quality, strategy evidence, and data health.
- A user can tell when picks are stale, incomplete, or injury-adjusted.

## Production Gates

A model version should not be promoted unless all are true:

- Evaluation sample is large enough for the target market.
- Calibration metrics are stored and within the sport-specific tolerance.
- Strategy ROI is acceptable on a realistic odds source.
- Serving validation passes for the current day/window.
- Predictions are deduped and linked to final scores when games complete.
- Known major player absences are either modeled or explicitly marked missing.

## Immediate Next Steps

1. Apply the injury availability migration to Supabase.
2. Run `scripts/sync_injury_reports_to_supabase.py` against a normalized
   NFL/NBA availability file.
3. Run `scripts/predict_week.py --injury-aware` once `player_impact_estimates`
   has rows for the target NFL week.
4. Add an injury-aware backtest mode for NFL and NBA.
5. Extend the dashboard to surface evaluation rows, strategy rows, and injury
   flags from Supabase.
