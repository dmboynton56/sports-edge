# NFL Week 1 Prediction Readiness Plan

Generated: 2026-06-08

The 2026 NFL regular season is currently scheduled to open on Wednesday,
September 9, 2026. The readiness checkpoint for Sports Edge Week 1 predictions
is therefore Wednesday, September 2, 2026, one week before kickoff.

Schedule source: [NFL Football Operations, 2026 schedule announcement](https://operations.nfl.com/updates/the-game/2026-nfl-schedule-announced/index.html).

## Objective

Have NFL Week 1 predictions ready, explainable, and safe to publish once the
regular-season schedule, rosters, injury context, market lines, and weather
signals are stable enough to trust.

Week 1 should be treated as a prior-heavy forecast. Current-season form will be
sparse or unavailable, so the system must lean on prior-season team strength,
EPA priors, market information, roster context, and injury-adjusted deltas.

## Target Timeline

### T-7 Days: Wednesday, September 2, 2026

- Confirm the NFL season gate turns on for the September 9 opener.
- Run the daily refresh workflow in dry-run mode.
- Run a forced NFL feature rebuild with enough historical context.
- Confirm Week 1 games are present in BigQuery raw schedules.
- Confirm the prediction window includes every Week 1 game from Wednesday
  through Monday.
- Check that NFL feature snapshots load prior-season context instead of relying
  only on empty current-season history.

Acceptance checks:

- `plan_daily_refresh.py` returns `run_nfl=true`.
- `sports_edge_raw.raw_schedules` contains all Week 1 NFL games.
- `sports_edge_curated.feature_snapshots` has one latest NFL row per Week 1
  game.
- `sports_edge_curated.model_predictions` can be populated in dry-run without
  missing game IDs.

### T-5 Days: Friday, September 4, 2026

- Load or verify opening market spreads for all Week 1 games.
- Confirm team abbreviations and matchup keys match across BigQuery, Supabase,
  and odds sources.
- Check for stale prior-season odds or predictions that could be accidentally
  served.
- Review the model's Week 1 feature values for obviously bad defaults.

Acceptance checks:

- Supabase serving window has all expected NFL games.
- Current book spreads are attached where available.
- No duplicate NFL game rows exist for Week 1 matchups.
- Predictions are linked to the correct `game_id`.

### T-3 Days: Sunday, September 6, 2026

- Review 53-man rosters, starting QBs, major depth chart changes, and coaching
  changes.
- Load or manually prepare high-confidence injury/player-impact adjustments.
- Confirm known suspensions and expected inactive players.
- Run base predictions and injury-adjusted predictions side by side.
- Flag games where injury uncertainty materially changes the edge.

Acceptance checks:

- Player impact rows exist for high-confidence QB and major offensive/defensive
  absences.
- Feature snapshots preserve base and injury-adjusted deltas.
- Games with uncertain key-player status are labeled as lower-confidence or
  held back.

### T-1 Day: Tuesday, September 8, 2026

- Run a normal daily refresh.
- Run strict Supabase validation.
- Re-sync odds and repair missing book spreads.
- Check weather for outdoor games.
- Review prediction explanations and top feature drivers.
- Prepare the Week 1 release note.

Acceptance checks:

- `validate_supabase_sync.py --strict` passes.
- Every published Week 1 game has one latest prediction.
- Every published prediction has current `prediction_ts`, `model_version`, and
  market context where available.
- No stale NFL predictions from the prior season appear in the serving window.

### Game Day: Wednesday, September 9, 2026

- Refresh odds and injury context before the opener.
- Confirm the opener prediction remains linked to the correct game row.
- Mark any late-inactive uncertainty in the release note.
- Publish only games that pass data-quality checks.

## Data Inputs Needed

### Required

- Official Week 1 schedule.
- BigQuery raw NFL schedule rows for the 2026 season.
- Prior-season NFL schedules and play-by-play.
- Current model artifacts for NFL spread and win probability.
- Current market spreads from the odds sync path.
- Supabase game and prediction sync credentials.

### Strongly Preferred

- Confirmed starting QBs.
- 53-man roster and depth chart changes.
- Injury reports and player availability.
- Player impact estimates for high-leverage absences.
- Weather and venue data for outdoor games.
- Closing/opening line comparison for sanity checks.

### Nice To Have

- Power-rating priors from the prior season.
- Coaching and coordinator-change indicators.
- Offseason roster movement flags.
- Team-level continuity metrics.
- Market totals, not only spreads.

## Implementation Follow-Ups Before September

1. Add NFL history support to incremental feature snapshots.

   The prediction path already loads `season - 3` through the current season.
   The feature snapshot builder should support the same behavior with something
   like `--history-seasons-back 3`, so Week 1 snapshots are not built from
   current-season rows only.

2. Make the NFL prediction window explicit.

   Add `--start-date` and `--lookahead-days` support to the workflow call so
   Week 1 can be scored from September 2 instead of waiting for the current
   Tuesday/Wednesday NFL-week-start heuristic.

3. Add a Week 1 readiness script.

   Create a small audit command that checks expected games, predictions, odds,
   duplicate rows, stale rows, and missing injury-impact rows for NFL Week 1.

4. Confirm injury-aware NFL prediction wiring.

   Ensure player impact estimates are loaded into the scheduled NFL prediction
   path, not only ad hoc scripts.

5. Confirm market spread coverage.

   Validate that `sync_odds.py --league NFL` and
   `populate_existing_book_odds.py --league NFL` attach spreads to Week 1 games
   in Supabase and BigQuery.

## Suggested Commands

Dry-run the gates:

```bash
cd data-core
python scripts/plan_daily_refresh.py \
  --date 2026-09-02 \
  --lookback-days 1 \
  --lookahead-days 10
```

Run an NFL feature snapshot rebuild once historical context is wired:

```bash
cd data-core
python scripts/build_feature_snapshots.py \
  --project "$PROJECT_ID" \
  --league NFL \
  --seasons 2026 \
  --date 2026-09-02 \
  --lookahead-days 12
```

Generate Week 1 predictions:

```bash
cd data-core
python -m src.pipeline.refresh_nfl \
  --project "$PROJECT_ID" \
  --model-version v1 \
  --start-date 2026-09-09 \
  --window-days 5 \
  --season 2026
```

Sync and validate serving data:

```bash
cd data-core
python scripts/sync_bq_to_supabase.py \
  --project "$PROJECT_ID" \
  --league NFL \
  --start-date 2026-09-09 \
  --window-days 5 \
  --append

python scripts/validate_supabase_sync.py --strict
```

## Go/No-Go Criteria

Publish Week 1 predictions only if all are true:

- Every published game has one current prediction row.
- Every published game has the correct matchup, date, and season.
- No stale prior-season NFL predictions appear in the serving window.
- Market spreads are present or the release clearly labels them unavailable.
- High-impact injuries are represented or explicitly called out as unresolved.
- Feature values pass a manual sanity check for Week 1 sparsity.
- Supabase strict validation passes.

Hold back or label predictions as preliminary if:

- Starting QB status is unresolved.
- A major injury materially changes the prediction.
- Odds are missing for a game where the pick depends on spread edge.
- The prediction row cannot be tied cleanly to the official schedule game ID.
- Validation reports duplicate, orphaned, or stale serving rows.
