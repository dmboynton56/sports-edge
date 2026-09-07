# Model and Data Quality Reset — NFL v2 Work Packet

Generated: 2026-09-06

## Scope and intended grain

This audit begins the team-model reset with NFL as the first challenger. The
expected curated feature grain is exactly one row per `(league, season,
game_id)`. A feature row must originate from a raw schedule row with the same
league and must contain only information available before kickoff.

The three contaminated BigQuery partitions were replaced only after the staged
validation described below passed. NFL v2 was subsequently activated as an
explicitly approved monitored rollout; this does not retroactively mark its
formal promotion gates as passed.

## Checks performed

- Joined BigQuery feature keys back to raw schedules to check league integrity.
- Profiled duplicate feature keys by league and season.
- Re-ran the NFL v1 model across all 285 completed 2025 games.
- Separated the direct classifier, spread-link probability, and final blended
  probability.
- Compared probability and margin outputs with constant baselines.
- Ran the complete Python unit-test suite after adding contract tests.

## Findings

### Critical: feature snapshots were not isolated by league

The snapshot query filtered only on `season`, then assigned the requested
league after feature construction. Live evidence before the fix:

| Declared partition | Feature keys | Correct raw league | Wrong raw league |
| --- | ---: | ---: | ---: |
| NFL 2026 | 85 | 31 | 54 MLB |
| NFL 2025 | 1,563 | 285 | 1,277 NBA |
| NBA 2025 | 1,563 | 1,277 | 285 NFL |

The NFL and NBA 2025 partitions each contained 84,946 physical rows for 1,563
keys. Nearly every key was duplicated, with as many as 55 rows per key.

Impact: the table is unsafe as a training source, feature-lineage source, or
historical explanation source until it is rebuilt. Daily prediction jobs build
features independently, so this finding does not by itself prove that every
published prediction used a cross-league row.

Remediation implemented: raw schedule and NFL play-by-play reads now filter by
league, builds fail when another league is present, and builds fail unless they
produce exactly one feature row per requested game.

The corrected partitions are now staged in
`sports_edge_curated.feature_snapshots_stage_20260906_v2`. The read-only
promotion audit passes with exact grain:

| Staged partition | Rows | Distinct games | Duplicates | Null keys |
| --- | ---: | ---: | ---: | ---: |
| NBA 2025 | 1,276 | 1,276 | 0 | 0 |
| NFL 2025 | 285 | 285 | 0 | 0 |
| NFL 2026 | 272 | 272 | 0 | 0 |

The staging pass also exposed a same-team/same-date join blowup in the NBA
All-Star mini-tournament. Rest features now join by `game_id` and team rather
than team and date, preserving the requested game grain. A failed scratch
stage with inferred numeric types was deleted; the retained v2 stage clones
the canonical schema before loading.

After review, the staged partitions were promoted transactionally. The prior
rows are retained in the dated backup table
`sports_edge_curated.feature_snapshots_backup_20260906T213707Z`. Post-promotion
verification reproduced the staged counts, found zero duplicate game keys,
zero missing schedule keys, and zero cross-league rows. The backup preserves
the original contaminated counts (84,946 rows each for NBA 2025 and NFL 2025,
plus 85 NFL 2026 rows).

### High: NFL v1 home inflation comes primarily from the spread path

The refreshed 2025 backtest produced:

| Output | Mean home probability | Home bias vs actual | Brier | AUC |
| --- | ---: | ---: | ---: | ---: |
| Direct classifier | 48.9% | -4.5 pp | 0.2689 | 0.6329 |
| Spread-to-probability link | 68.6% | +15.3 pp | 0.2927 | 0.5577 |
| Final published blend | 63.4% | +10.0 pp | 0.2668 | 0.5965 |

Actual home win rate was 53.3%. NFL v1 predicted the home team as the favorite
in 85.3% of games. Its average predicted home margin was 3.58 points versus an
actual average of 2.03, a +1.55 point home-margin bias.

The final Brier score was worse than both a 50/50 forecast (0.2500) and the
empirical home-rate baseline (0.2489). Confidence in NFL v1 probabilities is
therefore not justified. The spread link and legacy heuristic blend are the
largest source of the observed home inflation; this is not merely an intercept
problem in the direct classifier.

### High: legacy training and artifact lineage are not promotion-grade

- Inner spread OOF generation and probability calibration use shuffled folds
  despite time-ordered sports data.
- The stacking path contains in-sample and outcome-derived margin inputs in
  code labeled OOF.
- Link calibration is fitted on the full data frame rather than a calibration
  window isolated from the final test.
- Current NFL artifacts were serialized with scikit-learn 1.7.2 while the base
  local environment is pinned to 1.6.1.
- Training and inference maintain separate copies of feature-building logic.

Impact: saved training metrics cannot serve as sufficient promotion evidence,
even when later backtests are directionally useful.

## NFL v2 implementation and locked result

The new immutable export contains 1,693 completed games from the 2020–2025
seasons and exactly 3,386 team-game play-by-play aggregates. Both schedule and
PBP coverage are 100%; feature grain, league isolation, and history-date
leakage checks pass. Injuries, roster continuity, coaching, and weather were
excluded because their historical point-in-time coverage has not passed a
coverage gate.

The runner selected a small HistGradientBoosting probability model using the
2022 and 2023 rolling-origin folds and selected isotonic calibration using only
2024 predictions. The frozen run then evaluated the same 285 games from 2025.
Because 2025 results had already been inspected during the implementation and
v1 audit, this is recorded as consumed retrospective evidence rather than
claimed as a pristine untouched holdout:

| Output | Brier | Log loss | AUC | ECE | Home-probability bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| NFL v2 direct calibrated probability | 0.2322 | 0.7051 | 0.6608 | 0.0802 | -0.0 pp |
| NFL v1 published blend | 0.2668 | 0.7439 | 0.5965 | 0.1673 | +10.0 pp |
| Constant historical home rate | 0.2489 | 0.6910 | 0.5000 | 0.0050 | +0.5 pp |
| Aggregate closing market benchmark | 0.2104 | 0.6059 | 0.7245 | 0.0472 | +0.9 pp |

NFL v2 removes the large home inflation: its mean probability was 53.4%
against a 53.3% observed rate. The separate margin head reached 10.12 MAE with
-0.48 points of home-margin bias, improving on NFL v1's 11.15 MAE and +1.55
point bias. The margin-derived probability was better than the direct head on
Brier and log loss, but it is not ensemble-eligible because it did not improve
every required diagnostic, including absolute home bias.

The challenger did **not** pass the automatic shadow-promotion decision. It failed the locked
log-loss gate against the constant baselines, the ECE <= 0.05 gate, and the
Weeks 1–4 home-bias segment gate by 0.04 percentage points (-8.04 pp versus an
8.00 pp limit). Removing the explicit home-field term produced identical
rolling-fold Brier, so the registered home-field ablation also did not support
the hypothesis. This is a recorded negative result, not a cue to tune on 2025.
The next iteration must return to rolling folds through 2025; 2025 is now
consumed evidence. After review of these tradeoffs, the model was explicitly
activated as `nfl-v2-live-20260906` for a monitored 2026 rollout. That rollout
publishes the direct calibrated probability and independent margin head,
retains v1 as a rollback comparison, and does not use the legacy disagreement
blend. Weekly grading requires 64 games and four distinct weeks before a new
promotion assessment.

The nflverse closing fields had complete benchmark coverage but remain
benchmark-only: they are aggregate closing observations without a verified
prediction-time timestamp. The market-anchored track now requires separate
timestamped `market_home_prob`, `market_spread`, `market_as_of_ts`, and
`kickoff_ts` fields with at least 90% joint coverage. Those fields are absent,
so the residual-log-odds model failed closed and was not fit.

## Next implementation slice

1. Register the next single hypothesis and evaluate it on rolling-origin folds
   through 2025. Do not revisit the locked configuration using 2025 results.
2. Improve calibration robustness without selecting against 2025—for example,
   by testing sigmoid versus less brittle isotonic variants only on rolling OOF
   predictions.
3. Audit a timestamp-safe pre-kickoff odds source. Do not train the residual
   market track unless coverage reaches 90% on the intended rows.

## Verification

All 322 unit tests pass. The scoped live partitions now pass the same grain and
lineage checks as staging, and the replaced rows remain recoverable from the
dated backup.
