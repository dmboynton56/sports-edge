# Odds Market Coverage Implementation Plan

Generated: 2026-08-08

## 1. Objective

Build a repeatable market platform in which Sports Edge can:

1. discover what The Odds API actually offers by sport, market, bookmaker, and
   lead time;
2. ingest current and historical prices without league-specific one-off code;
3. train and test outcome models against leakage-safe data;
4. backtest prices, edges, expected value, and closing-line value consistently;
5. promote only markets with adequate data, model evidence, and serving health;
6. publish new markets through a common website contract rather than a new
   bespoke pipeline for every page.

The goal is broad, dependable **coverage**, not the largest possible list of
market keys. A market is covered only when its odds, labels, features, model,
evaluation, daily scoring, grading, and website presentation all work.

## 0. MLB vertical slice shipped

The first implementation slice is now available through
`scripts/run_mlb_vertical_evaluation.py`. It refreshes the v2 feature-store
window, normalizes the free MLB moneyline exports, runs the held-out winner,
run-line, totals, pitcher-strikeout, and batter-home-run evaluations, and
writes:

- `notebooks/cache/mlb_vertical_evaluation.json` — quality checks, split
  metrics, odds coverage, production gates, and gaps;
- `notebooks/cache/mlb_vertical_edges.csv` — one common edge schema for
  market-price EV and statistical reference signals;
- `web/public/data/mlb_vertical_evaluation.json` — the artifact consumed by
  the MLB market page.

Run it from the repository root with:

```bash
PYTHONPATH=data-core python data-core/scripts/run_mlb_vertical_evaluation.py
```

The evaluator never passes odds into model features. When a price is absent,
the edge file records a statistical probability-vs-reference signal and leaves
market probability/EV null. The current free odds adapters are useful for
historical benchmarking, but public comparison/consensus rows do not identify
a single sportsbook; run-line, totals, and player-prop historical prices still
need a licensed or explicitly archived source before those markets can pass an
odds coverage gate.

## 2. Current Starting Point

The repository already contains more of the solution than the current website
exposes:

| Area | Current state |
| --- | --- |
| Current odds | The Odds API is used for NBA featured odds and MLB batter home-run props. |
| Historical odds | League-specific backfills exist for The Odds API, OddsPapi, nflverse, FantasyData, and free MLB comparison data. |
| Existing team models | NBA spread/winner, NFL spread/winner, MLB moneyline, PGA tournament outcomes, and CBB bracket probabilities. |
| Existing derivative research | MLB totals, home -1.5 run line, starter strikeouts, and batter home runs. |
| Evaluation | Model-evaluation and strategy-backtest tables exist, but the strategy runner is not yet a shared market framework. |
| Serving | BigQuery is the warehouse, Supabase is the public serving cache, and Next.js already has generic prediction/performance types. |
| Main limitation | Odds schemas and pipelines are still specialized by league/market. The original `odds_snapshots` contract only allows spread, moneyline, and total, while player/tournament markets have separate tables. |

The July MLB expansion is the clearest immediate opportunity: totals, run line,
and starter-strikeout models already have held-out research results, but totals,
run-line, and strikeout price history is missing. This makes odds acquisition
and standardized backtesting higher leverage than starting another model first.

## 3. What The Odds API Can Provide

Sources reviewed:

- API guide: <https://the-odds-api.com/liveapi/guides/v4/>
- Market catalog: <https://the-odds-api.com/sports-odds-data/betting-markets.html>
- Plans: <https://the-odds-api.com/#pricing>

### Endpoint and history constraints

| Capability | API behavior | Planning implication |
| --- | --- | --- |
| Sports and events | `/sports` and current `/events` do not consume quota. | Use them for scheduling and event identity discovery. |
| Event market discovery | `/events/{eventId}/markets` costs 1 credit and only reports recently seen markets. More appear as game time approaches. | Run a sampled audit at fixed lead times; do not treat the documentation list as guaranteed coverage. |
| Featured current odds | `h2h`, `spreads`, and `totals` are available through the sport odds endpoint. Cost is markets x regions. | This is the cheapest path for team markets. |
| Additional current odds | Props, alternate lines, team totals, and period markets use the event odds endpoint. Cost is unique returned markets x regions. | Fetch only enabled markets and only for relevant events. |
| Historical featured odds | Paid plans; snapshots begin 2020-06-06, at 10-minute intervals initially and 5-minute intervals since September 2022. Cost is 10 x markets x regions per snapshot request. | Backfill core team markets by timestamp/slate batches where possible. |
| Historical additional markets | Paid plans; props, alternate lines, and period markets are available after 2023-05-03. Cost is 10 x unique returned markets x regions per event request. | Player-prop history is much more credit intensive and must be prioritized. |
| Scores | Live/upcoming scores cost 1 credit; including up to three days of completed events costs 2. | Keep league-authoritative result feeds where they are stronger; use this as reconciliation/fallback. |
| Current plans observed | Free: 500 credits; $30: 20K; $59: 100K; $119: 5M; $249: 15M per month. | Use the free tier for the discovery spike. If a broad historical backfill passes the spike, a single 5M month is much better value than stretching a 20K/100K plan across event-level prop history. Recheck pricing before purchase. |

### Supported market families relevant to Sports Edge

The documented catalog includes:

- Featured: moneyline (`h2h`), spread, total, and selected outrights.
- Derivatives: alternate spreads/totals, team totals, and alternate team totals.
- Periods: quarter/half markets for basketball and football, period markets for
  hockey, and first 1/3/5/7 inning markets for baseball.
- NFL props: passing, rushing, receiving, touchdown, kicking, and selected
  defensive markets.
- NBA/NCAAB/WNBA props: points, rebounds, assists, threes, stocks, turnovers,
  combination props, first basket, double-double, and triple-double.
- MLB props: batter home runs/hits/total bases/RBIs/runs/walks/strikeouts and
  pitcher strikeouts/outs/hits/walks/earned runs/win.
- NHL props: points, assists, goals, shots on goal, blocked shots, goalie saves,
  and scorer markets.
- Golf: mainly tournament winner outrights for the four majors through this
  provider. Existing Sports Edge PGA placement work therefore needs another
  source for broad top-5/top-10/top-20 price coverage.

## 4. Recommended Market Roadmap

### Priority matrix

| Priority | Sport / market | Model and label readiness | Why it belongs here |
| --- | --- | --- | --- |
| P0 | MLB total | Research model exists; game total labels exist; historical prices missing. | Fastest path to turn completed research into a real market evaluation. |
| P0 | MLB run line | Home -1.5 classifier exists; margin labels exist; historical prices missing. | Same feature store and result feed as MLB moneyline/total. |
| P0 | MLB pitcher strikeouts | Research model and starter-K labels exist; historical prop prices missing. | Best first player prop because the model, player identity, and result labels already exist. |
| P0 | MLB moneyline and batter HR | Already modeled and partly served. | Migrate these into the common contract and use them to prove backward compatibility. |
| P1 | NBA moneyline, spread, total | Spread/winner pipeline exists; score history exists; total needs a dedicated target/head. | High user interest, mature repo pipeline, inexpensive featured odds. |
| P1 | NFL moneyline, spread, total | Team pipeline and play-by-play exist; historical featured odds are available. | Restore reliable odds coverage before the season and re-evaluate the weak current model. |
| P1 | NBA points, rebounds, assists | Odds are documented; box scores and player availability must become reproducible. | Popular markets, but player identity/minutes/injury work is a prerequisite. |
| P1 | NFL anytime TD plus rush/receiving yards | Existing TD scripts and nflverse data provide a base. | Natural extension of existing play-by-play work; re-train rather than revive unvalidated artifacts. |
| P2 | NHL moneyline, puck line, total | No model is wired today; odds are available. | Good next sport after the shared platform works; needs an NHL results/feature pipeline first. |
| P2 | NHL shots on goal and goalie saves | Documented prop coverage; player-game labels must be sourced. | Attractive user-facing markets, but not a platform proof point. |
| P2 | MLB hits/total bases and NBA combo props | Labels are obtainable, but lineup/minutes and correlated-market handling add complexity. | Add only after single-stat props are reliable. |
| Later | Period markets, alternates, first scorer/basket, long-tail props | Odds are available but samples, sparsity, grading, and calibration are harder. | These multiply complexity without first proving durable core coverage. |

### First release scope

The first implementation release should cover exactly these canonical markets:

```text
MLB: moneyline, run_line, total, batter_home_runs, pitcher_strikeouts
NBA: moneyline, spread, total
NFL: moneyline, spread, total
PGA: tournament_winner (retain existing placement probabilities as model-only where odds are absent)
```

Do not begin with every documented prop. The first release should prove that a
new market can be enabled through configuration and receive the same ingestion,
evaluation, grading, and serving behavior as an existing market.

## 5. Target Architecture

### 5.1 Canonical market registry

Add one data-core registry, proposed as `data-core/config/markets.yaml`, with:

```yaml
- sport: MLB
  provider_sport_key: baseball_mlb
  market_id: pitcher_strikeouts
  provider_market_key: pitcher_strikeouts
  subject_type: player
  target_type: count
  sides: [over, under]
  result_source: mlb_stats_api
  model_name: mlb_pitcher_strikeouts
  model_version: v1
  live_enabled: false
  backtest_enabled: true
  min_books: 2
  max_price_age_minutes: 20
```

This registry should drive ingestion, model dispatch, grading, serving, and the
web market registry. It replaces market-name conditionals scattered across
league scripts.

### 5.2 Normalized warehouse contracts

Keep provider payloads raw, then normalize them into general-purpose tables.

**`raw_odds_events`**

- provider, provider_event_id, sport_key, commence_time
- home/away participant names and canonical IDs
- first_seen_at, last_seen_at, raw_record

**`raw_odds_snapshots`**

- provider, provider_event_id, canonical_event_id
- sport, league, market_key, market_family, period
- subject_type, subject_provider_name, canonical_subject_id
- outcome name/side, line/point, American and decimal price
- bookmaker key/title, region
- provider_last_update, requested_snapshot_ts, effective_snapshot_ts,
  ingested_at, raw_record

**`market_results`**

- canonical_event_id, market_key, canonical_subject_id
- observed_value, winning_side, push/void status
- result source, source timestamp, graded_at

**`market_predictions`**

- canonical event/subject/market, line and side
- model probability or projected value/distribution
- model/version, prediction timestamp, feature snapshot ID, quality flags

**`market_edges`** (derived)

- selected book and best available price
- raw and no-vig implied probability
- model probability, probability edge, expected value, optional Kelly fraction
- line age, book count, consensus line, market hold, quality flags

Also add `odds_ingestion_runs`, `odds_quota_ledger`, and
`market_coverage_daily` audit tables. BigQuery remains the source of truth;
Supabase receives only current edges, latest predictions, evaluation summaries,
and coverage/health data needed by the website.

Partition odds by effective snapshot date and cluster by sport, market key,
provider event ID, and bookmaker. Never overwrite historical snapshots.

### 5.3 Identity mapping

Create explicit aliases for:

- provider event ID -> canonical game/event ID;
- provider team name -> canonical team ID;
- provider player description/name -> canonical player ID;
- provider market key -> canonical market ID.

Unmatched identities must be quarantined and visible in data quality. Do not
silently join props on player display name alone. The existing MLB normalized
name behavior can seed this layer but should not remain the final identity key.

### 5.4 Generic client and ingestion jobs

Replace specialized fetching logic with a tested `TheOddsApiClient` that owns:

- retry/backoff and 429 handling;
- response validation and raw payload capture;
- request-cost estimation before execution;
- `x-requests-used`, `x-requests-remaining`, and `x-requests-last` logging;
- current sports/events, event markets, sport odds, event odds, historical
  events, historical sport odds, and historical event odds;
- resume manifests and idempotent load keys.

League adapters should only supply team/player mapping and result semantics.

### 5.5 Snapshot policy and leakage protection

Store at least three semantic snapshots when available:

- **open/reference:** approximately 24 hours before start;
- **prediction-time:** the exact line available when the model prediction is
  generated (initial target: 60 minutes before start);
- **close:** last valid snapshot strictly before start (initial target: 5-10
  minutes before start).

Use prediction-time odds for simulated decisions and closing odds for CLV. Do
not backtest a 60-minute prediction against a price first seen five minutes
before the game. Any feature derived from odds must have
`effective_snapshot_ts <= prediction_ts`; initial models should treat the
market as benchmark/price rather than a training feature.

## 6. Data Acquisition Plan

### Phase A: Coverage discovery (free tier, 3-5 days)

Build `scripts/audit_odds_market_coverage.py` before buying historical data.

1. Fetch active events without quota cost.
2. Sample approximately 25-40 events per in-season priority sport.
3. Call event-market discovery at consistent lead-time bands: greater than 24
   hours, 6-24 hours, 1-6 hours, and under 1 hour where scheduling permits.
4. Record market availability by book, event, and lead-time band.
5. Fetch a small sample of actual odds to validate outcome shape, player-name
   mapping, lines, prices, and timestamps.
6. Produce `artifacts/odds-market-coverage-YYYY-MM-DD.{csv,json,md}`.

Pass criteria for a proposed market:

- at least 80% of eligible sampled events return the market near prediction
  time;
- median of at least two usable books per event;
- at least 95% of outcomes map to a canonical team/player in the sample;
- prices and lines pass pair/completeness/hold sanity checks;
- projected current-season credit cost fits the selected refresh schedule.

These are initial operational gates and can be tightened with evidence.

### Phase B: Paid historical probe (1-2 days)

Before a broad backfill, probe 25-50 completed events from each target window.
Request the exact snapshot policy and market keys planned for production.

Measure:

- event match rate;
- market/book/participant coverage;
- earliest usable date by market and book;
- line availability at 24h, 60m, and 5-10m;
- missing/void/duplicate/anomalous rows;
- actual credits per returned market;
- overlap with existing free archives.

Do not purchase/backfill on the assumption that every documented market exists
for every historical game. The provider explicitly limits market history to the
date each sport/book/market entered its collection.

### Phase C: Targeted historical backfill

Backfill in this order:

1. MLB 2025-2026 total, run line, and pitcher strikeouts.
2. MLB moneyline and batter HR into the common schema for comparison.
3. NBA 2023-2026 featured markets, then points/rebounds/assists only after the
   player-label pipeline is ready.
4. NFL 2023-2025 featured markets, then TD/rushing/receiving props.
5. NHL only after its result and feature pipeline exists.

Use sport-level historical snapshot requests for featured markets when one
request can cover the needed slate. Use historical event odds only for props,
periods, alternates, or precise per-event timestamps. Every backfill must have a
dry-run cost estimate, max-credit guard, checkpoint, resume mode, and audit
report.

### Budget recommendation

- The 500-credit free plan is enough for a disciplined discovery sample, not a
  full production polling schedule.
- A 20K plan is plausible for moderate current polling of a deliberately small
  registry.
- A broad player-prop backfill becomes expensive because each historical event
  market costs 10 credits. If the historical probe passes, use a one-month 5M
  plan for the bounded backfill, verify all artifacts, then downgrade to the
  current-data tier appropriate to the measured polling budget.
- Add a hard monthly quota budget per sport/market so a scheduler bug cannot
  consume the plan.

## 7. Shared Training and Backtesting Framework

### 7.1 Standard task types

Support four model contracts:

| Contract | Examples | Required output |
| --- | --- | --- |
| Binary probability | moneyline, home -1.5, anytime TD/HR | Calibrated probability for each outcome. |
| Continuous regression | game total, margin, player points/yards/Ks | Expected value plus residual/distribution metadata. |
| Count distribution | HR, strikeouts, goals, TDs | Probability mass or threshold probability for the offered line. |
| Tournament/ranking | PGA winner/placements | Mutually coherent field probabilities. |

A regression prediction alone is not enough for an over/under edge. The model
must turn the predicted distribution into `P(over line)`, `P(under line)`, and
`P(push)` for the book's actual line.

### 7.2 Reproducible experiment runner

Add a market-aware command such as:

```text
python -m src.experiments.run \
  --sport MLB \
  --market pitcher_strikeouts \
  --train-through 2024-12-31 \
  --validation-season 2025 \
  --test-season 2026 \
  --odds-snapshot prediction_time \
  --config configs/experiments/mlb_pitcher_strikeouts_v1.yaml
```

Each run should persist:

- immutable config and source-data fingerprint;
- train/validation/test windows and row counts;
- feature list and leakage audit;
- model artifact and calibration artifact;
- prediction-level output;
- model, calibration, line-coverage, CLV, and strategy metrics;
- comparison to naive and market-implied baselines;
- promotion recommendation and failed gates.

Use chronological or rolling-origin validation. Never random-split games or
player rows that can leak the same game/season context across folds.

### 7.3 Required metrics

**Probability quality:** Brier score, log loss, AUC/ranking metric, ECE,
reliability buckets, and sharpness/distribution.

**Point/count quality:** MAE, RMSE or Poisson deviance, residual calibration,
and threshold probability metrics at the lines books actually offer.

**Market quality:** event coverage, player coverage, book count, hold/no-vig
method, line staleness, and price distribution.

**Strategy quality:** bets/no-bets, win/loss/push/void, units, ROI, CLV, maximum
drawdown, performance by edge bucket, and bootstrap confidence intervals.

Always compare with:

- a constant/base-rate or rolling-stat baseline;
- the no-vig market consensus;
- the currently promoted model on identical rows;
- a simple decision rule fixed before viewing the test window.

### 7.4 Promotion gates

A market can move through `research -> shadow -> candidate -> live` only when:

1. data coverage and identity-match gates pass;
2. labels and all prediction-time features are leakage-safe;
3. a frozen held-out or rolling-forward test beats the appropriate naive
   baseline on the primary model metric;
4. calibration is acceptable or corrected on validation data only;
5. price-backed results exist on a useful sample and include uncertainty, not
   just headline ROI;
6. the daily job runs in shadow mode with complete predictions, prices, and
   grading for at least 2-4 weeks;
7. stale-line, missing-lineup/injury, duplicate, and quota alerts pass;
8. the website clearly labels model version, sample size, freshness, and
   research/candidate/live status.

Positive historical ROI is not sufficient by itself, and a negative honest
result is still useful information that should remain visible in research.

## 8. Website and Product Plan

Preserve the current design, but make the data behind it generic.

### Market board

Extend the existing market registry so each sport page can render common tabs:

- best available price and book;
- consensus line and no-vig implied probability;
- model projection/probability;
- edge, expected value, and optional conservative stake guidance;
- price/model timestamps and freshness state;
- book count, data quality flags, lineup/injury status;
- model status: research, shadow, candidate, or live.

Do not show an edge when the line is stale, the player is not mapped, a required
lineup/injury input is missing, or the model is not approved for that market.

### Results and evidence

For every market exposed on the board, add:

- recent graded picks including no-bet decisions;
- season and trailing-window model metrics;
- ROI and CLV by fixed edge threshold;
- calibration/reliability summary;
- coverage and freshness history;
- model-version changelog and known gaps.

This turns “more information” into evidence users can judge, rather than simply
increasing the number of picks.

## 9. Delivery Phases

### Milestone 0 - Coverage report and budget decision (3-5 days)

- Implement the quota-aware client skeleton and market audit command.
- Sample current MLB plus any active NFL/NBA/NHL events.
- Publish actual market/book/lead-time coverage and a historical probe budget.
- Decide whether to purchase a one-month backfill plan.

**Exit:** ranked market list is based on observed coverage, not docs alone.

### Milestone 1 - Common odds platform (1-2 weeks)

- Add canonical registry and generalized BigQuery schemas.
- Implement current and historical featured/event ingestion.
- Add identity aliases, quarantine tables, quota ledger, audits, and tests.
- Dual-write current MLB HR and NBA odds to old and new contracts temporarily.

**Exit:** MLB HR and NBA spread outputs match the existing pipelines while the
new system retains every timestamp/book/outcome.

### Milestone 2 - Historical acquisition and generic backtester (1-2 weeks)

- Run the bounded P0 backfill.
- Implement line selection, no-vig math, grading, CLV, and strategy registry.
- Write evaluation and strategy results to the existing evidence tables.
- Add run-level manifests, data fingerprints, and reproducible commands.

**Exit:** the same runner can evaluate at least one team market and one player
prop without market-specific backtest code.

### Milestone 3 - MLB five-market vertical (1-2 weeks)

- Re-evaluate moneyline, total, run line, batter HR, and pitcher Ks on matched
  historical prices.
- Fix pregame-weather timing for totals before promotion.
- Add daily shadow predictions/odds/grading for all five.
- Promote only the markets that pass; keep the others visibly research-grade.

**Exit:** Sports Edge has one fully closed-loop sport demonstrating the pattern.

### Milestone 4 - NBA and NFL core markets (2 weeks)

- Migrate spread/winner paths.
- Add dedicated totals heads and probability distributions.
- Backfill featured prices and re-run chronological evaluations.
- Begin shadow serving before the relevant regular-season windows.

**Exit:** moneyline/spread/total share the same contracts across three sports.

### Milestone 5 - Player props and NHL (later, market by market)

- NBA points/rebounds/assists after minutes, lineup, injury, and player-ID
  pipelines are reliable.
- NFL TD/rushing/receiving after re-training and calibration.
- NHL core markets, then shots/saves after an NHL data/model foundation exists.

**Exit:** a new configured market requires an adapter/model, not a parallel data
platform or bespoke website.

## 10. Initial Engineering Backlog

1. Add `data-core/config/markets.yaml` and validation tests.
2. Add `src/data/the_odds_api.py` as the generic client; keep existing fetchers
   as compatibility wrappers during migration.
3. Add `scripts/audit_odds_market_coverage.py` with CSV/JSON/Markdown output.
4. Add generalized odds/event/result/prediction/edge BigQuery migrations.
5. Add team/player/event alias tables and unmatched-identity reporting.
6. Add quota estimation, max-credit guards, response-header logging, and resume
   manifests to every API job.
7. Add raw-to-normalized contract tests using saved API fixtures for featured,
   period, player-prop, alternate, and missing-market responses.
8. Add shared no-vig, best-price, consensus-line, freshness, and grading logic.
9. Add the experiment/strategy registry and persist its output to the existing
   evaluation tables.
10. Migrate MLB HR and NBA spreads first as regression tests.
11. Backfill and evaluate MLB total, run line, and pitcher Ks.
12. Add generic Supabase current-market tables/views and Next.js adapters.
13. Add market health, quota, unmatched player, stale odds, and grading alerts.
14. Remove old specialized writes only after a full dual-write comparison and
    website cutover.

## 11. Definition of Done for “Coverage”

A sport/market is covered only when all of the following are true:

- registry entry and provider mapping exist;
- current acquisition meets measured event/book/lead-time targets;
- historical data and result labels support an honest evaluation;
- identity mapping and grading meet quality thresholds;
- model artifact, configuration, features, and test evidence are reproducible;
- prediction-time price selection cannot see future information;
- shadow/live jobs have quota, freshness, duplicate, and failure monitoring;
- current predictions and final results reach BigQuery, Supabase, and the site;
- users can see the line, model output, edge/EV if approved, timestamps, model
  version, sample size, performance, and limitations;
- a failed or unprofitable model remains research-grade instead of silently
  becoming a public edge.

That definition is what makes broad coverage sustainable rather than a growing
set of disconnected scripts.
