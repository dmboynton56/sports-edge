# NFL October readiness — September 29, 2026

Implementation and research readout, with pre-merge verification updated September 30. The rollout checks below distinguish code verification from live account and serving-data coverage.

## Credit math and October budget

The free plan supplies **500 monthly usage credits**, rather than 500 interchangeable HTTP calls. A bulk NFL request for `h2h,spreads,totals` covers the available slate for **3 credits** with one region. Up to ten explicitly requested bookmakers count as one region. `team_totals` requires the event endpoint and costs **1 credit per game** when returned; that market includes both teams. Empty responses cost zero, and `/sports` and `/events` are free. See the [official usage documentation](https://the-odds-api.com/liveapi/guides/v4/) and [market definitions](https://the-odds-api.com/sports-odds-data/betting-markets.html).

Your estimate of roughly 200 credits for four weeks would be correct for three separate credit-consuming markets across every game. Bulk featured markets make this much cheaper: 16 games need 3 slate credits, plus 16 event credits for team totals, or **19 credits for one weekly capture**. A second featured-market capture costs another 3, not another 48.

October 2026 has four Tuesdays and four Sundays. The October 1 reset catch-up adds one featured-market capture. Five forecast cycles overlap the month: September 29, October 6, 13, 20, and 27. Assuming the maximum 16 games per cycle and every team-total capture charged in October:

| Source | October planning bound | Enforced monthly cap |
| --- | ---: | ---: |
| NFL featured markets | 9 captures × 3 = 27 | Included below |
| NFL team totals | 5 cycles × 16 = 80 | Included below |
| **NFL combined** | **107** | **140** |
| NBA featured markets | At most 31 daily captures × 3 = 93 | 100 |
| MLB research fallback | At most 31 daily captures × 3 = 93 | 100 |
| CFB featured markets | 10 Thursday/Saturday captures × 3 = 30 | 40 |
| PGA outright markets | At most 20 credits | 20 |
| **Combined planning bound** | **343** | **400 across sources** |

The 343-credit bound leaves 157 credits unspent. A separate 50-credit account reserve is protected, and other sources cannot consume NFL's unused allocation. NFL's 140-credit cap also covers months with six overlapping forecast cycles. Actual October usage should be lower because of bye weeks, inactive slates, and PropLine supplying MLB research. Provider availability still determines whether every requested field exists.

Credits reset on the first of the month according to the [provider FAQ](https://the-odds-api.com/manage/faqs.html). Accounting uses UTC budget months; scheduling and game dates use America/Denver.

## Implemented cadence and protection

- NFL forecasts run Tuesday for the Tuesday–Monday cycle. This covers Thursday, Sunday, Monday, and any Saturday games actually scheduled. Serving queries stop at that Monday, and forecasts from earlier cycles are withheld.
- Featured NFL prices refresh Tuesday and Sunday, plus the first day of the month. The October 1 capture recovers from an exhausted September key. Sunday updates prices without rerunning the forecast.
- Successful team-total responses are reused for the cycle. Zero-credit empty responses may be checked again on a later scheduled day. Partial provider coverage remains visible as a gap.
- `force_nfl_refresh` permits an explicit recovery/data update within the current cycle. It does not bypass credit caps or add injury adjustments to NFL v2.
- Daily NFL source data, scores, and roster/availability context continue refreshing. NFL touchdown output remains model probabilities and fair odds; no scheduled NFL TD or MLB HR sportsbook-prop requests were added.
- CFB buys featured odds Thursday/Saturday. PGA responses are reused within each Denver day. NBA has one canonical odds-buying step; prediction generation explicitly skips its redundant odds fetch.
- The shared backend ledger commits a reservation before buying odds. A transaction advisory lock coordinates workflows; the free quota probe includes account usage outside this ledger. Repeated requests reuse the captured response, and ambiguous failures retain their reservation. Missing credentials, ledger, or quota headers prevent paid requests.
- Cached responses retain their original timestamps. Replaying a response cannot make old quotes look fresh; new captures of unchanged prices remain recorded. Missing books or paired prices leave model-only rows with no invented edge/EV.
- NFL publishing, odds fetching, and the weekly readiness audit can proceed after the observed downstream MLB research failure. The account audit runs after the scheduled odds fetches.

Relevant entry points: `src/data/odds_api_client.py`, `scripts/plan_daily_refresh.py`, `scripts/sync_odds.py`, `scripts/audit_odds_api_credits.py`, and `.github/workflows/daily-refresh.yml`.

## What the first three weeks say

The latest [NFL live performance report](https://github.com/dmboynton56/sports-edge/blob/main/data-core/notebooks/cache/nfl_v2_live_performance.json), generated September 29 at 19:36 UTC, covers **48 games across 3 weeks**:

| Measure | Result |
| --- | ---: |
| Winner accuracy | 23/48 = 47.9% |
| Probability Brier score | 0.2617 |
| Fixed 50% probability benchmark Brier score | 0.2500 |
| Probability AUC | 0.5018 |
| Mean absolute margin error | 11.22 points |
| Mean predicted home margin / actual home margin | +2.28 / +0.88 points |

These results do not demonstrate a useful probability advantage. The sample also has not reached the existing 64-game/four-week gate. Keep the model under observation; do not interpret displayed prices and forecasts as a profitable strategy.

Recent Actions logs also explain the board gaps: the September 27 run encountered `OUT_OF_USAGE_CREDITS`, and September 29 Daily runs failed in MLB research after generating NFL predictions. The old NFL generation window extended 14 days. The cadence, budget guard, and publishing changes address those specific operational problems.

## Current season versus last season

NFL v2's rolling team features use `(sum(current window) + 4 × prior) / (current observations + 4)`. After three valid current-season observations, that is approximately **43% current data and 57% prior**. The prior already shrinks last season toward league averages: with 17 prior-season observations, it is 68% previous-season data and 32% league average. Those components translate to about 43% current season, 39% last season, and 18% league average for that particular rolling feature.

There is a real rigidity to investigate: the three-game feature stays at 43% current/57% prior because its window never grows. Five-game and ten-game features eventually reach 56% and 71% current data. These are feature weights, not a decomposition of the final win probability.

I would retain last season after only three games, but compare a declining prior pseudo-count against the current fixed four. Run that comparison across historical seasons with chronological splits, inspect Weeks 1–4 and later weeks separately, then retrain and recalibrate the artifact if it improves held-out probability and margin metrics. The production weights were left unchanged; an ad hoc input-weight change would alter the distribution seen by the trained model.

## Injury and starting-QB priority

This is already material. Mayfield is expected to miss at least three weeks, and Tampa Bay plans to start Jalon Daniels in Week 4. [NFL's September 28 report](https://amp.nfl.com/news/buccaneers-qb-baker-mayfield-thumb-out-at-least-three-weeks-jalon-daniels)

The official Week 3 report listed Caleb Williams out with a hamstring injury; Philadelphia also listed Dallas Goedert and Marquise Brown out. These are prior-week statuses, not confirmed Week 4 designations. [Official NFL injury report, checked September 29](https://amp.nfl.com/injuries/)

NFL v2 explicitly excludes injury adjustments pending historical point-in-time coverage. Its expected QB is the most frequent starter in the last three starts, so one backup start can still leave the injured starter selected. A forced forecast rerun alone does not solve that. Prioritize verified pregame QB identity and timestamped availability before adding numerical injury effects; preserve the existing historical coverage gate. Daily roster context helps explain the forecast's limitations, but is not equivalent to modeling these absences.

## Additional fields worth pursuing

1. **Book disagreement and Tuesday-to-Sunday movement:** the same three requested books cost no extra credits. Their responses are retained in the ledger; extending the comparison readout can reuse them.
2. **Verified starting QB, availability, and game weather:** improve context and model inputs using existing/free sources. QB changes take priority over more paid market coverage.
3. **Selective first-half spread/total snapshots:** two additional event markets cost up to two credits per selected game. Only consider this after core coverage is healthy, within the NFL cap; first-half prices would remain research-only without a validated model.

Keep player props and historical Odds API pulls outside the scheduled plan.

## Rollout and verification

Local credentials are absent, so I could inspect Actions and published performance output but could not inspect the live Sports Edge database or current account headers. No paid API requests were made during this work.

After the required explicit pre-merge reviews and deployment, the workflow applies the extended `sql/023_odds_api_usage.sql` ledger. Run Daily Refresh once with `force_nfl_refresh=true` and `skip_notifications=true` to publish the current cycle if the previous failed run left it missing, then check the independent NFL readiness and credit summaries. October 1 odds catch-up is automatic. The web changes must be deployed for the one-week serving window and team-total rows to take effect.

Verification: 409 Python tests (408 full-suite plus the added quiet-recovery workflow check); 53 frontend tests; frontend lint and production build (including TypeScript checking); PostgreSQL verification of repeatable schema application, reservations, restricted access, current-week monitoring, and stale forecast exclusion. Two sklearn/SciPy optimization warnings were emitted locally; CI retains the repository's isolated model runtimes. Live account and serving-data coverage require the rollout checks above.

Both required reviews (`/thermo-nuclear-review` and `/thermo-nuclear-code-quality-review`) were explicitly completed September 30. Review fixes preserve CFB captured recommendations on days without an odds fetch, reject invalid team-total price pairs, and report touchdown feed freshness from eligible current-cycle rows. Quota reservation and response recording now have separate transaction responsibilities, with one connection cleanup path. The rebase preserves the landed CFB reliability fix and newer PGA/performance exports.

The first remote security check identified a newly published critical Next.js dependency advisory. Next.js and its matching ESLint configuration are updated to the patched 16.3.6 release. See the [maintainer advisory](https://github.com/vercel/next.js/security/advisories/GHSA-vcvr-r3jv-pc5j).

Rollout found all 16 Week 4 serving games with September 27 forecasts and no captured book lines. A quiet forced refresh regenerates this cycle and applies the credit ledger. Touchdown probabilities refresh alongside NFL odds captures, including October 1 catch-up: missing game totals block qualified TD rows, so those rows must be rebuilt once the real game-total input arrives. This makes no player-prop API request; team forecasts retain the Tuesday cadence.

The September 30 recovery run published 16 fresh team forecasts and 372 TD model rows, applied the ledger, and confirmed account usage of 500 with zero remaining credits. Paid NFL calls were blocked without buying a snapshot. Qualified TD rows remain withheld pending game totals; the availability audit also reports 40 stale rows and one eligible absence without an impact estimate. The overall Daily job failed its existing MLB research audit because no September 30 MLB slate was found; NFL publishing and its audit still completed independently. The workflow-level quiet toggle skipped final notification steps, but the prediction sync script sent three nested Discord alerts. The follow-up also clears the webhook environment variable during quiet runs to suppress those nested senders.
