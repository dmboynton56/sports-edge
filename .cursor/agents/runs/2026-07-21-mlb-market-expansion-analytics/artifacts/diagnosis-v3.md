# MLB moneyline v3 underperformance diagnosis

## Technical summary

- **v3 adds little reliable discrimination beyond the free line.** On the same 673 games, the model's AUC was 0.5431 versus 0.5501 for the market. When their favored sides differed, v3 was correct 46.9% of the time versus 53.1% for the market; the same split persisted in 2025 (46.0% versus 54.0%).
- **The −3.1% flat ROI is concentrated in April, favorite picks, and two edge regions.** April lost -22.47 units, favorites lost -24.41, and the 2–4% plus 6%+ buckets lost -24.22 units. May and underdog picks were slightly positive.
- **Calibration is superficially good because probabilities are compressed.** ECE was 0.0120, but 85.0% of predictions sat between 0.45 and 0.60; p5/p50/p95 were 0.456/0.528/0.631. Low ECE therefore does not imply useful ranking or betting edges.
- **The green 0–2% edge bucket is not stable evidence.** It returned 3.7% on n=168 in 2026, with a rough 95% Wilson interval of 51.4%–66.1% for its 58.9% win rate. In 2025 the same bucket returned -6.1% on n=616 (win-rate interval 52.1%–59.9%).

## Reliability is good in aggregate but based on a narrow prediction range

The table uses ten fixed-width bins of the predicted home-win probability. Empty bins are retained so the bin counts reconcile exactly to all 673 games. ECE is the row-count-weighted absolute gap between mean prediction and empirical home-win rate.

| Home-win probability | n | Mean prediction | Observed home wins | Absolute gap |
| --- | --- | --- | --- | --- |
| 0.0-0.1 | 0 | — | — | — |
| 0.1-0.2 | 0 | — | — | — |
| 0.2-0.3 | 0 | — | — | — |
| 0.3-0.4 | 2 | 39.6% | 0.0% | 39.6% |
| 0.4-0.5 | 183 | 47.2% | 47.5% | 0.3% |
| 0.5-0.6 | 412 | 54.1% | 52.7% | 1.5% |
| 0.6-0.7 | 76 | 62.8% | 61.8% | 1.0% |
| 0.7-0.8 | 0 | — | — | — |
| 0.8-0.9 | 0 | — | — | — |
| 0.9-1.0 | 0 | — | — | — |

2026 YTD ECE is **0.0120**. The apparent calibration is compatible with weak discrimination: only 15.0% of predictions fall outside 0.45–0.60. As a stability check, the 2025 p5/p50/p95 were 0.418/0.522/0.652, with 69.3% inside 0.45–0.60 and ECE 0.0255.

## The free market ranks winners better when model and market disagree

Market probabilities are the no-vig home probabilities already stored in the committed free-line cache. Accuracy uses a 0.50 home/away threshold; Brier and AUC use the probabilities directly. The 2025 comparison is restricted to its 2309 non-null odds rows.

| Window | n | Corr. | Mean abs. gap | Model acc. | Model Brier | Model AUC | Market acc. | Market Brier | Market AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 YTD | 673 | 0.490 | 5.2% | 53.8% | 0.2478 | 0.5431 | 55.6% | 0.2485 | 0.5501 |
| 2025 | 2309 | 0.719 | 4.9% | 54.0% | 0.2460 | 0.5554 | 55.9% | 0.2421 | 0.5826 |

In 2026 the model and market chose the same side 70.9% of the time. On the 196 disagreements, the market's accuracy advantage was 6.1%; in 2025 it was 8.0% across 550 disagreements. v3's 2026 Brier was marginally better than the market by 0.0007, but this coexists with worse accuracy and AUC and is consistent with conservative probability compression rather than superior ranking.

## The ROI loss sits mainly in April and favorite picks

Flat staking risks one unit per game and records American-odds profit on wins and −1 on losses. Re-derived 2026 profit is **-21.09 units / 673 bets = -3.1% ROI**, differing from the committed `flat_roi` by 6.94e-18.

### Pick side

| Pick side | n | Wins | Win rate | Units | ROI |
| --- | --- | --- | --- | --- | --- |
| home | 488 | 264 | 54.1% | -16.96 | -3.5% |
| away | 185 | 98 | 53.0% | -4.14 | -2.2% |

### Favorite versus underdog

Favorite/underdog status is based on the selected side's price: negative is favorite and positive is underdog.

| Selected price class | n | Wins | Win rate | Units | ROI |
| --- | --- | --- | --- | --- | --- |
| favorite | 536 | 297 | 55.4% | -24.41 | -4.6% |
| underdog | 137 | 65 | 47.4% | +3.32 | 2.4% |

### Month

| Month | n | Wins | Win rate | Units | ROI |
| --- | --- | --- | --- | --- | --- |
| 2026-04 | 391 | 203 | 51.9% | -22.47 | -5.7% |
| 2026-05 | 282 | 159 | 56.4% | +1.38 | 0.5% |

April accounts for more than the full net loss because May offset it slightly. Model discrimination also changed sharply by month:

| Month | n | Accuracy | Brier | AUC |
| --- | --- | --- | --- | --- |
| 2026-04 | 391 | 51.9% | 0.2502 | 0.5099 |
| 2026-05 | 282 | 56.4% | 0.2446 | 0.5879 |

The April-to-May reversal supports an early-season-state hypothesis, but two months are insufficient to identify a causal season-reset effect.

### Existing edge buckets

| Absolute edge | n | Wins | Win rate | Units | ROI |
| --- | --- | --- | --- | --- | --- |
| 0-2% | 168 | 99 | 58.9% | +6.20 | 3.7% |
| 2-4% | 159 | 82 | 51.6% | -13.73 | -8.6% |
| 4-6% | 107 | 57 | 53.3% | -3.08 | -2.9% |
| 6%+ | 239 | 124 | 51.9% | -10.49 | -4.4% |

The 2–4% bucket lost -13.73 units, the largest named-bucket loss. The 0–2% bucket's positive result is contradicted by 2025 and its win-rate confidence interval is wide relative to the edge being claimed.

### Absolute-edge deciles

Deciles are equal-count ranks computed separately within the 2026 odds sample; their edge ranges are descriptive and are not reusable thresholds.

| Decile | Observed abs-edge range | n | Wins | Units | ROI |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0%–0.8% | 68 | 39 | +0.68 | 1.0% |
| 2 | 0.8%–1.5% | 67 | 39 | +1.52 | 2.3% |
| 3 | 1.5%–2.4% | 67 | 42 | +5.78 | 8.6% |
| 4 | 2.5%–3.2% | 67 | 35 | -5.18 | -7.7% |
| 5 | 3.2%–4.1% | 68 | 31 | -11.18 | -16.4% |
| 6 | 4.1%–5.3% | 67 | 33 | -7.74 | -11.5% |
| 7 | 5.3%–6.8% | 67 | 41 | +7.96 | 11.9% |
| 8 | 6.8%–8.3% | 67 | 37 | -0.95 | -1.4% |
| 9 | 8.3%–10.9% | 67 | 35 | -1.69 | -2.5% |
| 10 | 10.9%–28.0% | 68 | 30 | -10.31 | -15.2% |

Losses are not monotonic in claimed edge. Decile 5 (roughly 3.2%–4.1%) and decile 10 (>10.9%) lost -21.49 units combined, while decile 7 was positive. That jagged pattern is evidence against treating `abs_edge` as a calibrated staking signal.

## Error cohorts show a costly mid-confidence reversal

Pick confidence is `max(home_win_prob, 1-home_win_prob)`; loss rate is the fraction of model picks that lost.

| Predicted pick probability | n | Loss rate | Units | ROI |
| --- | --- | --- | --- | --- |
| 0.500-0.525 | 232 | 48.3% | -5.77 | -2.5% |
| 0.525-0.550 | 200 | 46.5% | -6.57 | -3.3% |
| 0.550-0.575 | 91 | 42.9% | +2.40 | 2.6% |
| 0.575-0.600 | 72 | 52.8% | -14.03 | -19.5% |
| 0.600-0.650 | 65 | 41.5% | -2.42 | -3.7% |
| 0.650-1.000 | 13 | 15.4% | +5.30 | 40.7% |

The 0.575–0.600 confidence band was the clearest 2026 miss: it lost -14.03 units at -19.5% ROI. This was not stable in 2025, when the comparable band's loss rate was 41.1%; it is a 2026 failure cohort, not a validated structural rule.

Selected-side price bands expose a second costly cohort:

| Selected-side price | n | Loss rate | Units | ROI |
| --- | --- | --- | --- | --- |
| heavy favorite (<= -200) | 42 | 45.2% | -9.20 | -21.9% |
| favorite (-199 to -110) | 435 | 43.2% | -9.02 | -2.1% |
| short favorite (-109 to -1) | 59 | 54.2% | -6.20 | -10.5% |
| underdog (+1 to +150) | 129 | 50.4% | +8.51 | 6.6% |
| long underdog (> +150) | 8 | 87.5% | -5.19 | -64.9% |

Heavy favorites lost -9.20 units on only 42 bets. Of those, home heavy-favorite picks were n=33, with 45.5% losses and -22.4% ROI. The same broad heavy-favorite band was approximately flat in 2025 (0.4%), so this too is a concentrated 2026 miss rather than stable proof of a bad cohort.

## Ranked hypotheses for weak AUC

1. **Feature ceiling, especially direct starter-quality signal — strongest hypothesis.** v3 loses the disagreement test to even the free market in both windows, while its feature manifest contains probable-starter history expressed mainly through team results and runs allowed/support, not starter K/9, ERA/FIP proxies, handedness, bullpen state, or confirmed lineup quality. That is consistent with omitted baseball-strength information, but the existing cache cannot isolate the contribution of any one missing feature.
2. **Season-reset states weaken April — supported, not proven.** April 2026 AUC was 0.5099 versus 0.5879 in May, and April generated -22.47 units of loss. The feature process resets team/starter rolling state by season and drops the first five games rather than carrying prior-year strength forward. Month effects and a short two-month test can confound this pattern, so a carryover-state ablation is required.
3. **RF probability compression / limited separation — clearly observed, causal role uncertain.** The 0.45–0.60 concentration and low ECE show conservative outputs with few strong predictions. Compression alone does not mathematically lower AUC because AUC depends on ordering, but it signals weak separation and makes tiny market-relative edges sensitive to noise. Compare RF ranking with logistic/boosting models and calibrate only after model selection.
4. **Free odds are soft or stale — plausible for ROI, not an explanation for weak AUC.** Line quality can corrupt the size and profitability of `model_edge_home`, and the jagged decile ROI supports caution. However, AUC is computed from game outcomes without odds, and the free market itself had higher AUC in both windows. Closing/sharp lines are needed to judge true betting edge, not to explain v3's outcome-ranking ceiling.

## Scope and method

- Population: completed regular-season games in the committed v3 prediction caches; 2026 YTD spans April 1–May 21 (673 games), and the 2025 stability window spans April 1–September 28 (2350 model rows; 2309 odds rows).
- Calibration: ten fixed probability bins and standard count-weighted ECE. Empty bins remain visible and contribute zero weight.
- Market comparison: same-row model and no-vig market probabilities; no external odds lookup or rebuild.
- ROI: one unit staked on every model pick using stored `profit`; all splits are descriptive post hoc cuts, with no multiple-testing correction.
- Uncertainty: Wilson 95% intervals are rough binomial intervals for win rate, not odds-aware intervals for ROI. Tables are used instead of charts because the required cuts are small, exact audit partitions.

## Needs v2 store to answer

- Whether direct probable-starter form and quality (prior-start K/9, ERA/FIP proxies, handedness, and matchup differentials) explains the model–market disagreement gap.
- Whether weather, temperature, wind, and park-relative conditions identify systematic residuals, especially for run environment and derivative markets.
- Whether leakage-safe prior-season carryover or preseason priors repair the April AUC collapse versus a pure season reset.
- Whether bullpen workload/availability, confirmed lineups, injuries, travel, and umpire context explain the heavy-favorite and mid-confidence losses; these require v2 or additional enrichment if v2 does not include them.
- Whether the apparent edge survives against timestamped sharp closing prices; the free comparison line cannot answer this.

## Recommended next steps and further questions

Packet 08 should treat v3 as a probability baseline, not a betting signal. Test starter-quality and prior-season-carryover ablations first, report AUC/Brier changes on the identical 2026 rows, then repeat the market-disagreement and edge-decile checks. Do not promote the 0–2% bucket without out-of-sample replication and sharper closing odds. The open questions are whether improved baseball features close the stable disagreement gap and whether any residual market edge remains after line-quality and timestamp controls.
