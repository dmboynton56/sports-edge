# NFL v2 bucket strategy backtest

**Generated:** 2026-10-19T20:54:22.093572+00:00
**Live model (2026 only):** `nfl-v2-live-20260906` — not modified
**Walk-forward candidate:** `hgb_small` + `ridge_a50` + walk-forward `auto` (live serving still `isotonic`)
**Rules tested:** 75

This is analysis, not a serving change and not a betting product. Missing prices stay unscored. In-sample numbers are labeled and are **not** an edge.

## Headline

**Q1:** In the 40–60 toss-up band (OOS n=614), backing the book underdog is still not a licensed strategy. Dog ML 220-390-2 / -9.1% (CI -18.9% to 0.7%). Model-side ML 322-290-2 / -8.1%. Book-favorite ML 390-220-2 / -0.5%. Dog ATS -110 300-297-20 / -4.1%. None of these CIs both exclude 0 and survive the season + Bonferroni bar.

**Q2:** Confident buckets are not a usable betting surface. Book-favorite ML when the model is >60% home: 378-146-1 / -2.2% (CI -7.5% to 3.5%). When the model is <40% home: 188-119-2 / -10.9%. Model-minus-book ≥5pp ML: 258-267-2 / -3.0% (CI -11.9% to 6.1%). The raw confident buckets and the disagreement filter both fail the strategy bar. That matches the health-check diagnosis: the model is overconfident, not quietly +EV at short prices.

Highest-ROI OOS rule: **80+: back model on ATS at actual/close price** — n=53, 33-18-2, ROI 26.4%, 95% CI -0.1% to 52.7%, Bonferroni p=1.000. This does **not** qualify as a strategy: the bar is OOS + same-sign ROI in ≥3 seasons with n≥20 + 95% CI excluding 0 + Bonferroni-adjusted p<0.05.

Largest-n near-edge (n≥200): **60-70: back model on moneyline at actual/close price** — n=256, 180-76-0, ROI 5.0%, 95% CI -4.0% to 13.8%. Still includes 0.

## Sample

| Slice | Kind | n | Seasons |
| --- | --- | ---: | --- |
| Walk-forward 2021–2025 (train on prior seasons only) | **out-of-sample** | 1359 | 2021, 2022, 2023, 2024, 2025 |
| Frozen `nfl-v2-live-20260906` on 2026 REG | **out-of-sample** | 92 | 2026 weeks 1–6 |
| Pooled OOS used for all headline tables | **out-of-sample** | 1451 | 2021–2026 |
| Frozen artifact on 2025 REG | **in-sample** (trained_through=2025) | 272 | 2025 |

2026 n≈92 is real OOS for the frozen artifact but too small to license a rule by itself. 2021 uses a single prior season, so OOF calibration is impossible; that year is raw HGB. 2022–2025 pick a calibrator by expanding-group OOS Brier **and** log loss (identity / Platt / temperature / unregularized isotonic / min-bin-40 isotonic). 2026 still uses the frozen live isotonic artifact. 2025 reconstructed through that artifact is in-sample and is appendix-only.

PR #41 health overlap on 2026: joined=92, max |Δp|=0.000000.

### Why 2022 skipped 40–60

The 2022 hole is a degenerate unregularized PAV isotonic fit on one OOF season (~2021), not a 2022 feature bug. Raw 2021 HGB had 272 unique probabilities and 71 games in 40–60; forcing sklearn `IsotonicRegression` on those OOF labels collapsed 2022 to 15 unique p-values and jumped from plateaus near 0.34/0.37 to 0.64, so no game landed in 40–60. Frozen `nfl-v2-live-20260906` is the same coarseness family: 31 PAV thresholds, 16 unique y, 17 unique probabilities on 92 2026 games (PR #41). More OOF seasons populate 40–60 live (38/92) but still only four unique values in-band on a 99-grid. Walk-forward now selects a smoother monotone calibrator when it beats identity on both Brier and log loss. Live serving is unchanged.

PR #43 occupancy under forced unregularized isotonic (before this fix):

| Season | n | Unique p | Calibrator | 0-20 | 20-30 | 30-40 | 40-60 | 60-70 | 70-80 | 80+ |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | 272 | 272 | none | 26 | 46 | 33 | 71 | 23 | 35 | 38 |
| 2022 | 271 | 15 | isotonic | 0 | 0 | 159 | 0 | 74 | 24 | 14 |
| 2023 | 272 | 22 | isotonic | 0 | 0 | 79 | 83 | 27 | 79 | 4 |
| 2024 | 272 | 14 | isotonic | 0 | 0 | 73 | 82 | 56 | 54 | 7 |
| 2025 | 272 | 21 | isotonic | 0 | 0 | 64 | 83 | 75 | 41 | 9 |
| 2026 | 92 | 17 | isotonic | 0 | 0 | 21 | 38 | 13 | 20 | 0 |

### Bucket occupancy (OOS, selected calibrator)

| Season | n | Unique p | Calibrator | 0-20 | 20-30 | 30-40 | 40-60 | 60-70 | 70-80 | 80+ |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | 272 | 272 | none | 26 | 46 | 33 | 71 | 23 | 35 | 38 |
| 2022 | 271 | 271 | temperature | 0 | 12 | 60 | 123 | 44 | 31 | 1 |
| 2023 | 272 | 272 | sigmoid | 0 | 2 | 48 | 115 | 53 | 51 | 3 |
| 2024 | 272 | 271 | sigmoid | 0 | 0 | 31 | 125 | 64 | 50 | 2 |
| 2025 | 272 | 272 | sigmoid | 0 | 0 | 30 | 145 | 59 | 29 | 9 |
| 2026 | 92 | 17 | isotonic | 0 | 0 | 21 | 38 | 13 | 20 | 0 |

Frozen `nfl-v2-live-20260906` isotonic map: 31 PAV thresholds, 16 unique y, 18 unique values on a 99-grid, 4 unique in 40–60, largest jump 0.222.

Walk-forward calibrator selection (prior seasons only; does not rewrite the live joblib):

| Test season | Selected | OOF unique raw p | OOF unique calibrated p | Reason |
| ---: | --- | ---: | ---: | --- |
| 2021 | none | 0 | None | single_prior_season_no_oof |
| 2022 | temperature | 285 | 285 | improved OOS Brier and log loss versus identity |
| 2023 | sigmoid | 569 | 569 | improved OOS Brier and log loss versus identity |
| 2024 | sigmoid | 854 | 854 | improved OOS Brier and log loss versus identity |
| 2025 | sigmoid | 1138 | 1138 | improved OOS Brier and log loss versus identity |

Candidate OOS deltas vs identity (negative is better):

#### 2022

| Method | Folds | Brier | Δ Brier | Log loss | Δ Log loss | Unique p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 17 | 0.2289 | 0.0000 | 0.6546 | 0.0000 | 205 |
| sigmoid | 17 | 0.2262 | -0.0027 | 0.6447 | -0.0099 | 205 |
| temperature | 17 | 0.2259 | -0.0030 | 0.6438 | -0.0107 | 204 |
| isotonic | 17 | 0.2278 | -0.0011 | 0.8127 | 0.1581 | 62 |
| isotonic_min_bin_40 | 17 | 0.2278 | -0.0011 | 0.6483 | -0.0063 | 125 |

#### 2023

| Method | Folds | Brier | Δ Brier | Log loss | Δ Log loss | Unique p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 1 | 0.2260 | 0.0000 | 0.6465 | 0.0000 | 284 |
| sigmoid | 1 | 0.2238 | -0.0022 | 0.6392 | -0.0073 | 284 |
| temperature | 1 | 0.2247 | -0.0013 | 0.6411 | -0.0053 | 284 |
| isotonic | 1 | 0.2369 | 0.0109 | 0.8215 | 0.1750 | 12 |
| isotonic_min_bin_40 | 1 | 0.2284 | 0.0024 | 0.6481 | 0.0016 | 170 |

#### 2024

| Method | Folds | Brier | Δ Brier | Log loss | Δ Log loss | Unique p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 2 | 0.2320 | 0.0000 | 0.6582 | 0.0000 | 569 |
| sigmoid | 2 | 0.2278 | -0.0043 | 0.6471 | -0.0111 | 568 |
| temperature | 2 | 0.2290 | -0.0030 | 0.6499 | -0.0084 | 569 |
| isotonic | 2 | 0.2343 | 0.0023 | 0.7383 | 0.0800 | 29 |
| isotonic_min_bin_40 | 2 | 0.2315 | -0.0005 | 0.6547 | -0.0035 | 313 |

#### 2025

| Method | Folds | Brier | Δ Brier | Log loss | Δ Log loss | Unique p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 3 | 0.2285 | 0.0000 | 0.6511 | 0.0000 | 853 |
| sigmoid | 3 | 0.2262 | -0.0023 | 0.6442 | -0.0069 | 852 |
| temperature | 3 | 0.2269 | -0.0016 | 0.6458 | -0.0052 | 853 |
| isotonic | 3 | 0.2295 | 0.0010 | 0.7026 | 0.0515 | 38 |
| isotonic_min_bin_40 | 3 | 0.2279 | -0.0006 | 0.6476 | -0.0035 | 423 |


## Prices and vig

- Close: nflverse `spread_line`, `home_spread_odds`, `away_spread_odds`, `home_moneyline`, `away_moneyline`.
- Home spread = `-spread_line` (home favorite negative), same convention as the health check.
- Book win probability = no-vig split of the two moneylines (`no_vig_home_probability`).
- ATS `-110` always stakes at -110. ATS actual uses the nflverse close price and falls back to -110 only when that side's price is missing.
- Moneyline uses the actual close. Ties push. Fail closed if a required price is missing.

Line shopping (±1.5 across books) is not measurable from these files. nflverse exposes one PFR/consensus close (`spread_line`, `home_spread_odds`, `away_spread_odds`, moneylines), not a book panel. Serving toss-up copy in #42 mentions books disagreeing at ±1.5; that is a live-book observation, not a historical series in this backtest.

Spread-price coverage: 100.0%. Moneyline coverage: 100.0%. Multi-book rows: 0.

## Multiple comparisons

75 rules were tested on the same OOS pool (toss-up sides × markets, confident buckets × model/book-favorite × markets, and model-minus-book thresholds 3/5/8pp). A 95% CI that excludes 0 is not enough. Bonferroni uses `75` tests. A rule is called a **strategy** only if it is out-of-sample, the CI excludes 0, Bonferroni-adjusted p < 0.05, and ROI keeps the same sign in at least three seasons with n≥20.

## Q1 — toss-up band (model home or away win prob 40–60%)

Serving still withholds a moneyline pick in the narrower ±5pp band from #42. This table is the owner's 40–60 analysis band, not a serving change.

| Rule | Sample | n | W-L-P | ROI | Units | Bootstrap 95% CI | Strategy? |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Toss-up 40-60: back model on ATS at -110 | out_of_sample | 614 | 285-311-18 | -8.7% | -51.91 | -16.1% to -1.0% | False |
| Toss-up 40-60: back model on ATS at actual/close price | out_of_sample | 614 | 285-311-18 | -8.1% | -48.35 | -15.7% to -0.3% | False |
| Toss-up 40-60: back model on moneyline at actual/close price | out_of_sample | 614 | 322-290-2 | -8.1% | -49.74 | -15.9% to -0.9% | False |
| Toss-up 40-60: back book dog on ATS at -110 | out_of_sample | 617 | 300-297-20 | -4.1% | -24.27 | -11.7% to 3.6% | False |
| Toss-up 40-60: back book dog on ATS at actual/close price | out_of_sample | 617 | 300-297-20 | -3.2% | -19.03 | -10.9% to 4.6% | False |
| Toss-up 40-60: back book dog on moneyline at actual/close price | out_of_sample | 612 | 220-390-2 | -9.1% | -55.68 | -18.9% to 0.7% | False |
| Toss-up 40-60: back book fav on ATS at -110 | out_of_sample | 617 | 297-300-20 | -5.0% | -30.00 | -12.7% to 2.6% | False |
| Toss-up 40-60: back book fav on ATS at actual/close price | out_of_sample | 617 | 297-300-20 | -4.6% | -27.26 | -12.3% to 3.2% | False |
| Toss-up 40-60: back book fav on moneyline at actual/close price | out_of_sample | 612 | 390-220-2 | -0.5% | -2.78 | -6.5% to 5.6% | False |

Per-season for the three moneyline sides:

### Toss-up 40-60: back model on moneyline at actual/close price

| Season | n | W-L-P | ROI | Units | 95% CI |
| ---: | ---: | --- | ---: | ---: | --- |
| 2021 | 71 | 40-31-0 | 6.4% | 4.58 | -17.0% to 28.9% |
| 2022 | 123 | 63-59-1 | -8.4% | -10.30 | -24.9% to 9.5% |
| 2023 | 115 | 63-52-0 | 2.2% | 2.55 | -16.5% to 23.5% |
| 2024 | 125 | 68-57-0 | -11.0% | -13.73 | -25.5% to 3.7% |
| 2025 | 145 | 73-72-0 | -18.8% | -27.28 | -33.1% to -5.7% |
| 2026 | 35 | 15-19-1 | -16.4% | -5.57 | -48.7% to 16.5% |

### Toss-up 40-60: back book dog on moneyline at actual/close price

| Season | n | W-L-P | ROI | Units | 95% CI |
| ---: | ---: | --- | ---: | ---: | --- |
| 2021 | 71 | 34-37-0 | 20.8% | 14.76 | -8.3% to 51.5% |
| 2022 | 122 | 47-74-1 | -2.4% | -2.95 | -24.0% to 19.2% |
| 2023 | 113 | 35-78-0 | -23.7% | -26.80 | -45.4% to -0.6% |
| 2024 | 124 | 33-91-0 | -35.1% | -43.55 | -54.4% to -14.6% |
| 2025 | 144 | 53-91-0 | -2.6% | -3.71 | -23.4% to 20.8% |
| 2026 | 38 | 18-19-1 | 17.7% | 6.56 | -24.3% to 60.0% |

### Toss-up 40-60: back book fav on moneyline at actual/close price

| Season | n | W-L-P | ROI | Units | 95% CI |
| ---: | ---: | --- | ---: | ---: | --- |
| 2021 | 71 | 37-34-0 | -18.9% | -13.44 | -37.0% to -1.3% |
| 2022 | 122 | 74-47-1 | -4.5% | -5.46 | -17.5% to 9.0% |
| 2023 | 113 | 78-35-0 | 10.8% | 12.19 | -3.8% to 24.3% |
| 2024 | 124 | 91-33-0 | 13.5% | 16.71 | 0.9% to 25.7% |
| 2025 | 144 | 91-53-0 | -4.1% | -5.93 | -16.9% to 8.0% |
| 2026 | 38 | 19-18-1 | -18.5% | -6.85 | -44.4% to 7.8% |

Model-minus-book inside the toss-up band (does any edge sit in disagreement, not the raw 40–60 bucket?):

| Rule | Sample | n | W-L-P | ROI | Units | Bootstrap 95% CI | Strategy? |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Toss-up 40-60 and model-side edge ≥ 3pp: back model side on ATS at -110 | out_of_sample | 250 | 118-128-4 | -8.4% | -20.73 | -20.1% to 4.0% | False |
| Toss-up 40-60 and model-side edge ≥ 3pp: back model side on ATS at actual/close price | out_of_sample | 250 | 118-128-4 | -7.6% | -18.60 | -19.1% to 4.8% | False |
| Toss-up 40-60 and model-side edge ≥ 3pp: back model side on moneyline at actual/close price | out_of_sample | 250 | 92-157-1 | -14.6% | -36.29 | -29.6% to -0.4% | False |
| Toss-up 40-60 and model-side edge ≥ 5pp: back model side on ATS at -110 | out_of_sample | 228 | 106-118-4 | -9.7% | -21.64 | -21.6% to 3.1% | False |
| Toss-up 40-60 and model-side edge ≥ 5pp: back model side on ATS at actual/close price | out_of_sample | 228 | 106-118-4 | -8.8% | -19.77 | -21.2% to 3.9% | False |
| Toss-up 40-60 and model-side edge ≥ 5pp: back model side on moneyline at actual/close price | out_of_sample | 228 | 79-148-1 | -16.9% | -38.40 | -31.8% to -0.3% | False |
| Toss-up 40-60 and model-side edge ≥ 8pp: back model side on ATS at -110 | out_of_sample | 191 | 87-100-4 | -11.2% | -20.91 | -25.5% to 2.1% | False |
| Toss-up 40-60 and model-side edge ≥ 8pp: back model side on ATS at actual/close price | out_of_sample | 191 | 87-100-4 | -10.5% | -19.60 | -24.6% to 2.9% | False |
| Toss-up 40-60 and model-side edge ≥ 8pp: back model side on moneyline at actual/close price | out_of_sample | 191 | 62-128-1 | -18.4% | -35.03 | -35.6% to -1.0% | False |

## Q2 — confident bands

Under 40 / over 60, then 10-point bins and their mirrors. Back the model side and the book favorite, ATS and ML.

| Rule | Sample | n | W-L-P | ROI | Units | Bootstrap 95% CI | Strategy? |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| under 40: back model on ATS at -110 | out_of_sample | 309 | 154-150-5 | -3.3% | -10.00 | -14.0% to 7.4% | False |
| under 40: back model on ATS at actual/close price | out_of_sample | 309 | 154-150-5 | -1.9% | -5.85 | -12.7% to 9.0% | False |
| under 40: back model on moneyline at actual/close price | out_of_sample | 309 | 193-114-2 | 0.5% | 1.54 | -9.7% to 9.5% | False |
| under 40: back book fav on ATS at -110 | out_of_sample | 309 | 134-170-5 | -15.8% | -48.18 | -25.9% to -5.2% | False |
| under 40: back book fav on ATS at actual/close price | out_of_sample | 309 | 134-170-5 | -14.9% | -45.28 | -25.3% to -4.3% | False |
| under 40: back book fav on moneyline at actual/close price | out_of_sample | 309 | 188-119-2 | -10.9% | -33.36 | -18.8% to -3.0% | False |
| over 60: back model on ATS at -110 | out_of_sample | 525 | 265-246-14 | -1.0% | -5.09 | -9.2% to 7.2% | False |
| over 60: back model on ATS at actual/close price | out_of_sample | 525 | 265-246-14 | -0.0% | -0.14 | -8.2% to 8.5% | False |
| over 60: back model on moneyline at actual/close price | out_of_sample | 525 | 379-145-1 | 0.2% | 1.01 | -5.6% to 5.8% | False |
| over 60: back book fav on ATS at -110 | out_of_sample | 525 | 262-249-14 | -2.1% | -10.82 | -10.0% to 6.1% | False |
| over 60: back book fav on ATS at actual/close price | out_of_sample | 525 | 262-249-14 | -1.2% | -5.88 | -9.3% to 7.1% | False |
| over 60: back book fav on moneyline at actual/close price | out_of_sample | 525 | 378-146-1 | -2.2% | -11.31 | -7.5% to 3.5% | False |
| 0-20 (80+ mirror): back model on ATS at -110 | out_of_sample | 26 | 11-15-0 | -19.2% | -5.00 | -55.9% to 17.5% | False |
| 0-20 (80+ mirror): back model on ATS at actual/close price | out_of_sample | 26 | 11-15-0 | -16.9% | -4.40 | -54.7% to 20.8% | False |
| 0-20 (80+ mirror): back model on moneyline at actual/close price | out_of_sample | 26 | 17-9-0 | 4.0% | 1.04 | -29.7% to 38.5% | False |
| 0-20 (80+ mirror): back book fav on ATS at -110 | out_of_sample | 26 | 9-17-0 | -33.9% | -8.82 | -70.6% to 2.8% | False |
| 0-20 (80+ mirror): back book fav on ATS at actual/close price | out_of_sample | 26 | 9-17-0 | -32.2% | -8.36 | -69.6% to 5.6% | False |
| 0-20 (80+ mirror): back book fav on moneyline at actual/close price | out_of_sample | 26 | 14-12-0 | -27.5% | -7.15 | -54.2% to -2.4% | False |
| 20-30 (70-80 mirror): back model on ATS at -110 | out_of_sample | 60 | 33-26-1 | 6.8% | 4.00 | -16.0% to 29.4% | False |
| 20-30 (70-80 mirror): back model on ATS at actual/close price | out_of_sample | 60 | 33-26-1 | 9.7% | 5.71 | -14.8% to 34.0% | False |
| 20-30 (70-80 mirror): back model on moneyline at actual/close price | out_of_sample | 60 | 39-21-0 | 0.2% | 0.13 | -20.0% to 20.4% | False |
| 20-30 (70-80 mirror): back book fav on ATS at -110 | out_of_sample | 60 | 32-27-1 | 3.5% | 2.09 | -19.1% to 26.2% | False |
| 20-30 (70-80 mirror): back book fav on ATS at actual/close price | out_of_sample | 60 | 32-27-1 | 5.7% | 3.34 | -18.2% to 29.9% | False |
| 20-30 (70-80 mirror): back book fav on moneyline at actual/close price | out_of_sample | 60 | 42-18-0 | 3.5% | 2.09 | -13.8% to 21.2% | False |
| 30-40 (60-70 mirror): back model on ATS at -110 | out_of_sample | 223 | 110-109-4 | -4.1% | -9.00 | -16.3% to 8.1% | False |
| 30-40 (60-70 mirror): back model on ATS at actual/close price | out_of_sample | 223 | 110-109-4 | -3.3% | -7.16 | -15.4% to 9.2% | False |
| 30-40 (60-70 mirror): back model on moneyline at actual/close price | out_of_sample | 223 | 137-84-2 | 0.2% | 0.38 | -10.9% to 11.4% | False |
| 30-40 (60-70 mirror): back book fav on ATS at -110 | out_of_sample | 223 | 93-126-4 | -18.9% | -41.45 | -31.1% to -6.7% | False |
| 30-40 (60-70 mirror): back book fav on ATS at actual/close price | out_of_sample | 223 | 93-126-4 | -18.4% | -40.26 | -30.6% to -6.3% | False |
| 30-40 (60-70 mirror): back book fav on moneyline at actual/close price | out_of_sample | 223 | 132-89-2 | -12.8% | -28.30 | -22.3% to -3.2% | False |
| 60-70: back model on ATS at -110 | out_of_sample | 256 | 129-121-6 | -1.5% | -3.73 | -13.7% to 10.7% | False |
| 60-70: back model on ATS at actual/close price | out_of_sample | 256 | 129-121-6 | -0.7% | -1.73 | -13.2% to 11.5% | False |
| 60-70: back model on moneyline at actual/close price | out_of_sample | 256 | 180-76-0 | 5.0% | 12.71 | -4.0% to 13.8% | False |
| 60-70: back book fav on ATS at -110 | out_of_sample | 256 | 127-123-6 | -3.0% | -7.55 | -14.5% to 9.2% | False |
| 60-70: back book fav on ATS at actual/close price | out_of_sample | 256 | 127-123-6 | -2.2% | -5.60 | -14.1% to 9.9% | False |
| 60-70: back book fav on moneyline at actual/close price | out_of_sample | 256 | 178-78-0 | 0.4% | 1.02 | -8.1% to 8.8% | False |
| 70-80: back model on ATS at -110 | out_of_sample | 216 | 103-107-6 | -6.4% | -13.36 | -19.1% to 7.3% | False |
| 70-80: back model on ATS at actual/close price | out_of_sample | 216 | 103-107-6 | -5.7% | -11.89 | -18.8% to 8.0% | False |
| 70-80: back model on moneyline at actual/close price | out_of_sample | 216 | 157-58-1 | -5.6% | -12.08 | -13.4% to 2.0% | False |
| 70-80: back book fav on ATS at -110 | out_of_sample | 216 | 102-108-6 | -7.3% | -15.27 | -20.0% to 6.4% | False |
| 70-80: back book fav on ATS at actual/close price | out_of_sample | 216 | 102-108-6 | -6.5% | -13.72 | -19.4% to 6.8% | False |
| 70-80: back book fav on moneyline at actual/close price | out_of_sample | 216 | 158-57-1 | -5.5% | -11.84 | -13.2% to 2.2% | False |
| 80+: back model on ATS at -110 | out_of_sample | 53 | 33-18-2 | 23.5% | 12.00 | -2.7% to 49.7% | False |
| 80+: back model on ATS at actual/close price | out_of_sample | 53 | 33-18-2 | 26.4% | 13.47 | -0.1% to 52.7% | False |
| 80+: back model on moneyline at actual/close price | out_of_sample | 53 | 42-11-0 | 0.7% | 0.37 | -15.8% to 14.6% | False |
| 80+: back book fav on ATS at -110 | out_of_sample | 53 | 33-18-2 | 23.5% | 12.00 | -2.7% to 49.7% | False |
| 80+: back book fav on ATS at actual/close price | out_of_sample | 53 | 33-18-2 | 26.3% | 13.44 | 0.2% to 52.6% | False |
| 80+: back book fav on moneyline at actual/close price | out_of_sample | 53 | 42-11-0 | -0.9% | -0.49 | -16.0% to 12.9% | False |

### Model more confident than the book

Threshold is model-favorite probability minus no-vig book probability on that same side. Compared with the raw bucket table above.

| Rule | Sample | n | W-L-P | ROI | Units | Bootstrap 95% CI | Strategy? |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Model-side edge vs no-vig book ≥ 3pp: back model side on ATS at -110 | out_of_sample | 603 | 306-282-15 | -0.6% | -3.82 | -8.8% to 6.8% | False |
| Model-side edge vs no-vig book ≥ 3pp: back model side on ATS at actual/close price | out_of_sample | 603 | 306-282-15 | 0.7% | 3.92 | -7.4% to 8.3% | False |
| Model-side edge vs no-vig book ≥ 3pp: back model side on moneyline at actual/close price | out_of_sample | 603 | 310-291-2 | -1.7% | -10.34 | -9.8% to 6.6% | False |
| Model-side edge vs no-vig book ≥ 5pp: back model side on ATS at -110 | out_of_sample | 527 | 265-249-13 | -1.6% | -8.09 | -9.7% to 6.2% | False |
| Model-side edge vs no-vig book ≥ 5pp: back model side on ATS at actual/close price | out_of_sample | 527 | 265-249-13 | -0.2% | -1.23 | -8.3% to 7.9% | False |
| Model-side edge vs no-vig book ≥ 5pp: back model side on moneyline at actual/close price | out_of_sample | 527 | 258-267-2 | -3.0% | -15.87 | -11.9% to 6.1% | False |
| Model-side edge vs no-vig book ≥ 8pp: back model side on ATS at -110 | out_of_sample | 424 | 216-199-9 | -0.6% | -2.64 | -9.8% to 8.1% | False |
| Model-side edge vs no-vig book ≥ 8pp: back model side on ATS at actual/close price | out_of_sample | 424 | 216-199-9 | 0.7% | 2.86 | -8.5% to 9.7% | False |
| Model-side edge vs no-vig book ≥ 8pp: back model side on moneyline at actual/close price | out_of_sample | 424 | 200-222-2 | -1.8% | -7.59 | -11.7% to 9.0% | False |

![roi-by-bucket](nfl-bucket-strategy/roi-by-bucket.png)
![roi-by-model-minus-book-edge](nfl-bucket-strategy/roi-by-model-minus-book-edge.png)

## In-sample 2025 (not an edge)

Frozen `nfl-v2-live-20260906` scored on 2025 REG. Trained through 2025. Do not read ROI here as a tradable rule. The in-sample toss-up dog ML printed **46.3%**; the same rule is **-9.1%** out of sample. That is the reason this appendix exists.

| Rule | Sample | n | W-L-P | ROI | Units | Bootstrap 95% CI | Strategy? |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Toss-up 40-60: back model on ATS at -110 | in_sample | 113 | 58-55-0 | -2.0% | -2.27 | -18.9% to 16.6% | False |
| Toss-up 40-60: back model on ATS at actual/close price | in_sample | 113 | 58-55-0 | -1.7% | -1.96 | -18.8% to 16.8% | False |
| Toss-up 40-60: back model on moneyline at actual/close price | in_sample | 113 | 64-49-0 | 9.2% | 10.39 | -9.0% to 27.8% | False |
| Toss-up 40-60: back book dog on ATS at -110 | in_sample | 121 | 76-45-0 | 19.9% | 24.09 | 2.6% to 37.3% | False |
| Toss-up 40-60: back book dog on ATS at actual/close price | in_sample | 121 | 76-45-0 | 20.2% | 24.47 | 2.8% to 37.0% | False |
| Toss-up 40-60: back book dog on moneyline at actual/close price | in_sample | 120 | 66-54-0 | 46.3% | 55.54 | 20.4% to 72.7% | False |
| Toss-up 40-60: back book fav on ATS at -110 | in_sample | 121 | 45-76-0 | -29.0% | -35.09 | -46.4% to -11.6% | False |
| Toss-up 40-60: back book fav on ATS at actual/close price | in_sample | 121 | 45-76-0 | -29.1% | -35.22 | -46.3% to -11.8% | False |
| Toss-up 40-60: back book fav on moneyline at actual/close price | in_sample | 120 | 54-66-0 | -29.8% | -35.76 | -44.4% to -15.4% | False |
| under 40: back model on ATS at -110 | in_sample | 54 | 34-20-0 | 20.2% | 10.91 | -4.5% to 44.9% | False |
| under 40: back model on ATS at actual/close price | in_sample | 54 | 34-20-0 | 19.7% | 10.66 | -4.9% to 44.8% | False |
| under 40: back model on moneyline at actual/close price | in_sample | 54 | 45-8-1 | 25.2% | 13.34 | 8.7% to 38.7% | False |
| under 40: back book fav on ATS at -110 | in_sample | 54 | 33-21-0 | 16.7% | 9.00 | -8.1% to 41.4% | False |
| under 40: back book fav on ATS at actual/close price | in_sample | 54 | 33-21-0 | 16.4% | 8.84 | -8.5% to 41.0% | False |
| under 40: back book fav on moneyline at actual/close price | in_sample | 54 | 44-9-1 | 20.9% | 11.08 | 4.0% to 35.0% | False |
| over 60: back model on ATS at -110 | in_sample | 97 | 56-40-1 | 11.4% | 10.91 | -8.5% to 31.2% | False |
| over 60: back model on ATS at actual/close price | in_sample | 97 | 56-40-1 | 11.3% | 10.89 | -8.0% to 31.0% | False |
| over 60: back model on moneyline at actual/close price | in_sample | 97 | 82-15-0 | 15.4% | 14.96 | 3.7% to 27.4% | False |
| over 60: back book fav on ATS at -110 | in_sample | 97 | 53-43-1 | 5.4% | 5.18 | -14.5% to 23.3% | False |
| over 60: back book fav on ATS at actual/close price | in_sample | 97 | 53-43-1 | 5.3% | 5.08 | -14.1% to 23.8% | False |
| over 60: back book fav on moneyline at actual/close price | in_sample | 97 | 79-18-0 | 6.8% | 6.63 | -4.1% to 16.5% | False |
| Model-side edge vs no-vig book ≥ 3pp: back model side on ATS at -110 | in_sample | 94 | 67-26-1 | 37.5% | 34.91 | 19.1% to 54.0% | False |
| Model-side edge vs no-vig book ≥ 3pp: back model side on ATS at actual/close price | in_sample | 94 | 67-26-1 | 37.9% | 35.27 | 19.9% to 54.6% | False |
| Model-side edge vs no-vig book ≥ 3pp: back model side on moneyline at actual/close price | in_sample | 94 | 78-16-0 | 60.4% | 56.76 | 42.6% to 78.6% | False |
| Model-side edge vs no-vig book ≥ 5pp: back model side on ATS at -110 | in_sample | 82 | 58-24-0 | 35.0% | 28.73 | 14.1% to 51.3% | False |
| Model-side edge vs no-vig book ≥ 5pp: back model side on ATS at actual/close price | in_sample | 82 | 58-24-0 | 35.5% | 29.11 | 15.0% to 52.6% | False |
| Model-side edge vs no-vig book ≥ 5pp: back model side on moneyline at actual/close price | in_sample | 82 | 66-16-0 | 60.0% | 49.22 | 39.0% to 79.0% | False |
| Model-side edge vs no-vig book ≥ 8pp: back model side on ATS at -110 | in_sample | 51 | 36-15-0 | 34.8% | 17.73 | 8.6% to 57.2% | False |
| Model-side edge vs no-vig book ≥ 8pp: back model side on ATS at actual/close price | in_sample | 51 | 36-15-0 | 35.1% | 17.90 | 9.6% to 57.5% | False |
| Model-side edge vs no-vig book ≥ 8pp: back model side on moneyline at actual/close price | in_sample | 51 | 40-11-0 | 73.9% | 37.70 | 45.5% to 102.6% | False |

## Method

1. Build the leakage-safe v2 feature store from nflverse schedules + PBP (same path as Daily / PR #41).
2. For each season 2021–2025: fit `hgb_small` + `ridge_a50` on **prior seasons only**. Collect rolling OOF raw HGB probabilities, then pick among identity / Platt / temperature / unregularized isotonic / `isotonic_min_bin_40` by expanding-group OOS Brier **and** log loss. A method must beat identity on both; otherwise raw probabilities stay. 2021 has only 2020, so no OOF calibrator.
3. 2026: reconstruct the frozen live artifact (`isotonic`). That is the production model, trained through 2025. This run does not rewrite `nfl_v2_live.joblib` or serving.
4. Grade flat 1u bets. Bootstrap 2000 resamples of per-bet units (pushes excluded from ROI denominator). Seed 20261019.
5. No live serving writes. `predict_live_games` is untouched.

## Reproduce

```bash
PYTHONPATH=data-core python data-core/scripts/nfl_bucket_strategy_backtest.py
```

Reuse saved predictions:

```bash
PYTHONPATH=data-core python data-core/scripts/nfl_bucket_strategy_backtest.py \
  --from-games data-core/notebooks/cache/nfl_bucket_strategy_games.csv
```

Artifacts: `data-core/notebooks/cache/nfl_bucket_strategy_games.csv`, `nfl_bucket_strategy_backtest.json`.

