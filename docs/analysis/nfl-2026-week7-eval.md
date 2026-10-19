# NFL v2 live eval — 2026 Oct 15–19 cycle

**Model:** `nfl-v2-live-20260906`  
**Slate:** Thursday 2026-10-15 SEA@DEN (SEA 14, DEN 17 OT) and Sunday 2026-10-18 (12 games). MNF WAS@SF had not kicked off at evaluation time.  
**nflverse week:** 6 (week 1 starts 2026-09-09/10). The Tuesday–Monday Sports Edge cycle is 2026-10-13 → 2026-10-19.

This is a readout, not a claim that the model is a betting strategy.

## How the numbers were built

Supabase/BigQuery credentials are not in this environment. Forecasts were **reconstructed** from the frozen live artifact (`data-core/models/nfl_v2_live.joblib`, sklearn 1.7.2 serving path) plus nflverse schedules/PBP, which is what Daily Refresh uses after `backfill_nfl_raw.py`.

Reconstruction checks:

| Check | Result |
| --- | --- |
| Public `/models/results` Sunday winner + model-line ATS grades (12 games) | **12/12 match** |
| Live `/markets/nfl` WAS@SF | model **-14.0 / 78.2%** vs reconstructed **-14.03 / 78.16%** |
| Public v2 sample / winner W-L | **92 games, 51-41** matches |
| Public v2 model-line ATS | published **39-53**; reconstruction **40-52** (one-game discrepancy, likely rounding vs the stored `my_spread`) |

Book lines are nflverse closing/consensus, converted to Sports Edge home-spread convention (`home_spread = -spread_line`). Magnitudes match the live DK WAS@SF **-7.5**. They are **not** the Tuesday Odds API snapshot (that write path crashed; see ops).

Finals: nflverse, ESPN scoreboard (`Final/OT` for SEA@DEN and PIT@TB), and the public results page.

Repro: `PYTHONPATH=data-core python data-core/scripts/eval_nfl_2026_week7.py`

---

## Headline numbers

Spreads are home-team (negative = home favored). `p_home` is model home win probability. Book is nflverse closing.

### Oct 15–18 (n=13)

| | Model | Book / market |
| --- | ---: | ---: |
| Margin MAE | **7.79** | **7.04** |
| Margin RMSE | **8.75** | **7.99** |
| Home-margin bias (pred − actual) | +2.90 | +1.81 |
| Brier | **0.2680** | **0.2741** |
| Log loss | **0.736** | **0.756** |
| Winner accuracy | 7-6 (**53.8%**) | 6-7 (46.2% market favorite) |
| ATS vs **model** line | 5-8 (38.5%) | — |
| ATS vs **book** line, model side, -110 | **4-9** (−41.3% ROI) | — |
| Flat 1u ML on model favorite | **−1.98u** (−15.2%) | — |

Book still wins the point-spread error contest this week. Model Brier is a coin-flip edge over market on n=13; do not overread it. Five games finished inside 3 points; model went **1-4** on those moneylines.

### Season-to-date 2026 (n=92, weeks 1–6)

| | Model | Book / market |
| --- | ---: | ---: |
| Margin MAE | **9.61** | **9.39** |
| Margin RMSE | **12.26** | **11.90** |
| Home-margin bias | +1.79 | +1.53 |
| Brier | **0.2414** | **0.2369** |
| Log loss | **0.677** | **0.669** |
| ECE (10-bin) | **0.088** | 0.056 |
| Winner accuracy | **51-41 (55.4%)** | 60.9% market favorite |
| AUC | 0.627 | 0.649 |
| ATS vs **model** line | 40-52 (43.5%); public site **39-53 (42.4%)** | — |
| ATS vs **book** line, model side, -110 | **41-45-6** (−9.0% ROI) | — |
| Flat 1u ML on model favorite | **−11.4u** (−12.4%) | — |
| ML only when model ≠ market favorite | 21 bets, **−2.66u** (−12.7%) | — |

The public Results spread row is **home-covers-the-model-line**, not a sportsbook ATS. That is why it can sit at 42% while book-line ATS is near 48%. Through week 5 (79 games) the Oct 13 `grade_nfl_v2_live.py` dump was Brier 0.240 / MAE 10.00; this reconstruction is 0.237 / 9.91 on the same 79.

---

## Game log (TNF + Sunday)

`err` is predicted home margin minus actual home margin. Book ATS is the model side versus the closing home spread.

| Game | Score | Model spread | Book | p_home | mkt p_home | QBs (away / home) | ML | Book ATS | err |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| SEA@DEN | 14-17 OT | +2.5 | +2.5 | 0.494 | 0.417 | **Lock** / Nix | L | L | −5.5 |
| TEN@IND | 26-20 | −6.5 | −6.5 | 0.764 | 0.718 | Ward / Jones | **L** | W | +12.5 |
| PIT@TB | 26-19 OT | −0.6 | +2.5 | 0.364 | 0.437 | Rodgers / Daniels | W | W | +7.6 |
| CAR@PHI | 25-14 | +5.1 | +3.0 | 0.431 | 0.407 | Young / **McKee** | W | W | +5.9 |
| CHI@ATL | 31-23 | +1.0 | +1.5 | 0.431 | 0.483 | Williams / Penix | W | W | +7.0 |
| NO@NYG | 34-27 | −3.7 | +1.5 | 0.494 | 0.504 | Shough / Winston | W | L | +10.7 |
| BUF@LV | 33-30 | +6.5 | +5.5 | 0.364 | 0.325 | Allen / Cousins | W | W | −3.5 |
| BAL@CLE | 10-13 | +9.3 | +3.0 | 0.364 | 0.396 | **Huntley** / Watson | **L** | L | −12.3 |
| ARI@LA | 18-16 | −11.1 | −12.5 | 0.764 | 0.840 | Brissett / Stafford | **L** | L | +13.1 |
| NYJ@NE | 19-26 | −8.9 | −7.5 | 0.764 | 0.744 | Smith / Maye | W | W | +1.9 |
| DAL@GB | 13-23 | −0.3 | +1.5 | 0.494 | 0.489 | Prescott / Love | **L** | L | −9.7 |
| HOU@JAX | 34-31 | −7.9 | −2.5 | 0.652 | 0.563 | Stroud / Lawrence | **L** | L | +10.9 |
| LAC@KC | 6-22 | −15.3 | −7.5 | 0.800 | 0.787 | Herbert / Mahomes | W | L | −0.7 |

MNF leftover (not graded): WAS@SF model **SF −14.0 / 78.2%**, book **SF −7.5**. Same over-favorite pattern as KC.

---

## Calibration

S2D model buckets (home win):

| Bucket | n | Mean p | Actual | Bias |
| --- | ---: | ---: | ---: | ---: |
| (0.3, 0.4] | 21 | 0.36 | 0.33 | +0.03 |
| (0.4, 0.5] | 33 | 0.48 | 0.52 | −0.04 |
| (0.5, 0.6] | 5 | 0.55 | 0.40 | +0.15 |
| (0.6, 0.7] | 13 | 0.66 | 0.54 | +0.12 |
| (0.7, 0.8] | 20 | 0.76 | 0.65 | +0.11 |

Favorites are too confident. ECE 0.088 on 92 games already exceeds the live-rollout 0.05 alert. This week the isotonic classifier emitted **six unique probabilities** across 13 games (0.364, 0.431, 0.494, 0.652, 0.764, 0.800). IND, LA, and NE all got **0.764**; that bucket went 1-2 this week.

---

## What threw the moneylines

Six ML misses: SEA, IND, BAL, LA, DAL, JAX. Grounded vs speculation labeled.

### 1. Tuesday freeze is real (ops, not a feature bug)

`plan_daily_refresh.py` sets `run_nfl_predictions` only on Tuesday (or `force_nfl_refresh`). Actions:

- **2026-10-13 19:55 UTC** (scheduled Tuesday): wrote **14** NFL v2 predictions; log line `NFL v2 excludes injuries until historical point-in-time coverage passes its gate.`; synced 14 rows to Supabase.
- **2026-10-14 through 2026-10-18:** `run_nfl_predictions: false`. Availability/roster context still ran; the team model did not.

Live board: “forecasts updated Oct 13, 1:55 PM MDT.” That is the afternoon Tuesday run, not a Sunday reopen.

Odds: The Odds API **did** return 14 NFL events Tuesday (~13:07 UTC capture, reused all afternoon). `games.book_spread` + ML/spread/total snapshots were already present (`missing_book_spread: 0`, 14/14 paired ML/spread/total in the 20:05 UTC audit). The job then died inserting **`team_total`** rows:

`odds_snapshots_market_check` only allows `spread|moneyline|total` (`sql/001_initial_schema.sql`). Failing row: `market=team_total, selection=home_over`. That is why Daily Refresh is red every day and why Sunday CLV snapshots are unreliable, not why the Tuesday spread quotes were missing.

**Leverage of the freeze on this specific slate:** no team played TNF and Sunday, so missing SEA@DEN PBP did not leak into Sunday team features. The freeze mattered for **availability that changed after Tuesday**, not for in-week EPA.

### 2. Injuries / QB identity — BAL is the clean hit; SEA is a posted-starter miss

v2 `expected_qb()` is the **mode of the last three PBP dropback QBs**, not the posted starter (`nfl_v2.py`). Injury deltas are explicitly no-ops in `refresh_nfl.py`.

| Team | Last 3 PBP QBs before this slate | Posted starter | Injury report (week 6) |
| --- | --- | --- | --- |
| BAL | Lamar, Lamar, **Huntley** | Huntley | Lamar **Out** (ankle) weeks 5 and 6 |
| SEA | Darnold, Darnold, Darnold | **Lock** | Lock limited; Darnold not on the week-6 report |
| PHI | (Hurts-led) | McKee | Hurts **Out** (calf) |
| TB | (Mayfield-led) | Jalon Daniels | Mayfield **Out** |

**BAL@CLE (grounded):** model CLE +9.3 / BAL 63.6% vs market CLE +3 / BAL −170. Huntley already started week 5, but 2/3 last QBs are still Lamar, so the model keeps Lamar’s prior EPA. It was **more** on Baltimore than the book, which had already moved to Huntley. CLE 13–10. Close game, but the *line* was the tell.

**SEA@DEN (grounded starter mismatch, speculative price impact):** model and market both had SEA (model DEN +2.5 / 49.4% DEN; market DEN +2.5 / SEA −155). DEN won in OT. A Darnold→Lock swap should move a line; it did not move *this* reconstruction because last-3 QBs were Darnold and injuries are off. Whether a Lock-aware model would have flipped the pick is **speculation** — the market also stayed on Seattle.

**PHI / TB:** Hurts and Mayfield out. Model *and* market were already on CAR and PIT. Those were wins, not misses. Injury blindness did not cause this week’s ML 0-fer on those two.

**IND, LA, DAL, JAX:** posted QBs match last-3 PBP QBs (Jones, Stafford, Prescott, Lawrence). Not a QB-identity story.

### 3. Overconfidence and uncoupled spread vs ML

- IND 76% / LA 76% / JAX 65% are the same isotonic plateaus used for much bigger and much smaller edges. JAX vs market was the overconfidence case: model JAX −7.9 / 65% vs book JAX −2.5 / 56%, London/Wembley (`home_field=0`, correct). HOU 34–31.
- **Spread model and probability model disagree** on PIT@TB, NO@NYG, DAL@GB (margin favorite ≠ `p>=0.5` favorite). Winner grading uses probability only. DAL was a 49.4% pick of Dallas while the margin model had GB −0.3; GB won by 10. That is a serving-contract issue, not just variance.

### 4. Close-game variance (grounded, not a bug)

Four of six misses were one-score games: SEA@DEN OT, ARI@LA 18–16, BAL@CLE 13–10, HOU@JAX 34–31. IND (6) and DAL (10) were not. LA was a 12.5-point favorite on the book; an 18–16 dog win is mostly variance. Market was *more* on the Rams (84%) than the model (76%).

### 5. Home field / rest — not the miss driver

- JAX at Wembley: `location=Neutral` → `home_field=0`. Correct, and it *reduced* the JAX lean. The miss would have been larger with a fake home-field boost.
- Rest: PHI `rest_diff=-7` (CAR bye), KC `+7` (KC bye), DAL `−3`, PIT `+3`. PHI/CAR and KC were ML wins. No rest-feature smoking gun on the misses.
- `games_before_diff` is 0 almost every game; bye handling is in rest/travel, not that channel.

### 6. Feature bugs worth naming

- **Not a feature-store leakage bug.** Pregame rows are emitted before the game updates team state; WAS@SF and the 12 Sunday public grades matching is strong evidence the published snapshot is this artifact.
- **team_total check constraint** breaks Daily Refresh after a successful NFL forecast. Separate from model quality; it hides Sunday odds movement.
- **sklearn 1.6.1 artifact loaded on 1.7.2** in CI and here. Serving does this on purpose; it is still an InconsistentVersionWarning.

---

## Prioritized recommendations

1. **Posted-starter QB identity as an inference-time override** (do not wait on a full injury backtest). Last-3 mode cannot see Lamar Out + Huntley, or a Darnold→Lock scratch. The already-running `refresh_nfl_availability.py` feed is the obvious join key. Keep the historical coverage gate for *training*; fail closed if the posted starter is missing rather than silently using Lamar.
2. **Regenerate the current cycle on confirmed QB/injury changes**, not only Tuesday. `force_nfl_refresh` already exists. Wire it to the availability delta (Lamar Out, Hurts Out, Lock starting) instead of hoping a human dispatches it. TNF results are low leverage unless that team plays again; availability is not.
3. **Fix `odds_snapshots_market_check` to allow `team_total`** (and `home_over` / `home_under` selections) or stop sending team totals into that table. This is why the workflow is red and why closing/Sunday quotes are fragile. Featured ML/spread already captured; don’t take the job down after that.
4. **Couple or constrain spread vs ML.** Ridge margin and isotonic HGB independently picking opposite sides on coin-flip games is a product bug. Either derive `p_home` from the margin (with a residual sigma you already use on the board) or refuse to publish a moneyline favorite that disagrees with `my_spread`.
5. **Recalibrate / replace the isotonic plateaus.** Six unique p-values in a 13-game week, 76% bucket actual 65% S2D, ECE 0.088 with the sample gate already reached. Next experiment family: probability calibration only, 2026 live held out, 2020–2025 rolling origin. Do not retune on this week’s residuals.
6. **Treat public spread ATS as a line-calibration metric** and add a second column: model side vs book, fail-closed when `book_spread` is null. The 42% number is being read as “the model is 42% ATS” when it is “homes cover the model’s own line 42%.”
7. **Home-margin bias (+1.8 S2D, +2.9 this week).** Check `home_field` against 2026 scoring; this is the existing live alert (`home-margin bias exceeds ±0.75`). Not this week’s ML story, but it is why KC/SF get −15/−14 when the book is −7.5.

Do not ship an injury *magnitude* into v2 until the point-in-time coverage gate passes. Posted starter identity is a discrete override, not that gate.

---

## Sources

- Live board: https://sports-edge.drewboynton.com/models/results and `/markets/nfl`, `/models/performance/nfl`
- Artifact + code: `data-core/src/models/nfl_live.py`, `nfl_v2.py`, `src/pipeline/refresh_nfl.py`, `scripts/plan_daily_refresh.py`
- Actions: runs `38457969062` (Tue 10/13 predictions), `38568277017` (Wed skip), `39129688994` (Sun skip + odds crash)
- nflverse schedules, PBP, injury reports; ESPN scoreboard for 2026-10-15 / 18 / 19
- Cached metrics: `data-core/notebooks/cache/nfl_2026_week7_eval.json`
