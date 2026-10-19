# NFL injury and starting-QB news

Live v2 (`nfl-v2-live-20260906`) currently freezes team forecasts on Tuesday and
reads **last-three PBP dropback QBs**, not the posted starter. That is the BAL
Huntley miss from the week-7 eval. This design is how Sports Edge picks up
current injury and starter news before kickoff and **re-prices only the games
that news (or a real line move) actually changes**.

Not a betting strategy. Fail closed on missing news, missing starters, and
missing book prices.

## (a) Data sources — hit on 2026-10-19 ~18:54 UTC

| Source | Endpoint | Key / cost | Freshness observed | Coverage observed | Verdict |
| --- | --- | --- | --- | --- | --- |
| **ESPN injuries** | `https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries` | Public, no key | `Cache-Control: max-age=9`; payload `timestamp` 2026-10-19T18:54:19Z | 32 teams; status mix `{active: 598, out: 88, questionable: 59, Injured Reserve: 49, doubtful: 3, Suspension: 3}` including Lamar Out, Hurts IR, Mayfield Out, Huntley/McKee/Daniels/Darnold Active | **Primary status** |
| **ESPN depth charts** | `https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/teams/{id}/depthcharts` | Public, no key | `max-age` 88–822s | Ranked QBs per team. SEA rank 1 = Sam Darnold, rank 2 = Drew Lock. **BAL rank 1 is still Lamar while he is Out** — depth chart alone is not a starter | **Primary order**, walked with injury status |
| **ESPN scoreboard / summary** | site.api scoreboard + `/summary?event={id}` | Public, no key | Seconds | Spreads for line-move detection; box score + drives for eval. SEA@DEN event `401872995`: Darnold 27/51, 278 yards. NYG event `401873000` Q1 15:00: `J.Winston pass` | Spreads + post-game verification, not a live IR |
| **Sleeper players** | `https://api.sleeper.app/v1/players/nfl` (already in `src/fantasy/sleeper.py`) | Free. Docs: full directory ≤1×/day | `news_updated` epoch ms; Huntley `depth_chart_order=1` while Lamar is Out | 12k players; `injury_status` + `depth_chart_order` + **gsis_id** (the v2 PBP key) | **Fallback status/order**; **ID join even when ESPN is primary** |
| **nflverse injuries** | GitHub release `injuries_2026.csv` (nflreadpy `load_injuries`) | Free | Week-grained. Latest week on this hit = **6**. Lamar week 5+6 Out (ankle). Drew Lock week 6 limited, no game status. Darnold **absent** from week 6 | Official-report shaped, not game-day inactives | Historical / eval only. Too slow for kickoff |
| **Official NFL practice / game status** | nfl.com / league IR, not a stable JSON API we will scrape | n/a | Wed/Thu/Fri practice reports. **TNF teams’ final designation is Wednesday.** Inactives ~**90 minutes before kickoff** | Authoritative, unstructured for us | Cadence only. Do not scrape |

**Primary:** ESPN injuries (status) + ESPN depth chart (order). Skip Out / IR /
Doubtful / Suspension / PUP and take the next healthy QB. That is how Lamar #1
+ Out becomes Huntley.

**Fallback:** Sleeper `injury_status` + `depth_chart_order` when ESPN 4xx/5xx,
empty team, or no usable QB ranks. Sleeper already had Huntley at order 1 on
this hit, so it is a real backup, not a toy.

**ID enrichment:** map ESPN name+team → Sleeper `gsis_id` so v2 `qb_prior_epa`
joins PBP. If Sleeper is down, keep the ESPN name, **do not invent a GSIS**, and
leave last-3 mode for that team (fail closed on the EPA channel).

**Not used live:** nflverse injuries (week lag). Odds API is not an injury
source.

## (b) How news enters the model

v2 was trained on `qb_stability_diff` and `qb_prior_epa_diff`, which today come
from `TeamState.expected_qb()` = mode of last three PBP QBs
(`data-core/src/models/nfl_v2.py`). Injury **magnitudes** are a no-op on the
live path (`refresh_nfl.py` logs that they stay off until the point-in-time
coverage gate). That gate still holds.

Live inference override, discrete, no new trained weights:

1. Resolve posted starter GSIS per team from the snapshot.
2. If it differs from last-3 mode, **replace** `expected_qb` for that game only.
3. `qb_prior_epa` = mean of that QB’s last 10 stored off-EPA values, else `0.0`
   (same default the feature store already uses for an unknown QB). That is
   backup quality.
4. `qb_stability` = fraction of the last three PBP QBs that **are** the posted
   starter (Huntley with Lamar/Lamar/Huntley → `1/3`, not Lamar’s `2/3`).
5. Do **not** also shove `player_impact_estimates` EPA into `adj_off_epa_*`.
   That would double-count QB-out through an untrained channel. The existing
   `injury_impact.py` / `what_if_qb_injury.py` / v1 `--injury-aware` path stays
   for research; v2 live does not inherit it.

Key non-QB skill injuries (RB/WR/TE/FB with Out/IR/Doubtful/Sus/PUP) are stored
on the snapshot so an eval can see them. They do not move v2 numbers until the
historical coverage gate passes.

Missing posted starter → last-3 mode, snapshot flagged `starter_unresolved`.
Never invent a name.

Tuesday Daily still builds the full slate. Midweek windows only rebuild the
dirty subset, with the same override.

## (c) Schedule (America/Denver)

Cloud-agent `gh workflow run` 403s in this environment. **Cron in the workflow
is the trigger.** `workflow_dispatch` stays for a human in the Actions UI.

New workflow `.github/workflows/nfl-injury-refresh.yml`, not Daily. Daily at
07:05 already runs MLB/NBA; stuffing TNF/Sunday reopens into it would rerun
those leagues and fight the Odds budget.

GitHub Actions cron is UTC and does not follow `America/Denver`. The times
below are **MDT** (UTC−6, most of the regular season through early November).
MST (UTC−7, November–January) shifts a single UTC cron **one hour earlier**.
Morning pre-slate windows can absorb that; **post-inactives windows cannot** —
TNF/MNF inactives land ~4:45 PM MT, and a 4:00 PM MST fire would miss them.

So each post-inactives window is **two UTC crons**: the MDT time, plus an MST
companion one UTC hour later. During the other DST period that companion is a
cheap extra look (5–6 PM MT, still before typical 6:15 PM MT kicks). Change
detection is unchanged, so the extra fires spend **zero Odds API credits** when
QB/injury/line fingerprints match the last used snapshot.

ET and MT both observe DST, so kickoff and inactives stay at the same **MT
clock time** year-round. TNF is 8:15 PM ET = **6:15 PM MT**; inactives ~90
minutes prior ≈ **4:45 PM MT**. Early Sunday 1:00 PM ET inactives ≈ **9:30 AM
MT**. 4:05/4:25 PM ET inactives ≈ **12:35–12:55 PM MT**. SNF ~8:20 PM ET
inactives ≈ **4:50 PM MT**. MNF matches TNF.

| UTC cron | MDT (MT) | MST (MT) | Window | Why |
| --- | --- | --- | --- | --- |
| `15 13 * * 4` | Thu **7:15 AM** | 6:15 AM | `thu-am` | After Wednesday TNF designations. Morning; MST hour-early is still before kick |
| `0 23 * * 4` | Thu **5:00 PM** | 4:00 PM | `thu-tnf` | ~15 min after TNF inactives (MDT). 4:00 PM MST is the too-early extra |
| `0 0 * * 5` | Thu 6:00 PM | Thu **5:00 PM** | `thu-tnf` | MST companion |
| `15 13 * * 0` | Sun **7:15 AM** | 6:15 AM | `sun-am` | After Friday practice reports, before the early slate |
| `45 15 * * 0` | Sun **9:45 AM** | 8:45 AM | `sun-inactives` | ~15 min after 1:00 PM ET inactives (MDT) |
| `45 16 * * 0` | Sun 10:45 AM | Sun **9:45 AM** | `sun-inactives` | MST companion |
| `0 19 * * 0` | Sun **1:00 PM** | 12:00 PM | `sun-late` | 4:05/4:25 PM ET inactives (MDT) |
| `0 20 * * 0` | Sun 2:00 PM | Sun **1:00 PM** | `sun-late` | MST companion |
| `15 23 * * 0` | Sun **5:15 PM** | 4:15 PM | `sun-snf` | SNF inactives (MDT) |
| `15 0 * * 1` | Sun 6:15 PM | Sun **5:15 PM** | `sun-snf` | MST companion (Monday 00:15 UTC) |
| `0 23 * * 1` | Mon **5:00 PM** | 4:00 PM | `mon-mnf` | MNF inactives (MDT) |
| `0 0 * * 2` | Mon 6:00 PM | Mon **5:00 PM** | `mon-mnf` | MST companion |

Tuesday 07:05 Daily still publishes the cycle. It snapshots ESPN/Sleeper
**before** `refresh_nfl` so the first write is not last-3-blind.

## (d) Change detection and odds

Fingerprint per remaining game:

- posted home/away starter GSIS (or normalized name if GSIS missing)
- key injuries: skill positions with unavailable status, plus **questionable QBs**

Re-predict iff, versus the last snapshot **used for a published prediction**:

- starter identity changed on either side, or
- key-injury set/status changed, or
- ESPN home spread moved by **≥ 1.5 points** (same band books disagree on at
  ±1.5; toss-up labeling uses the same idea)

No prior used snapshot (first Tuesday): write the whole slate as today.
Later windows: if posted starter matches last-3 and nothing key is hurt, **skip**.

Odds: ESPN spread is the free movement sensor. The Odds API is only called when
the dirty set is non-empty (QB/key-injury or line-move). Featured
`h2h,spreads,totals` on the **slate** endpoint costs **3 credits for the whole
slate** — the same as one event call — so we fetch the slate, then **write
snapshots only for dirty games**. Skip `team_totals` (already captured for the
cycle). `cache_period = nfl-injury-{denver_date}-{window}` so we do not silently
reuse Tuesday’s quote when we actually need a new one. Same `budgeted_get` /
`odds_api_request_cache` path; NFL monthly cap 140 + 50 contingency unchanged.
PropLine is MLB-only here.

Zero dirty games → zero Odds API credits.

## (e) Persistence

`sql/025_nfl_injury_prediction_snapshots.sql` → table
`nfl_injury_news_snapshots`. Applied from Daily (with the other serving SQL)
and from the injury workflow, same auto-apply pattern as `024`.

One row per fetch per game. Columns: supabase `game_id`, nflverse
`provider_game_id`, posted starters, last-3 expected QBs (filled at predict
time), `key_injuries` jsonb, `status_fingerprint`, ESPN and book spreads,
`source`, `fallback_used`, `window_name`, `used_for_prediction`,
`prediction_asof_ts`, `model_version`, `change_reasons`, `raw_record`.

Evals join `model_predictions.asof_ts` ≈ `prediction_asof_ts` to separate
**stale news** from **model error**. A row with `used_for_prediction=false` is
the news log only.

## Toss-up serving rule (already decided)

If `|p_home - 0.5| ≤ 0.05`, the game is a **toss-up**. Do not publish a
moneyline pick. Spread still publishes. Spread vs ML disagreement in that band
is not a product bug — books disagree at ±1.5. Week 7 reconstruction: SEA 0.494,
NO@NYG 0.494, DAL 0.494.

## Out of scope / deferred

- Training v2 on injury magnitudes / point-in-time IR (coverage gate).
- Scraping nfl.com inactives HTML.
- Recalibrating the isotonic plateaus (eval rec 5).
- Changing Tuesday-only **full** slate rebuild; this only adds targeted reopens.
