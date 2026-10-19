# NFL props — free historical lines + stats-only fair odds

**Date:** 2026-10-19  
**Scope:** analysis and prototype only. Live site, Daily Refresh, Player Market Refresh, and published predictions were not changed.  
**Companion files:** `nfl-props-fair-odds-metrics.json`, `nfl-props-week-live-comparison.json`, `nfl-propline-graded-14d-compact.json`

This is not a betting product and not a claim of edge.

```bash
PYTHONPATH=data-core python data-core/scripts/ingest_free_nfl_prop_lines.py /path/to/nfl_graded_props_14d.csv.gz
PROPLINE_API_KEY=$PROPLINE_API_KEY PYTHONPATH=data-core python data-core/scripts/nfl_prop_fair_odds_prototype.py
```

---

## Part A — free historical NFL prop lines

Every candidate below was checked for a downloadable, joinable receiving-yards or receptions archive covering **2023–2025**. None of the free files we could actually pull cover those three seasons. EndzoneAPI *has* the archive behind a free-tier key we do not have (500 credits/month ≈ 50 games or 50 player histories; a full 2023–2025 rec-yard close pull is thousands of credits). No paid Odds API / PropLine / Endzone spend.

### Source ledger

| Source | URL | Years | Markets | Books | Coverage / week | License / ToS | Download + join to nflverse? | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PropLine Kaggle / Hugging Face graded sample | [Kaggle](https://www.kaggle.com/datasets/andyt338/graded-player-props-mlb-nfl-23-sportsbooks), [HF](https://huggingface.co/datasets/propline/graded-player-props-sample) | **2026-09-18 to 2026-09-29 only** | rec yds, receptions, rush yds, pass yds | 23 books + DFS | ~8 NFL dates; ~888 rec-yard player-games | **CC BY 4.0** | **Yes.** Downloaded `nfl_graded_props_14d.csv.gz` (213,190 rows). Compacted to 1,840 sportsbook two-way mains. Join on normalized name + teams + date. Player ids are ESPN, not GSIS. | **Ingested as a 2026 sample.** Not 2023–2025. |
| PropLine API history / CSV archive | [prop-line.com](https://prop-line.com/historical-backfill) | Archive from **April 2026**; NFL grading starts 2026 | All current props | 20+ | Full slate when paid | Paid Hobby+ / $99 backfill. Free demo history is redacted. | Demo `/odds/history` does not persist 2023–2025 closes. | **No 2023–2025.** |
| The Odds API historical event odds | [the-odds-api.com](https://the-odds-api.com/sports-odds-data/betting-markets.html) | Props from 2023-05-03 | Same five keys + alternates | US books | Per event, 10 credits × markets × regions | Commercial ToS; Starter 500/mo | Joinable if purchased. 2023–2025 rec-yard close ≈ **8,211 credits**. | **Not free. Not spent.** |
| EndzoneAPI player props | [endzoneapi.com/data/player-props](https://endzoneapi.com/data/player-props), [get_prop_history](https://endzoneapi.com/docs/get-prop-history) | **2023–2025 + 2026 from week 5** (100% of finished games) | rec yds, receptions, rush/pass yds, anytime TD | 7 (DK, FD, MGM, ESPN BET, Pinnacle, …) | Opening + closing; GSIS ids | 500 free credits/month; 10 credits/game or /player-season. Unauthenticated POST → 401. | Public HTML player pages show **last 10 games only**. Full history needs a key. 855 REG games × 10 cr = 8,550 cr for 2023–2025. | **Exists, not freely bulk-downloadable.** Free tier is a sample, not a backfill. |
| Endzone public player pages | [endzoneapi.com/players](https://endzoneapi.com/players), e.g. [Chase](https://endzoneapi.com/players/ja-marr-chase-000036900) | Last 10 games (late 2025–2026) | Same | Median close | Last 10, not full season | Public marketing pages; API ToS for bulk | Name + GSIS in the slug. Last-10 only. | **Not 2023–2025.** |
| The Odds Gap props export | [theoddsgap.com/data/player-props](https://theoddsgap.com/data/player-props), free CSV [props-export.csv](https://theoddsgap.com/api/props-export.csv) | Capture from **2026-08-04**; free file is **2026-09-15 to 2026-09-21** | rec yds (104 rows), receptions (74), others | Best book + n_books | One sample week | Free sample week; rest is subscription | Downloaded 8,112 rows. NFL `player_id` prefix `nfl:`. Mostly MLB. | **2026 sample week only.** |
| Fieldwork weekly archives | [fieldworklines.com/nfl](https://fieldworklines.com/nfl) | **2026 weeks 1–7 only** (sitemap). `/nfl/2023/week-1` and `/2024`, `/2025` → 404 | Rec yds / receptions on WR/TE boards | DK, FD, MGM, BetRivers via Odds API | Median pregame observation, not necessarily close | Public pages; `/api/` disallowed | HTML boards, no CSV. No 2023–2025 URLs. | **2026 only.** |
| Nebula Insights player pages | [nebulainsights.net/players/nfl/](https://nebulainsights.net/players/nfl/), e.g. [Chase](https://nebulainsights.net/players/nfl/jamarr-chase.html) | Summaries **2023–2026**; HTML game log is **last 14 lined games** | Rec yds, receptions, ATD | Median **opening** line | 192 players, 30+ lined-game cutoff | Terms page; no CSV/JSON endpoints (404) | Opening lines, not closes. Last 14 only in HTML. | **Summaries exist; no bulk 2023–2025 download.** |
| nflverse / nflreadpy | [nflverse](https://github.com/nflverse) | Stats 1999+ | Box scores, snaps, `spread_line` / `total_line` | Game lines only | Full | Open | Weekly stats join cleanly. **No player-prop lines.** | **Outcomes yes, props no.** |
| sportsbookreviewsonline archives | [NFL odds archives](https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm) | Historical **game** lines | ML / spread / total / 2H | Consensus | Season files | Public archive, no longer updated | No player props. | **Game lines only.** |
| Action Network receiving props | [actionnetwork.com/nfl/props/receiving](https://www.actionnetwork.com/nfl/props/receiving) | Live / articles | Rec yds, receptions | Consensus | Current week | Site ToS; no bulk archive | Live table only. 2023 articles are picks, not a dataset. | **No archive.** |
| Wayback Machine DK league pages | [CDX](https://web.archive.org/cdx/search/cdx?url=sportsbook.draftkings.com/leagues/football/nfl&matchType=prefix&from=202309&to=202312&limit=5) | Snapshots exist (e.g. 2023-09) | JS sportsbook shell | DK | HTML snapshots ~370 KB | IA ToS; DK ToS forbid scraping | Snapshots are the SPA chrome, not a prop table we could join. | **Not usable.** |
| GitHub scrapers (quantgalore/nfl-props, yzRobo/draftkings_api_explorer, pureslurp/DKNFL, traplandlord/nfl-props-stats) | various | Current-week scrapers or nflverse stats mirrors | Rec yds / receptions (live) or box-score stats | DK / Prop-Odds | Live or stats-only | Code only; Prop-Odds is paid | DKNFL overwrites past weeks. traplandlord ships weekly *stats*, not lines. No committed 2023–2025 prop CSV. | **No historical file.** |
| ScoreTape | [scoretape.com/nfl](https://scoretape.com/nfl) | Game lines via nflverse to 1999; PM tape is forward-capture | Props mentioned | PM + books | Commercial | Paid API | Not a free 2023–2025 prop CSV. | **Not free bulk.** |
| LineTerminal rec-yard leaderboard | [lineterminal.com](https://lineterminal.com/players/nfl/leaderboards/receiving-yards) | **2026-27 only** (320 recorded props through Oct 18) | Rec yds | Unspecified close | Current season HTML | Site ToS; no bulk CSV | HTML leaderboard, not a joinable archive. | **2026 only. No download.** |
| Props.Cash | [props.cash](https://props.cash/) | App claims historical trends | Rec yds / receptions | Books + DFS | Current + in-app history | Paid after 7-day trial ($20/mo) | No public CSV/API. Would mix DFS. | **Not free bulk.** |

### What we ingested

`docs/analysis/nfl-propline-graded-14d-compact.json` (CC BY 4.0, PropLine sample):

- 1,840 preferred-book two-way closes
- Markets: `player_reception_yds`, `player_receptions`
- Books: DK / FD / Pinnacle / BetMGM / Bovada (and fallbacks)
- DFS dropped
- Dates: 2026-09-18 … 2026-09-29

Walk-forward ROI/CLV on **2023–2025 closes: not run** (no free file). The 2026 sample is graded when it joins nflverse and labeled **small-sample, not 2023–2025**.

---

## Part B — stats-only fair odds

`data-core/src/models/nfl_prop_fair_odds.py`

- **Mean:** season-to-date yards/receptions, **shrunk toward last-season / career** with strength 6 games, then blended with trailing-8. Recency weight keys off **season games**, not career games. Baseline for the MAE comparison is still the plain season-average used in #46 (17.63 on receiving yards).
- **Residual:** shrunk opportunity ratio (expected targets vs trailing target rate) and v2-implied pass-volume scale, clipped to 0.82–1.18. ESPN / nflverse injury status scales availability (Out 0.20, Doubtful 0.50, Questionable 0.88).
- **Distribution:** same negative binomial as the first prototype. `P(over line)` and fair American odds for any line.
- **Live:** PropLine sportsbook two-way only. Modal line, best over/under price, no-vig book `P(over)`. Never mixed with DFS. `|our P − book P| > 0.20` is flagged `check_data`, not an edge. Written to `docs/analysis` only.

### Walk-forward 2023–2025 (n=14,963, same universe as #46)

| Stat | Model MAE | Season-avg MAE | Beat 17.63? | Trailing-median P(over) vs empirical |
| --- | ---: | ---: | --- | --- |
| Receiving yards | **17.04** | 17.39 | **Yes** | 0.416 vs 0.385 (gap 0.031) |
| Receptions | **1.330** | 1.341 | n/a (different units) | 0.435 vs 0.408 (gap 0.028) |

Fixed-line calibration:

| Stat | Line | Mean P(over) | Empirical | \|gap\| |
| --- | ---: | ---: | ---: | ---: |
| Yards | 24.5 | 0.331 | 0.367 | 0.036 |
| Yards | 49.5 | 0.146 | 0.173 | 0.026 |
| Yards | 74.5 | 0.073 | 0.079 | 0.007 |
| Receptions | 2.5 | 0.362 | 0.365 | 0.003 |
| Receptions | 4.5 | 0.157 | 0.158 | 0.000 |
| Receptions | 6.5 | 0.066 | 0.059 | 0.007 |

The 24.5 yard line still **under-predicts** (0.331 vs 0.367). Shrinkage closed part of the old 0.049 gap; it did not eliminate it. Trailing-median calibration is a bit worse than the unshrunk residual (the prior pulls P(over) up). MAE still beats 17.63.

### 2026 PropLine 14-day sample — ROI / CLV (small-sample)

| Market | n | Win-loss | Flat ROI | Mean close−open toward model side |
| --- | ---: | --- | ---: | ---: |
| Receiving yards | 335 (218 players) | 167–168 | **−4.8%** | −0.86 yd (n=315) |
| Receptions | 332 (217 players) | 178–154 | **−0.6%** | −0.05 (n=309) |

Labeled small-sample. Not 2023–2025. Not a strategy. Books are DK/FD/Pinnacle/MGM plus fallbacks; DFS excluded.

### Live-table bug (fixed)

The first live table showed 30–40 point gaps (A.J. Brown 0.084 vs book 0.518). That was **not an edge**.

`early = games_before < 4` used **career** games, so every veteran got a 72% weight on season-to-date. A.J. Brown’s 2026 season-to-date was one 26-yard game after a team change; career / 2025 mean (~67–78) was ignored. Same pattern: Okonkwo 2-game 10.5-yard mean, Loveland a cold start vs a 44.6 2025 mean. Live also scored the last **played** game’s prior (did not roll that actual into the next-game mean).

Fix: shrink season-to-date toward last-season / career (strength 6); key recency off season games; roll the last actual into the live prior. `|ΔP| > 0.20` → `check_data`.

Inputs after the fix (2026 games include the last played actual):

| Player | 2026 n | Season mean | Trail-8 | 2025 mean | Resid | Avail | μ | α |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A.J. Brown | 2 | 26.5 | 52.3 | 66.9 | 0.97 | 1.00 | 53.1 | 0.35 |
| Colston Loveland | 6 | 29.2 | 44.7 | 44.6 | 0.96 | 1.00 | 37.6 | 0.35 |
| Chig Okonkwo | 3 | 27.0 | 37.0 | 32.9 | 0.97 | 1.00 | 33.0 | 0.35 |
| Kalif Raymond | 6 | 40.7 | 38.9 | 19.3 | 1.06 | 1.00 | 34.6 | 0.35 |
| Austin Ekeler | 2 | 13.0 | 20.5 | 19.0 | 0.94 | 1.00 | 17.9 | 0.35 |

Brown and Ekeler remain `check_data` (role / thin 2026 sample vs a sharp line). Not joins, not availability, not the wrong market.

### This week's live PropLine comparison (not published)

16 PropLine event calls (cached; remaining 272). **49** sportsbook two-way rows. 7 flagged `check_data`. 2026 rosters from nflverse. No DFS. `ok` rows first, then `|ΔP|`.

| Player | Game | Market | Line | Our P(over) | Book no-vig | Fair over | Best over | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Rachaad White | WAS @ SF | rec yds | 14.5 | 0.314 | 0.504 | +219 | FD −113 | ok |
| George Kittle | SF vs WAS | rec yds | 66.5 | 0.317 | 0.500 | +216 | FD −113 | ok |
| Colston Loveland | CHI vs NE | rec yds | 44.5 | 0.316 | 0.497 | +216 | MGM −111 | ok |
| Efton Chism III | NE @ CHI | rec yds | 19.5 | 0.670 | 0.493 | −203 | MGM −111 | ok |
| Antonio Williams | WAS @ SF | rec yds | 21.5 | 0.632 | 0.498 | −172 | DK −111 | ok |
| Chig Okonkwo | WAS @ SF | rec yds | 24.5 | 0.600 | 0.487 | −150 | Pinnacle −103 | ok |
| Christian McCaffrey | SF vs WAS | rec yds | 40.5 | 0.387 | 0.497 | +159 | Pinnacle −108 | ok |
| Kalif Raymond | CHI vs NE | rec yds | 25.5 | 0.604 | 0.500 | −153 | FD −114 | ok |

A.J. Brown 67.5 is 0.272 vs book 0.518 (`check_data`). Ekeler 7.5 is 0.830 vs 0.506 (`check_data`). Full table: `nfl-props-week-live-comparison.json`. This is model vs book, not a slip.

---

## Recommended next step

1. Do not publish this board. The residual mean is only a **0.35 yard** MAE lift, and `|ΔP| > 0.20` still means check the data.
2. For 2023–2025 closes: get an EndzoneAPI free key and spend the 500-credit month on a **player-history sample** (top ~50 WR/TE), not Odds API historical.
3. Keep PropLine 2026 CC-BY samples as the cheap CLV path; they already join.
