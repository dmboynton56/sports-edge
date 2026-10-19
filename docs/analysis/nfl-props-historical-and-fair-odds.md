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

- **Mean:** season-to-date yards/receptions, blended with trailing-8 (more weight on recent form in the first three games). Baseline for the MAE comparison is still the plain season-average used in #46 (17.63 on receiving yards).
- **Residual:** shrunk opportunity ratio (expected targets vs trailing target rate) and v2-implied pass-volume scale, clipped to 0.82–1.18. ESPN / nflverse injury status scales availability (Out 0.20, Doubtful 0.50, Questionable 0.88).
- **Distribution:** same negative binomial as the first prototype. `P(over line)` and fair American odds for any line.
- **Live:** PropLine sportsbook two-way only. Modal line, best over/under price, no-vig book `P(over)`. Never mixed with DFS. Written to `docs/analysis` only.

### Walk-forward 2023–2025 (n=14,963, same universe as #46)

| Stat | Model MAE | Season-avg MAE | Beat 17.63? | CRPS | Trailing-median P(over) vs empirical |
| --- | ---: | ---: | --- | ---: | --- |
| Receiving yards | **17.12** | 17.39 | **Yes** (also beats this run's 17.39 baseline) | 12.21 | 0.382 vs 0.385 (gap 0.003) |
| Receptions | **1.336** | 1.341 | n/a (different units) | 0.91 | 0.425 vs 0.408 (gap 0.018) |

Fixed-line calibration:

| Stat | Line | Mean P(over) | Empirical | \|gap\| |
| --- | ---: | ---: | ---: | ---: |
| Yards | 24.5 | 0.318 | 0.367 | 0.049 |
| Yards | 49.5 | 0.143 | 0.173 | 0.030 |
| Yards | 74.5 | 0.066 | 0.079 | 0.013 |
| Receptions | 2.5 | 0.354 | 0.365 | 0.012 |
| Receptions | 4.5 | 0.156 | 0.158 | 0.001 |
| Receptions | 6.5 | 0.068 | 0.059 | 0.009 |

The residual mean is a small MAE win, not a new board. P(over) at a player's own trailing median is well calibrated for yards.

### 2026 PropLine 14-day sample — ROI / CLV (small-sample)

| Market | n | Win-loss | Flat ROI | Mean close−open toward model side |
| --- | ---: | --- | ---: | ---: |
| Receiving yards | 335 (218 players) | 163–172 | **−7.0%** | −0.47 yd (n=315) |
| Receptions | 332 (217 players) | 155–177 | **−15.0%** | −0.06 (n=309) |

Labeled small-sample. Not 2023–2025. Not a strategy. Books are DK/FD/Pinnacle/MGM plus fallbacks; DFS excluded.

### This week's live PropLine comparison (not published)

16 PropLine event calls (cached; remaining 272). **49** sportsbook two-way rows after stripping book `Player (TEAM)` suffixes and requiring the nflverse team to be on that event. 2026 rosters (A.J. Brown NE, Mike Evans SF, Diggs/White WAS) come from nflverse, not 2025 memory. No DFS. Sorted by |our P(over) − no-vig book P(over)|.

| Player | Game | Market | Line | Our P(over) | Book no-vig P(over) | Fair over | Best over |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A.J. Brown | NE @ CHI | rec yds | 67.5 | 0.084 | 0.518 | +1088 | FD −114 |
| Colston Loveland | CHI vs NE | rec yds | 44.5 | 0.136 | 0.497 | +635 | MGM −111 |
| Chig Okonkwo | WAS @ SF | rec yds | 24.5 | 0.139 | 0.487 | +619 | Pinnacle −103 |
| Kalif Raymond | CHI vs NE | rec yds | 25.5 | 0.766 | 0.500 | −327 | FD −114 |
| Austin Ekeler | WAS @ SF | rec yds | 7.5 | 0.752 | 0.506 | −303 | DK −110 |
| Christian McCaffrey | SF vs WAS | rec yds | 40.5 | 0.259 | 0.497 | +286 | Pinnacle −108 |
| Mike Evans | SF vs WAS | rec yds | 65.5 | 0.296 | 0.500 | +238 | Bovada −115 |
| Terry McLaurin | WAS @ SF | rec yds | 49.5 | 0.352 | 0.500 | +184 | Pinnacle −110 |

Evans availability 0.88 (ESPN Questionable). Full table: `nfl-props-week-live-comparison.json`. This is model vs book, not a slip.

---

## Recommended next step

1. Do not publish this board. The residual mean is only a **0.27 yard** MAE lift.
2. For 2023–2025 closes: get an EndzoneAPI free key and spend the 500-credit month on a **player-history sample** (top ~50 WR/TE), not Odds API historical.
3. Keep PropLine 2026 CC-BY samples as the cheap CLV path; they already join.
