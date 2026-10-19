# NFL player props — data availability + receiving-yards prototype

**Date:** 2026-10-19  
**Scope:** analysis and prototype only. Live site, Daily Refresh, Player Market Refresh, and published predictions were not changed.  
**Raw probe / metrics:** `docs/analysis/nfl-props-market-probe.json`, `docs/analysis/nfl-receiving-yards-prototype-metrics.json`  
**Repro:**

```bash
PYTHONPATH=data-core python data-core/scripts/probe_nfl_prop_markets.py
PYTHONPATH=data-core python data-core/scripts/nfl_receiving_yards_prototype.py
```

This is not a betting product and not a claim of edge.

---

## (a) Current markets — PropLine live, Odds API documented

### Credits spent

| Provider | Calls | Credits | Notes |
| --- | ---: | ---: | --- |
| PropLine | 8 | n/a (request quota, not Odds API credits) | `/sports`, `/events`, `/scores`, `/markets`, 2× `/odds`, `/odds/history`, `/odds/closing`. Demo-key `X-RateLimit-Remaining` ended at **330**. |
| The Odds API | **0** | **0** | `ODDS_API_KEY` is not in this environment. Paid current-odds sample was not spent. Historical sample was not spent. |

Daily Refresh `audit_odds_api_credits.py` on 2026-10-19 19:20 UTC: account **used=63, remaining=437**, NFL ledger **63 / 140**, monthly cap 500, contingency 50. Shared Denver-day MLB budget was unused (PropLine-first research). A 1-event current prop pull would be ≤5 credits (`cost = unique markets returned × regions`). A 1-week receiving-yards *historical* close sample is **~161 credits** and is not cheap against the NFL allocation. Not run.

### PropLine now (2026-10-19, two upcoming games)

Sport key `football_nfl` (`americanfootball_nfl` is an alias). `/events/{id}/markets` listed all five requested keys plus extras (`player_reception_targets`, `player_rush_reception_yds`, `player_1st_td`, …).

Sampled WAS@SF (TNF/MNF) and NE@CHI.

| Market | Present | Sportsbooks (union) | DFS | Players / game (mean) |
| --- | --- | --- | --- | ---: |
| `player_reception_yds` | yes | DK, FD, Pinnacle, BetMGM, Bovada, Fanatics, Hard Rock, Fliff, BetRivers, BetOnline, LowVig, Kalshi, Novig, ProphetX | PrizePicks, Underdog, Sleeper, Dabble, ParlayPlay, Pick6 | **30** |
| `player_receptions` | yes | same minus Fanatics on the second game | same minus Pick6 | **29** |
| `player_rush_yds` | yes | same 14 | same 6 | **13** |
| `player_pass_yds` | yes | same 14 | same 6 | **4.5** |
| `player_anytime_td` | yes | 14 (Pinnacle missed the second game) | same 6 | **62.5** |

Do not mix DFS pick'em prices with sportsbook two-way odds. Probe output splits them.

### Odds API (docs, no live spend)

[Betting markets](https://the-odds-api.com/sports-odds-data/betting-markets.html) lists the same five keys for NFL, plus `_alternate` ladders. Additional markets are **per-event** (`/v4/sports/americanfootball_nfl/events/{id}/odds`). Current-odds cost is **1 credit × unique markets returned × regions**. Featured `h2h/spreads/totals` stay on the cheaper sport-level endpoint. Event list `/events` is free.

Books in the Odds API US region for these props are typically DK / FD / BetMGM / Caesars / Bovada and a shorter additional-market set than PropLine's 20+ books. We did not spend credits to count players-per-game on Odds API; PropLine already answered that.

---

## (b) Historical prop odds — cost, do not backfill 2023–2025 on Starter

### PropLine

- `/odds/history` and `/odds/closing` exist. Paid tiers: Hobby 30 days → Enterprise unlimited. CSV export (`/v1/exports/resolved-props`) is Pro+ and **archive starts April 2026**. NFL player-prop *grading* starts with the 2026 season.
- **2023–2025 closing lines are not in PropLine.**
- This run: one completed 2026 event (GB vs DAL, 2026-10-19). History listed 26 bookmakers and an `upgrade_url`. Compact main-line ticks were not persisted (closing body did not yield two-way DK/FD/Pinnacle mains on this key). 2026 CLV is a paid-tier problem, not a 2023–2025 backfill.

### The Odds API historical event odds

- Player props from **2023-05-03**. Per-event, **10 credits × markets × regions**.
- 2023–2025 REG receiving yards, one snapshot near close, `us` only:

| Item | Count |
| --- | ---: |
| Games (17×16×3) | 816 |
| Credits @ 10 / event | 8,160 |
| Weekly `/historical/.../events` discovery | 51 |
| **Total** | **8,211** |
| 1-week sample | **161** |
| 2-week sample | **322** |
| Starter month | 500 (50 reserved, NFL 140) |

**Not cheap. Not run.** 161 credits would exceed remaining NFL allocation (77) on the 2026-10-19 ledger. A 2023–2025 rec-yard close backfill needs a 20k-credit month, not Starter.

ROI/CLV on historical lines: **not run** (small-sample label in the prototype metrics). No Odds API historical spend.

---

## (c) Free outcomes / usage history

| Source | What | Years | Used in prototype? |
| --- | --- | --- | --- |
| nflverse / `nflreadpy.load_player_stats(week)` (nfl_data_py `import_weekly_data` fallback) | targets, target_share, receptions, receiving_yards, air yards, WOPR, attempts | 1999+ | **yes** |
| `load_snap_counts` | offense snaps / `offense_pct` since 2012 | 2012+ | **yes** (96% join) |
| `load_schedules` | `spread_line`, `total_line`, scores | full | **yes** (total always; spread fallback) |
| Frozen NFL v2 `nfl_v2_live.joblib` (read-only) | predicted home margin from leakage-safe pregame features | trained through 2025 | **yes** (pass-volume tilt). **v2 has no totals model.** |
| `load_ff_opportunity` | expected rec yards / team rec attempts | recent | documented, not required |
| `load_nextgen_stats(receiving)` | Cushion, YAC, catch % — not routes | 2016+ | no |
| `load_pfr_advstats(rec)` | drops, broken tackles — **no routes** | 2018+ | no |
| `load_ftn_charting` | play-action / screen flags — **no per-player routes** | 2022+ | no |
| ESPN scoreboard + box / `nfl_injury_news.py` | current scores, injuries, depth chart, posted starters | live | not in this prototype; v2 still does not apply injury *magnitudes* |
| ESPN box as historical weekly | redundant with nflverse player stats | — | skip |

**Routes:** public nflverse does not ship routes-run. Prototype uses lagged `offense_pct` as a route/snap proxy. FTN/PFR rec do not fill the hole.

---

## Prototype — receiving yards walk-forward

`data-core/src/models/nfl_receiving_yards.py` + `data-core/scripts/nfl_receiving_yards_prototype.py`

- WR/TE/RB only. Features lagged (`shift(1)` before rolling), including team points at **team-game** grain so same-game teammate order cannot leak. QB attempts feed team pass volume.
- Mean: `expected_targets = expected_team_pass_attempts × shrunk target share`; `mu = expected_targets × shrunk YPT`. Team pass volume scales lagged attempts by implied team points from **v2 margin + nflverse closing total**.
- Distribution: negative binomial, `var = mu + α mu²`. `α` fit on prior seasons only (≈0.60).
- Baseline: player's season-to-date receiving yards (then career / 5-game roll).
- Eval: 2023–2025 REG, n=14,963 player-games. Snap coverage 0.96. Live artifact scored read-only (sklearn pickle 1.6.1 loaded under 1.7.2, same as Daily's NFL venv).

### Metrics vs baseline

| Slice | n | Model MAE | Season-avg MAE | CRPS | P(over median) vs empirical |
| --- | ---: | ---: | ---: | ---: | --- |
| All WR/TE/RB | 14,963 | **19.31** | **17.63** | 13.09 | 0.48 vs 0.42 |
| Board-like (recent targets/yards) | 12,352 | 20.84 | 19.42 | 14.23 | 0.49 vs 0.45 |

Season-average **wins MAE** in every year 2023–2025 and at WR/TE/RB. Honest result: a shrunk opportunity×efficiency mean is not better than “what did this player do this season.”

Calibration at fixed lines is the part that works:

| Line | Mean P(over) | Empirical over | |gap| |
| ---: | ---: | ---: | ---: |
| 24.5 | 0.364 | 0.367 | 0.003 |
| 49.5 | 0.158 | 0.173 | 0.015 |
| 74.5 | 0.078 | 0.079 | 0.001 |

Pinball 10/50/90: 2.90 / 9.07 / 5.70. Median is a bit high (right-skewed NB). ROI/CLV: **not run** (no cheap historical closes).

---

## Recommended next step

Do **not** publish this as a board and do **not** buy Odds API historical for 2023–2025 rec yards on Starter.

1. Keep **season-average (or last-8-game yards) as the mean**. Use opportunity (target share, snaps, v2-implied pass volume, ESPN posted-starter/injury) as a *residual* / availability gate, not a replacement mean.
2. For lines: PropLine **2026** `/odds/closing` on a paid tier once GB/DAL-style games grade — that is the CLV path. Odds API historical is a 8k-credit one-off, not this month.
3. Next market if the residual mean works: `player_receptions` (same usage features, easier integer NB). Leave rush/pass yards until the mean actually beats a trailing average.

---

## What this PR does not do

- No dashboard / Vercel / Supabase / BigQuery writes.
- No Daily or PMR workflow edits.
- No Odds API spend, no `odds_api_usage` claim.
- No change to `nfl_v2_live.joblib` or anytime-TD artifacts.
