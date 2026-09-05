---
name: sports-edge-ops
description: When to re-run Daily Refresh vs Player Market Refresh, MLB HR odds once/day plus PropLine fallback, and the afternoon cron after 2pm MT. Use for refresh ownership, stale boards, missing HR prices, research MLB audit failures, Odds API credits, or workflow_dispatch questions.
---

# Sports Edge ops

Short ownership map. Read the workflow YAML if a step name matters.

## Which workflow

| Symptom | Re-run | Do not |
| --- | --- | --- |
| Research MLB missing / audit fail (`audit_mlb_research_readiness`) | **Daily Refresh** | PMR |
| MLB HR odds / priced rows / PropLine / 2pm board gate | **Player Market Refresh** | Daily (`run_mlb_hr` is a deprecated escape hatch) |
| NBA / NFL / CFB slate, injuries, team predictions | Daily Refresh | PMR |
| PGA tournament board | `pga-tournament-refresh.yml` | Daily / PMR unless you also need those |

## MLB HR odds

- Canonical fetch is PMR, afternoon cron **after 2:00 PM MT** (`cron: 15 20 * * *` ≈ 2:15 PM MT) so the board clears the 2pm eligibility gate.
- The Odds API is **once per Denver day**. A second run that day should skip Odds and use PropLine (`PROPLINE_API_KEY`).
- Fallback when Odds is missing, already used today, 401/429, quota 0, or 0 priced rows.
- Both fail → fail closed (empty / `provider=failed`). Do not invent prices or EV.

## Manual dispatch

- Daily: morning catch-up or research MLB. Leave `run_mlb_hr` false.
- PMR: afternoon/ad-hoc HR. Default `run_mlb_hr=true`. Do not force Odds twice in one Denver day unless Drew explicitly wants `--force-odds-api`.
