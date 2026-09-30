---
name: sports-edge-ops
description: Refresh ownership for MLB HR and NFL TD model probabilities, MLB game odds, and the afternoon cron after 2pm MT. Use for stale boards, research MLB audit failures, Odds API credits, or workflow_dispatch questions.
---

# Sports Edge ops

Short ownership map. Read the workflow YAML if a step name matters.

## Which workflow

| Symptom | Re-run | Do not |
| --- | --- | --- |
| Research MLB missing / audit fail (`audit_mlb_research_readiness`) | **Daily Refresh** | PMR |
| MLB HR probabilities / fair odds / 2pm board gate | **Player Market Refresh** | Daily (`run_mlb_hr` is a deprecated escape hatch) |
| NBA / NFL / CFB slate, injuries, team predictions | Daily Refresh | PMR |
| PGA tournament board | `pga-tournament-refresh.yml` | Daily / PMR unless you also need those |

## Player probabilities and game odds

- PMR publishes MLB HR probabilities after 2:00 PM MT (`cron: 15 20 * * *` ≈ 2:15 PM MT) so the board clears the afternoon eligibility gate. It makes no player prop odds request.
- Daily publishes NFL anytime-TD probabilities without player prop odds requests. The dashboard derives and labels fair odds from model probabilities.
- MLB game research (Daily) is PropLine-first. It calls Odds only when PropLine misses and the Denver-day budget is free.
- Game odds fail closed if both providers fail. Do not invent sportsbook prices or EV. Model fair odds are never sportsbook prices.

## Manual dispatch

- NFL team predictions run Tuesday for one Tuesday–Monday cycle. Touchdown probabilities refresh with NFL odds captures so the model can use newly available game totals. Availability context still runs daily within that cycle.
- NFL moneyline/spread/game-total prices run Tuesday and Sunday, plus a first-of-month catch-up. Team-total event responses are reused for the cycle.
- All scheduled current game-odds fetches share `odds_api_request_cache` from `sql/023_odds_api_usage.sql`: 500 account credits, 50 reserved, NFL allocation protected. Missing ledger/quota headers skips paid calls.
- `audit_odds_api_credits.py` reads monthly reservations and free account quota headers; Daily publishes the report in its run summary.
- `force_nfl_refresh=true` regenerates only the current NFL cycle. It does not enable injury adjustments in NFL v2 or bypass odds caching/credit caps.
- `skip_notifications=true` suppresses Discord and portfolio notifications for a manual recovery run; scheduled notification behavior stays enabled.

- Daily: morning catch-up or research MLB. Leave `run_mlb_hr` false.
- PMR: afternoon/ad-hoc HR. Default `run_mlb_hr=true`; no player prop price fetch is scheduled.
