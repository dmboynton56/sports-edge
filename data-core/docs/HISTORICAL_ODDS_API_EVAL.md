# Historical Odds API Evaluation

Generated: 2026-05-23

## Verdict

Do not buy OddsPapi paid solely to raise quota unless they confirm a deeper pre-2026 archive or a one-time bulk export. The public v4 historical docs say all historical odds data is available since January 2026, which can help the NBA February-May 2026 gap and MLB 2026 YTD, but cannot satisfy NFL 2025 or MLB 2025.

Update after free-route audit: NFL 2025 spreads, NBA 2026 tail spreads, MLB 2025 moneylines, and MLB 2026 YTD moneylines can be filled without a paid odds API. See `docs/FREE_HISTORICAL_ODDS_AUDIT.md`.

Best value paid fallback: The Odds API paid for one month, but only if the product requires supported API terms, sharp-book provenance, or exact 100% historical coverage. It is not needed for ROI backtesting coverage.

Best quality finalist: a Pinnacle-first archive/feed, preferably a vendor that can prove 2021-present MLB/NBA/NFL ML/spread coverage in a 10-game spike. BettingIsCool is the most directly aligned public Pinnacle archive found; OpticOdds/OddsJam are higher-confidence enterprise alternatives if a sales quote is acceptable.

## Free Route

Free and cheap sources now cover more than 80 percent of the current gaps.

Known cheap/free coverage:

- Existing NBA Pinnacle/Kaggle path covers BigQuery through 2026-02-12 only.
- FantasyData public team odds pages cover 353 of the 357 local NBA tail rows from 2026-02-13 through 2026-05-21; the four misses are All-Star/exhibition rows.
- CheckBestOdds public MLB historical pages cover 2,387 of 2,430 local MLB 2025 regular-season rows.
- FantasyData public MLB team odds pages cover 750 of 750 local MLB 2026 YTD regular-season rows.
- OddsPapi free has already produced recent Pinnacle joins, but the free quota is about 250 requests/month and local testing only produced recent-window coverage.
- nflreadpy/nflverse covers NFL 2025 spreads for all 285 target games; exported to `notebooks/cache/nfl_nflverse_spreads_2025.csv`.
- Spreadspoke can cheaply cover NFL 2025 spreads too, but nflverse already removed the immediate NFL need.

Coverage math after the free-route audit: repo-generated free sources cover 3,775 of 3,822 target rows, or 98.8%. This clears the 80 percent gate. The remaining misses are 43 MLB 2025 games and four NBA All-Star/exhibition rows, neither of which blocks ROI backtesting.

## Scorecard

Legend: Pass, Partial, Fail, Unknown.

| Vendor/source | Closing history | Markets | Coverage | Joinable | Scriptable | Sharp book | Predictable cost | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FantasyData public pages | Partial consensus close | NBA spreads | Pass NBA tail except exhibition rows | Pass `game_id` | Pass HTML scrape | No | Pass/free | Use now for NBA ROI |
| CheckBestOdds public pages | Partial consensus/best odds | MLB moneylines | Pass MLB 2025 | Pass `game_pk` | Pass HTML scrape | No | Pass/free | Use for MLB 2025 ROI |
| FantasyData MLB pages | Partial consensus close | MLB moneylines | Pass MLB 2026 YTD | Pass `game_pk` | Pass HTML scrape | No | Pass/free | Use for MLB 2026 ROI |
| OddsPapi paid | Pass for line movement | Pass | Fail unless custom pre-2026 archive exists | Pass; repo already has mapping | Pass REST | Pass Pinnacle | Unknown/custom | Not recommended yet |
| The Odds API paid | Pass snapshots near close | Pass | Pass since 2020 | Pass by team/date/event | Pass REST | Partial; Pinnacle via EU, DK/FD via US | Pass | Best value |
| OpticOdds | Pass full movement | Pass | Likely pass, verify exact league/book depth | Pass normalized IDs | Pass REST/bulk | Likely pass, verify Pinnacle | Unknown/custom | Quality finalist |
| OddsJam | Pass full historical database claimed | Pass | Likely pass | Likely pass | Pass/custom API | Likely pass | Fail; contact sales | Enterprise only |
| SportsDataIO | Pass closing and line movements | Pass | Likely pass for US books | Strong official IDs | Pass REST/archive | Partial; US books, no Pinnacle shown | Unknown/sales | Good US-book fallback |
| OddsMatrix | Pass multi-year archive | Pass | Likely pass | Unknown | API/feed/bulk | Unknown book list | Unknown/sales | Overkill |
| Existing NBA CSV/Pinnacle | Pass | NBA spreads only | Fail; stale after 2026-02-12 | Pass; already loaded | Pass CSV | Pass | Pass/free | Keep, but not enough |
| nflreadpy/nflverse | Partial; closing spread/price only | NFL spreads only | Pass NFL 2025 | Pass `game_id` | Pass Python | No named sportsbook | Pass/free | Use now for NFL ROI |
| Spreadspoke | Partial; closing spread only | NFL spreads only | Pass NFL 2025 | Pass team/date | Pass CSV | Unknown | Pass $24.99/year | Redundant fallback |
| Scottfree historical CSV | Partial; closing/opening where included | All target sports | Unknown current-season depth | Pass team/date | Pass CSV/API | Unknown | Pass $19-$59/mo plus datasets | Spike only |

## Cost Model

Original backfill target in local repo files: 3,822 rows across NFL 2025, NBA 2026 tail, and MLB 2025-2026 YTD. Current free sources cover 3,775 rows, leaving 47 non-blocking rows.

OddsPapi:

- Expected requests: about one historical call per matched fixture plus fixture discovery, roughly 4,100-4,300 requests.
- Runtime: historical endpoint cooldown is 5 seconds, so about 5-6 hours for historical calls.
- Cost: free tier cannot do this in one month; paid is custom/opaque. Fails if January 2026 retention is hard.

The Odds API:

- Historical cost: 10 credits per region per market.
- Best implementation for only the remaining 47-row gap: one sport-level snapshot per distinct close timestamp, one market, one region. Estimate under 100 snapshots = under 1,000 credits plus event discovery, so $30/20K should fit.
- Conservative implementation for the original full backfill: per-event historical odds = about 38,220 credits, so use $59/100K for one month.
- Ongoing cost after backfill: $30/month is enough for occasional audits; live polling needs separate sizing.

Enterprise/custom vendors:

- OpticOdds, OddsJam, SportsDataIO, OddsMatrix: assume custom monthly minimum. Do not proceed without a written quote, export option, retention statement, and commercial-use terms.
- Spreadspoke NFL fallback: $24.99/year, but use only if -110 spread ROI is acceptable.

## Spike Plan

Run this for each finalist before purchase beyond one month.

1. Select 5 MLB games: two from 2025 regular season, three from 2026 YTD. Query last available moneyline snapshot before first pitch. Output `game_pk,home_moneyline,away_moneyline,book,snapshot_ts`.
2. Select 5 NBA games already present in `sports_edge_raw.raw_nba_odds` before 2026-02-12. Query last available spread before tip. Convert to home perspective and decimal/American price.
3. Assert MLB joins by `game_pk` through team/date matching and NBA joins by `game_id`; require 10/10 joined.
4. For NBA, compare candidate spread and price against BigQuery Pinnacle rows. Require exact side orientation, spread difference <= 0.5, and price within 5 American cents or 0.03 decimal.
5. Record request count, elapsed time, missing markets, stale timestamps, and quota headers.
6. Only then run the full backfill into `data-core/.env`-backed scripts; never place secrets outside `data-core/.env`.

## Source Notes

- OddsPapi historical docs: https://oddspapi.io/us/docs/get-historical-odds
- OddsPapi product/pricing overview: https://oddspapi.io/us/
- The Odds API pricing: https://the-odds-api.com/
- The Odds API historical docs: https://the-odds-api.com/liveapi/guides/v4/index.html
- OpticOdds historical: https://opticodds.com/historical-odds
- OpticOdds pricing lead form: https://opticodds.com/pricing
- OddsJam API: https://oddsjam.com/odds-api
- SportsDataIO odds API: https://sportsdata.io/live-odds-api
- SportsDataIO historical guide: https://sportsdata.io/help/historical-data-integration-guide
- OddsMatrix: https://oddsmatrix.com/
- Spreadspoke: https://www.spreadspoke.com/
- Scottfree pricing/sample: https://scottfreellc.com/sports/pricing
- BettingIsCool Pinnacle archive: https://api.bettingiscool.com/
