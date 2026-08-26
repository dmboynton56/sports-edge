# MLB Home Run Odds Fetcher — PropLine Fallback

## Overview

The MLB HR odds fetcher (`data-core/scripts/fetch_mlb_home_run_odds.py`) now supports **PropLine** as a fallback when The Odds API is unavailable or quota-exhausted.

## Fallback Behavior

The fetcher tries **The Odds API first** (if `ODDS_API_KEY` is set), then falls back to **PropLine** (if `PROPLINE_API_KEY` is set) when:

1. `ODDS_API_KEY` is missing or empty
2. The Odds API returns HTTP 401 (unauthorized) or 429 (rate limit)
3. The Odds API quota is exhausted (`x-requests-remaining: 0`)
4. The Odds API returns 0 priced rows (empty board)

If both keys are missing or both APIs fail, the script writes an empty board with an audit trail explaining the failure.

## Setup

### 1. Get a PropLine API Key

PropLine offers a **free tier** with 1,000 requests/day (no credit card required):

- Sign up at [prop-line.com](https://prop-line.com)
- Copy your API key from the dashboard

### 2. Add GitHub Secret

In the Sports Edge repository:

1. Go to **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `PROPLINE_API_KEY`
4. Value: your PropLine API key
5. Save

The secret is already wired into the GitHub Actions workflows:
- `.github/workflows/player-markets-refresh.yml` (daily HR fetch at 11:00 AM MDT)
- `.github/workflows/daily-refresh.yml` (deprecated escape hatch, off by default)

### 3. Local Development

For local testing, add to `.env` in the repo root or `data-core/.env`:

```bash
PROPLINE_API_KEY=your-key-here
```

Do **not** commit `.env` files or API keys to git.

## How It Works

### Event Matching

PropLine and The Odds API use **different event IDs**. The fetcher joins PropLine events to the MLB schedule by:

1. Normalizing team names (e.g., "Los Angeles Dodgers" → `losangelesdodgers`)
2. Matching on `home_team`, `away_team`, and `commence_time` (with +/- 1 day fuzzy window)

### Provider Tracking

All odds rows include a `provider` field:
- `"the_odds_api"` when fetched from The Odds API
- `"propline"` when fetched from PropLine fallback

The audit JSON also records `"provider"` and (if fallback occurred) `"fallbackReason"`.

### Markets

The script fetches the same markets from both providers:
- `batter_home_runs` (standard 0.5 line)
- `batter_home_runs_alternate` (alternate lines like 1.5, 2.5)

PropLine's JSON format is compatible with The Odds API, so normalization logic is shared.

## Throttling

### The Odds API Starter Plan (500 credits/month)

- **Before:** Player-markets-refresh ran twice daily, burning ~30 credits/day (~900/month)
- **After:** Runs once daily at **11:00 AM America/Denver** (17:00 UTC), using ~15 credits/day (~450/month)
- Manual runs via `workflow_dispatch` are still supported

### PropLine Free Tier (1,000 requests/day)

- One request per event (typically ~15 MLB games/day = 15 requests)
- Daily budget is 1,000, so PropLine won't hit quota during normal MLB season operation

### Game-Line Odds (separate path)

The game-line fetch (`scripts/fetch_mlb_game_odds.py`) requests h2h/spreads/totals for the full slate in **one API call** (cheap). It runs daily and is **not affected** by this change.

## Testing

Run the unit tests:

```bash
cd data-core
pytest tests/test_mlb_hr_odds_propline_fallback.py -v
```

Key tests:
- `test_fetch_day_hr_odds_sets_provider_field` — ensures The Odds API sets `provider="the_odds_api"`
- `test_fetch_day_hr_odds_propline_sets_provider_field` — ensures PropLine sets `provider="propline"`
- `test_propline_client_uses_header_auth` — verifies PropLine uses `X-API-Key` header (not query param)
- `test_match_events_to_schedule_by_teams_and_time` — validates event join logic

## Audit Trail

Every run writes `data-core/notebooks/cache/mlb_home_run_odds_audit.json` with:

```json
{
  "generatedAt": "2026-08-26T17:30:00Z",
  "gameDate": "2026-08-26",
  "provider": "propline",
  "fallbackReason": "The Odds API quota exhausted (0 credits remaining)",
  "oddsRows": 234,
  "eventsMatched": 15,
  "gaps": []
}
```

If both APIs fail, `provider` will be `"failed"` and `gaps` will explain both errors.

## FAQ

### Why PropLine instead of other APIs?

- **Free tier:** 1,000 requests/day, no credit card
- **JSON compatibility:** Same format as The Odds API
- **MLB coverage:** Includes batter_home_runs and alternates
- **No scraping:** Official API with rate limits

### What if both keys are missing?

The script writes an empty board and audit, then exits cleanly. The board publish job (`manage_mlb_hr_board_run.py`) still succeeds with 0 priced rows (honest behavior).

### Does PropLine replace The Odds API?

No. The Odds API is still the **primary source**. PropLine is a **fallback** to prevent the board from going dark when The Odds API quota is exhausted.

### Can I force PropLine instead of The Odds API?

Yes, unset `ODDS_API_KEY` (or don't add the GitHub secret). The script will skip The Odds API and go straight to PropLine.

### Where is the PropLine client code?

- **Client:** `data-core/src/data/propline_client.py`
- **Fetcher integration:** `data-core/src/data/mlb_hr_odds_fetcher.py`
- **Script with fallback logic:** `data-core/scripts/fetch_mlb_home_run_odds.py`

## References

- PropLine API docs: [prop-line.com/llms-full.txt](https://prop-line.com/llms-full.txt)
- The Odds API docs: [the-odds-api.com/liveapi/guides/v4](https://the-odds-api.com/liveapi/guides/v4/)
- Sports Edge HR board: [sports-edge.drewboynton.com/markets/mlb/home-runs](https://sports-edge.drewboynton.com/markets/mlb/home-runs)
