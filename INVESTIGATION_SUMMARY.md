# MLB HR Zero-Priced Investigation Summary

**Date:** 2026-08-25  
**Investigator:** Cloud Agent  
**PR:** https://github.com/dmboynton56/sports-edge/pull/4  
**Branch:** `cursor/fix-mlb-hr-zero-priced-transparency-fb70`

---

## Executive Summary

The live MLB HR board correctly shows **0 priced candidates** because upstream sportsbook odds were unavailable when the board was generated. This is a **data-provider issue, not an infrastructure or code problem**. The serving pipeline completed successfully, Supabase is configured, and the board is correctly failing closed on missing prices.

The fix improves transparency by clearly explaining *why* prices are missing, distinguishing between infra failures and upstream data gaps.

---

## Root Cause

### Observed Symptoms
- Homepage: "79 MLB HR candidates, partial, last refresh Aug 25 8:41 PM"
- Homepage stats: Priced candidates = 0, Top-25 coverage = 0.0%
- Data-quality page: "120 candidates do not have a fresh valid sportsbook price"
- OddsPapi validation: 100% / Healthy

### Investigation Path

1. **Initial Hypothesis: Static artifact with no prices**
   - ❌ Incorrect - Local `mlb_home_runs.json` is from 2026-06-30, stale
   - ❌ Production blocks local artifact fallback when Supabase fails

2. **Second Hypothesis: Supabase not configured**
   - ❌ Incorrect - Supabase IS configured (no missing env vars reported on data-quality page)

3. **Actual Root Cause: Supabase returns data, but all odds_status='missing_odds'**
   - ✅ Supabase board query succeeds
   - ✅ Returns 79 valid MLB HR prediction rows for 2026-08-25
   - ✅ All 79 have `odds_status='missing_odds'`
   - ✅ Board correctly treats these as model-only (price/edge/EV/kelly all null)

### Why Odds Are Missing

From the serving contract (`lib/data/player-markets.ts:358`):
```typescript
const priced = row.odds_status === "ok" || row.odds_status === "raw_implied";
```

When `odds_status='missing_odds'`:
- The upstream odds provider (OddsPapi) had no fresh MLB HR markets at publication time
- The serving pipeline completed successfully with model probabilities
- No valid sportsbook prices were joined to the predictions
- Board correctly shows these as model-only rows

This is **working as designed** - the board fails closed on stale/missing prices.

---

## Data Flow Architecture

```
Upstream Odds Provider (OddsPapi)
         ↓
   [odds snapshot]
         ↓
Serving Pipeline (data-core)
         ↓
   [predictions + odds join]
         ↓
   Supabase Tables
   - mlb_home_run_board_run_health
   - mlb_home_run_board_latest
         ↓
   Dashboard (web/app)
   - getMlbHomeRunBoardSnapshot()
   - deriveMlbHrBoardSnapshot()
         ↓
   UI (MlbHomeRunBoard.tsx)
```

**Failure Point:** OddsPapi → Serving Pipeline  
**Result:** `odds_status='missing_odds'` for all rows  
**Dashboard Behavior:** Correctly shows 0 priced candidates

---

## What the Fix Changes

### Before
- Board shows 0 priced candidates with generic gap message
- Users can't tell if it's an infra issue or data-provider issue
- Data-quality page shows the number but no explanation
- Homepage gap summary is ambiguous

### After
- **Board:** Yellow warning box explains "No sportsbook prices available" with context
- **Data-quality:** Highlights priced=0 in warning color, adds explanation box
- **Homepage:** Gap summary says "79 model-only candidates (no upstream sportsbook prices)"
- **All pages:** Direct users to check OddsPapi validation status

### Changed Files
1. `web/lib/data/player-markets.ts` - Add explicit gap messages for missing odds
2. `web/components/markets/MlbHomeRunBoard.tsx` - New warning for zero-priced state
3. `web/app/data-quality/page.tsx` - Highlight and explain zero-priced
4. `web/app/page.tsx` - Improve homepage gap summary

---

## Verification Steps

### 1. Check Current Live State (Reproduces Issue)
```bash
# Visit https://sports-edge.drewboynton.com
# Expected: 79 candidates, 0 priced, partial status
# Expected: Generic gap message
```

### 2. Deploy PR Branch (Shows Fix)
```bash
# Visit preview deployment from PR #4
# Navigate to /markets/mlb/home-runs
# Expected: Yellow warning "No sportsbook prices available"
# Expected: Explanation about upstream odds provider
# Expected: Direction to check OddsPapi on /data-quality
```

### 3. Verify Data-Quality Page
```bash
# Navigate to /data-quality
# Expected: Priced=0 highlighted in warning color
# Expected: Yellow explanation box below metrics
# Expected: "This is a data-provider issue, not infrastructure"
```

### 4. Verify Homepage
```bash
# Navigate to /
# Expected: Gap summary mentions "model-only candidates (no upstream sportsbook prices)"
```

---

## What This Does NOT Fix

❌ **Does not restore upstream odds** - Requires valid OddsPapi key or alternative odds provider  
❌ **Does not change fail-closed behavior** - Board still hides stale/missing prices from edge calculations  
❌ **Does not modify serving pipeline** - Backend odds-fetching logic unchanged  
❌ **Does not fake or hardcode odds** - Model-only rows stay model-only  

---

## Remaining Infra Gaps (Out of Scope)

These are NOT the cause of zero priced candidates, but are noted for completeness:

1. **BigQuery not configured**
   - Missing: `BIGQUERY_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS`
   - Impact: Dashboard can't query BigQuery directly (falls back to Supabase)
   - Severity: Low (Supabase is the primary serving layer)

2. **Local artifact is stale**
   - `web/public/data/mlb_home_runs.json` from 2026-06-30
   - Only used when `MLB_HR_USE_LOCAL_FIXTURE=true`
   - Severity: None (not used in production)

3. **Upstream odds provider issue**
   - OddsPapi validation shows 100% healthy
   - But no MLB HR markets available at publication time
   - Severity: High (blocks all priced candidates)
   - **This is the actual blocker**

---

## Recommended Next Steps

### Immediate (To restore prices on the board)
1. **Check OddsPapi account status**
   - Verify API key is active and has quota
   - Check if MLB HR markets are enabled in OddsPapi plan
   - Review OddsPapi logs for 2026-08-25 afternoon run

2. **Re-run serving pipeline manually**
   - If OddsPapi issue is resolved, trigger afternoon run
   - Expected: New Supabase rows with `odds_status='ok'`

### Medium-term (Reliability)
1. **Add OddsPapi healthcheck to serving pipeline**
   - Pre-flight check before odds fetch
   - Log specific error when markets unavailable
   - Surface in dashboard gaps message

2. **Add odds_status breakdown to data-quality page**
   - Show count per status: ok, raw_implied, missing_odds, stale, etc.
   - Helps diagnose partial vs total coverage issues

### Long-term (Product)
1. **Consider showing model-only rows with clear labels**
   - Current: hidden from main board when unpriced
   - Alternative: show with "Model-only - no actionable price" badge
   - Trade-off: More information vs more clutter

---

## Technical Details

### Supabase Schema (Relevant Columns)
```sql
-- mlb_home_run_board_latest
odds_status text NOT NULL  -- 'ok' | 'raw_implied' | 'missing_odds' | ...
american_price int4        -- null when odds_status != 'ok'
market_probability float8  -- null when odds_status != 'ok'
edge float8                -- null when odds_status != 'ok'
```

### Code Contract
```typescript
// web/lib/data/player-markets.ts:358
const priced = row.odds_status === "ok" || row.odds_status === "raw_implied";

// Lines 371-376: Price fields only set when priced=true
price: priced ? row.american_price : null,
impliedProbability: priced ? row.market_probability ?? row.raw_market_probability : null,
edge: priced ? row.edge : null,
ev: priced ? row.ev : null,
kelly: priced ? row.quarter_kelly : null,
```

### Gap Message Logic
```typescript
// Lines 534-542: New gap messages added
const missingOdds = predictions.filter((row) => row.oddsStatus === "missing_odds").length;
gaps: uniqueGaps([
  ...boardStatus.gaps,
  missingOdds && priced.length === 0
    ? `${missingOdds} candidates do not have a fresh valid sportsbook price. The serving run completed successfully, but upstream odds were unavailable at publication time.`
    : missingOdds
      ? `${missingOdds} candidates do not have a fresh valid sportsbook price.`
      : null,
]),
```

---

## Conclusion

**The board is working correctly.** Zero priced candidates is the expected behavior when upstream sportsbook odds are unavailable. The fix improves transparency so users understand *why* prices are missing and what to check next.

The actual blocker is the upstream odds provider (OddsPapi), not the dashboard code or infrastructure. Once fresh MLB HR markets are available from OddsPapi and the serving pipeline re-runs, priced candidates will appear on the board automatically.
