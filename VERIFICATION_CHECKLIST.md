# Verification Checklist - MLB HR Zero Priced Fix

**PR:** https://github.com/dmboynton56/sports-edge/pull/4  
**Branch:** `cursor/fix-mlb-hr-zero-priced-transparency-fb70`

---

## Pre-Merge Checklist

### ✅ Code Quality
- [x] TypeScript types are correct (no `tsc --noEmit` errors)
- [x] ESLint passes with no warnings
- [x] Next.js build succeeds
- [x] All changed files follow existing code style
- [x] No console errors or warnings introduced

### ✅ Functionality
- [x] Gap messages added to `player-markets.ts` when `missing_odds` detected
- [x] Board component shows appropriate warning for zero-priced state
- [x] Data-quality page highlights priced=0 and explains cause
- [x] Homepage gap summary distinguishes data-provider vs infra issues

### ✅ Testing
- [x] Build completes successfully (`npm run build`)
- [x] Dev server starts without errors (`npm run dev`)
- [x] No TypeScript errors (`npx tsc --noEmit`)
- [x] No ESLint errors (`npm run lint`)

### ⏳ Manual Testing (Requires Deployment)
- [ ] Navigate to `/markets/mlb/home-runs` - verify yellow warning appears
- [ ] Check warning message includes "upstream odds unavailable at publication time"
- [ ] Verify link/reference to data-quality page for OddsPapi status
- [ ] Navigate to `/data-quality` - verify priced=0 highlighted in warning color
- [ ] Check yellow explanation box appears with data-provider clarification
- [ ] Navigate to `/` - verify gap summary says "model-only candidates (no upstream sportsbook prices)"
- [ ] Test with partial-coverage scenario (if available) - verify existing notice still works

---

## Smoke Test Scenarios

### Scenario 1: Zero Priced (Current Production State)
**Given:** 79 candidates, 0 priced, odds_status='missing_odds' for all rows  
**When:** User visits `/markets/mlb/home-runs`  
**Then:**
- Board shows 79 candidates
- Priced count is 0
- Yellow warning notice appears: "No sportsbook prices available"
- Message explains upstream odds unavailable
- Directs to data-quality page

**When:** User visits `/data-quality`  
**Then:**
- Priced count highlighted in warning color
- Yellow explanation box visible
- OddsPapi validation status shown in table

**When:** User visits `/`  
**Then:**
- Gap summary mentions "79 model-only candidates (no upstream sportsbook prices)"

---

### Scenario 2: Partial Coverage
**Given:** 79 candidates, 40 priced, 39 with odds_status='missing_odds'  
**When:** User visits `/markets/mlb/home-runs`  
**Then:**
- Board shows 79 candidates
- Priced count is 40
- Yellow warning notice: "Partial pricing coverage"
- Gap badge shows "39 candidates do not have a fresh valid sportsbook price"
- Top-25 coverage percentage shown

---

### Scenario 3: Healthy State
**Given:** 79 candidates, 75 priced, top-25 coverage 96%  
**When:** User visits `/markets/mlb/home-runs`  
**Then:**
- Board shows 79 candidates
- Priced count is 75
- No yellow warning (healthy state)
- Normal metrics and table display

---

## Regression Tests

### Existing Functionality
- [ ] Board still hides rows for games that started or begin within 5 minutes
- [ ] Fail-closed behavior unchanged (stale prices not used for edge/EV)
- [ ] Model-only rows still visible in "all" filter
- [ ] Priced/model-only filters still work
- [ ] Pagination still works
- [ ] Desktop and mobile layouts render correctly

### Edge Cases
- [ ] Board handles empty rows (0 candidates) gracefully
- [ ] Board handles unavailable/stale status correctly
- [ ] Board handles no_slate status correctly
- [ ] Gap messages don't duplicate or overflow
- [ ] Warning notices stack properly when multiple apply

---

## Performance Checks

- [ ] Page load time not significantly increased
- [ ] No new console warnings in browser
- [ ] No layout shift from warning notices
- [ ] Mobile responsive layout works
- [ ] Metrics cards render without flicker

---

## Documentation

- [x] PR description explains root cause clearly
- [x] Investigation summary documents data flow and architecture
- [x] Verification steps included in PR
- [x] Code comments added where logic is non-obvious
- [x] Commit messages are descriptive

---

## Deployment Notes

### Pre-Deployment
1. Verify Supabase connection is healthy
2. Check most recent MLB HR board run status
3. Note current priced candidate count for comparison

### Post-Deployment
1. Visit live site immediately after deploy
2. Verify zero-priced warning appears (if still applicable)
3. Check console for any runtime errors
4. Test on mobile device
5. Verify data-quality page renders correctly

### Rollback Plan
If issues occur:
1. Revert to main branch
2. Redeploy previous version
3. Report specific error in PR comments
4. Fix will be small (UI messaging only, no schema changes)

---

## Follow-Up Actions (Out of Scope)

These are NOT required for this PR but should be tracked separately:

1. **Restore upstream odds**
   - Check OddsPapi account status and quota
   - Verify MLB HR markets are available in plan
   - Re-run serving pipeline once resolved

2. **Add odds_status breakdown to data-quality page**
   - Show count per status: ok, raw_implied, missing_odds, stale
   - Helps diagnose partial vs total coverage

3. **Add OddsPapi healthcheck to serving pipeline**
   - Pre-flight check before odds fetch
   - Log specific error when markets unavailable
   - Surface in dashboard gaps

---

## Sign-Off

**Code Review:** Ready for review  
**Testing:** Automated tests pass, manual testing pending deployment  
**Documentation:** Complete  
**Risk Level:** Low (UI messaging only, no schema/backend changes)  

**Merge Recommendation:** Approve after manual smoke test on preview deployment
