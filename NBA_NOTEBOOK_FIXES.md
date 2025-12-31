# NBA Notebook Fixes

## Issues Fixed

### 1. TypeError in Team Strength Features
**Problem**: Comparing `None` values when scores are missing
**Fix**: Added proper NaN handling using `pd.to_numeric()` and `pd.isna()` checks
**Location**: `notebooks/nba_eda.ipynb` Cell 7

### 2. Missing Defensive/Net Rating Values
**Problem**: Game logs don't have opponent points directly - only points scored
**Fix**: 
- Updated `nba_game_logs_loader.py` to match game logs with schedule data
- Uses `game_date + team` matching (more reliable than game_id)
- Computes `points_allowed` from schedule scores
- Then computes `net_rating` and `defensive_rating` from point differential

**Changes**:
- `_compute_net_rating_from_logs()` now accepts `schedule_df` parameter
- Matches game logs to schedule using date + team (not just game_id)
- Computes ratings after combining all logs (more efficient)

### 3. Notebook Cell Order
**Fix**: Updated Cell 3 to pass `schedule_df` to `load_nba_game_logs()` so opponent points can be computed

## How It Works Now

1. **Load Schedule** (Cell 2) → Gets all games with scores
2. **Load Game Logs** (Cell 3) → Gets team game-by-game stats
   - Passes schedule to enable opponent points lookup
   - Matches logs to schedule using `game_date + team`
   - Computes `points_allowed` from schedule scores
3. **Compute Form Features** (Cell 5) → Uses cached computation
   - Net rating = point differential (points_scored - points_allowed)
   - Offensive rating = points_scored
   - Defensive rating = points_allowed

## Testing

After these fixes, you should see:
- ✅ No TypeError when computing team strength
- ✅ Non-zero values for defensive ratings
- ✅ Non-zero values for net ratings
- ✅ Proper form features for all windows (3/5/10 games)

## Next Steps

1. Clear the cache and re-run the notebook:
   ```python
   from src.utils.cache import clear_cache
   clear_cache('nba_form_features*')
   ```

2. Re-run cells 2-5 to reload data with fixes

3. Continue with feature analysis and model training

