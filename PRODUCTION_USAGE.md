# Production Usage Guide

## Overview

This guide explains how to go from EDA notebook results to production predictions for specific games.

## Architecture

```
Notebook (EDA) → Model Training → Saved Models → Production Pipeline → Supabase → Website
```

## Step-by-Step Process

### 1. Train and Export Models (Notebook)

Run the notebook cells in order. The **"Export Models for Production Use"** cell will:
- Train win probability and spread models
- Calibrate the link function
- Save everything to `models/` directory with version tags

**Output files:**
- `models/win_prob_model_nfl_v1.pkl`
- `models/spread_model_nfl_v1.pkl`
- `models/link_function_nfl_v1.pkl`
- `models/feature_medians_nfl_v1.pkl`

### 2. Use Production Predictor (Python)

#### Single Game Prediction

```python
from src.models.predictor import GamePredictor
import pandas as pd
from src.data import nfl_fetcher

# Load historical data
schedule_df = nfl_fetcher.fetch_nfl_schedule(2024)
schedule_df['game_date'] = pd.to_datetime(schedule_df['gameday'])

# Initialize predictor
predictor = GamePredictor('NFL', 'v1')

# Predict a specific game
game_row = pd.DataFrame({
    'home_team': ['KC'],
    'away_team': ['BUF'],
    'game_date': ['2025-01-15'],
    'season': [2025]
})

prediction = predictor.predict(game_row, schedule_df)
print(f"Spread: {prediction['predicted_spread']:.2f}")
print(f"Home Win Prob: {prediction['home_win_probability']:.1%}")
```

#### Batch Predictions

```python
# Predict multiple games
future_games = schedule_df[schedule_df['game_date'] > pd.Timestamp.now()].head(10)
predictions_df = predictor.predict_batch(future_games, schedule_df)
```

### 3. Use Production Pipeline (CLI)

The `refresh.py` pipeline automatically:
- Fetches games for a date
- Loads historical data
- Builds features (matching notebook)
- Makes predictions
- Exports to Supabase

```bash
# Predict and export for a specific date
python -m src.pipeline.refresh --league NFL --date 2025-01-15 --model-version v1
```

### 4. Feature Contract

The production system uses the same features as the notebook:

**Core Features:**
- `rest_home`, `rest_away` - Days of rest
- `b2b_home`, `b2b_away` - Back-to-back flags
- `opp_strength_home_season`, `opp_strength_away_season` - Opponent strength

**Team Strength Features:**
- `home_team_win_pct`, `away_team_win_pct` - Season-to-date win %
- `home_team_point_diff`, `away_team_point_diff` - Season-to-date point differential

**Interaction Features:**
- `rest_differential` - Rest advantage
- `win_pct_differential` - Win % advantage
- `point_diff_differential` - Point diff advantage
- `opp_strength_differential` - Opponent strength difference
- `week_number`, `month`, `is_playoff` - Time features

**Form Features (if PBP data available):**
- `form_home_epa_off_3/5/10`, `form_away_epa_off_3/5/10`
- `form_home_epa_def_3/5/10`, `form_away_epa_def_3/5/10`
- `form_epa_off_diff_3/5/10`, `form_epa_def_diff_3/5/10`

### 5. Integration with Supabase

The pipeline automatically:
1. Upserts games to `games` table
2. Inserts odds snapshots to `odds_snapshots` table
3. Inserts predictions to `model_predictions` table
4. Inserts features to `features` table (as JSONB)
5. Logs run to `model_runs` table

**Query today's games:**
```sql
SELECT * FROM games_today_enriched;
```

This view shows:
- `book_spread` - Latest sportsbook spread
- `my_spread` - Our model's spread
- `edge_pts` - Difference (our_spread - book_spread)
- `my_home_win_prob` - Our win probability
- `model_version` - Model version used

### 6. Next.js Integration

The website can query the `games_today_enriched` view via API:

```typescript
// app/api/sports-edges/route.ts
const { data } = await supabase
  .from('games_today_enriched')
  .select('*')
  .order('game_time_utc');
```

## Workflow Summary

1. **Development/EDA**: Use notebook to explore, train, validate
2. **Export**: Run export cell to save models
3. **Production**: Use `GamePredictor` class or `refresh.py` CLI
4. **Automation**: Schedule `refresh.py` to run every 15 minutes
5. **Display**: Website reads from Supabase view

### Backfill sportsbook odds for existing games

When games already live in Supabase but lack recent sportsbook lines (e.g., after importing schedules), run:

```bash
python populate_existing_book_odds.py --markets spreads
```

This script:
1. Finds future games where `games.book_spread` is still `NULL`.
2. Calls The Odds API (`ODDS_API_KEY` required) to fetch current spreads/totals (pass `--bookmakers fanduel` etc. to restrict).
3. Updates each game's `book_spread` column with the bookmaker's home-team spread so downstream queries can use it without joining odds snapshots.

## Key Files

- **Notebook**: `notebooks/nfl_eda.ipynb` - EDA and model training
- **Predictor**: `src/models/predictor.py` - Production prediction class
- **Training**: `src/pipeline/train_models.py` - Model training/export
- **Pipeline**: `src/pipeline/refresh.py` - CLI for daily predictions
- **Models**: `models/*.pkl` - Saved model artifacts

## Example: Complete Workflow

```python
# 1. In notebook: Train and export models
# (Run export cell)

# 2. In production script or CLI:
from src.models.predictor import GamePredictor
from src.data import nfl_fetcher

# Load data
schedule = nfl_fetcher.fetch_nfl_schedule(2024)
schedule['game_date'] = pd.to_datetime(schedule['gameday'])

# Predict
predictor = GamePredictor('NFL', 'v1')
game = pd.DataFrame({
    'home_team': ['KC'],
    'away_team': ['BUF'],
    'game_date': ['2025-01-15'],
    'season': [2025]
})

result = predictor.predict(game, schedule)
print(f"{result['away_team']} @ {result['home_team']}")
print(f"Spread: {result['predicted_spread']:.1f}")
print(f"Home Win: {result['home_win_probability']:.1%}")

# 3. Or use CLI:
# python -m src.pipeline.refresh --league NFL --date 2025-01-15 --model-version v1
```

## Notes

- Models are versioned (e.g., `v1`, `v2`) for easy rollback
- Feature contract is stable - new features require model retraining
- Missing features are filled with medians from training data
- Form features are optional (work without PBP data, just less accurate)

