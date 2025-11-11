<!-- cbdcab3c-4710-4aae-bde6-04335e722a7e 96e5220a-67cc-470b-8102-de58525840bd -->
# Sports-Edge: Complete Analysis & Display Implementation Plan

## Overview

Build an end-to-end system to analyze NFL/NBA games, compute model spreads vs sportsbook lines, and display results interactively on the personal portfolio. The system consists of three main parts: (1) Python analysis pipeline, (2) Supabase database & export, (3) Next.js display integration.

## Phase 1: Python Analysis Pipeline Setup

### 1.1 Project Structure (`sports-edge/`)

```
sports-edge/
├── notebooks/
│   ├── nfl_eda.ipynb
│   └── nba_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nfl_fetcher.py      # nflreadpy/nfl_data_py integration
│   │   ├── nba_fetcher.py      # nba_api integration
│   │   └── odds_fetcher.py     # The Odds API integration
│   ├── features/
│   │   ├── __init__.py
│   │   ├── rest_schedule.py    # Rest days, B2B, travel distance
│   │   ├── form_metrics.py     # Rolling net rating (NBA), EPA (NFL)
│   │   └── strength.py         # Opponent strength adjustments
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spread_model.py     # Regression for point spread
│   │   ├── win_prob_model.py   # Logistic regression for win prob
│   │   └── link_function.py    # Spread ↔ win% conversion
│   └── pipeline/
│       ├── __init__.py
│       └── refresh.py          # CLI: python -m src.pipeline.refresh
├── data/
│   ├── raw/
│   │   ├── nfl/
│   │   └── nba/
│   └── curated/
│       ├── nfl/
│       └── nba/
├── models/                      # Saved model artifacts (.pkl)
├── requirements.txt
├── .env.example
└── README.md
```

### 1.2 Initial Setup

- Create `requirements.txt` with: `nflreadpy`, `nba_api`, `requests`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `python-dotenv`, `supabase`
- Create `.env.example` with placeholders for `ODDS_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
- Set up data directory structure with `.gitkeep` files

### 1.3 EDA Notebooks

- `notebooks/nfl_eda.ipynb`: Load historical NFL data, explore distributions, compute EPA metrics, analyze rest/schedule features, check correlations
- `notebooks/nba_eda.ipynb`: Load NBA game logs, compute net ratings, analyze pace/totals, rest features, travel distance (Haversine)

### 1.4 Feature Engineering

- Implement feature contract from analysis plan (rest days, B2B flags, travel distance, rolling form metrics, opponent strength)
- Ensure no data leakage (all features pre-game only)
- Cache features to `/data/curated/{league}/YYYY-MM-DD/`

### 1.5 Model Development

- Start with simple models: Ridge regression for spread, LogisticRegression for win probability
- Train on rolling 2-3 seasons with time-based validation split
- Save model artifacts with version tags (e.g., `spread_model_v0.1.0.pkl`)
- Implement spread ↔ win% link function

## Phase 2: Supabase Database Setup

### 2.1 Schema Migration

Create `sports-edge/sql/001_initial_schema.sql`:

- Tables: `games`, `odds_snapshots`, `model_predictions`, `features`, `model_runs`
- View: `games_today_enriched` (latest odds + predictions per game)
- RLS policies: public read-only for all tables

### 2.2 Migration Execution

- Option A: Run SQL directly in Supabase dashboard SQL editor
- Option B: Create migration script that connects and executes (for CI/CD later)

## Phase 3: Data Export Pipeline

### 3.1 CLI Module (`src/pipeline/refresh.py`)

Implement `python -m src.pipeline.refresh --league NFL --date 2025-11-06`:

1. Fetch schedule for date (league-specific)
2. Fetch latest odds from The Odds API
3. Build features using feature contract
4. Load model artifacts and run inference
5. Upsert to Supabase: `games`, `odds_snapshots`, `model_predictions`, `features`
6. Write audit row to `model_runs`

### 3.2 Idempotency

- Use composite keys for upserts: `(league, season, home_team, away_team, game_time_utc)` for games
- Handle duplicate odds/predictions gracefully

### 3.3 Error Handling

- Log errors to `model_runs.error_text`
- Set `success=false` on failures
- Continue processing other games if one fails

## Phase 4: Next.js Integration

### 4.1 API Route (`personal-portfolio/app/api/sports-edges/route.ts`)

- GET endpoint that queries `games_today_enriched` view
- Use Supabase service role key (server-only)
- Add cache headers: `Cache-Control: public, s-maxage=60, stale-while-revalidate=120`
- Return JSON array of games with spreads, win probabilities, edge calculations

### 4.2 Display Component (`personal-portfolio/components/SportsEdgeCard.tsx`)

- Client component that fetches from `/api/sports-edges`
- Display games in grid layout (similar to DailyBiasDisplay pattern)
- Show: league, teams, book spread vs our spread, edge points, home win probability
- Highlight significant edges (e.g., ≥1.0 point difference)
- Show last updated timestamp and model version
- Handle loading/error states gracefully

### 4.3 Integration into WorkSection

- Add new project entry to `mockProjects` array in `WorkSection.tsx`
- Create interactive display similar to DailyBiasDisplay pattern
- Use DeviceFrameset for consistent presentation
- Add project description highlighting ML modeling and real-time odds comparison

## Phase 5: Scheduling & Automation

### 5.1 Local Testing

- Run refresh CLI manually for a test date
- Verify data appears in Supabase
- Test API route and display component

### 5.2 Automation Options

- **Option A**: GitHub Actions cron job (runs every 15 minutes during game days)
- **Option B**: Supabase Edge Function with pg_cron
- Store secrets in GitHub repository secrets or Supabase secrets

## Implementation Order

1. Set up Python project structure and dependencies
2. Create EDA notebooks to explore data
3. Implement feature engineering modules
4. Build and train initial models
5. Create Supabase schema and run migrations
6. Implement export pipeline CLI
7. Create Next.js API route
8. Build display component
9. Integrate into WorkSection
10. Set up automation/scheduling

## Key Files to Create/Modify

- `sports-edge/requirements.txt` (new)
- `sports-edge/src/pipeline/refresh.py` (new)
- `sports-edge/sql/001_initial_schema.sql` (new)
- `personal-portfolio/app/api/sports-edges/route.ts` (new)
- `personal-portfolio/components/SportsEdgeCard.tsx` (new)
- `personal-portfolio/components/WorkSection.tsx` (modify - add project entry)

## Assumptions

- Starting with both NFL and NBA (can prioritize one if needed)
- Using The Odds API (need API key)
- Supabase already configured in personal-portfolio
- Python 3.9+ environment available