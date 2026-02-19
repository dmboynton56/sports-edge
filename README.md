# Sports-Edge: NFL/NBA Betting Analysis Pipeline

![Build Status](https://github.com/dmboynton56/sports-edge/actions/workflows/test.yml/badge.svg)


A machine learning pipeline to compute model spreads and home win probabilities for NFL/NBA games, compare against sportsbook lines, and display results on a personal portfolio.

The system treats BigQuery as the source of truth for scoring and historical data, while Supabase remains the lightweight cache that the frontend website reads.

## Project Overview

- **Daily Automation**: GitHub Actions update raw data, rebuild features, and generate predictions every morning.
- **Machine Learning**: LightGBM ensembles trained on historical rolling performance metrics, strength of schedule, and situational rest spots.
- **Unified Pipeline**: Standardized approach for both NFL (weekly) and NBA (daily).
- **Data Architecture**: Raw data -> Curated BigQuery features -> Python ML scoring -> Supabase Postgres -> Next.js Frontend.

## Setup

### 1. Prerequisites

- **API Keys**:
  - The Odds API key (for sportsbook lines)
  - Supabase project URL and Service Role key
  - GCP Service Account JSON key (with BigQuery Admin permissions)
- **Python Environment**:
  - Python 3.11+
  - Dependencies: `pip install -r requirements.txt`

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Key Variables:
- `GCP_PROJECT_ID`: Your Google Cloud project ID.
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your GCP service account JSON.
- `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY`: Supabase API access.
- `SUPABASE_DB_*`: PostgreSQL connection details for bulk sync.
- `ODDS_API_KEY`: Key from the-odds-api.com.

### 3. Initialize BigQuery

Create the necessary datasets and tables:
```bash
python gcp_setup/create_bigquery_tables.py --project your-project-id
```

### 4. Set Up Supabase Database

Run the migration scripts in the `sql/` directory within your Supabase SQL editor in order:
1. `001_initial_schema.sql`
2. `002_add_week_column.sql`
3. ... and so on.

## Usage

### Run Production Pipeline Manually

The pipeline handles data fetching, feature building, and prediction in one flow.

**NBA (Daily)**:
```bash
python -m src.pipeline.refresh_nba --project your-project-id --model-version v3 --date 2026-01-14
```
The NBA refresh now fetches current odds from The Odds API (requires `ODDS_API_KEY`) and loads them into `raw_nba_odds` before generating predictions. Use `--skip-odds` to skip this step.

**NFL (Weekly)**:
```bash
python -m src.pipeline.refresh_nfl --project your-project-id --model-version v1
```

**Sync to Supabase**:
After predictions are in BigQuery, push them to the web database:
```bash
python scripts/sync_bq_to_supabase.py --project your-project-id --league NBA --append
```

### Training Models

Models are developed in Jupyter notebooks:
- NBA: `notebooks/nba_eda.ipynb`
- NFL: `notebooks/nfl_eda.ipynb`

Exported model artifacts (`.pkl`) are stored in the `models/` directory and versioned.

## GitHub Actions Automation

The repository includes a daily refresh workflow (`.github/workflows/daily-refresh.yml`) that runs at 13:00 UTC (8:00 AM ET). 

It requires the following GitHub Secrets:
- `GCP_PROJECT_ID`
- `GCP_SERVICE_ACCOUNT_KEY` (Entire JSON content)
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_DB_PASSWORD`
- `SUPABASE_DB_HOST`
- `SUPABASE_DB_USER`
- `ODDS_API_KEY`

## Project Structure

- `src/` - Core logic
  - `data/` - API fetchers (NBA, NFL, Odds)
  - `features/` - Engineered rolling stats and rest calculations
  - `models/` - Production predictor class and link functions
  - `pipeline/` - Production refresh scripts
- `scripts/` - Maintenance and utility scripts (backfills, syncs)
- `sql/` - Database schemas for Supabase and BigQuery
- `notebooks/` - Research and training environments
- `models/` - Production model artifacts (versioned)

## Data Sources

- **NFL**: nflreadpy / nfl_data_py
- **NBA**: nba_api
- **Odds**: The Odds API
