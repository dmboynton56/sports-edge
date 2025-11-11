# Sports-Edge: NFL/NBA Betting Analysis Pipeline

A machine learning pipeline to compute model spreads and home win probabilities for NFL/NBA games, compare against sportsbook lines, and display results on a personal portfolio.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

3. Run EDA notebooks to explore data:
```bash
jupyter notebook notebooks/
```

## Usage

### Run Analysis Pipeline

```bash
python -m src.pipeline.refresh --league NFL --date 2025-11-06
python -m src.pipeline.refresh --league NBA --date 2025-11-06
```

### Train Models

Models are trained via notebooks or separate training scripts. Saved artifacts go in `models/` directory.

## Project Structure

- `notebooks/` - Jupyter notebooks for EDA and exploration
- `src/` - Source code modules
  - `data/` - Data fetching modules (NFL, NBA, odds)
  - `features/` - Feature engineering modules
  - `models/` - Model training and inference
  - `pipeline/` - CLI and orchestration
- `data/` - Raw and curated data (gitignored)
- `models/` - Saved model artifacts (gitignored)
- `sql/` - Database migration scripts

## Data Sources

- **NFL**: `nfl_data_py`
- **NBA**: `nba_api`
- **Odds**: The Odds API
