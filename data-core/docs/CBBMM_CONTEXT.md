# CBB March Madness Module

## Architecture

```
data-core/
├── data/cbb/                      # Kaggle MMLM CSVs go here
├── src/data/
│   ├── cbb_data_loader.py         # Barttorvik API + Kaggle CSV loader
│   └── cbb_feature_engineering.py # Pairwise matchup feature store builder
├── src/models/
│   ├── cbb_train_matchup_model.py # LGBM/XGB ensemble, expanding window CV
│   └── cbb_bracket_simulator.py   # Monte Carlo bracket sim with constraints
├── notebooks/
│   └── cbb_march_madness.ipynb    # EDA, training, backtesting, simulation
└── models/saved/                  # Trained model artifacts

web/
├── app/cbb/page.tsx               # CBB page with client-side simulation
└── components/
    ├── BracketView.tsx            # Interactive bracket (bracket + table view)
    ├── SimulationControls.tsx     # Constraint builder, presets, sim runner
    └── TeamCard.tsx               # Team detail card with advancement probs
```

## Data Pipeline

1. Place Kaggle MMLM files in `data-core/data/cbb/`
2. Run `cbb_data_loader.py` to compute season stats
3. Run `cbb_feature_engineering.py` to build pairwise matchup store
4. Run `cbb_train_matchup_model.py` for expanding window CV + final model
5. Run notebook for full analysis and bracket simulation

## Web UI

The CBB page (`/cbb`) runs Monte Carlo simulations entirely client-side:
- Seed-based probability matrix (logistic: P = 1/(1+exp(-0.15*seed_diff)))
- Supports constraints: eliminate teams, force advances
- Preset scenarios: all #1 seeds lose, chalk bracket, etc.

## Key Dependencies

- `lightgbm`, `xgboost`: Base matchup models
- `torch`: GPU-accelerated simulation (optional)
- `beautifulsoup4`: Barttorvik HTML scraping
- `cbbpy`: Supplemental CBB data
