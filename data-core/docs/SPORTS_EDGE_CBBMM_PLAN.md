{
  "march_madness_execution_plan": {
    "phase_1_data_engineering_and_sequencing": {
      "objective": "Build a chronologically sound dataset that prevents future data from leaking into past predictions.",
      "steps": [
        {
          "step": "Daily Snapshotting",
          "action": "Calculate rolling averages and efficiency metrics (e.g., Adjusted Offensive Efficiency) for every team, but ONLY using games played up to Day T-1. Never use season-long averages to predict a game in the middle of the season."
        },
        {
          "step": "The '2020 Problem' Handling",
          "action": "Hardcode an exception for the 2020 season (tournament canceled due to COVID-19). Exclude 2020 from tournament backtesting to prevent null-value errors or skewed historical weighting."
        },
        {
          "step": "Pairwise Matrix Generation",
          "action": "For every historical tournament, generate a row for every actual matchup played, AND generate rows for all possible counterfactual matchups (for full bracket simulation testing)."
        }
      ]
    },
    "phase_2_cross_validation_strategy": {
      "objective": "Evaluate model accuracy without violating the arrow of time.",
      "concept": "Expanding Window Time-Series Split",
      "why_standard_k_fold_fails": "Standard K-Fold randomly shuffles data. If you shuffle, your model might train on a team's Elite Eight performance to predict their opening night game. This causes catastrophic target leakage.",
      "execution": [
        "Fold 1: Train on Seasons 2010-2015 -> Validate on Regular Season 2016.",
        "Fold 2: Train on Seasons 2010-2016 -> Validate on Regular Season 2017.",
        "Fold 3: Train on Seasons 2010-2017 -> Validate on Regular Season 2018.",
        "Rule": "The validation set is ALWAYS strictly in the future relative to the training set."
      ]
    },
    "phase_3_historical_tournament_backtesting": {
      "objective": "Simulate past tournaments as if you were living in that exact year on Selection Sunday.",
      "the_blind_test": "To backtest the 2022 tournament, your model must freeze all feature engineering on the exact date of Selection Sunday 2022. It cannot know that Saint Peter's makes an Elite Eight run.",
      "evaluation_steps": [
        {
          "step": "Log Loss Benchmarking",
          "action": "Predict the Win Probability for all 63 games actually played in the historical tournament. Calculate the Log Loss. (The Kaggle 'March Machine Learning Mania' benchmark for a great model is typically a Log Loss between 0.49 and 0.52)."
        },
        {
          "step": "Upset Detection Calibration",
          "action": "Isolate all historical 1-vs-16, 2-vs-15, and 3-vs-14 matchups. If your model assigns a 0% chance to the underdog, it is overfit. It should ideally assign between 1% to 5% to account for historical black swan events (e.g., UMBC, Fairleigh Dickinson)."
        },
        {
          "step": "Bracket Pool Scoring Simulation",
          "action": "Run your Monte Carlo simulator on the past tournament to generate your 'Optimal Bracket'. Score that bracket using standard 10-20-40-80-160-320 scoring against historical public bracket data to see what percentile your model would have finished in."
        }
      ]
    },
    "phase_4_live_execution_and_monte_carlo": {
      "objective": "Generate actionable edges for the current year's tournament using GPU-accelerated simulation.",
      "steps": [
        {
          "step": "The 63-Game Probability Matrix",
          "action": "On Selection Sunday, use your trained LightGBM/PyTorch ensemble to predict the Win Probability for every single possible combination of the 68 teams (2,278 possible unique matchups). Save this as a tensor/matrix."
        },
        {
          "step": "GPU Tensor Simulation",
          "action": "Load the probability matrix into PyTorch on your RTX 5070. Run 10,000+ simulated brackets by generating random numbers against your win probabilities for each round, advancing the winners logically through the bracket structure."
        },
        {
          "step": "Leverage and EV Extraction",
          "action": "Aggregate the 10,000 simulations to find the true odds of each team reaching the Final Four or winning the Championship. Compare these true odds against Vegas Futures and ESPN Public Pick percentages to highlight teams with positive Expected Value (+EV) and maximum bracket leverage."
        }
      ]
    }
  }
}