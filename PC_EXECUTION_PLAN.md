# RTX 5070 PC Execution Plan

This document serves as your master checklist and execution guide when you switch to your main PC. All data pipelines and foundational architecture were built on the MacBook; the PC is strictly for heavy ingestion, GPU training, and live tournament simulation.

---

## Phase 1: Environment & Data Sync

1.  **Pull Code & Env Variables**
    *   `git pull` the latest code.
    *   **Crucial:** Copy your `.env` file (containing `DATA_GOLF_API_KEY`) and your `gcp-service-account.json` to the PC.
2.  **GPU Python Environment Setup**
    *   Install the data stack: `pip install pandas numpy google-cloud-bigquery pandas-gbq`
    *   Install the GPU-enabled ML stack:
        ```bash
        pip install xgboost lightgbm scikit-learn
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

---

## Phase 2: Final Data Ingestion (Optional but Recommended)

If you haven't already pushed the massive historical files to BigQuery from the Mac, let the PC do it fast:
1. Open `src/data/ingest_to_bigquery.py`.
2. Uncomment the `ingest_kaggle_historical_to_bq("src/data/archive/pga_results_2001-2025.tsv")` line at the bottom.
3. Run `python src/data/ingest_to_bigquery.py`.

---

## Phase 3: Model Training Strategies

Because you have an RTX 5070, you should train an ensemble of models rather than relying on just one. The scripts in `src/models/` handle this.

### Strategy A: The Core Regression Engine (XGBoost/LightGBM)
*   **Goal:** Predict a player's `Expected Strokes Gained` for the next round based on their 50-round baseline and their recent momentum.
*   **Why GPU?** LightGBM and XGBoost natively support CUDA. Training on 150,000+ rows takes minutes instead of hours.
*   **Script to run:** `python src/models/train_models.py`

### Strategy B: Deep Learning for Course Fit (PyTorch)
*   **Goal:** Learn complex, non-linear interactions (e.g., how a player's spin rate and driving accuracy interact with wind conditions and specific grass types).
*   **Why GPU?** PyTorch neural networks require the 5070's Tensor Cores for matrix multiplication.
*   **Execution:** Incorporated in `train_models.py`.

---

## Phase 4: Monte Carlo Simulation (The Betting Edge)

Golf isn't about predicting an exact score; it's about probability distributions. Once the Regression Engine predicts an "Expected Score" and a "Variance" for each player, we run simulations.

*   **Goal:** Simulate the remaining rounds of the tournament 10,000 times to find the true probability of a player winning, finishing Top 20, or making the cut.
*   **Why GPU?** The 5070 can run 10,000 parallel simulations of a 150-man field in less than 5 seconds.
*   **Script to run:** `python src/models/monte_carlo_sim.py`

---

## Phase 5: Live Inference (THE PLAYERS Championship)

When tournament week arrives (or to test on the current PLAYERS tournament):

1.  **Ingest Live Leaderboard:**
    `python src/data/ingest_to_bigquery.py` (Grabs the live ESPN/DataGolf leaderboard).
2.  **Build Current State Dataset:**
    `python src/data/dataset_builder.py` (Pulls the live leaderboard + historical baselines from BigQuery).
3.  **Run Live Inference & Sim:**
    `python src/models/live_inference.py` (Feeds the current state into the trained models, runs the Monte Carlo sim, and spits out live betting probabilities).
