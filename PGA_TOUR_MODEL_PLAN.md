# PGA Tour Predictive Model & Betting Strategy Plan

## 1. Overview and Architecture
This document outlines the architecture and workflow for building a robust, live-updating PGA Tour predictive model. The system leverages historical data for baseline training and integrates live tournament data to re-evaluate player probabilities (e.g., after the cut).

The infrastructure is designed to utilize an RTX 5070 for intensive feature engineering, model training, and hyperparameter tuning, while relying on BigQuery and Supabase for data warehousing and application state.

## 2. Data Acquisition Strategy

To avoid stale data and missing features, we use a hybrid data sourcing approach:

### 2.1 Historical Foundation (Pre-training)
*   **Source:** Kaggle Datasets (e.g., "PGA Tour Golf Data (2015-2022/2024)", "PGA Tour Results").
*   **Purpose:** Deep learning pre-training and baseline model evaluation.
*   **Key Metrics:** Historical Strokes Gained (SG) metrics (Off-the-Tee, Approach, Around-the-Green, Putting).
*   **Integration:** Download CSVs, clean, and load into the BigQuery "Bronze/Silver" layers for historical modeling.

### 2.2 The Live Engine (In-Tournament Updates)
*   **Source:** Data Golf API.
*   **Purpose:** The gold standard for live Strokes-Gained data and probabilistic forecasts.
*   **Key Metrics:** Live SG: Putting, SG: Approach, baseline true skill.
*   **Integration:** Ingested into the BigQuery "Gold Layer". Used to refresh input features after Round 2 (the cut) for weekend predictions.

### 2.3 The Quick Scraper (Leaderboard Backup)
*   **Source:** Unofficial ESPN API.
*   **Purpose:** Lightweight, real-time leaderboard status ("Current Position", "Thru").
*   **Integration:** Quick verification of final/live scores to update the Supabase application database without hitting rate limits on premium APIs.

## 3. Feature Engineering

The core of the predictive edge lies in advanced feature engineering rather than just looking at raw scores.

*   **Regression to the Mean:** Identify players whose current performance is driven by unsustainable variance (e.g., SG: Putting > +3.0). The model should predict regression over the weekend.
*   **Expected Strokes Gained:** Calculate "Expected SG" for upcoming rounds based on a weighted average of long-term baseline skill and short-term form (recent rounds).
*   **Course Fit:** Interaction features between player historical SG profiles (e.g., "Bomber" vs. "Accuracy") and the specific course characteristics (e.g., driving distance importance, green size).
*   **Weather/Draw Bias:** Adjusting expectations based on tee times (AM/PM wave splits) and forecasted wind/weather.

## 4. Modeling Strategy: Multiple Targets and Markets

Instead of training a single "Winner" model, we will train specific models tailored to different prediction markets and outcomes.

### 4.1 Regression Model (The Core Engine)
*   **Target:** Predict a player's *Expected Strokes Gained* or *Expected Score* for the next round.
*   **Why:** This is the most stable metric. By predicting round-level performance, we can simulate the remainder of the tournament.
*   **Algorithm:** LightGBM / XGBoost for tabular data; PyTorch Neural Networks (leveraging the RTX 5070) for complex feature interactions and embeddings.

### 4.2 Classification Models (Placement Markets)
*   **Targets:** Make Cut (Binary), Top 20 (Binary), Top 10 (Binary), Top 5 (Binary).
*   **Why:** These are highly liquid betting markets and great for DFS (DraftKings/FanDuel) optimization.
*   **Algorithm:** Logistic Regression (for interpretable baselines), Gradient Boosting Classifiers.

### 4.3 Simulation Engine (Win Probability & Derivatives)
*   **Method:** Monte Carlo Simulations.
*   **Process:** Use the output from the Regression Model (Expected Score + Variance) to simulate the remaining rounds 10,000+ times.
*   **Output:** Calculate the exact probability of winning, placing Top X, or winning a specific Head-to-Head matchup based on the simulation distributions.

## 5. Development Workflow & Implementation Steps

1.  **Environment Setup:** Ensure Python `dotenv`, `google-cloud-bigquery`, `supabase`, `scikit-learn`, `lightgbm`, and `torch` (with CUDA support for the RTX 5070) are configured.
2.  **Data Ingestion Pipeline (The Wrapper):**
    *   Write a Python wrapper class (`PGADataloader`) that seamlessly toggles between reading Kaggle CSVs (local/BigQuery) for backtesting and hitting the Data Golf/ESPN APIs for live inference.
3.  **Historical Backtesting:**
    *   Train the base models on 2010–2024 data. Evaluate performance on predicting Top-20s and Outright Winners using cross-validation (grouped by tournament to prevent data leakage).
4.  **Live Inference Pipeline:**
    *   Set up a script triggered after Round 2 (Friday night). It pulls live Data Golf stats, updates the player feature vectors, runs the Regression Model, and outputs updated probabilities for the weekend.
5.  **Data Export:**
    *   Push final predictions and value bets (model probability vs. implied sportsbook odds) to the Supabase database for front-end consumption or alerts.
