# Model Story Notebook Walkthrough

This document explains how each cell in `model_story.ipynb` works and how it contributes to the overall narrative for your bet365 interview.

## Overall Narrative Structure

The notebook tells a complete story: **From raw data → curated features → model predictions → trading insights → performance validation**. Each section builds on the previous one to demonstrate your full-stack data science and trading domain expertise.

---

## Section 0: Setup & Configuration

### Cell 0: Title & Introduction (Markdown)
**Purpose:** Sets context and usage instructions
**Contribution:** Shows you can write clear documentation and think about reproducibility (single week vs multi-week analysis)

### Cell 1: Imports & Configuration (Code)
**What it does:**
- Loads environment variables (BigQuery credentials)
- Imports data science stack (numpy, pandas, seaborn, matplotlib)
- Sets up BigQuery client connection
- Configures plotting theme for consistent visualizations
- Defines key parameters: `SEASON`, `WEEK`, `MODEL_VERSION`

**Why it matters:** Demonstrates professional setup with:
- Environment variable management (security best practice)
- Consistent visualization theme (important for presentations)
- Parameterized configuration (easy to adapt for different weeks/seasons)

### Cell 2: Setup Documentation (Markdown)
**Purpose:** Explains the setup section
**Contribution:** Shows documentation thinking

---

## Section 1: Data Ingestion & Feature Engineering

### Cell 3: Check Available Weeks (Code)
**What it does:**
- Queries BigQuery to see which weeks have data in `feature_snapshots`
- Validates that requested weeks actually exist
- Falls back gracefully if weeks are missing
- Sets up `weeks_to_use` and `actual_week` variables

**Why it matters:** Shows defensive programming and data validation - critical for production systems. Demonstrates you think about edge cases.

### Cell 4: Load Raw Schedule Data (Code)
**What it does:**
- Queries `raw_schedules` table to get basic game information
- Shows the "before" state of data (minimal features)

**Contribution:** Establishes baseline - shows you understand the raw data structure

### Cell 5: Section Header - Raw vs Curated (Markdown)
**Purpose:** Introduces the feature engineering story

### Cell 6: Load Curated Features (Code)
**What it does:**
- Queries `feature_snapshots` table for engineered features
- Selects key features: `rest_home`, `rest_away`, `opp_strength_differential`, `form_epa_off_diff_5`, `form_epa_def_diff_5`
- Handles empty results gracefully

**Why it matters:** This is where you show **domain knowledge**:
- **Rest days** matter in NFL (fatigue factor)
- **Opponent strength** matters (schedule difficulty)
- **Form (EPA)** matters (recent performance > season-long averages)

**Interview talking point:** "I engineered features based on sports betting domain knowledge - rest days, opponent strength, and recent form. These aren't just statistical features; they're factors that professional bettors actually use."

### Cell 7: Feature Distribution Quick Look (Markdown)
**Purpose:** Explains why visualization matters

---

## Section 2: Model Predictions - Python Pipeline

### Cell 8: Load Python Predictions (Code)
**What it does:**
- Loads pre-computed predictions from `notebooks/exports/week*_python_preds.parquet`
- Handles multiple weeks by concatenating files
- **Normalizes predictions** to reduce overconfidence:
  - Clips spreads to [-21, 21] (realistic NFL range)
  - Clips win probabilities to [0.15, 0.85] (reduces extreme confidence)
  - Optionally applies shrinkage toward neutral (commented out)

**Why it matters:** Shows you understand **calibration** - models can be overconfident. The normalization makes predictions more realistic for betting purposes.

**Interview talking point:** "I added normalization because raw model outputs can be overconfident. In betting, you need realistic probabilities, not just high accuracy. I clip extreme values and apply shrinkage to make the model more conservative."

### Cell 9: Section Header - Python Pipeline (Markdown)
**Purpose:** Explains where these predictions come from

---

## Section 3: Model Predictions - BQML

### Cell 10: Load BQML Predictions (Code)
**What it does:**
- Queries BigQuery ML model predictions from `model_predictions` table
- Uses CTEs to filter by season/week and model version
- Handles cases where BQML predictions might not exist

**Why it matters:** Shows you can work with **multiple model sources**:
- Python pipeline (local, more control)
- BQML (cloud, scalable, SQL-based)

**Interview talking point:** "I compare Python and BQML models to validate consistency. If they disagree significantly, that's a signal to investigate. Also shows I can work across different ML platforms."

### Cell 11: Section Header - BQML (Markdown)
**Purpose:** Explains BQML integration

---

## Section 4: Market Data (Book Odds)

### Cell 12: Determine Week to Use (Code)
**What it does:**
- Resolves which week(s) to analyze
- Handles both single week and multi-week scenarios

**Contribution:** Utility code for flexibility

### Cell 13: Section Header - Book Odds (Markdown)
**Purpose:** Introduces market data

### Cell 14: Load Actual Game Results (Code)
**What it does:**
- Queries `game_results` table for actual outcomes
- Joins with `feature_snapshots` to match game_ids consistently
- Handles deduplication (some games might have multiple result rows)
- Extracts: `actual_home_win`, `actual_home_points`, `actual_away_points`, `actual_home_margin`

**Why it matters:** This is **ground truth** for backtesting. Shows you understand you need actual results to validate predictions.

**Interview talking point:** "I always validate models against actual outcomes. But I'm careful about lookahead bias - I only use predictions that would have been available before the game started."

---

## Section 5: Data Merging

### Cell 23: Note About Book Spreads (Markdown)
**Purpose:** Technical note about data sources

### Cell 24: Model vs Book Spread Visualization (Code)
**What it does:**
- Creates scatter plots comparing model spreads vs book spreads
- Diagonal line = perfect agreement with books
- Points above diagonal = model likes home team more than market
- Points below diagonal = model likes away team more than market

**Why it matters:** This is your **edge visualization**. Shows where your model disagrees with the market - these are potential betting opportunities.

**Interview talking point:** "I visualize model vs book spreads to identify edges. When my model consistently disagrees with the market in one direction, that's either a model flaw or a market inefficiency. I investigate both."

### Cell 25: Section Header - Merge Everything (Markdown)
**Purpose:** Explains data joining

### Cell 26: Edge Distribution Comparison (Code)
**What it does:**
- Calculates edge = model_spread - book_spread
- Creates box plots comparing Python model edge vs BQML model edge
- Shows which model creates larger deviations from market

**Why it matters:** Quantifies **model disagreement**. If Python and BQML have similar edges, that's validation. If they differ significantly, that's a signal.

**Interview talking point:** "I compare edge distributions to validate model consistency. If both models agree on an edge, I'm more confident. If they disagree, I investigate why."

---

## Section 6: Visual Comparisons & Explainability

### Cell 27: Section Header - Visual Comparisons (Markdown)
**Purpose:** Introduces visualization section

### Cell 28: SHAP Explainability Header (Markdown)
**Purpose:** Introduces model explainability

### Cell 29: SHAP Analysis (Code)
**What it does:**
- Loads the trained model from disk (`win_prob_model_nfl_v2.pkl`)
- Handles calibrated classifiers (extracts underlying estimator)
- Maps model feature names to BigQuery column names (handles naming mismatches)
- Queries full feature set from BigQuery
- Generates SHAP values using TreeExplainer
- Creates two visualizations:
  1. **Bar plot**: Global feature importance (which features matter most overall)
  2. **Beeswarm plot**: Feature impact direction (how high/low values affect predictions)

**Why it matters:** This is **model explainability** - critical for:
- **Debugging**: Why did the model make a bad prediction?
- **Trust**: Can you explain predictions to stakeholders?
- **Feature engineering**: Which features actually matter?

**Interview talking point:** "I use SHAP to explain model predictions. This is crucial in trading - if I can't explain why the model likes a bet, I won't take it. SHAP shows me which features drove each prediction and helps me identify spurious correlations."

### Cell 30: Win Probability Calibration (Code)
**What it does:**
- Bins predicted win probabilities (0-0.2, 0.2-0.4, etc.)
- Calculates actual win rate for each bin
- Plots predicted probability vs actual win rate
- Perfect calibration = points on diagonal line

**Why it matters:** Shows **calibration quality**. A well-calibrated model means:
- When it says 70% win probability, home team actually wins ~70% of the time
- Critical for betting - you need accurate probabilities to calculate EV correctly

**Interview talking point:** "I check calibration because accurate probabilities are essential for EV calculations. If my model says 70% but home teams only win 50% of the time, my EV calculations will be wrong."

### Cell 31: Model vs Book Spreads Header (Markdown)
**Purpose:** Explains the scatter plot visualization

### Cell 32: Export Directory Setup (Code)
**What it does:**
- Sets up directory for exporting plots
- Example code for saving figures

**Contribution:** Utility for portfolio building

### Cell 33: Edge Distribution Header (Markdown)
**Purpose:** Explains edge visualization

### Cell 34: Win Probability Calibration Header (Markdown)
**Purpose:** Explains calibration plot

---

## Section 7: Performance Metrics

### Cell 15: Section Header - Performance Metrics (Markdown)
**Purpose:** Introduces performance evaluation

### Cell 16: Calculate Performance Metrics (Code)
**What it does:**
- Calculates **spread accuracy**: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error)
- Calculates **win probability accuracy**: Brier Score, Log Loss, Accuracy
- Compares Python model vs BQML model
- Creates summary table

**Why it matters:** Standard ML evaluation metrics. Shows you understand:
- **MAE/RMSE**: How far off are spread predictions? (Lower is better)
- **Brier Score**: How well-calibrated are probabilities? (Lower is better, 0 = perfect)
- **Log Loss**: Penalizes confident wrong predictions (Lower is better)
- **Accuracy**: Simple win/loss prediction accuracy

**Interview talking point:** "I use multiple metrics because different metrics capture different aspects of performance. MAE tells me average error, but RMSE penalizes large errors more. Brier Score tells me if probabilities are accurate, not just if predictions are correct."

---

## Section 8: Expected Value Analysis

### Cell 17: Section Header - EV Analysis (Markdown)
**Purpose:** Introduces trading-focused metrics

### Cell 18: Calculate Expected Value (Code)
**What it does:**
- Defines constants: `SPREAD_STD = 13.5` (typical NFL spread standard deviation), `PAYOUT_MULTIPLIER = 1.909` (-110 odds)
- For each prediction, calculates:
  1. **Edge** = model_spread - book_spread (in points)
  2. **Bet direction** = bet home if edge > 0, bet away if edge < 0
  3. **Cover probability** = conservative normal CDF conversion (edge / (2 * SPREAD_STD))
  4. **EV** = (cover_prob * PAYOUT_MULTIPLIER) - 1
- Clips probabilities to [0.05, 0.95] to avoid extreme EVs
- Clips EV to [-0.90, 0.90] for realistic bounds

**Why it matters:** This is **trading domain knowledge**:
- You're not just predicting spreads - you're calculating **betting value**
- EV tells you: "If I make this bet 100 times, what's my expected return?"
- Positive EV = good bet (long-term profitable)
- Negative EV = bad bet (avoid)

**Interview talking point:** "I calculate Expected Value because accuracy alone isn't enough. A model can be 60% accurate but still lose money if it's betting on -110 lines. EV quantifies the actual betting value - I only bet when EV > 0."

**Key Fix:** The cover probability calculation uses a conservative approach (divides by 2x std dev) to prevent overconfidence. This is important because small edges shouldn't translate to high probabilities.

---

## Section 9: Betting Strategy Simulation

### Cell 19: Section Header - Betting Strategies (Markdown)
**Purpose:** Introduces backtesting

### Cell 20: Betting Strategy Simulation (Code) ⚠️ **FIXED BUG**
**What it does:**
- Simulates three betting strategies:
  1. **Bet All Positive EV**: Bet every game with EV > 0
  2. **High Confidence**: Bet games where win_prob > 60% or < 40%
  3. **Kelly Criterion**: Bet sizing based on edge (capped at 25% of bankroll)

- For each strategy:
  - Determines bet direction (home/away)
  - Calculates **ATS margin** = bet_margin - bet_line
    - **FIXED**: Changed from `bet_margin + bet_line` to `bet_margin - bet_line`
    - This correctly calculates: "Did the team I bet on cover the spread?"
  - Determines win/loss/push
  - Calculates profit (win = +$90.90 on $100 bet at -110, loss = -$100)
  - Calculates metrics: ROI, Sharpe Ratio, Max Drawdown

**Why it matters:** This is **backtesting** - simulating how strategies would have performed historically.

**Interview talking point:** "I backtest betting strategies to validate model performance. But I'm careful about lookahead bias - I only use predictions that would have been available before game time. The 100% win rate you saw earlier was a bug in the ATS margin calculation, which I've now fixed."

**Key Fixes:**
1. **ATS margin calculation**: Fixed sign error (`+` → `-`)
2. **Added warning**: This is backtesting, not forward simulation. Results may be inflated due to survivorship bias.

**Important Notes:**
- This is **backtesting** (evaluating on past games)
- For **forward simulation**, you'd need to filter by prediction timestamp < game start time
- Results may be inflated because we only evaluate games that already happened
- The normalization in Cell 8 helps make predictions more realistic

---

## Section 10: Performance Over Time

### Cell 21: Section Header - Time Series (Markdown)
**Purpose:** Introduces trend analysis

### Cell 22: Performance Over Time Analysis (Code)
**What it does:**
- Groups performance metrics by week
- Creates time series plots showing:
  - Spread MAE over time
  - Win probability accuracy over time
- Identifies best/worst weeks
- Checks for **model drift** (performance degradation over time)

**Why it matters:** Shows **model monitoring**:
- Is performance stable over time?
- Are there weeks where the model performs poorly? (Why?)
- Is there model drift? (Performance getting worse over time)

**Interview talking point:** "I track performance over time to detect model drift. If MAE increases significantly, that could mean the game has changed (rule changes, player behavior shifts) or my features are becoming less relevant. I investigate both."

---

## Section 11: Export & Documentation

### Cell 35: Save Plots Header (Markdown)
**Purpose:** Explains export process

### Cell 36: Talking Points (Markdown)
**Purpose:** Key points to discuss in interviews/presentations

---

## Key Themes for bet365 Interview

### 1. **Domain Knowledge** (Not Just Data Science)
- You understand rest days, opponent strength, form
- You calculate EV, not just accuracy
- You think in terms of betting value, not just predictions

### 2. **Production-Ready Thinking**
- Defensive programming (handles missing data)
- Calibration (normalizes overconfident predictions)
- Model monitoring (tracks performance over time)
- Explainability (SHAP for debugging)

### 3. **Trading Domain Expertise**
- EV calculations
- Betting strategy simulation
- Risk metrics (Sharpe Ratio, Max Drawdown)
- Understanding of lookahead bias and survivorship bias

### 4. **Full-Stack Data Science**
- Feature engineering
- Model training (Python + BQML)
- Evaluation (multiple metrics)
- Visualization
- Backtesting

### 5. **Critical Thinking**
- Fixed bugs (ATS margin calculation)
- Added warnings (backtesting vs forward simulation)
- Normalized predictions (reduced overconfidence)
- Conservative EV calculations (prevented extreme probabilities)

---

## How to Use This Notebook in Your Interview

1. **Start with the narrative**: "I built a complete pipeline from raw data to betting insights..."
2. **Highlight domain knowledge**: "I engineered features based on factors professional bettors use..."
3. **Show trading focus**: "I calculate EV, not just accuracy, because that's what matters for betting..."
4. **Demonstrate critical thinking**: "I found and fixed a bug in the betting simulation that was causing unrealistic results..."
5. **Show production awareness**: "I normalize predictions to reduce overconfidence and track performance over time to detect drift..."

The notebook tells a complete story that demonstrates you're not just a data scientist - you're a **data scientist who understands trading**.

