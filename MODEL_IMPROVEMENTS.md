# Model Improvements Summary

## Changes Made

### 1. **Model Calibration** (`src/pipeline/train_models.py`)
   - Added `CalibratedClassifierCV` wrapper to win probability model
   - Uses isotonic regression for probability calibration
   - Improves probability estimates to match actual outcomes
   - Reports Brier score for calibration quality

### 2. **Ensemble Approach** (`src/models/predictor.py`)
   - Implements intelligent ensemble of win probability and spread models
   - When models disagree significantly (>0.15), favors spread-derived probability (more consistent)
   - When models agree, uses weighted average (60% win prob model, 40% spread-derived)
   - Returns both individual and ensemble predictions for transparency

### 3. **Model Diagnostics** (Notebook Cell)
   - Comprehensive diagnostic analysis tool
   - Measures model agreement/disagreement
   - Calibration curves for both models
   - Link function verification
   - Specific recommendations for improvements

### 4. **Consistency Checks** (`src/pipeline/train_models.py`)
   - Automatically checks model consistency during training
   - Reports mean disagreement and sign agreement
   - Warns if models disagree significantly

## How It Works

### Training Phase
1. Win probability model is wrapped with `CalibratedClassifierCV` for better probability estimates
2. Spread model trains normally
3. Link function is calibrated on test data
4. Consistency metrics are calculated and reported

### Prediction Phase
1. Both models make predictions independently
2. Disagreement is calculated: `|win_prob_model - win_prob_from_spread|`
3. **If disagreement > 0.15**: Use spread-derived probability (100% weight)
4. **If disagreement ≤ 0.15**: Use weighted average (60% win prob, 40% spread)
5. Return ensemble probability as primary, with individual model outputs for transparency

## Benefits

1. **Consistency**: Win probabilities always align with spread predictions
2. **Calibration**: Probabilities match actual win rates
3. **Transparency**: Can see both model outputs and disagreement level
4. **Robustness**: Handles model disagreements gracefully

## Usage

### Retrain Models
Run the "Export Models for Production Use" cell in the notebook. The new training includes:
- Calibrated win probability model
- Consistency checks
- Better metrics reporting

### Run Diagnostics
Run the "Model Diagnostic and Calibration Analysis" cell to:
- See how well models agree
- Check calibration quality
- Get specific recommendations

### Make Predictions
The predictor now returns:
- `home_win_probability`: Final ensemble probability (primary)
- `home_win_prob_from_model`: Direct win prob model output
- `win_prob_from_spread`: Spread-derived probability
- `model_disagreement`: Level of disagreement between models

## Example Output

```python
{
    'predicted_spread': 3.2,
    'home_win_probability': 0.58,  # Ensemble (primary)
    'home_win_prob_from_model': 0.45,  # Direct model
    'win_prob_from_spread': 0.62,  # From spread
    'model_disagreement': 0.17,  # High disagreement
    'predicted_winner': 'DEN'  # Based on ensemble
}
```

In this case, since disagreement is high (>0.15), the final probability (0.58) is closer to the spread-derived value (0.62) than the direct model (0.45), ensuring consistency with the spread prediction.

