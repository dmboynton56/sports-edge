"""
Link function to convert between spread and win probability.
Calibrated on historical game margins.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Optional


def logistic_link(spread: float, a: float = 1.0, b: float = 0.0) -> float:
    """
    Convert point spread to win probability using logistic function.
    
    Args:
        spread: Point spread (home margin)
        a: Scaling parameter (default: 1.0, meaning ~1 point = ~2.5% win prob change)
        b: Offset parameter
    
    Returns:
        Win probability (0-1)
    """
    # Standard logistic: P = 1 / (1 + exp(-a * (spread - b)))
    # For spread of 0, win prob should be ~0.5 (home court advantage)
    # For spread of +3, win prob should be ~0.65-0.70
    return 1 / (1 + np.exp(-a * (spread - b)))


def fit_link_function(spread_like_signal: pd.Series, historical_wins: pd.Series,
                      max_abs_a: float = 0.3, min_abs_a: float = 0.05) -> tuple:
    """
    Fit link function parameters from historical data with slope clipping.
    
    Args:
        spread_like_signal: Series of spread predictions or market spreads
        historical_wins: Series of binary outcomes (1 = home wins, 0 = away wins)
        max_abs_a: Maximum absolute slope allowed for stability
        min_abs_a: Minimum absolute slope to avoid flat link
    
    Returns:
        Tuple of (a, b) parameters for logistic_link
    """
    def logistic_func(x, a, b):
        return 1 / (1 + np.exp(-a * (x - b)))
    
    x = np.asarray(spread_like_signal, dtype=float)
    y = np.asarray(historical_wins, dtype=float)
    
    if np.unique(y).size < 2:
        raise ValueError("Need both home and away wins to calibrate link function.")
    
    # Fit using curve_fit
    popt, _ = curve_fit(logistic_func, x, y, p0=[0.15, 2.5], maxfev=10000)
    
    a = float(np.clip(popt[0], -max_abs_a, max_abs_a))
    if abs(a) < min_abs_a:
        a = np.sign(a) * min_abs_a if a != 0 else min_abs_a
    b = float(popt[1])
    
    return a, b


def spread_to_win_prob(spread: float, a: float = 0.15, b: float = 2.5) -> float:
    """
    Convert point spread to win probability.
    
    Default parameters calibrated on NFL/NBA data:
    - Spread of 0 → ~55% win prob (home advantage)
    - Spread of +3 → ~65% win prob
    - Spread of +7 → ~75% win prob
    
    Args:
        spread: Point spread (home margin, positive = home favored)
        a: Scaling parameter (default: 0.15)
        b: Offset parameter (default: 2.5)
    
    Returns:
        Home win probability (0-1)
    """
    return logistic_link(spread, a, b)


def win_prob_to_spread(win_prob: float, a: float = 0.15, b: float = 2.5) -> float:
    """
    Convert win probability to point spread.
    
    Inverse of spread_to_win_prob.
    
    Args:
        win_prob: Home win probability (0-1)
        a: Scaling parameter (default: 0.15)
        b: Offset parameter (default: 2.5)
    
    Returns:
        Point spread (home margin)
    """
    # Inverse logistic: spread = b - (1/a) * ln((1/P) - 1)
    if win_prob <= 0:
        return -np.inf
    if win_prob >= 1:
        return np.inf
    
    spread = b - (1 / a) * np.log((1 / win_prob) - 1)
    return spread
