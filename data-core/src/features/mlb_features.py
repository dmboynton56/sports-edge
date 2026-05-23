"""
MLB feature engineering public API.

The implementation currently lives with the MLB winner-model utilities because
the first model and feature contract were built together. New pipelines should
import from this module so later feature refactors do not touch scripts.
"""

from src.models.mlb_winner_model import (
    build_mlb_prediction_features,
    build_mlb_winner_features,
    default_feature_columns,
)

__all__ = [
    "build_mlb_prediction_features",
    "build_mlb_winner_features",
    "default_feature_columns",
]
