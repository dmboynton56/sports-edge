"""
Spread prediction model (regression).
Predicts point spread (home margin).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import pickle
import os
from typing import Optional, Dict


class SpreadModel:
    """
    Model to predict point spread (home team margin).
    Positive values mean home team is favored.
    """
    
    def __init__(self, model_type: str = 'ridge', **kwargs):
        """
        Initialize spread model.
        
        Args:
            model_type: 'ridge', 'rf', or 'lightgbm'
            **kwargs: Model-specific hyperparameters
        """
        self.model_type = model_type
        
        if model_type == 'ridge':
            self.model = Ridge(alpha=kwargs.get('alpha', 1.0))
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 5),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target (actual home margin)
        """
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict spreads.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted spreads
        """
        if self.feature_names is None:
            raise ValueError("Model not trained yet")
        
        # Ensure columns match
        X_aligned = X[[col for col in self.feature_names if col in X.columns]]
        return self.model.predict(X_aligned.values)
    
    def save(self, filepath: str, version: str = 'v0.1.0'):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
            version: Model version string
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'version': version
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SpreadModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        
        Returns:
            Loaded SpreadModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        
        return instance

