"""
Win probability model (classification).
Predicts home team win probability.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import pickle
import os
from typing import Optional


class WinProbModel:
    """
    Model to predict home team win probability (0-1).
    """
    
    def __init__(self, model_type: str = 'logistic', **kwargs):
        """
        Initialize win probability model.
        
        Args:
            model_type: 'logistic', 'rf', or 'lightgbm'
            **kwargs: Model-specific hyperparameters
        """
        self.model_type = model_type
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=42
            )
        elif model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
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
            y: Target (1 if home wins, 0 if away wins)
        """
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict win probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of shape (n_samples, 2) with [P(away wins), P(home wins)]
        """
        if self.feature_names is None:
            raise ValueError("Model not trained yet")
        
        # Ensure columns match
        X_aligned = X[[col for col in self.feature_names if col in X.columns]]
        return self.model.predict_proba(X_aligned.values)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary outcomes.
        
        Args:
            X: Feature matrix
        
        Returns:
            Binary predictions (1 = home wins, 0 = away wins)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
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
    def load(cls, filepath: str) -> 'WinProbModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
        
        Returns:
            Loaded WinProbModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        
        return instance

