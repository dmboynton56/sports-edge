import pandas as pd
import numpy as np
import os
import joblib

# Optional GPU imports - wrapped in try/except so it doesn't crash on the Mac
try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    has_ml_libs = True
except ImportError:
    has_ml_libs = False
    print("Warning: ML libraries (lightgbm, xgboost, sklearn) not installed. Models will not train.")

class PGAModelTrainer:
    """
    Trains GPU-accelerated regression models to predict Expected Strokes Gained
    for the next round.
    """
    def __init__(self, data_path: str = "data/processed/training_dataset.csv"):
        self.data_path = data_path
        self.models_dir = "models/saved/"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_data(self):
        """Loads and prepares the dataset for training."""
        if not os.path.exists(self.data_path):
            print(f"Error: Training data not found at {self.data_path}")
            return None, None
            
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # We need a target variable. Let's assume we want to predict 'sg_total' 
        # based on historical features. In a real time-series setup, you'd shift this.
        # For this script, we'll assume the dataset builder has created a 'target_sg_next_round'
        # If not, we create a dummy target for the sake of the architecture script.
        
        if 'target_sg_next_round' not in df.columns:
            print("Warning: 'target_sg_next_round' not found. Creating a synthetic target for architecture testing.")
            # Synthetic target: Current SG + some random noise
            df['target_sg_next_round'] = df.get('sg_total', 0) * 0.8 + np.random.normal(0, 1, len(df))
            
        # Define features
        features = [col for col in df.columns if col not in ['target_sg_next_round', 'player_name', 'date', 'tournament_id', 'course_id']]
        
        # Drop rows with NaN targets or fill NaN features
        df = df.dropna(subset=['target_sg_next_round'])
        df = df.fillna(0) # Simple imputation for now
        
        X = df[features]
        y = df['target_sg_next_round']
        
        return X, y

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Trains a LightGBM model utilizing the GPU if available."""
        if not has_ml_libs:
            return None
            
        print("\n--- Training LightGBM Regressor ---")
        
        # Configure for GPU (This will seamlessly fall back to CPU on Mac if configured properly, 
        # but is intended for the RTX 5070)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            # 'device': 'gpu', # Uncomment on the PC
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Save model
        model_path = os.path.join(self.models_dir, 'lgbm_sg_model.joblib')
        joblib.dump(model, model_path)
        print(f"LightGBM model saved to {model_path}")
        
        return model

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Trains an XGBoost model utilizing the GPU if available."""
        if not has_ml_libs:
            return None
            
        print("\n--- Training XGBoost Regressor ---")
        
        # Configure for GPU (tree_method='hist', device='cuda' on PC)
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            # tree_method='hist', # Uncomment on the PC
            # device='cuda',      # Uncomment on the PC
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Save model
        model_path = os.path.join(self.models_dir, 'xgb_sg_model.joblib')
        joblib.dump(model, model_path)
        print(f"XGBoost model saved to {model_path}")
        
        return model

    def run_training_pipeline(self):
        """Executes the full training pipeline."""
        X, y = self.load_data()
        if X is None or y is None:
            return
            
        print(f"Data loaded. Features shape: {X.shape}, Target shape: {y.shape}")
        
        if not has_ml_libs:
            print("Cannot proceed with training: ML libraries missing.")
            return
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Evaluate
        if lgb_model and xgb_model:
            lgb_preds = lgb_model.predict(X_val)
            xgb_preds = xgb_model.predict(X_val)
            
            # Simple ensemble average
            ensemble_preds = (lgb_preds + xgb_preds) / 2.0
            
            print("\n--- Validation Results ---")
            print(f"LightGBM RMSE: {np.sqrt(mean_squared_error(y_val, lgb_preds)):.4f}")
            print(f"XGBoost RMSE:  {np.sqrt(mean_squared_error(y_val, xgb_preds)):.4f}")
            print(f"Ensemble RMSE: {np.sqrt(mean_squared_error(y_val, ensemble_preds)):.4f}")

if __name__ == "__main__":
    trainer = PGAModelTrainer(data_path="data/processed/test_training_dataset.csv") # Using the test set from earlier
    trainer.run_training_pipeline()
