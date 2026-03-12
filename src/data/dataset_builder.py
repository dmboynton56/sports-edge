import os
import pandas as pd
from typing import Optional
from dotenv import load_dotenv
from .queries import get_base_performance_query, get_course_environmental_query, get_player_baseline_query
from .feature_engineering import build_features
from pga_dataloader import PGADataloader
from google.cloud import bigquery

load_dotenv()

class TrainingDatasetBuilder:
    """
    Orchestrates the extraction of raw PGA Tour data, applies feature engineering,
    and outputs clean datasets ready for model training.
    """
    def __init__(self, use_api_fallback: bool = True):
        self.use_api_fallback = use_api_fallback
        self.client = None
        # Try to initialize BigQuery if credentials exist
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                self.client = bigquery.Client()
            except Exception as e:
                print(f"Failed to initialize BigQuery client: {e}")

    def fetch_from_bigquery(self, query: str) -> pd.DataFrame:
        """
        Executes a SQL query against BigQuery and returns a DataFrame.
        """
        if not self.client:
            print("BigQuery client not available.")
            return pd.DataFrame()
            
        try:
            query_job = self.client.query(query)
            return query_job.to_dataframe()
        except Exception as e:
            print(f"Error querying BigQuery: {e}")
            return pd.DataFrame()

    def fetch_from_api_fallback(self) -> pd.DataFrame:
        """
        Fallback to fetching data from Kaggle CSV or Live API (via PGADataloader)
        if BigQuery is not yet populated or available.
        """
        print("Using API/Kaggle fallback for data...")
        # For local testing, use a dummy csv or ESPN
        loader = PGADataloader(mode="training", kaggle_csv_path="dummy_kaggle_data.csv")
        df = loader.fetch_leaderboard_and_stats()
        
        # Ensure minimal columns exist for feature engineering
        if not df.empty:
            if 'date' not in df.columns:
                df['date'] = pd.Timestamp.now()
            if 'sg_total' not in df.columns:
                # Mock sg_total if it doesn't exist
                if 'sg_putt' in df.columns and 'sg_app' in df.columns:
                    df['sg_total'] = df['sg_putt'] + df['sg_app']
                else:
                    df['sg_total'] = 0.0
                    
        return df

    def get_raw_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Attempts to get raw data from BigQuery, falling back to APIs if needed.
        """
        # 1. Try BigQuery
        if self.client and not self.use_api_fallback:
            print("Fetching base performance from BigQuery...")
            query = get_base_performance_query(limit)
            df = self.fetch_from_bigquery(query)
            if not df.empty:
                return df
                
        # 2. Fallback to API/CSV
        return self.fetch_from_api_fallback()

    def build_dataset(self, limit: int = 1000, output_path: str = "data/processed/training_dataset.csv") -> pd.DataFrame:
        """
        Main pipeline execution.
        """
        print("Starting data pipeline...")
        
        # 1. Extract
        df = self.get_raw_data(limit=limit)
        
        if df.empty:
            print("No data available to process.")
            return df
            
        # 2. Transform (Feature Engineering)
        print("Applying feature engineering...")
        df_features = build_features(df)
        
        # 3. Load / Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving dataset to {output_path}...")
        df_features.to_csv(output_path, index=False)
        
        print("Pipeline execution complete.")
        return df_features

if __name__ == "__main__":
    builder = TrainingDatasetBuilder(use_api_fallback=True)
    builder.build_dataset()
