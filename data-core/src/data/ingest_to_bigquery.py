import os
import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv

load_dotenv()

class BigQueryIngestor:
    """
    Handles fetching data from external APIs or CSVs and uploading 
    them to Google BigQuery for permanent storage (The 'Extract & Load' step).
    """
    def __init__(self, project_id: str = None):
        """
        Initializes the BigQuery client. 
        It relies on the GOOGLE_APPLICATION_CREDENTIALS environment variable.
        """
        self.client = bigquery.Client(project=project_id) if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") else None
        if not self.client:
            print("Warning: GOOGLE_APPLICATION_CREDENTIALS not found. BigQuery upload will fail.")

    def upload_dataframe(self, df: pd.DataFrame, dataset_id: str, table_id: str, write_disposition: str = "WRITE_APPEND"):
        """
        Uploads a Pandas DataFrame to a specific BigQuery table.
        
        :param df: The Pandas DataFrame to upload.
        :param dataset_id: The target BigQuery dataset (e.g., 'sports_edge_raw').
        :param table_id: The target BigQuery table (e.g., 'raw_pga_leaderboards').
        :param write_disposition: 'WRITE_APPEND' (add to existing) or 'WRITE_TRUNCATE' (overwrite).
        """
        if self.client is None:
            print("BigQuery client is not initialized. Cannot upload data.")
            return False

        if df.empty:
            print("DataFrame is empty. Nothing to upload.")
            return False

        table_ref = f"{self.client.project}.{dataset_id}.{table_id}"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            # Automatically detect the schema from the DataFrame
            autodetect=True,
        )

        print(f"Uploading {len(df)} rows to {table_ref}...")
        
        try:
            job = self.client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )
            job.result()  # Wait for the job to complete.
            print(f"Successfully uploaded data to {table_ref}.")
            return True
        except Exception as e:
            print(f"Error uploading to BigQuery: {e}")
            return False

# --- Example Usage for Free Data Pipelines ---

def ingest_espn_leaderboard_to_bq():
    """
    Example function that grabs the free ESPN leaderboard and uploads it to BigQuery.
    """
    from pga_dataloader import PGADataloader
    
    print("Fetching free ESPN leaderboard data...")
    loader = PGADataloader(mode="live")
    raw_data = loader.fetch_quick_leaderboard_espn()
    
    if not raw_data or "events" not in raw_data:
        print("No live event data found.")
        return
        
    event = raw_data["events"][0]
    competitors = event.get("competitions", [{}])[0].get("competitors", [])
    
    # Transform the nested JSON into a flat DataFrame
    rows = []
    for player in competitors:
        athlete = player.get("athlete", {})
        status = player.get("status", {})
        
        rows.append({
            "tournament_name": event.get("name"),
            "player_id": athlete.get("id"),
            "player_name": athlete.get("displayName"),
            "score": player.get("score"),
            "position": status.get("position", {}).get("displayName"),
            "thru": status.get("thru"),
            "ingested_at": pd.Timestamp.utcnow()
        })
        
    df = pd.DataFrame(rows)
    
    print(f"Extracted {len(df)} players from the ESPN API.")
    
    # Now load it into BigQuery
    ingestor = BigQueryIngestor()
    # Pushing to a 'raw_pga_leaderboards' table in the 'sports_edge_raw' dataset
    ingestor.upload_dataframe(
        df=df, 
        dataset_id="sports_edge_raw", 
        table_id="raw_pga_leaderboards",
        write_disposition="WRITE_APPEND" # Keep historical records
    )

def ingest_kaggle_historical_to_bq(csv_path: str):
    """
    Example function to upload a large downloaded Kaggle CSV to BigQuery.
    """
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return
        
    print(f"Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Add a timestamp so we know when we ingested it
    df["ingested_at"] = pd.Timestamp.utcnow()
    
    ingestor = BigQueryIngestor()
    # Pushing to 'raw_pga_player_rounds' which our ML models will query later
    ingestor.upload_dataframe(
        df=df, 
        dataset_id="sports_edge_raw", 
        table_id="raw_pga_player_rounds",
        write_disposition="WRITE_TRUNCATE" # Usually truncate historical data to avoid duplicates
    )

if __name__ == "__main__":
    # Test the ESPN ingestion pipeline
    ingest_espn_leaderboard_to_bq()
    
    # ---------------------------------------------------------
    # Example: How to save PGA predictions to the SHARED table
    # ---------------------------------------------------------
    print("Generating sample PGA prediction to save to shared 'model_predictions' table...")
    
    # We must match the schema from `WHOLE PROJECT TABLES.json` for model_predictions
    pga_prediction_df = pd.DataFrame([{
        "prediction_id": "pga_pred_12345",
        "game_id": "masters_2024",       # Using tournament_id as game_id
        "league": "PGA",                 # Crucial: flag it as PGA!
        "model_version": "v1.0_pga",
        "predicted_spread": None,        # N/A for golf typically, use None
        "home_win_prob": 0.15,           # E.g., probability of player A winning / placing
        "prediction_ts": pd.Timestamp.utcnow(),
        "input_hash": "abcdef12345",
        "season": 2024,
        "season_week": 15,
        "model_number": "m1"
    }])
    
    ingestor = BigQueryIngestor()
    # Pushing to the existing 'model_predictions' table in 'sports_edge_curated'
    ingestor.upload_dataframe(
        df=pga_prediction_df, 
        dataset_id="sports_edge_curated", 
        table_id="model_predictions",
        write_disposition="WRITE_APPEND" # Append, so we don't delete NBA/NFL predictions!
    )

