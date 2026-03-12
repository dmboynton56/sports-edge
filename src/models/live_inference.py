import os
import sys
import pandas as pd
import numpy as np

# Ensure root paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset_builder import TrainingDatasetBuilder
from src.data.ingest_to_bigquery import ingest_espn_leaderboard_to_bq, BigQueryIngestor
from src.models.monte_carlo_sim import PGAMonteCarloSimulator

def run_live_inference():
    print("=== Starting Live Inference for Current Tournament ===")
    
    # 1. Update the latest leaderboard into BigQuery
    print("\n1. Fetching live leaderboard...")
    # NOTE: Normally you'd uncomment this to get the *latest* data
    # ingest_espn_leaderboard_to_bq()
    
    # 2. Extract state from BigQuery
    print("\n2. Pulling current tournament state from BigQuery...")
    builder = TrainingDatasetBuilder(use_api_fallback=False)
    
    if not builder.client:
        print("GCP Credentials missing. Falling back to dummy data.")
        # Create a mock leaderboard based on our earlier test
        df_live = pd.DataFrame({
            'player_name': ['Scottie Scheffler', 'Rory McIlroy', 'Jon Rahm', 'Tiger Woods', 'Justin Thomas'],
            'score': [-5, -4, -4, 0, +1]
        })
    else:
        # In a real scenario, you'd run a SQL query to get the latest leaderboard we just ingested,
        # joined with their historical baseline stats.
        query = """
            SELECT 
                l.player_name,
                l.score as score,
                COALESCE(w.overall_avg, 0) as historical_avg_to_par
            FROM `sports_edge_raw.raw_pga_leaderboards` l
            LEFT JOIN `sports_edge_curated.pga_player_wind_skill` w 
                ON LOWER(l.player_name) = LOWER(w.player_name)
            ORDER BY l.ingested_at DESC
            LIMIT 150
        """
        try:
            df_live = builder.fetch_from_bigquery(query)
            if df_live.empty:
                print("No live data found in BigQuery.")
                return
            
            # Clean up 'E' to 0 if it slipped through
            df_live['score'] = pd.to_numeric(df_live['score'], errors='coerce').fillna(0)
            
        except Exception as e:
            print(f"Failed to query BigQuery: {e}")
            return
            
    # 3. Use Model to predict Expected SG (Mocking this step for the script structure)
    print("\n3. Predicting Expected Strokes Gained for remaining rounds...")
    # In production, you would load `xgb_sg_model.joblib` and run `model.predict(df_live[features])`
    
    # Mocking model output: Better historical average -> Better Expected SG
    # E.g. A negative historical avg_to_par means they usually shoot under par.
    if 'historical_avg_to_par' in df_live.columns:
        df_live['expected_sg_per_round'] = -df_live['historical_avg_to_par']
    else:
        df_live['expected_sg_per_round'] = np.random.normal(0, 1, len(df_live))
        
    df_live['sg_variance'] = 2.5 # Average PGA Tour variance per round

    # 4. Run Monte Carlo Simulation
    print("\n4. Running Monte Carlo Simulation (10,000 runs)...")
    sim = PGAMonteCarloSimulator(num_simulations=10000)
    
    # Format for simulator
    df_sim_input = df_live.rename(columns={'score': 'current_score'})
    
    # Let's assume it's Friday night, 2 rounds remaining
    results = sim.run_simulation(df_sim_input, remaining_rounds=2)
    
    print("\n=== FINAL WIN PROBABILITIES (Top 10) ===")
    print(results.head(10))
    
    # 5. Push predictions back to BigQuery
    print("\n5. Saving predictions to sports_edge_curated.model_predictions...")
    if builder.client:
        results['prediction_id'] = f"live_sim_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M')}"
        results['game_id'] = "current_tournament"
        results['league'] = "PGA"
        results['model_version'] = "v1.0_mc_sim"
        results['prediction_ts'] = pd.Timestamp.utcnow()
        
        # Map columns to match the `model_predictions` schema
        bq_df = results[['prediction_id', 'game_id', 'league', 'model_version', 'win_prob', 'prediction_ts']].copy()
        bq_df.rename(columns={'win_prob': 'home_win_prob'}, inplace=True)
        
        ingestor = BigQueryIngestor()
        ingestor.upload_dataframe(
            df=bq_df,
            dataset_id="sports_edge_curated",
            table_id="model_predictions",
            write_disposition="WRITE_APPEND"
        )

if __name__ == "__main__":
    run_live_inference()
