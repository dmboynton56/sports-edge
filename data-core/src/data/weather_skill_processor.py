import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

# Ensure BigQueryIngestor is accessible
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest_to_bigquery import BigQueryIngestor

load_dotenv()

def process_weather_skill():
    print("Loading datasets...")
    # Load results
    results_path = "src/data/archive/pga_results_2001-2025.tsv"
    try:
        df_results = pd.read_csv(results_path, sep="\t")
    except Exception as e:
        print(f"Failed to load results: {e}")
        return

    # Load weather
    weather_path = "src/data/archive/Tournaments With Weather Data.csv"
    try:
        df_weather = pd.read_csv(weather_path)
    except Exception as e:
        print(f"Failed to load weather data: {e}")
        return

    print(f"Results shape: {df_results.shape}, Weather shape: {df_weather.shape}")

    # Standardize column names for merging
    # Weather: year, name
    # Results: season, tournament
    df_weather['season'] = df_weather['year']
    
    # We will do a merge on season and tournament name
    # Since tournament names might differ slightly, we'll try an exact merge first
    df_weather['tournament_lower'] = df_weather['name'].str.lower().str.strip()
    df_results['tournament_lower'] = df_results['tournament'].str.lower().str.strip()

    print("Merging datasets on season and tournament name...")
    df_merged = pd.merge(
        df_results, 
        df_weather, 
        left_on=['season', 'tournament_lower'], 
        right_on=['season', 'tournament_lower'], 
        how='inner'
    )
    
    print(f"Merged dataset shape (exact match): {df_merged.shape}")
    
    if df_merged.empty:
        print("Exact merge failed. Check tournament names.")
        # Print samples
        print("Results sample tournaments:", df_results['tournament_lower'].unique()[:5])
        print("Weather sample tournaments:", df_weather['tournament_lower'].unique()[:5])
        return

    print("Calculating Wind Sensitivity per Player...")
    
    # We will look at Round 1 and Round 2 scores, and match them with day0wind and day1wind
    # In df_results: 'round1', 'round2', 'round3', 'round4'
    # In df_weather: 'day0wind', 'day1wind', 'day2wind'
    
    # Let's melt the dataframe so we have one row per player per round
    # Keep essential columns
    cols_to_keep = ['season', 'tournament', 'name_x', 'position', 'Course Par', 'day0wind', 'day1wind', 'day2wind', 'round1', 'round2', 'round3']
    
    # 'name_x' is the player name (since 'name_y' would be tournament name from weather)
    df_sub = df_merged[cols_to_keep].copy()
    df_sub.rename(columns={'name_x': 'player_name'}, inplace=True)

    # Convert rounds to numeric, coercing errors (like 'WD', 'DQ') to NaN
    for r in ['round1', 'round2', 'round3']:
        df_sub[r] = pd.to_numeric(df_sub[r], errors='coerce')

    # Calculate score relative to par
    df_sub['r1_vs_par'] = df_sub['round1'] - df_sub['Course Par']
    df_sub['r2_vs_par'] = df_sub['round2'] - df_sub['Course Par']
    df_sub['r3_vs_par'] = df_sub['round3'] - df_sub['Course Par']

    # Create a long format table: player_name, round, score_vs_par, wind
    r1_data = df_sub[['player_name', 'r1_vs_par', 'day0wind']].rename(columns={'r1_vs_par': 'score_vs_par', 'day0wind': 'wind_speed'})
    r2_data = df_sub[['player_name', 'r2_vs_par', 'day1wind']].rename(columns={'r2_vs_par': 'score_vs_par', 'day1wind': 'wind_speed'})
    r3_data = df_sub[['player_name', 'r3_vs_par', 'day2wind']].rename(columns={'r3_vs_par': 'score_vs_par', 'day2wind': 'wind_speed'})

    df_long = pd.concat([r1_data, r2_data, r3_data]).dropna(subset=['score_vs_par', 'wind_speed'])

    # Categorize wind
    # Let's define: Low (<10mph), Medium (10-15mph), High (>15mph)
    def categorize_wind(wind):
        if wind < 10:
            return 'Low Wind'
        elif wind <= 15:
            return 'Medium Wind'
        else:
            return 'High Wind'

    df_long['wind_category'] = df_long['wind_speed'].apply(categorize_wind)

    # Calculate average score to par per player per wind category
    # Filter out players with too few rounds
    round_counts = df_long['player_name'].value_counts()
    valid_players = round_counts[round_counts >= 20].index
    
    df_filtered = df_long[df_long['player_name'].isin(valid_players)]
    
    print("Aggregating stats...")
    wind_stats = df_filtered.groupby(['player_name', 'wind_category'])['score_vs_par'].agg(['mean', 'count']).reset_index()
    
    # Pivot to make it nice: Player | Low Wind Avg | Med Wind Avg | High Wind Avg
    wind_pivot = wind_stats.pivot(index='player_name', columns='wind_category', values='mean').reset_index()
    
    # Add an "Overall Avg"
    overall_stats = df_filtered.groupby('player_name')['score_vs_par'].mean().reset_index().rename(columns={'score_vs_par': 'overall_avg'})
    
    final_df = pd.merge(overall_stats, wind_pivot, on='player_name', how='left')
    
    # Calculate High Wind "Premium" (High Wind Avg - Overall Avg)
    # A negative premium means they play BETTER relative to their own average when it's windy
    if 'High Wind' in final_df.columns:
        final_df['high_wind_premium'] = final_df['High Wind'] - final_df['overall_avg']
    
    print("\n--- Top 10 Wind Specialists (Lowest High Wind Premium, min 20 rounds) ---")
    if 'high_wind_premium' in final_df.columns:
        # Sort by high wind premium ascending (meaning they play best in wind relative to themselves)
        print(final_df.sort_values('high_wind_premium').head(10))
    else:
        print(final_df.head(10))

    # Clean up column names for BigQuery
    final_df.columns = [c.lower().replace(' ', '_') for c in final_df.columns]
    final_df['updated_at'] = pd.Timestamp.utcnow()

    # Upload to BigQuery
    print("\nUploading to BigQuery...")
    ingestor = BigQueryIngestor()
    # Pushing to the curated layer since this is feature-engineered
    ingestor.upload_dataframe(
        df=final_df, 
        dataset_id="sports_edge_curated", 
        table_id="pga_player_wind_skill",
        write_disposition="WRITE_TRUNCATE" # Overwrite to keep it fresh
    )

if __name__ == "__main__":
    # We may need fuzzywuzzy installed
    process_weather_skill()
