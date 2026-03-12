import os
import pandas as pd
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class PGADataloader:
    """
    A wrapper class to fetch PGA Tour data from either local/BigQuery Kaggle datasets (for testing/training)
    or live APIs (Data Golf, ESPN) for tournament weekend inference.
    """

    def __init__(self, mode: str = "training", kaggle_csv_path: Optional[str] = None):
        """
        :param mode: "training" for historical data, "live" for tournament weekend data.
        :param kaggle_csv_path: Path to the Kaggle dataset CSV if in "training" mode.
        """
        self.mode = mode
        self.kaggle_csv_path = kaggle_csv_path
        self.data_golf_api_key = os.getenv("DATA_GOLF_API_KEY")

        if self.mode not in ["training", "live"]:
            raise ValueError("Mode must be either 'training' or 'live'")

    def fetch_leaderboard_and_stats(self) -> pd.DataFrame:
        """
        Fetches the current leaderboard and Strokes Gained stats based on the initialized mode.
        """
        if self.mode == "training":
            return self._fetch_historical_data()
        elif self.mode == "live":
            return self._fetch_live_data_golf()
        else:
            raise NotImplementedError(f"Mode '{self.mode}' is not implemented.")

    def _fetch_historical_data(self) -> pd.DataFrame:
        """
        Loads historical Kaggle CSV data. In a full implementation, this could also query BigQuery.
        """
        if not self.kaggle_csv_path or not os.path.exists(self.kaggle_csv_path):
            print("Warning: kaggle_csv_path is not provided or file does not exist. Returning empty DataFrame.")
            return pd.DataFrame()

        print(f"Loading historical data from {self.kaggle_csv_path}...")
        df = pd.read_csv(self.kaggle_csv_path)
        
        # Ensure standard column names for compatibility with live data
        if 'player_name' not in df.columns and 'Player' in df.columns:
            df.rename(columns={'Player': 'player_name'}, inplace=True)
            
        return df

    def _fetch_live_data_golf(self) -> pd.DataFrame:
        """
        Hits the Data Golf API to get live tournament stats (Strokes Gained, etc.)
        """
        if not self.data_golf_api_key:
            raise ValueError("DATA_GOLF_API_KEY environment variable is missing for live mode.")

        print("Fetching live Strokes Gained data from Data Golf API...")
        
        # Example Data Golf Endpoint for live tournament stats
        # The exact endpoint depends on your subscription tier (e.g., live-tournament-stats)
        url = f"https://feeds.datagolf.com/preds/live-tournament-stats?file_format=json&key={self.data_golf_api_key}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Assuming 'data' key holds the list of player records in Data Golf's JSON response
            players_data = data.get("data", [])
            df = pd.DataFrame(players_data)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Data Golf: {e}")
            return pd.DataFrame()

    def fetch_quick_leaderboard_espn(self) -> Dict[str, Any]:
        """
        Hits the Unofficial ESPN API for a lightweight, real-time leaderboard.
        Useful for quick verification without using Data Golf API credits.
        """
        print("Fetching quick leaderboard from Unofficial ESPN API...")
        # ESPN's hidden API for golf leaderboards
        url = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from ESPN: {e}")
            return {}

# Example Usage:
if __name__ == "__main__":
    # --- 1. Training Mode (Historical Kaggle Data) ---
    # Create a dummy CSV for testing if it doesn't exist
    test_csv = "dummy_kaggle_data.csv"
    if not os.path.exists(test_csv):
        pd.DataFrame({
            "player_name": ["Scottie Scheffler", "Rory McIlroy"],
            "sg_putt": [0.5, 0.2],
            "sg_app": [1.2, 0.9],
            "round": [4, 4]
        }).to_csv(test_csv, index=False)

    training_loader = PGADataloader(mode="training", kaggle_csv_path=test_csv)
    historical_df = training_loader.fetch_leaderboard_and_stats()
    print("Historical Data:")
    print(historical_df.head(), "\n")

    # --- 2. Live Mode (Data Golf API) ---
    # Note: Requires DATA_GOLF_API_KEY in your .env file
    # live_loader = PGADataloader(mode="live")
    # live_df = live_loader.fetch_leaderboard_and_stats()
    # print(live_df.head())
    
    # --- 3. Quick ESPN Scraper ---
    quick_loader = PGADataloader(mode="live") # Mode doesn't strictly matter for this standalone function
    espn_data = quick_loader.fetch_quick_leaderboard_espn()
    events = espn_data.get("events", [])
    if events:
        event_name = events[0].get("name", "Unknown Event")
        print(f"Current ESPN Event: {event_name}")
    else:
        print("No active ESPN events found.")
