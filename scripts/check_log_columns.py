import pandas as pd
from src.data.nba_fetcher import fetch_nba_games_for_date
from src.data.nba_game_logs_loader import load_nba_game_logs

def check_columns():
    print("Fetching sample logs...")
    # Load a small sample of logs (e.g. current season)
    df = load_nba_game_logs(seasons=[2025])
    
    if df is not None and not df.empty:
        print(f"\nColumns ({len(df.columns)}):")
        print(list(df.columns))
        
        # Check for key stats needed for possessions
        needed = ['FGA', 'FTA', 'TOV', 'OREB', 'DREB', 'STL', 'BLK', 'PF']
        found = [c for c in needed if c in df.columns or c.upper() in df.columns]
        print(f"\nFound {len(found)}/{len(needed)} key stats: {found}")
        
        # Print a sample row
        print("\nSample Row:")
        print(df.iloc[0])
    else:
        print("No logs found.")

if __name__ == "__main__":
    check_columns()
