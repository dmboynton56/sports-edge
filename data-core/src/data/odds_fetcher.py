"""
Odds data fetcher using The Odds API.
Fetches spreads, totals, and moneylines from sportsbooks.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
from pathlib import Path
from src.data.odds_api_client import budgeted_get
from src.data.odds_api_errors import OddsApiQuotaExhausted

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def fetch_odds(
    league: str,
    date: Optional[str] = None,
    regions: str = 'us',
    markets: str = 'spreads,totals,h2h',
    bookmakers: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch odds from The Odds API.
    
    Args:
        league: 'nfl' or 'nba'
        date: Optional date string YYYY-MM-DD (default: today)
        regions: Comma-separated regions (default: 'us')
        markets: Comma-separated markets (default: 'spreads,totals,h2h')
    
    Returns:
        DataFrame with odds data
    """
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key:
        raise ValueError("ODDS_API_KEY not found in environment variables")
    
    # Convert league to API format
    sport = 'americanfootball_nfl' if league.upper() == 'NFL' else 'basketball_nba'
    
    # Build URL
    base_url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    
    params = {
        'apiKey': api_key,
        'regions': regions,
        'markets': markets,
    }
    if bookmakers:
        params['bookmakers'] = bookmakers
    
    if date:
        params['dateFormat'] = 'iso'
        # The Odds API expects dates in ISO format
        params['commenceTimeFrom'] = f"{date}T00:00:00Z"
        params['commenceTimeTo'] = f"{date}T23:59:59Z"
    
    try:
        response = budgeted_get(base_url, params=params, source=league.lower())
        
        data = response.json()
        
        # Transform to DataFrame
        rows = []
        for game in data:
            game_id = game.get('id')
            commence_time = game.get('commence_time')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            for bookmaker in game.get('bookmakers', []):
                book_name = bookmaker.get('key')
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key')
                    
                    for outcome in market.get('outcomes', []):
                        rows.append({
                            'game_id': game_id,
                            'commence_time': commence_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'book': book_name,
                            'market': market_key,
                            'outcome_name': outcome.get('name'),
                            'line': outcome.get('point'),
                            'price': outcome.get('price'),
                        })
        
        df = pd.DataFrame(rows)
        return df
        
    except OddsApiQuotaExhausted as e:
        print(f"Odds request skipped: {e}")
        return pd.DataFrame()


def fetch_odds_for_date(league: str, date: str) -> pd.DataFrame:
    """
    Fetch odds for games on a specific date.
    
    Args:
        league: 'nfl' or 'nba'
        date: Date string YYYY-MM-DD
    
    Returns:
        DataFrame with odds for that date
    """
    return fetch_odds(league, date=date)


def cache_odds_data(data: pd.DataFrame, league: str, date: str):
    """
    Cache odds data to disk.
    
    Args:
        data: DataFrame to cache
        league: 'nfl' or 'nba'
        date: Date string YYYY-MM-DD
    """
    cache_dir = f"data/raw/{league}/{date}"
    os.makedirs(cache_dir, exist_ok=True)
    
    filepath = f"{cache_dir}/odds.parquet"
    data.to_parquet(filepath, index=False)
    print(f"Cached odds to {filepath}")

