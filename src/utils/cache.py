"""
Caching utilities for feature computation.
Allows caching expensive computations (like form features) to speed up notebook iterations.
"""

import os
import pickle
import hashlib
import pandas as pd
from typing import Optional, Any, Dict
from datetime import datetime


CACHE_DIR = "notebooks/cache"


def _ensure_cache_dir():
    """Ensure cache directory exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_key(data_hash: str, function_name: str, params: Dict[str, Any]) -> str:
    """Generate cache key from data hash, function name, and parameters."""
    param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
    key_str = f"{function_name}_{data_hash}_{param_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Generate hash of DataFrame for cache invalidation."""
    # Hash based on shape and first/last few rows
    if df.empty:
        return "empty"
    
    # Use shape, column names, and sample of data
    sample = pd.concat([df.head(5), df.tail(5)]) if len(df) > 10 else df
    data_str = f"{df.shape}_{list(df.columns)}_{sample.to_string()}"
    return hashlib.md5(data_str.encode()).hexdigest()


def cache_form_features(
    games_df: pd.DataFrame,
    game_logs: pd.DataFrame,
    league: str,
    cache_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Cache form features computation.
    
    Args:
        games_df: Games DataFrame
        game_logs: Game logs DataFrame (for NBA) or PBP DataFrame (for NFL)
        league: 'NBA' or 'NFL'
        cache_name: Optional custom cache name
    
    Returns:
        DataFrame with form features (from cache if available)
    """
    _ensure_cache_dir()
    
    # Generate cache key
    games_hash = _hash_dataframe(games_df)
    logs_hash = _hash_dataframe(game_logs)
    combined_hash = hashlib.md5(f"{games_hash}_{logs_hash}".encode()).hexdigest()
    
    cache_key = cache_name or f"form_features_{league.lower()}_{combined_hash[:12]}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading form features from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                # Verify it matches our data
                if cached_data.get('games_hash') == games_hash and cached_data.get('logs_hash') == logs_hash:
                    print(f"  Cache hit! Using cached form features.")
                    return cached_data['features_df']
                else:
                    print(f"  Cache miss (data changed), recomputing...")
        except Exception as e:
            print(f"  Error loading cache: {e}, recomputing...")
    
    # Compute form features
    print(f"Computing form features (this may take a while)...")
    from src.features import form_metrics
    
    features_df = games_df.copy()
    
    if league.upper() == 'NBA':
        for window in [3, 5, 10]:
            features_df = form_metrics.add_form_features_nba(features_df, game_logs, window=window)
    else:  # NFL
        for window in [3, 5, 10]:
            features_df = form_metrics.add_form_features_nfl(features_df, game_logs, window=window)
    
    # Save to cache
    cache_data = {
        'features_df': features_df,
        'games_hash': games_hash,
        'logs_hash': logs_hash,
        'league': league,
        'cached_at': datetime.now().isoformat()
    }
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  Cached form features to: {cache_path}")
    except Exception as e:
        print(f"  Warning: Could not save cache: {e}")
    
    return features_df


def clear_cache(pattern: Optional[str] = None):
    """
    Clear cached files.
    
    Args:
        pattern: Optional pattern to match (e.g., 'form_features_nba_*')
    """
    _ensure_cache_dir()
    
    if pattern:
        import glob
        files = glob.glob(os.path.join(CACHE_DIR, pattern))
    else:
        files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    
    print(f"Cleared {len(files)} cache files")


def get_cache_info() -> Dict[str, Any]:
    """Get information about cached files."""
    _ensure_cache_dir()
    
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    info = {
        'cache_dir': CACHE_DIR,
        'num_files': len(cache_files),
        'files': []
    }
    
    for file in cache_files:
        file_path = os.path.join(CACHE_DIR, file)
        try:
            stat = os.stat(file_path)
            info['files'].append({
                'name': file,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        except Exception as e:
            info['files'].append({'name': file, 'error': str(e)})
    
    return info

