import pandas as pd
from typing import Iterable, Optional, Sequence


def _normalize_game_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a game_date column exists and is datetime."""
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
    elif 'gameday' in df.columns:
        df['game_date'] = pd.to_datetime(df['gameday'])
    elif 'game_start_time' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_start_time'], errors='coerce')
    return df


def load_pbp(seasons: Sequence[int], strict: bool = False) -> Optional[pd.DataFrame]:
    """
    Load play-by-play rows for the requested seasons.
    
    Attempts nflreadpy first (DuckDB cache) and falls back to nfl_data_py.
    
    Args:
        seasons: Iterable of season years (e.g., [2024, 2025])
        strict: If True, raise if data cannot be loaded. Otherwise return None.
    """
    seasons = list(seasons)
    last_error: Optional[Exception] = None
    
    # Try nflreadpy
    try:
        import nflreadpy as nflr
        try:
            pbp_rel = nflr.load_pbp(seasons=seasons)
            pbp_df = pbp_rel.to_pandas()
            if not pbp_df.empty:
                return _normalize_game_dates(pbp_df)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    except ImportError as exc:  # noqa: BLE001
        last_error = exc
    
    # Fallback to nfl_data_py
    try:
        import nfl_data_py as nfl
        pbp_df = nfl.import_pbp_data(seasons)
        if not pbp_df.empty:
            return _normalize_game_dates(pbp_df)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
    
    message = f"Unable to load play-by-play data ({last_error})"
    if strict:
        raise RuntimeError(message)
    print(f"WARNING: {message}")
    return None
