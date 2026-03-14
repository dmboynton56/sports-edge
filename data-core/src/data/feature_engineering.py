import pandas as pd
import numpy as np

def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate recent form (e.g., exponential moving average of SG over the last 3, 5, and 10 tournaments).
    Assumes df is sorted by date/tournament for each player.
    """
    if df.empty or 'sg_total' not in df.columns:
        return df

    # Example: Calculate EMA for Total Strokes Gained
    df = df.sort_values(by=['player_name', 'date'])
    
    # Calculate exponentially weighted moving averages for different spans
    for span in [3, 5, 10]:
        df[f'sg_total_ema_{span}'] = df.groupby('player_name')['sg_total'].transform(
            lambda x: x.ewm(span=span, adjust=False).mean()
        )
        
    return df

def calculate_course_fit(df: pd.DataFrame, course_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features comparing a player's historical strengths with specific course requirements.
    """
    if df.empty or course_data.empty:
        return df
        
    # Merge course profile data with player data
    # Assuming course_data has columns like 'course_id', 'driving_distance_importance', etc.
    # and df has 'driving_distance_avg'
    
    df = pd.merge(df, course_data, on='course_id', how='left')
    
    if 'driving_distance_avg' in df.columns and 'driving_distance_importance' in df.columns:
        df['course_fit_driving'] = df['driving_distance_avg'] * df['driving_distance_importance']
        
    return df

def flag_regression_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify statistical anomalies in single rounds (e.g., extreme positive SG: Putting with negative SG: Approach)
    to flag players likely to regress.
    """
    if df.empty:
        return df
        
    if 'sg_putt' in df.columns and 'sg_app' in df.columns:
        # Flag players who are putting unsustainably well but striking the ball poorly
        df['negative_regression_candidate'] = np.where(
            (df['sg_putt'] > 2.0) & (df['sg_app'] < 0), 1, 0
        )
        
        # Flag players who are striking it great but putting terribly (positive regression)
        df['positive_regression_candidate'] = np.where(
            (df['sg_putt'] < -2.0) & (df['sg_app'] > 1.5), 1, 0
        )
        
    return df

def build_features(df: pd.DataFrame, course_data: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    """
    df = calculate_momentum_features(df)
    df = calculate_course_fit(df, course_data)
    df = flag_regression_candidates(df)
    return df
