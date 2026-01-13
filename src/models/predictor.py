"""
Production prediction module for NFL/NBA games.
Provides functions to predict win probabilities and spreads for specific games.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Optional, List
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features import rest_schedule, form_metrics, strength
from src.models.link_function import spread_to_win_prob, win_prob_to_spread

DEFAULT_LINK_PARAMS = (0.15, 2.5)


class GamePredictor:
    """
    Production predictor for game outcomes.
    Loads saved models and makes predictions for specific games.
    """
    
    def __init__(self, league: str, model_version: str = 'v1'):
        """
        Initialize predictor for a league.
        
        Args:
            league: 'NFL' or 'NBA'
            model_version: Model version string (e.g., 'v1')
        """
        self.league = league.upper()
        self.model_version = model_version
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Load models
        self.win_prob_model = None
        self.spread_model = None
        self.feature_names = None
        self.win_feature_names = None
        self.spread_feature_names = None
        self.link_params = None
        self.meta_ensemble = None
        self._median_cache = None
        
        self._load_models()
    
    def _load_models(self):
        """Load saved models from disk."""
        win_prob_path = os.path.join(self.models_dir, f'win_prob_model_{self.league.lower()}_{self.model_version}.pkl')
        spread_path = os.path.join(self.models_dir, f'spread_model_{self.league.lower()}_{self.model_version}.pkl')
        link_path = os.path.join(self.models_dir, f'link_function_{self.league.lower()}_{self.model_version}.pkl')
        meta_path = os.path.join(self.models_dir, f'meta_ensemble_{self.league.lower()}_{self.model_version}.pkl')
        
        if not os.path.exists(win_prob_path):
            raise FileNotFoundError(f"Win probability model not found: {win_prob_path}")
        if not os.path.exists(spread_path):
            raise FileNotFoundError(f"Spread model not found: {spread_path}")
        
        # Load win probability model
        with open(win_prob_path, 'rb') as f:
            win_prob_data = pickle.load(f)
            self.win_prob_model = win_prob_data['model']
            self.win_feature_names = win_prob_data.get('win_feature_names') or win_prob_data.get('feature_names')
            self.is_ensemble = win_prob_data.get('ensemble', False)
        
        # Load spread model
        with open(spread_path, 'rb') as f:
            spread_data = pickle.load(f)
            self.spread_model = spread_data['model']
            self.spread_feature_names = spread_data.get('spread_feature_names') or spread_data.get('feature_names')
        
        if self.win_feature_names is None:
            self.win_feature_names = self.spread_feature_names
        if self.spread_feature_names is None:
            self.spread_feature_names = self.win_feature_names
        self.feature_names = sorted(set(self.win_feature_names or []) | set(self.spread_feature_names or []))
        
        # Load link function parameters
        if os.path.exists(link_path):
            with open(link_path, 'rb') as f:
                link_data = pickle.load(f)
                a = float(link_data.get('a', DEFAULT_LINK_PARAMS[0]))
                b = float(link_data.get('b', DEFAULT_LINK_PARAMS[1]))
                if (not np.isfinite(a)) or (not np.isfinite(b)) or abs(a) > 1:
                    print(f"Warning: Link parameters out of bounds (a={a}, b={b}). Using defaults.")
                    self.link_params = DEFAULT_LINK_PARAMS
                else:
                    self.link_params = (a, b)
        else:
            self.link_params = DEFAULT_LINK_PARAMS
            
        # Load meta-ensemble if available
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                self.meta_ensemble = pickle.load(f)
                print(f"Loaded meta-ensemble from {meta_path}")

    def _load_feature_medians(self) -> Dict[str, float]:
        """Load cached feature medians from disk."""
        if self._median_cache is not None:
            return self._median_cache
        
        median_path = os.path.join(self.models_dir, f'feature_medians_{self.league.lower()}_{self.model_version}.pkl')
        if os.path.exists(median_path):
            with open(median_path, 'rb') as f:
                self._median_cache = pickle.load(f)
        else:
            self._median_cache = {}
        return self._median_cache
    
    @staticmethod
    def _prepare_feature_matrix(features_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Project features_df onto the requested columns, defaulting missing ones to NaN."""
        X = pd.DataFrame(index=features_df.index)
        for col in columns:
            if col in features_df.columns:
                X[col] = pd.to_numeric(features_df[col], errors='coerce')
            else:
                X[col] = np.nan
        return X

    @staticmethod
    def _fill_with_medians(df: pd.DataFrame, medians: Dict[str, float]) -> pd.DataFrame:
        """Fill NaNs with provided medians, defaulting to zero when missing."""
        # Set option to avoid future warnings about downcasting
        with pd.option_context('future.no_silent_downcasting', True):
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(medians.get(col, 0.0))
                # Ensure it's numeric after filling
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df
    
    def build_features_for_game(self, game_row: pd.DataFrame, historical_games: pd.DataFrame,
                                play_by_play: Optional[pd.DataFrame] = None,
                                game_logs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build features for a single game or games.
        
        Args:
            game_row: DataFrame with game(s) to predict (must have home_team, away_team, game_date)
            historical_games: Historical games DataFrame for computing rest/strength features
            play_by_play: Optional PBP data for NFL form features
            game_logs: Optional game logs for NBA form features
        
        Returns:
            DataFrame with features
        """
        df = game_row.copy()
        
        # Ensure game_date is datetime
        if 'game_date' not in df.columns:
            if 'gameday' in df.columns:
                df['game_date'] = pd.to_datetime(df['gameday'])
            else:
                raise ValueError("game_date or gameday column required")
        else:
            df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Add rest features
        df = rest_schedule.add_rest_features(df, historical_games)
        
        # Add opponent strength features
        df = strength.add_opponent_strength_features(df, historical_games, league=self.league.lower())
        
        # Add team strength features (win %, point diff)
        df = self._add_team_strength_features(df, historical_games)
        
        # Add interaction features
        df = self._add_interaction_features(df)
        
        # Add form features if available
        if self.league == 'NFL' and play_by_play is not None:
            for window in [3, 5, 10]:
                df = form_metrics.add_form_features_nfl(df, play_by_play, window=window)
        elif self.league == 'NBA' and game_logs is not None:
            for window in [3, 5, 10]:
                df = form_metrics.add_form_features_nba(df, game_logs, window=window)
        
        # Add form interaction features if form features exist
        df = self._add_form_interactions(df)
        
        return df
    
    def _add_team_strength_features(self, games_df: pd.DataFrame, historical_games: pd.DataFrame) -> pd.DataFrame:
        """Add team strength features (win %, point diff).
        
        IMPORTANT: Only uses current season data. No fallback to previous seasons.
        Teams change year-to-year, so using old data would be inaccurate.
        """
        df = games_df.copy()
        df['home_team_win_pct'] = np.nan
        df['away_team_win_pct'] = np.nan
        df['home_team_point_diff'] = np.nan
        df['away_team_point_diff'] = np.nan
        
        for idx, row in df.iterrows():
            game_date = pd.to_datetime(row['game_date'])
            home_team = row['home_team']
            away_team = row['away_team']
            season = row.get('season', game_date.year)
            
            # Only use current season games (completed games only)
            season_games = historical_games[
                (pd.to_datetime(historical_games['game_date']) < game_date) &
                (historical_games.get('season', pd.to_datetime(historical_games['game_date']).dt.year) == season)
            ]
            
            # Filter to only completed games (have scores)
            if 'home_score' in season_games.columns and 'away_score' in season_games.columns:
                season_games = season_games[
                    season_games['home_score'].notna() & season_games['away_score'].notna()
                ]
            
            # If no current season completed games, leave as NaN
            # This signals that we don't have enough data for accurate prediction
            if len(season_games) == 0:
                continue  # Leave features as NaN
            
            # Home team stats
            home_games = season_games[
                (season_games['home_team'] == home_team) | (season_games['away_team'] == home_team)
            ]
            if len(home_games) > 0:
                home_wins = sum(1 for _, g in home_games.iterrows()
                              if (g['home_team'] == home_team and g.get('home_score', 0) > g.get('away_score', 0)) or
                                 (g['away_team'] == home_team and g.get('away_score', 0) > g.get('home_score', 0)))
                
                home_at_home = home_games[home_games['home_team'] == home_team]
                home_wins_at_home = sum(1 for _, g in home_at_home.iterrows() if g.get('home_score', 0) > g.get('away_score', 0))
                
                home_point_diff = [
                    g.get('home_score', 0) - g.get('away_score', 0) if g['home_team'] == home_team
                    else g.get('away_score', 0) - g.get('home_score', 0)
                    for _, g in home_games.iterrows()
                ]
                df.loc[idx, 'home_team_win_pct'] = home_wins / len(home_games)
                df.loc[idx, 'home_team_win_pct_at_home'] = home_wins_at_home / len(home_at_home) if len(home_at_home) > 0 else 0.5
                df.loc[idx, 'home_team_point_diff'] = np.mean(home_point_diff) if home_point_diff else 0
                df.loc[idx, 'home_team_point_diff_std'] = np.std(home_point_diff) if len(home_point_diff) > 1 else 10
            
            # Away team stats
            away_games = season_games[
                (season_games['home_team'] == away_team) | (season_games['away_team'] == away_team)
            ]
            if len(away_games) > 0:
                away_wins = sum(1 for _, g in away_games.iterrows()
                              if (g['home_team'] == away_team and g.get('home_score', 0) > g.get('away_score', 0)) or
                                 (g['away_team'] == away_team and g.get('away_score', 0) > g.get('home_score', 0)))
                
                away_on_road = away_games[away_games['away_team'] == away_team]
                away_wins_on_road = sum(1 for _, g in away_on_road.iterrows() if g.get('away_score', 0) > g.get('home_score', 0))
                
                away_point_diff = [
                    g.get('home_score', 0) - g.get('away_score', 0) if g['home_team'] == away_team
                    else g.get('away_score', 0) - g.get('home_score', 0)
                    for _, g in away_games.iterrows()
                ]
                df.loc[idx, 'away_team_win_pct'] = away_wins / len(away_games)
                df.loc[idx, 'away_team_win_pct_on_road'] = away_wins_on_road / len(away_on_road) if len(away_on_road) > 0 else 0.4
                df.loc[idx, 'away_team_point_diff'] = np.mean(away_point_diff) if away_point_diff else 0
                df.loc[idx, 'away_team_point_diff_std'] = np.std(away_point_diff) if len(away_point_diff) > 1 else 10
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction and derived features."""
        # Rest differentials
        if 'rest_home' in df.columns and 'rest_away' in df.columns:
            df['rest_differential'] = df['rest_home'] - df['rest_away']
            df['rest_advantage_home'] = (df['rest_home'] > df['rest_away']).astype(int)
        
        # Team strength differentials
        if 'home_team_win_pct' in df.columns and 'away_team_win_pct' in df.columns:
            df['win_pct_differential'] = df['home_team_win_pct'] - df['away_team_win_pct']
        if 'home_team_point_diff' in df.columns and 'away_team_point_diff' in df.columns:
            df['point_diff_differential'] = df['home_team_point_diff'] - df['away_team_point_diff']
            df['point_diff_gap'] = df['away_team_point_diff'] - df['home_team_point_diff']
            df['abs_point_diff_gap'] = df['point_diff_gap'].abs()
            df['point_diff_gap_flag'] = (df['point_diff_gap'] > 5).astype(int)
        
        # Opponent strength differential
        if 'opp_strength_home_season' in df.columns and 'opp_strength_away_season' in df.columns:
            df['opp_strength_differential'] = df['opp_strength_home_season'] - df['opp_strength_away_season']
        
        # Time features
        if 'game_date' in df.columns:
            if 'game_type' in df.columns:
                df['is_playoff'] = df['game_type'].str.contains('POST', case=False, na=False).astype(int)
            else:
                df['is_playoff'] = 0
        
        return df
    
    def _add_form_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add form feature interactions."""
        for window in [3, 5, 10]:
            if self.league == 'NFL':
                # EPA Offensive diffs
                h_off = f'form_home_epa_off_{window}'
                a_off = f'form_away_epa_off_{window}'
                if h_off in df.columns and a_off in df.columns:
                    df[f'form_epa_off_diff_{window}'] = df[h_off] - df[a_off]
                
                # EPA Defensive diffs
                h_def = f'form_home_epa_def_{window}'
                a_def = f'form_away_epa_def_{window}'
                if h_def in df.columns and a_def in df.columns:
                    df[f'form_epa_def_diff_{window}'] = df[h_def] - df[a_def]
            
            elif self.league == 'NBA':
                # Net Rating differential
                h_net = f'form_home_net_rating_{window}'
                a_net = f'form_away_net_rating_{window}'
                if h_net in df.columns and a_net in df.columns:
                    df[f'form_net_rating_diff_{window}'] = df[h_net] - df[a_net]
                
                # Offensive rating differential
                h_off = f'form_home_off_rating_{window}'
                a_off = f'form_away_off_rating_{window}'
                if h_off in df.columns and a_off in df.columns:
                    df[f'form_off_rating_diff_{window}'] = df[h_off] - df[a_off]
                
                # Pace differential
                h_pace = f'form_home_pace_{window}'
                a_pace = f'form_away_pace_{window}'
                if h_pace in df.columns and a_pace in df.columns:
                    df[f'form_pace_diff_{window}'] = df[h_pace] - df[a_pace]
        
        return df
    
    def predict(self, game_row: pd.DataFrame, historical_games: pd.DataFrame,
                play_by_play: Optional[pd.DataFrame] = None,
                game_logs: Optional[pd.DataFrame] = None,
                fill_missing_with_median: bool = True,
                include_explanations: bool = False) -> Dict:
        """
        Predict outcome for a game.
        
        Args:
            game_row: DataFrame with game to predict
            historical_games: Historical games for feature computation
            play_by_play: Optional PBP data (NFL)
            game_logs: Optional game logs (NBA)
            fill_missing_with_median: Whether to fill missing features with median values
            include_explanations: Whether to include feature importance/explanations
        
        Returns:
            Dictionary with predictions
        """
        # Build features
        features_df = self.build_features_for_game(game_row, historical_games, play_by_play, game_logs)
        
        medians = self._load_feature_medians() if fill_missing_with_median else None
        spread_cols = self.spread_feature_names or self.feature_names
        win_cols = self.win_feature_names or self.feature_names
        
        X_spread = self._prepare_feature_matrix(features_df, spread_cols)
        
        # Neutralize features the user wants to ignore (without re-training)
        ignore_features = ['week_number', 'month', 'point_diff_differential']
        for feat in ignore_features:
            if feat in X_spread.columns:
                X_spread[feat] = 0.0
                
        if fill_missing_with_median and medians is not None:
            X_spread = self._fill_with_medians(X_spread, medians)
        else:
            X_spread = X_spread.fillna(0)
            
        # Spread prediction
        if self.is_ensemble:
            spread_preds = [m.predict(X_spread.values.astype(float)) for m in self.spread_model.values()]
            spread_pred = np.mean(spread_preds, axis=0)
        else:
            spread_pred = self.spread_model.predict(X_spread)
        
        X_win = self._prepare_feature_matrix(features_df, win_cols)
        
        # Neutralize features in win model as well
        for feat in ignore_features:
            if feat in X_win.columns:
                X_win[feat] = 0.0
                
        if 'model_spread_feature' in X_win.columns:
            X_win['model_spread_feature'] = spread_pred
        if fill_missing_with_median and medians is not None:
            X_win = self._fill_with_medians(X_win, medians)
        else:
            X_win = X_win.fillna(0)
        
        # Predict win probability
        if self.is_ensemble:
            win_probs = [m.predict_proba(X_win.values.astype(float))[:, 1] for m in self.win_prob_model.values()]
            win_prob_proba = np.mean(win_probs, axis=0)
        else:
            win_prob_proba = self.win_prob_model.predict_proba(X_win)[:, 1]  # Probability of home win
        
        # Convert to win prob using link function (for consistency)
        link_a, link_b = self.link_params
        win_prob_from_spread = spread_to_win_prob(spread_pred, link_a, link_b)
        
        # Blend models smoothly; never fully discard either side so extreme disagreements stay informative
        disagreement = np.abs(win_prob_proba - win_prob_from_spread)
        
        if self.meta_ensemble:
            # Use trained meta-model for final win probability
            meta_X = np.column_stack([win_prob_proba, win_prob_from_spread, spread_pred])
            final_win_prob = self.meta_ensemble['model'].predict_proba(meta_X)[:, 1]
        else:
            # Fallback to heuristic blending
            max_disagreement = 0.30
            base_weight = 0.65
            min_weight = 0.25
            weight_drop = (disagreement / max_disagreement) * 0.4
            ensemble_weight = np.clip(base_weight - weight_drop, min_weight, base_weight)
            final_win_prob = (ensemble_weight * win_prob_proba +
                             (1 - ensemble_weight) * win_prob_from_spread)
        
        # Feature explanations if requested
        explanations = []
        if include_explanations:
            explanations = self._get_local_explanations(X_spread, spread_cols)

        # Return predictions
        results = []
        for idx in range(len(game_row)):
            disagreement_val = disagreement[idx]
            model_disagreement = disagreement_val > 0.15
            
            home_margin = float(spread_pred[idx])  # positive => home advantage
            home_spread = -home_margin  # betting convention: favorite has negative spread
            result = {
                'home_team': game_row.iloc[idx]['home_team'],
                'away_team': game_row.iloc[idx]['away_team'],
                'game_date': pd.to_datetime(game_row.iloc[idx]['game_date']).strftime('%Y-%m-%d'),
                'predicted_spread': home_spread,
                # Use ensemble probability as primary
                'home_win_probability': float(final_win_prob[idx]),
                'away_win_probability': float(1 - final_win_prob[idx]),
                # Keep individual model outputs for transparency
                'home_win_prob_from_model': float(win_prob_proba[idx]),
                'win_prob_from_spread': float(win_prob_from_spread[idx]),
                'model_disagreement': float(disagreement_val),
                'predicted_winner': game_row.iloc[idx]['home_team'] if final_win_prob[idx] > 0.5 else game_row.iloc[idx]['away_team'],
                'confidence': float(abs(final_win_prob[idx] - 0.5) * 2),
                'spread_interpretation': (
                    f"{game_row.iloc[idx]['home_team']} by {abs(home_spread):.1f}"
                    if home_spread < 0
                    else f"{game_row.iloc[idx]['away_team']} by {abs(home_spread):.1f}"
                )
            }
            if include_explanations and idx < len(explanations):
                result['top_features'] = explanations[idx]
                
            results.append(result)
        
        return results[0] if len(results) == 1 else results

    def _get_local_explanations(self, X: pd.DataFrame, feature_names: List[str]) -> List[List[Dict]]:
        """
        Get local feature importance for each prediction.
        Falls back to feature values vs medians if model doesn't support contributions.
        """
        explanations = []
        
        # Try to get SHAP-like contributions if it's a LightGBM or XGBoost model
        # We focus on the spread model for explanations as it's often the most intuitive
        model = self.spread_model
        if self.is_ensemble:
            # For ensembles, use the first model for a representative explanation
            # or could average them, but first is faster and usually representative
            model = list(self.spread_model.values())[0]
            
        try:
            import lightgbm as lgb
            if isinstance(model, (lgb.LGBMRegressor, lgb.Booster)):
                # LightGBM supports fast SHAP values via pred_contrib
                if isinstance(model, lgb.LGBMRegressor):
                    booster = model.booster_
                else:
                    booster = model
                
                # pred_contrib=True returns (n_samples, n_features + 1)
                contribs = booster.predict(X, pred_contrib=True)
                
                for i in range(len(X)):
                    # Get feature names and their contributions
                    row_contribs = contribs[i, :-1]  # skip the expected value at the end
                    feat_impact = []
                    # Exclude time-based features and blowout-skewed features
                    exclude_explanations = ['week_number', 'month', 'point_diff_differential']
                    
                    for name, val, impact in zip(feature_names, X.iloc[i], row_contribs):
                        if name in exclude_explanations:
                            continue
                            
                        feat_impact.append({
                            'feature': name,
                            'value': float(val),
                            'impact': float(impact)
                        })
                    
                    # Sort by absolute impact
                    feat_impact = sorted(feat_impact, key=lambda x: abs(x['impact']), reverse=True)
                    explanations.append(feat_impact[:10])  # Top 10
                return explanations
        except Exception as e:
            # print(f"DEBUG: Could not use LightGBM contributions: {e}")
            pass

        # Fallback: Just return feature values relative to medians for the most important-sounding features
        # This is a heuristic but better than nothing
        medians = self._load_feature_medians()
        important_prefixes = ['win_pct', 'point_diff', 'rest', 'form_', 'opp_strength']
        
        for i in range(len(X)):
            feat_impact = []
            for name, val in X.iloc[i].items():
                if any(name.startswith(p) for p in important_prefixes):
                    median = medians.get(name, 0)
                    # Heuristic "impact": difference from median
                    # We don't know the direction/weight for sure here, so just report the delta
                    diff = val - median
                    feat_impact.append({
                        'feature': name,
                        'value': float(val),
                        'impact': float(diff), # This is not true impact, just delta
                        'is_heuristic': True
                    })
            
            # Sort by absolute delta
            feat_impact = sorted(feat_impact, key=lambda x: abs(x['impact']), reverse=True)
            explanations.append(feat_impact[:10])
            
        return explanations
    
    def predict_batch(self, games_df: pd.DataFrame, historical_games: pd.DataFrame,
                     play_by_play: Optional[pd.DataFrame] = None,
                     game_logs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict outcomes for multiple games.
        
        Returns:
            DataFrame with predictions
        """
        results = []
        for idx, game in games_df.iterrows():
            try:
                pred = self.predict(
                    pd.DataFrame([game]),
                    historical_games,
                    play_by_play,
                    game_logs
                )
                results.append(pred)
            except Exception as e:
                print(f"Error predicting {game.get('home_team')} vs {game.get('away_team')}: {e}")
                continue
        
        return pd.DataFrame(results)


def predict_single_game(home_team: str, away_team: str, game_date: str,
                       league: str, historical_games: pd.DataFrame,
                       play_by_play: Optional[pd.DataFrame] = None,
                       game_logs: Optional[pd.DataFrame] = None,
                       model_version: str = 'v1') -> Dict:
    """
    Convenience function to predict a single game.
    
    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        game_date: Game date (YYYY-MM-DD)
        league: 'NFL' or 'NBA'
        historical_games: Historical games DataFrame
        play_by_play: Optional PBP data (NFL)
        game_logs: Optional game logs (NBA)
        model_version: Model version
    
    Returns:
        Prediction dictionary
    """
    predictor = GamePredictor(league, model_version)
    
    game_row = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'game_date': [game_date],
        'season': [pd.to_datetime(game_date).year]
    })
    
    return predictor.predict(game_row, historical_games, play_by_play, game_logs)
