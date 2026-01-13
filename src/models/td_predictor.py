import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Dict, Optional
import lightgbm as lgb
from src.data import nfl_fetcher

from src.data.pbp_loader import load_pbp

class TDScorerPredictor:
    """
    Predicts the probability of players scoring a touchdown in a given game.
    """
    
    def __init__(self, model_version: str = 'v1'):
        self.model_version = model_version
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        self.model = None
        self.feature_names = None
        
        # Positions we care about for TD scoring
        self.target_positions = ['QB', 'RB', 'WR', 'TE']
    
    def load_model(self):
        """Load the trained TD scorer model."""
        model_path = os.path.join(self.models_dir, f'td_scorer_model_nfl_{self.model_version}.pkl')
        if not os.path.exists(model_path):
            return False
            
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
        return True

    def _add_redzone_features(self, df: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Add player-level red zone opportunity features and team-level RZ efficiency from PBP data.
        """
        if pbp is None or pbp.empty:
            # Add columns with zeros if PBP is missing
            for col in ['rolling_rz_opp_3', 'rolling_rz_opp_5', 'rolling_ten_opp_3', 'rolling_ten_opp_5', 
                        'rolling_five_opp_3', 'rolling_five_opp_5']:
                df[col] = 0.0
            return df, pd.DataFrame()

        # Filter PBP to plays that could result in a TD (run/pass)
        pbp_plays = pbp[pbp['play_type'].isin(['run', 'pass'])].copy()
        
        # Red zone flags
        pbp_plays['is_rz'] = (pbp_plays['yardline_100'] <= 20).astype(int)
        pbp_plays['is_ten'] = (pbp_plays['yardline_100'] <= 10).astype(int)
        pbp_plays['is_five'] = (pbp_plays['yardline_100'] <= 5).astype(int)
        
        # Player opportunities
        # Rusher
        rusher_rz = pbp_plays[pbp_plays['is_rz'] == 1].groupby(['rusher_player_id', 'game_id']).size().reset_index(name='rz_carries')
        rusher_ten = pbp_plays[pbp_plays['is_ten'] == 1].groupby(['rusher_player_id', 'game_id']).size().reset_index(name='ten_carries')
        rusher_five = pbp_plays[pbp_plays['is_five'] == 1].groupby(['rusher_player_id', 'game_id']).size().reset_index(name='five_carries')
        
        # Receiver (targets)
        receiver_rz = pbp_plays[pbp_plays['is_rz'] == 1].groupby(['receiver_player_id', 'game_id']).size().reset_index(name='rz_targets')
        receiver_ten = pbp_plays[pbp_plays['is_ten'] == 1].groupby(['receiver_player_id', 'game_id']).size().reset_index(name='ten_targets')
        receiver_five = pbp_plays[pbp_plays['is_five'] == 1].groupby(['receiver_player_id', 'game_id']).size().reset_index(name='five_targets')

        # Process rushers
        rusher_stats = rusher_rz.merge(rusher_ten, on=['rusher_player_id', 'game_id'], how='outer')
        rusher_stats = rusher_stats.merge(rusher_five, on=['rusher_player_id', 'game_id'], how='outer').fillna(0)
        rusher_stats.rename(columns={'rusher_player_id': 'player_id'}, inplace=True)
        
        # Process receivers
        receiver_stats = receiver_rz.merge(receiver_ten, on=['receiver_player_id', 'game_id'], how='outer')
        receiver_stats = receiver_stats.merge(receiver_five, on=['receiver_player_id', 'game_id'], how='outer').fillna(0)
        receiver_stats.rename(columns={'receiver_player_id': 'player_id'}, inplace=True)
        
        # Combine rush and rec
        pbp_stats = pd.concat([
            rusher_stats.assign(rz_opp=rusher_stats['rz_carries'], ten_opp=rusher_stats['ten_carries'], five_opp=rusher_stats['five_carries']),
            receiver_stats.assign(rz_opp=receiver_stats['rz_targets'], ten_opp=receiver_stats['ten_targets'], five_opp=receiver_stats['five_targets'])
        ]).groupby(['player_id', 'game_id']).agg({
            'rz_opp': 'sum',
            'ten_opp': 'sum',
            'five_opp': 'sum'
        }).reset_index()

        # Team RZ Efficiency (Offense)
        # % of games where team scored a TD when in RZ
        team_rz_off = pbp_plays[pbp_plays['is_rz'] == 1].groupby(['posteam', 'game_id']).agg({
            'touchdown': 'max'
        }).reset_index().groupby('posteam')['touchdown'].mean().reset_index(name='team_rz_eff')
        
        # Team RZ Allowed Efficiency (Defense)
        # % of games where defense allowed a TD when opponent in RZ
        team_rz_def = pbp_plays[pbp_plays['is_rz'] == 1].groupby(['defteam', 'game_id']).agg({
            'touchdown': 'max'
        }).reset_index().groupby('defteam')['touchdown'].mean().reset_index(name='opp_rz_allowed_eff')
        
        return df, pbp_stats, team_rz_off, team_rz_def

    def _prepare_features(self, weekly_data: pd.DataFrame, schedule: pd.DataFrame, pbp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for training or inference.
        """
        df = weekly_data.copy()
        
        # Calculate total TDs (rushing + receiving)
        df['total_tds'] = df['rushing_tds'].fillna(0) + df['receiving_tds'].fillna(0)
        df['has_td'] = (df['total_tds'] > 0).astype(int)
        
        # Sort for rolling calculations
        df = df.sort_values(['player_id', 'season', 'week'])
        
        # Rolling features per player
        for window in [3, 5]:
            df[f'rolling_tds_{window}'] = df.groupby('player_id')['total_tds'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f'rolling_yards_{window}'] = (df.groupby('player_id')['rushing_yards'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()) + 
                                           df.groupby('player_id')['receiving_yards'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()))
            if 'targets' in df.columns:
                df[f'rolling_targets_{window}'] = df.groupby('player_id')['targets'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            if 'carries' in df.columns:
                df[f'rolling_carries_{window}'] = df.groupby('player_id')['carries'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # If PBP is provided, add red zone opportunities
        if pbp is not None:
            if schedule is not None:
                sched_map = schedule[['season', 'week', 'home_team', 'away_team', 'game_id']].copy()
                home_map = sched_map[['season', 'week', 'home_team', 'away_team', 'game_id']].rename(columns={'home_team': 'recent_team', 'away_team': 'opponent'})
                away_map = sched_map[['season', 'week', 'away_team', 'home_team', 'game_id']].rename(columns={'away_team': 'recent_team', 'home_team': 'opponent'})
                team_game_map = pd.concat([home_map, away_map])
                
                df = df.merge(team_game_map, on=['season', 'week', 'recent_team'], how='left')
                
                # Get RZ stats from PBP
                _, rz_stats, team_eff, opp_eff = self._add_redzone_features(df, pbp)
                
                # Merge RZ stats into df
                df = df.merge(rz_stats, on=['player_id', 'game_id'], how='left')
                df['rz_opp'] = df['rz_opp'].fillna(0)
                df['ten_opp'] = df['ten_opp'].fillna(0)
                df['five_opp'] = df['five_opp'].fillna(0)
                
                # Calculate rolling RZ features
                for window in [3, 5]:
                    df[f'rolling_rz_opp_{window}'] = df.groupby('player_id')['rz_opp'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    df[f'rolling_ten_opp_{window}'] = df.groupby('player_id')['ten_opp'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                    df[f'rolling_five_opp_{window}'] = df.groupby('player_id')['five_opp'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                
                # Add team RZ efficiency
                df = df.merge(team_eff, left_on='recent_team', right_on='posteam', how='left').drop(columns=['posteam'])
                df['team_rz_eff'] = df['team_rz_eff'].fillna(0.5)
                
                # Add opponent RZ allowed efficiency
                df = df.merge(opp_eff, left_on='opponent', right_on='defteam', how='left').drop(columns=['defteam'])
                df['opp_rz_allowed_eff'] = df['opp_rz_allowed_eff'].fillna(0.5)
            else:
                for window in [3, 5]:
                    df[f'rolling_rz_opp_{window}'] = 0.0
                    df[f'rolling_ten_opp_{window}'] = 0.0
                    df[f'rolling_five_opp_{window}'] = 0.0
                df['team_rz_eff'] = 0.5
                df['opp_rz_allowed_eff'] = 0.5
        else:
            for window in [3, 5]:
                df[f'rolling_rz_opp_{window}'] = 0.0
                df[f'rolling_ten_opp_{window}'] = 0.0
                df[f'rolling_five_opp_{window}'] = 0.0
            df['team_rz_eff'] = 0.5
            df['opp_rz_allowed_eff'] = 0.5

        return df

    def train(self, seasons: List[int]):
        """Train the TD scorer model."""
        print(f"Loading weekly data for seasons: {seasons}")
        weekly = nfl_fetcher.fetch_nfl_weekly_data(seasons)
        all_schedules = pd.concat([nfl_fetcher.fetch_nfl_schedule(s) for s in seasons])
        
        print(f"Loading PBP data for seasons: {seasons}")
        pbp = load_pbp(seasons)
        
        # Filter to relevant positions
        weekly = weekly[weekly['position'].isin(self.target_positions)].copy()
        
        print("Preparing features...")
        df = self._prepare_features(weekly, all_schedules, pbp)
        
        # Define features
        features = [
            'rolling_tds_3', 'rolling_tds_5', 
            'rolling_yards_3', 'rolling_yards_5',
            'rolling_targets_3', 'rolling_targets_5',
            'rolling_carries_3', 'rolling_carries_5',
            'rolling_rz_opp_3', 'rolling_rz_opp_5',
            'rolling_ten_opp_3', 'rolling_ten_opp_5',
            'rolling_five_opp_3', 'rolling_five_opp_5',
            'team_rz_eff', 'opp_rz_allowed_eff'
        ]
        
        # Drop rows with NaNs in features (beginning of season for players)
        train_df = df.dropna(subset=features)
        
        X = train_df[features]
        y = train_df['has_td']
        
        print(f"Training model on {len(train_df)} samples...")
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X, y)
        
        self.model = model
        self.feature_names = features
        
        # Save model
        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, f'td_scorer_model_nfl_{self.model_version}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'feature_names': features}, f)
        print(f"Model saved to {model_path}")

    def predict(self, players_df: pd.DataFrame, include_explanations: bool = False) -> pd.DataFrame:
        """
        Predict TD probabilities for a list of players.
        players_df should have the required features.
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Model not trained or loaded.")
                
        X = players_df[self.feature_names]
        probs = self.model.predict_proba(X)[:, 1]
        
        result = players_df.copy()
        result['td_probability'] = probs
        
        if include_explanations:
            explanations = self._get_local_explanations(X)
            result['top_features'] = explanations
            
        return result[['player_id', 'player_display_name', 'position', 'recent_team', 'td_probability'] + (['top_features'] if include_explanations else [])]

    def _get_local_explanations(self, X: pd.DataFrame) -> List[List[Dict]]:
        """
        Get local feature importance for each prediction.
        """
        explanations = []
        try:
            # For LightGBM Classifier, predict_contrib returns raw score contributions (logits)
            # The last value in each row is the expected value (base score)
            contribs = self.model.booster_.predict(X, pred_contrib=True)
            
            for i in range(len(X)):
                # Get feature names and their contributions
                row_contribs = contribs[i, :-1]  # skip the expected value at the end
                feat_impact = []
                for name, val, impact in zip(self.feature_names, X.iloc[i], row_contribs):
                    feat_impact.append({
                        'feature': name,
                        'value': float(val),
                        'impact': float(impact)
                    })
                
                # Sort by absolute impact
                feat_impact = sorted(feat_impact, key=lambda x: abs(x['impact']), reverse=True)
                explanations.append(feat_impact[:5])  # Top 5 for players
        except Exception:
            # Fallback to empty if anything fails
            explanations = [[] for _ in range(len(X))]
            
        return explanations
