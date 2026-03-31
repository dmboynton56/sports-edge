"""
CBB Feature Engineering
=======================
Builds pairwise matchup feature store for NCAA tournament prediction.

Each row represents a potential Team A vs Team B matchup with features
computed using ONLY data available before the game date (no leakage).
"""

import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'cache')


def compute_strength_of_schedule(
    team_stats: pd.DataFrame, results_df: pd.DataFrame, season: int
) -> pd.Series:
    """
    SOS = average win_pct of opponents.
    Uses only regular season data for the given season.
    """
    season_results = results_df[results_df['Season'] == season]
    season_stats = team_stats[team_stats['Season'] == season].set_index('TeamID')

    sos = {}
    for tid in season_stats.index:
        opp_ids = []
        wins = season_results[season_results['WTeamID'] == tid]['LTeamID'].tolist()
        losses = season_results[season_results['LTeamID'] == tid]['WTeamID'].tolist()
        opp_ids = wins + losses
        if opp_ids:
            opp_wp = [
                season_stats.loc[oid, 'win_pct']
                for oid in opp_ids if oid in season_stats.index
            ]
            sos[tid] = np.mean(opp_wp) if opp_wp else 0.5
        else:
            sos[tid] = 0.5

    return pd.Series(sos, name='sos')


def compute_recent_form(
    results_df: pd.DataFrame, season: int, last_n: int = 10
) -> pd.DataFrame:
    """
    Computes recent form metrics from the last N games of the regular season.
    Returns win_pct, avg point margin, avg offensive/defensive efficiency.
    """
    season_results = results_df[results_df['Season'] == season].copy()
    if season_results.empty:
        return pd.DataFrame()

    season_results = season_results.sort_values('DayNum')

    form_data = {}
    for _, row in season_results.iterrows():
        for prefix, opp_prefix, won in [('W', 'L', True), ('L', 'W', False)]:
            tid = int(row[f'{prefix}TeamID'])
            if tid not in form_data:
                form_data[tid] = []
            margin = row[f'{prefix}Score'] - row[f'{opp_prefix}Score']
            form_data[tid].append({
                'day': row['DayNum'], 'won': won, 'margin': margin,
                'score': row[f'{prefix}Score'],
                'opp_score': row[f'{opp_prefix}Score'],
            })

    rows = []
    for tid, games in form_data.items():
        recent = games[-last_n:]
        if not recent:
            continue
        rows.append({
            'Season': season, 'TeamID': tid,
            'recent_win_pct': np.mean([g['won'] for g in recent]),
            'recent_avg_margin': np.mean([g['margin'] for g in recent]),
            'recent_ppg': np.mean([g['score'] for g in recent]),
            'recent_opp_ppg': np.mean([g['opp_score'] for g in recent]),
        })

    return pd.DataFrame(rows)


def compute_tourney_experience(
    seeds_df: pd.DataFrame, current_season: int, lookback: int = 5
) -> pd.DataFrame:
    """
    Tournament experience: appearances, avg seed in last N years.
    """
    if seeds_df.empty:
        return pd.DataFrame()

    min_season = current_season - lookback
    hist = seeds_df[
        (seeds_df['Season'] >= min_season) &
        (seeds_df['Season'] < current_season)
    ]

    if hist.empty:
        return pd.DataFrame()

    exp = hist.groupby('TeamID').agg(
        tourney_appearances=('Season', 'count'),
        avg_seed=('SeedNum', 'mean'),
        best_seed=('SeedNum', 'min'),
    ).reset_index()
    exp['Season'] = current_season

    return exp


def build_team_features(
    team_stats: pd.DataFrame,
    results_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    season: int
) -> pd.DataFrame:
    """
    Builds comprehensive per-team features for a single season.
    All features use only data available BEFORE the tournament.
    """
    stats = team_stats[team_stats['Season'] == season].copy()
    if stats.empty:
        return pd.DataFrame()

    sos = compute_strength_of_schedule(team_stats, results_df, season)
    stats = stats.set_index('TeamID')
    stats['sos'] = sos
    stats = stats.reset_index()

    form = compute_recent_form(results_df, season)
    if not form.empty:
        stats = stats.merge(form, on=['Season', 'TeamID'], how='left')

    exp = compute_tourney_experience(seeds_df, season)
    if not exp.empty:
        stats = stats.merge(
            exp[['TeamID', 'tourney_appearances', 'avg_seed', 'best_seed']],
            on='TeamID', how='left'
        )

    season_seeds = seeds_df[seeds_df['Season'] == season] if not seeds_df.empty else pd.DataFrame()
    if not season_seeds.empty:
        stats = stats.merge(
            season_seeds[['TeamID', 'SeedNum', 'Region']],
            on='TeamID', how='left'
        )

    fill_cols = ['tourney_appearances', 'avg_seed', 'best_seed']
    for col in fill_cols:
        if col in stats.columns:
            stats[col] = stats[col].fillna(0)

    return stats


MATCHUP_FEATURE_COLS = [
    'off_eff_diff', 'def_eff_diff', 'net_eff_diff',
    'tempo_diff', 'tempo_mismatch',
    'off_efg_diff', 'off_to_rate_diff', 'off_or_rate_diff', 'off_ftr_diff',
    'def_efg_diff', 'def_to_rate_diff', 'def_or_rate_diff', 'def_ftr_diff',
    'win_pct_diff', 'sos_diff',
    'recent_win_pct_diff', 'recent_avg_margin_diff',
    'tourney_exp_diff',
    'a_off_eff', 'a_def_eff', 'a_net_eff', 'a_tempo',
    'a_off_efg', 'a_off_to_rate', 'a_off_or_rate', 'a_off_ftr',
    'a_def_efg', 'a_def_to_rate', 'a_def_or_rate', 'a_def_ftr',
    'a_win_pct', 'a_sos',
    'b_off_eff', 'b_def_eff', 'b_net_eff', 'b_tempo',
    'b_off_efg', 'b_off_to_rate', 'b_off_or_rate', 'b_off_ftr',
    'b_def_efg', 'b_def_to_rate', 'b_def_or_rate', 'b_def_ftr',
    'b_win_pct', 'b_sos',
]


def build_matchup_row(
    team_a: pd.Series, team_b: pd.Series
) -> dict:
    """
    Creates a single matchup feature row from two team feature Series.
    Features include both differentials (A - B) and raw values for each team.
    """
    row = {}

    seed_a = team_a.get('SeedNum', 8)
    seed_b = team_b.get('SeedNum', 8)
    row['seed_diff'] = seed_a - seed_b

    diff_features = [
        ('off_eff', 'off_eff_diff'),
        ('def_eff', 'def_eff_diff'),
        ('net_eff', 'net_eff_diff'),
        ('tempo', 'tempo_diff'),
        ('off_efg', 'off_efg_diff'),
        ('off_to_rate', 'off_to_rate_diff'),
        ('off_or_rate', 'off_or_rate_diff'),
        ('off_ftr', 'off_ftr_diff'),
        ('def_efg', 'def_efg_diff'),
        ('def_to_rate', 'def_to_rate_diff'),
        ('def_or_rate', 'def_or_rate_diff'),
        ('def_ftr', 'def_ftr_diff'),
        ('win_pct', 'win_pct_diff'),
        ('sos', 'sos_diff'),
    ]
    for src, dst in diff_features:
        a_val = team_a.get(src, 0) or 0
        b_val = team_b.get(src, 0) or 0
        row[dst] = a_val - b_val

    row['tempo_mismatch'] = abs(
        (team_a.get('tempo', 0) or 0) - (team_b.get('tempo', 0) or 0)
    )

    row['recent_win_pct_diff'] = (
        (team_a.get('recent_win_pct', 0.5) or 0.5) -
        (team_b.get('recent_win_pct', 0.5) or 0.5)
    )
    row['recent_avg_margin_diff'] = (
        (team_a.get('recent_avg_margin', 0) or 0) -
        (team_b.get('recent_avg_margin', 0) or 0)
    )

    row['tourney_exp_diff'] = (
        (team_a.get('tourney_appearances', 0) or 0) -
        (team_b.get('tourney_appearances', 0) or 0)
    )
    row['avg_seed_hist_diff'] = (
        (team_a.get('avg_seed', 8) or 8) -
        (team_b.get('avg_seed', 8) or 8)
    )

    raw_features = [
        'off_eff', 'def_eff', 'net_eff', 'tempo',
        'off_efg', 'off_to_rate', 'off_or_rate', 'off_ftr',
        'def_efg', 'def_to_rate', 'def_or_rate', 'def_ftr',
        'win_pct', 'sos',
    ]
    for feat in raw_features:
        row[f'a_{feat}'] = team_a.get(feat, 0) or 0
        row[f'b_{feat}'] = team_b.get(feat, 0) or 0

    return row


def build_tournament_matchups(
    team_features: pd.DataFrame,
    tourney_results: pd.DataFrame,
    season: int
) -> pd.DataFrame:
    """
    Builds matchup rows for all actual tournament games in a season.
    Each game generates TWO rows (A vs B and B vs A) for symmetry.
    Target: team_a_wins (1 if Team A won).
    """
    season_tourney = tourney_results[tourney_results['Season'] == season]
    season_feats = team_features[team_features['Season'] == season].set_index('TeamID')

    if season_tourney.empty or season_feats.empty:
        return pd.DataFrame()

    rows = []
    for _, game in season_tourney.iterrows():
        w_id = int(game['WTeamID'])
        l_id = int(game['LTeamID'])

        if w_id not in season_feats.index or l_id not in season_feats.index:
            continue

        team_w = season_feats.loc[w_id]
        team_l = season_feats.loc[l_id]

        row_wl = build_matchup_row(team_w, team_l)
        row_wl['Season'] = season
        row_wl['TeamA_ID'] = w_id
        row_wl['TeamB_ID'] = l_id
        row_wl['team_a_wins'] = 1
        rows.append(row_wl)

        row_lw = build_matchup_row(team_l, team_w)
        row_lw['Season'] = season
        row_lw['TeamA_ID'] = l_id
        row_lw['TeamB_ID'] = w_id
        row_lw['team_a_wins'] = 0
        rows.append(row_lw)

    return pd.DataFrame(rows)


def build_full_feature_store(
    team_stats: pd.DataFrame,
    results_df: pd.DataFrame,
    tourney_results: pd.DataFrame,
    seeds_df: pd.DataFrame,
    min_season: int = 2010,
    max_season: int = 2025
) -> pd.DataFrame:
    """
    Builds the full pairwise matchup feature store across all seasons.
    This is the primary training dataset.
    """
    print("Building full CBB matchup feature store...")

    all_matchups = []
    for season in range(min_season, max_season + 1):
        if season == 2020:
            continue

        team_feats = build_team_features(team_stats, results_df, seeds_df, season)
        if team_feats.empty:
            print(f"  Season {season}: No team features, skipping")
            continue

        matchups = build_tournament_matchups(team_feats, tourney_results, season)
        if not matchups.empty:
            all_matchups.append(matchups)
            print(f"  Season {season}: {len(matchups)} matchup rows")
        else:
            print(f"  Season {season}: No tournament matchups found")

    if not all_matchups:
        print("No matchup data found across any season.")
        return pd.DataFrame()

    feature_store = pd.concat(all_matchups, ignore_index=True)

    cache_path = os.path.join(CACHE_DIR, 'cbb_matchup_feature_store.csv')
    os.makedirs(CACHE_DIR, exist_ok=True)
    feature_store.to_csv(cache_path, index=False)
    print(f"\nFeature store saved to {cache_path}: {feature_store.shape}")

    return feature_store


def build_prediction_matchups(
    team_features: pd.DataFrame,
    team_ids: List[int],
    season: int
) -> pd.DataFrame:
    """
    Builds ALL possible pairwise matchups among the given teams for prediction.
    Used to generate the 68x68 probability matrix on Selection Sunday.
    """
    season_feats = team_features[team_features['Season'] == season].set_index('TeamID')

    rows = []
    for i, tid_a in enumerate(team_ids):
        for j, tid_b in enumerate(team_ids):
            if i == j:
                continue
            if tid_a not in season_feats.index or tid_b not in season_feats.index:
                continue

            team_a = season_feats.loc[tid_a]
            team_b = season_feats.loc[tid_b]

            row = build_matchup_row(team_a, team_b)
            row['Season'] = season
            row['TeamA_ID'] = tid_a
            row['TeamB_ID'] = tid_b
            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    from cbb_data_loader import CBBDataManager

    dm = CBBDataManager()
    team_stats, tourney, seeds = dm.build_historical_dataset()

    if not team_stats.empty and not tourney.empty:
        results = dm.kaggle.load_regular_season()
        feature_store = build_full_feature_store(
            team_stats, results, tourney, seeds
        )
        print(f"\nFinal feature store: {feature_store.shape}")
        print(feature_store.head())
    else:
        print("No data available. Place Kaggle MMLM CSVs in data-core/data/cbb/")
