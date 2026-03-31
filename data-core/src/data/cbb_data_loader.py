"""
CBB Data Loader
===============
Fetches college basketball data from multiple sources:
- Barttorvik API (T-Rank ratings, four factors, tournament results)
- Kaggle March Machine Learning Mania CSVs (historical game data)
- ESPN (current bracket/seeds)
"""

import os
import time
import json
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cbb')
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'cache')

BARTTORVIK_BASE = "https://barttorvik.com"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

CURRENT_SEASON = 2026


# ---------------------------------------------------------------------------
# Barttorvik Scraper
# ---------------------------------------------------------------------------
class BarttovikClient:
    """Fetches T-Rank ratings, four factors, and tournament data from Barttorvik."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_ratings(self, year: int = CURRENT_SEASON, use_cache: bool = True) -> pd.DataFrame:
        cache_path = os.path.join(self.cache_dir, f'bart_ratings_{year}.csv')
        if use_cache and os.path.exists(cache_path):
            return pd.read_csv(cache_path)

        url = f"{BARTTORVIK_BASE}/trank.php"
        params = {
            'year': year, 'sort': '', 'top': 0,
            'conlimit': 'All', 'venue': 'All', 'type': 'All',
        }
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            df = self._parse_ratings_html(resp.text, year)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            print(f"Error fetching Barttorvik ratings for {year}: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path)
            return pd.DataFrame()

    def get_ratings_range(
        self, start_year: int = 2010, end_year: int = CURRENT_SEASON,
        use_cache: bool = True
    ) -> pd.DataFrame:
        frames = []
        for year in range(start_year, end_year + 1):
            if year == 2020:
                continue
            df = self.get_ratings(year, use_cache=use_cache)
            if not df.empty:
                frames.append(df)
            time.sleep(1.0)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    def _parse_ratings_html(self, html: str, year: int) -> pd.DataFrame:
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table', {'id': 'trank-table'})
        if table is None:
            tables = soup.find_all('table')
            table = tables[0] if tables else None
        if table is None:
            return pd.DataFrame()

        rows_data = []
        rows = table.find_all('tr')
        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 10:
                continue
            text = [c.get_text(strip=True) for c in cells]
            rows_data.append(text)

        if not rows_data:
            return pd.DataFrame()

        col_count = len(rows_data[0])
        base_cols = [
            'rank', 'team', 'conf', 'record', 'adj_o', 'adj_d',
            'barthag', 'adj_t', 'wab', 'seed'
        ]
        if col_count > len(base_cols):
            cols = base_cols + [f'col_{i}' for i in range(len(base_cols), col_count)]
        else:
            cols = base_cols[:col_count]

        df = pd.DataFrame(rows_data, columns=cols)
        df['year'] = year

        for col in ['adj_o', 'adj_d', 'barthag', 'adj_t', 'wab']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'rank' in df.columns:
            df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

        return df

    def get_four_factors(self, year: int = CURRENT_SEASON, use_cache: bool = True) -> pd.DataFrame:
        cache_path = os.path.join(self.cache_dir, f'bart_factors_{year}.csv')
        if use_cache and os.path.exists(cache_path):
            return pd.read_csv(cache_path)

        url = f"{BARTTORVIK_BASE}/teamsheets.php"
        params = {'year': year, 'sort': 1, 'conlimit': 'All'}
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            df = self._parse_factors_html(resp.text, year)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            print(f"Error fetching four factors for {year}: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path)
            return pd.DataFrame()

    def _parse_factors_html(self, html: str, year: int) -> pd.DataFrame:
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table')
        if not tables:
            return pd.DataFrame()

        rows_data = []
        table = tables[0]
        for row in table.find_all('tr')[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 5:
                continue
            text = [c.get_text(strip=True) for c in cells]
            rows_data.append(text)

        if not rows_data:
            return pd.DataFrame()

        col_count = len(rows_data[0])
        base_cols = [
            'team', 'conf', 'adj_o', 'adj_d', 'barthag',
            'off_efg', 'off_to', 'off_or', 'off_ftr',
            'def_efg', 'def_to', 'def_or', 'def_ftr',
            'tempo', 'wins', 'losses'
        ]
        cols = base_cols[:col_count] if col_count <= len(base_cols) else \
            base_cols + [f'col_{i}' for i in range(len(base_cols), col_count)]

        df = pd.DataFrame(rows_data, columns=cols)
        df['year'] = year

        numeric_cols = [
            'adj_o', 'adj_d', 'barthag', 'off_efg', 'off_to', 'off_or',
            'off_ftr', 'def_efg', 'def_to', 'def_or', 'def_ftr', 'tempo'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_tourney_results(
        self, min_year: int = 2010, max_year: int = CURRENT_SEASON
    ) -> pd.DataFrame:
        cache_path = os.path.join(self.cache_dir, f'bart_tourney_{min_year}_{max_year}.csv')
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Kaggle MMLM CSV Loader
# ---------------------------------------------------------------------------
class KaggleMMLMLoader:
    """
    Loads Kaggle March Machine Learning Mania dataset CSVs.

    Expected files in data_dir:
        MRegularSeasonDetailedResults.csv
        MNCAATourneyDetailedResults.csv
        MNCAATourneySeeds.csv
        MTeams.csv
        MSeasons.csv (optional)
        MGameCities.csv (optional)
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_teams(self) -> pd.DataFrame:
        return self._load_csv('MTeams.csv')

    def load_seeds(self) -> pd.DataFrame:
        df = self._load_csv('MNCAATourneySeeds.csv')
        if df.empty:
            return df
        df['SeedNum'] = df['Seed'].str.extract(r'(\d+)').astype(int)
        df['Region'] = df['Seed'].str[0]
        return df

    def load_regular_season(self) -> pd.DataFrame:
        return self._load_csv('MRegularSeasonDetailedResults.csv')

    def load_tourney_results(self) -> pd.DataFrame:
        return self._load_csv('MNCAATourneyDetailedResults.csv')

    def load_regular_season_compact(self) -> pd.DataFrame:
        return self._load_csv('MRegularSeasonCompactResults.csv')

    def load_tourney_compact(self) -> pd.DataFrame:
        return self._load_csv('MNCAATourneyCompactResults.csv')

    def compute_season_stats(
        self, results_df: pd.DataFrame, season: int
    ) -> pd.DataFrame:
        """
        Computes per-team season-level stats from detailed game results.
        Calculates four factors, efficiency, tempo from raw box scores.
        """
        df = results_df[results_df['Season'] == season].copy()
        if df.empty:
            return pd.DataFrame()

        team_stats = {}
        for _, row in df.iterrows():
            for prefix, opp_prefix in [('W', 'L'), ('L', 'W')]:
                tid = int(row[f'{prefix}TeamID'])
                if tid not in team_stats:
                    team_stats[tid] = {
                        'games': 0, 'wins': 0, 'points_for': 0, 'points_against': 0,
                        'fgm': 0, 'fga': 0, 'fgm3': 0, 'fga3': 0,
                        'ftm': 0, 'fta': 0, 'or': 0, 'dr': 0,
                        'ast': 0, 'to': 0, 'stl': 0, 'blk': 0, 'pf': 0,
                        'opp_fgm': 0, 'opp_fga': 0, 'opp_fgm3': 0, 'opp_fga3': 0,
                        'opp_ftm': 0, 'opp_fta': 0, 'opp_or': 0, 'opp_dr': 0,
                        'opp_to': 0, 'opp_fta_r': 0,
                    }
                s = team_stats[tid]
                s['games'] += 1
                s['wins'] += 1 if prefix == 'W' else 0
                s['points_for'] += row[f'{prefix}Score']
                s['points_against'] += row[f'{opp_prefix}Score']
                s['fgm'] += row[f'{prefix}FGM']
                s['fga'] += row[f'{prefix}FGA']
                s['fgm3'] += row[f'{prefix}FGM3']
                s['fga3'] += row[f'{prefix}FGA3']
                s['ftm'] += row[f'{prefix}FTM']
                s['fta'] += row[f'{prefix}FTA']
                s['or'] += row[f'{prefix}OR']
                s['dr'] += row[f'{prefix}DR']
                s['ast'] += row[f'{prefix}Ast']
                s['to'] += row[f'{prefix}TO']
                s['stl'] += row[f'{prefix}Stl']
                s['blk'] += row[f'{prefix}Blk']
                s['pf'] += row[f'{prefix}PF']
                s['opp_fgm'] += row[f'{opp_prefix}FGM']
                s['opp_fga'] += row[f'{opp_prefix}FGA']
                s['opp_fgm3'] += row[f'{opp_prefix}FGM3']
                s['opp_fga3'] += row[f'{opp_prefix}FGA3']
                s['opp_ftm'] += row[f'{opp_prefix}FTM']
                s['opp_fta'] += row[f'{opp_prefix}FTA']
                s['opp_or'] += row[f'{opp_prefix}OR']
                s['opp_dr'] += row[f'{opp_prefix}DR']
                s['opp_to'] += row[f'{opp_prefix}TO']

        rows = []
        for tid, s in team_stats.items():
            g = s['games']
            if g == 0:
                continue

            poss = s['fga'] - s['or'] + s['to'] + 0.475 * s['fta']
            opp_poss = s['opp_fga'] - s['opp_or'] + s['opp_to'] + 0.475 * s['opp_fta']

            poss_pg = poss / g if g > 0 else 1
            opp_poss_pg = opp_poss / g if g > 0 else 1

            off_eff = (s['points_for'] / poss * 100) if poss > 0 else 0
            def_eff = (s['points_against'] / opp_poss * 100) if opp_poss > 0 else 0
            tempo = (poss + opp_poss) / (2 * g) if g > 0 else 0

            off_efg = (s['fgm'] + 0.5 * s['fgm3']) / s['fga'] if s['fga'] > 0 else 0
            off_to_rate = s['to'] / poss if poss > 0 else 0
            off_or_rate = s['or'] / (s['or'] + s['opp_dr']) if (s['or'] + s['opp_dr']) > 0 else 0
            off_ftr = s['ftm'] / s['fga'] if s['fga'] > 0 else 0

            def_efg = (s['opp_fgm'] + 0.5 * s['opp_fgm3']) / s['opp_fga'] if s['opp_fga'] > 0 else 0
            def_to_rate = s['opp_to'] / opp_poss if opp_poss > 0 else 0
            def_or_rate = s['opp_or'] / (s['opp_or'] + s['dr']) if (s['opp_or'] + s['dr']) > 0 else 0
            def_ftr = s['opp_ftm'] / s['opp_fga'] if s['opp_fga'] > 0 else 0

            win_pct = s['wins'] / g

            rows.append({
                'Season': season, 'TeamID': tid,
                'games': g, 'wins': s['wins'], 'losses': g - s['wins'],
                'win_pct': win_pct,
                'points_per_game': s['points_for'] / g,
                'points_allowed_per_game': s['points_against'] / g,
                'off_eff': off_eff, 'def_eff': def_eff,
                'net_eff': off_eff - def_eff,
                'tempo': tempo,
                'off_efg': off_efg, 'off_to_rate': off_to_rate,
                'off_or_rate': off_or_rate, 'off_ftr': off_ftr,
                'def_efg': def_efg, 'def_to_rate': def_to_rate,
                'def_or_rate': def_or_rate, 'def_ftr': def_ftr,
                'ast_per_game': s['ast'] / g,
                'stl_per_game': s['stl'] / g,
                'blk_per_game': s['blk'] / g,
            })

        return pd.DataFrame(rows)

    def compute_all_season_stats(
        self, min_season: int = 2010, max_season: int = CURRENT_SEASON
    ) -> pd.DataFrame:
        results = self.load_regular_season()
        if results.empty:
            return pd.DataFrame()

        frames = []
        for season in range(min_season, max_season + 1):
            if season == 2020:
                continue
            stats = self.compute_season_stats(results, season)
            if not stats.empty:
                frames.append(stats)
                print(f"  Season {season}: {len(stats)} teams")

        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# ESPN Bracket Fetcher
# ---------------------------------------------------------------------------
class ESPNBracketFetcher:
    """Fetches current NCAA tournament bracket from ESPN."""

    def __init__(self):
        self.session = requests.Session()

    def get_tournament_bracket(self, year: int = CURRENT_SEASON) -> Dict:
        url = f"{ESPN_BASE}/tournament/bracket"
        params = {'season': year}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error fetching ESPN bracket: {e}")
            return {}

    def get_team_rankings(self, limit: int = 100) -> pd.DataFrame:
        url = f"{ESPN_BASE}/rankings"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = []
            for ranking in data.get('rankings', []):
                for rank_entry in ranking.get('ranks', []):
                    team = rank_entry.get('team', {})
                    rows.append({
                        'rank': rank_entry.get('current'),
                        'team': team.get('location', ''),
                        'team_id': team.get('id', ''),
                        'record': rank_entry.get('recordSummary', ''),
                    })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error fetching ESPN rankings: {e}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Unified Data Manager
# ---------------------------------------------------------------------------
class CBBDataManager:
    """
    Orchestrates data loading from all sources and builds
    a unified dataset for model training and prediction.
    """

    def __init__(self, data_dir: str = DATA_DIR, cache_dir: str = CACHE_DIR):
        self.kaggle = KaggleMMLMLoader(data_dir)
        self.barttorvik = BarttovikClient(cache_dir)
        self.espn = ESPNBracketFetcher()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def build_historical_dataset(
        self, min_season: int = 2010, max_season: int = CURRENT_SEASON
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Builds the core datasets needed for training:
        1. team_stats: per-team per-season stats (from Kaggle detailed results)
        2. tourney_results: historical tournament game results
        3. seeds: tournament seedings
        """
        print("Building historical CBB dataset...")

        print("\n1. Computing season-level team stats from game data...")
        team_stats = self.kaggle.compute_all_season_stats(min_season, max_season)

        print("\n2. Loading tournament results...")
        tourney = self.kaggle.load_tourney_results()
        if tourney.empty:
            tourney = self.kaggle.load_tourney_compact()
        if not tourney.empty:
            tourney = tourney[
                (tourney['Season'] >= min_season) &
                (tourney['Season'] <= max_season) &
                (tourney['Season'] != 2020)
            ]

        print("\n3. Loading tournament seeds...")
        seeds = self.kaggle.load_seeds()
        if not seeds.empty:
            seeds = seeds[
                (seeds['Season'] >= min_season) &
                (seeds['Season'] <= max_season) &
                (seeds['Season'] != 2020)
            ]

        print("\n4. Loading team names...")
        teams = self.kaggle.load_teams()

        if not team_stats.empty and not teams.empty:
            team_stats = team_stats.merge(
                teams[['TeamID', 'TeamName']], on='TeamID', how='left'
            )

        cache_path = os.path.join(self.cache_dir, 'cbb_team_stats.csv')
        if not team_stats.empty:
            team_stats.to_csv(cache_path, index=False)
            print(f"\nSaved team stats to {cache_path}: {len(team_stats)} rows")

        return team_stats, tourney, seeds

    def get_current_ratings(self, year: int = CURRENT_SEASON) -> pd.DataFrame:
        return self.barttorvik.get_ratings(year)

    def get_current_factors(self, year: int = CURRENT_SEASON) -> pd.DataFrame:
        return self.barttorvik.get_four_factors(year)


if __name__ == '__main__':
    dm = CBBDataManager()
    print("CBB Data Manager initialized")
    print(f"Data dir: {DATA_DIR}")
    print(f"Cache dir: {CACHE_DIR}")

    team_stats, tourney, seeds = dm.build_historical_dataset()
    if not team_stats.empty:
        print(f"\nTeam stats shape: {team_stats.shape}")
        print(team_stats.head())
    if not tourney.empty:
        print(f"\nTourney results shape: {tourney.shape}")
    if not seeds.empty:
        print(f"\nSeeds shape: {seeds.shape}")
