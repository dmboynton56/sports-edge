import argparse
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.predictor import GamePredictor
from src.data import nfl_fetcher
from src.data.pbp_loader import load_pbp

MODEL_VERSION = 'v1'

# Playoff Seeds for 2025 (Inferred from standings and WC matchups)
AFC_SEEDS = {
    1: 'DEN',
    2: 'NE',
    3: 'JAX',
    4: 'PIT',
    5: 'HOU',
    6: 'BUF',
    7: 'LAC'
}

NFC_SEEDS = {
    1: 'SEA',
    2: 'PHI',
    3: 'CHI',
    4: 'CAR',
    5: 'LA',
    6: 'SF',
    7: 'GB'
}

def get_team_seed(team: str, conference_seeds: Dict[int, str]) -> int:
    for seed, t in conference_seeds.items():
        if t == team:
            return seed
    return 99

def simulate_game(predictor: GamePredictor, home: str, away: str, schedule: pd.DataFrame, 
                  pbp: pd.DataFrame, is_neutral: bool = False) -> Dict:
    """Predict a single game and return the results."""
    # Use a dummy date in Jan 2026 for playoff features
    game_date = '2026-01-15'
    
    game_row = pd.DataFrame([{
        'home_team': home,
        'away_team': away,
        'game_date': game_date,
        'season': 2025,
        'game_type': 'POST'
    }])
    
    pred = predictor.predict(game_row, schedule, play_by_play=pbp)
    
    if is_neutral:
        # Simple neutral site adjustment: predict with home/away swapped and average
        swapped_row = pd.DataFrame([{
            'home_team': away,
            'away_team': home,
            'game_date': game_date,
            'season': 2025,
            'game_type': 'POST'
        }])
        swapped_pred = predictor.predict(swapped_row, schedule, play_by_play=pbp)
        
        # Average the probabilities
        original_home_win_prob = pred['home_win_probability']
        swapped_away_win_prob = swapped_pred['away_win_probability'] # Prob of 'away' team winning
        
        neutral_home_win_prob = (original_home_win_prob + swapped_away_win_prob) / 2
        pred['home_win_probability'] = neutral_home_win_prob
        pred['away_win_probability'] = 1 - neutral_home_win_prob
        pred['predicted_winner'] = home if neutral_home_win_prob > 0.5 else away
        pred['confidence'] = abs(neutral_home_win_prob - 0.5) * 2
        pred['predicted_spread'] = (pred['predicted_spread'] + (-swapped_pred['predicted_spread'])) / 2
        pred['spread_interpretation'] = f"Neutral: {pred['predicted_winner']} by {abs(pred['predicted_spread']):.1f}"

    return pred

def simulate_round(predictor: GamePredictor, matchups: List[Tuple[str, str]], 
                   round_name: str, schedule: pd.DataFrame, pbp: pd.DataFrame, 
                   is_neutral: bool = False) -> List[str]:
    print(f"\n{'='*20} {round_name.upper()} {'='*20}")
    winners = []
    for away, home in matchups:
        res = simulate_game(predictor, home, away, schedule, pbp, is_neutral=is_neutral)
        winner = res['predicted_winner']
        winners.append(winner)
        # Prob of the winner
        win_prob = res['home_win_probability'] if winner == home else res['away_win_probability']
        print(f"{away:3} @ {home:3} | Winner: {winner:3} | Prob: {win_prob:.1%} | Spread: {res['predicted_spread']:.1f}")
    return winners

def get_divisional_matchups(seeds: Dict[int, str], wc_winners: List[str]) -> List[Tuple[str, str]]:
    """NFL Divisional re-seeding: #1 hosts lowest remaining seed."""
    remaining = sorted(wc_winners + [seeds[1]], key=lambda x: get_team_seed(x, seeds))
    # Matchups: 1 vs lowest (last in sorted list), 2nd vs 3rd
    # sorted(remaining) will give [seed_high, ..., seed_low]
    return [(remaining[3], remaining[0]), (remaining[2], remaining[1])]

def main():
    parser = argparse.ArgumentParser(description="Simulate NFL Playoffs")
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    
    print(f"Starting 2025 NFL Playoff Simulation...")
    predictor = GamePredictor('NFL', MODEL_VERSION)
    
    # Load and normalize schedule
    schedule = nfl_fetcher.fetch_nfl_schedule(args.season)
    if 'gameday' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['gameday'])
    elif 'game_date' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    schedule['season'] = args.season
    
    pbp = load_pbp([args.season])
    
    # Wild Card Round
    afc_wc_matchups = [('HOU', 'PIT'), ('BUF', 'JAX'), ('LAC', 'NE')]
    nfc_wc_matchups = [('LA', 'CAR'), ('SF', 'PHI'), ('GB', 'CHI')]
    
    afc_wc_winners = simulate_round(predictor, afc_wc_matchups, "AFC Wild Card", schedule, pbp)
    nfc_wc_winners = simulate_round(predictor, nfc_wc_matchups, "NFC Wild Card", schedule, pbp)
    
    # Divisional Round
    afc_div_matchups = get_divisional_matchups(AFC_SEEDS, afc_wc_winners)
    nfc_div_matchups = get_divisional_matchups(NFC_SEEDS, nfc_wc_winners)
    
    afc_div_winners = simulate_round(predictor, afc_div_matchups, "AFC Divisional", schedule, pbp)
    nfc_div_winners = simulate_round(predictor, nfc_div_matchups, "NFC Divisional", schedule, pbp)
    
    # Conference Championships
    # Higher seed hosts
    afc_cc_matchup = sorted(afc_div_winners, key=lambda x: get_team_seed(x, AFC_SEEDS))
    nfc_cc_matchup = sorted(nfc_div_winners, key=lambda x: get_team_seed(x, NFC_SEEDS))
    
    afc_champ = simulate_round(predictor, [(afc_cc_matchup[1], afc_cc_matchup[0])], "AFC Championship", schedule, pbp)[0]
    nfc_champ = simulate_round(predictor, [(nfc_cc_matchup[1], nfc_cc_matchup[0])], "NFC Championship", schedule, pbp)[0]
    
    # Super Bowl (Neutral Site)
    sb_winner = simulate_round(predictor, [(afc_champ, nfc_champ)], "Super Bowl LIX", schedule, pbp, is_neutral=True)[0]
    
    print(f"\n" + "!" * 50)
    print(f"  SUPER BOWL CHAMPION: {sb_winner}")
    print("!" * 50 + "\n")

if __name__ == "__main__":
    main()
