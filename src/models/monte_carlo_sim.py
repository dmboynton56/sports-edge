import numpy as np
import pandas as pd

class PGAMonteCarloSimulator:
    """
    Simulates the remaining rounds of a PGA tournament to calculate probabilities
    for win, top 5, top 10, and top 20 finishes.
    """
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations

    def run_simulation(self, current_leaderboard: pd.DataFrame, remaining_rounds: int) -> pd.DataFrame:
        """
        Runs the Monte Carlo simulation.
        
        Expects a DataFrame with at least:
        - 'player_name'
        - 'current_score' (to par)
        - 'expected_sg_per_round' (from our ML model)
        - 'sg_variance' (historical variance of the player's performance)
        """
        print(f"Running {self.num_simulations} simulations for the remaining {remaining_rounds} rounds...")
        
        num_players = len(current_leaderboard)
        player_names = current_leaderboard['player_name'].values
        current_scores = current_leaderboard['current_score'].values
        expected_sg = current_leaderboard['expected_sg_per_round'].values
        variances = current_leaderboard['sg_variance'].values

        # Initialize simulation matrices
        # shape: (num_simulations, num_players)
        simulated_total_scores = np.tile(current_scores, (self.num_simulations, 1)).astype(float)
        
        # Simulate each remaining round
        for r in range(remaining_rounds):
            # Normal distribution based on expected SG and variance
            # A higher SG means a LOWER (better) score relative to par
            # So simulated score for round = Expected Score to Par - simulated SG
            
            # For simplicity, let's assume 'expected_sg_per_round' is their Expected Score relative to the field average
            # and 'sg_variance' is the std dev of their round scores.
            
            round_sim = np.random.normal(loc=-expected_sg, scale=variances, size=(self.num_simulations, num_players))
            simulated_total_scores += round_sim

        # Now we have self.num_simulations possible final leaderboards
        # Let's calculate probabilities
        
        win_counts = np.zeros(num_players)
        top5_counts = np.zeros(num_players)
        top10_counts = np.zeros(num_players)
        top20_counts = np.zeros(num_players)
        
        for i in range(self.num_simulations):
            # Get the simulated final scores for this simulation
            scores = simulated_total_scores[i]
            
            # Sort players by score (lowest is best)
            # argsort returns indices that would sort the array
            ranks = np.argsort(scores)
            
            # Add to counts
            win_counts[ranks[0]] += 1
            
            for j in range(min(5, num_players)):
                top5_counts[ranks[j]] += 1
                
            for j in range(min(10, num_players)):
                top10_counts[ranks[j]] += 1
                
            for j in range(min(20, num_players)):
                top20_counts[ranks[j]] += 1

        # Compile results
        results = pd.DataFrame({
            'player_name': player_names,
            'current_score': current_scores,
            'win_prob': win_counts / self.num_simulations,
            'top5_prob': top5_counts / self.num_simulations,
            'top10_prob': top10_counts / self.num_simulations,
            'top20_prob': top20_counts / self.num_simulations
        })
        
        # Sort by win probability
        return results.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    # Test the simulator with dummy data
    np.random.seed(42)
    dummy_data = pd.DataFrame({
        'player_name': ['Scottie Scheffler', 'Rory McIlroy', 'Jon Rahm', 'Tiger Woods', 'Justin Thomas'],
        'current_score': [-5, -4, -4, 0, +1],
        'expected_sg_per_round': [2.0, 1.5, 1.5, -0.5, 0.5], # Better players have higher expected SG
        'sg_variance': [2.0, 2.5, 2.2, 3.0, 2.8] # Standard deviation of their round scores
    })
    
    print("Initial State:")
    print(dummy_data)
    print("\n--- Starting Simulation ---")
    
    sim = PGAMonteCarloSimulator(num_simulations=10000)
    # Simulate the weekend (2 rounds remaining)
    results = sim.run_simulation(dummy_data, remaining_rounds=2)
    
    print("\n--- Simulation Results ---")
    print(results)
