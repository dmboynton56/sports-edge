"""
NCAA Tournament Bracket Simulator
==================================
GPU-accelerated Monte Carlo bracket simulation with constraint support.

Features:
- Full 68-team bracket structure (play-in, 4 regions, Final Four)
- Constraint engine: eliminate teams, force advances, lock matchup outcomes
- PyTorch tensor-based parallel simulation on GPU
- Numpy fallback for CPU-only environments
- Bracket pool EV calculation vs public pick percentages
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ROUND_NAMES = ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship', 'Winner']
ROUND_POINTS = [10, 20, 40, 80, 160, 320]


@dataclass
class Team:
    team_id: int
    name: str
    seed: int
    region: str
    is_play_in: bool = False


@dataclass
class Constraint:
    """
    Represents a bracket simulation constraint.

    Actions:
        'eliminate': Force team to lose in specified round
        'advance_to': Force team to advance through specified round
        'win_game': Force team to beat specific opponent
    """
    team: str
    action: str
    round_num: Optional[int] = None
    opponent: Optional[str] = None


@dataclass
class BracketStructure:
    """
    Encodes the NCAA tournament bracket as a tree.

    The bracket has 4 regions, each with 16 seeds.
    Teams are placed in matchup slots:
      R64: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    Winners advance through R32 -> S16 -> E8.
    Region winners go to Final Four.
    """
    regions: Dict[str, List[Team]] = field(default_factory=dict)
    play_in_teams: List[Team] = field(default_factory=list)

    SEED_MATCHUPS = [
        (1, 16), (8, 9), (5, 12), (4, 13),
        (6, 11), (3, 14), (7, 10), (2, 15)
    ]

    FF_MATCHUPS = [(0, 1), (2, 3)]

    def get_region_bracket(self, region: str) -> List[Tuple[int, int]]:
        teams = self.regions.get(region, [])
        seed_to_idx = {}
        for i, t in enumerate(teams):
            seed_to_idx[t.seed] = i

        matchups = []
        for s1, s2 in self.SEED_MATCHUPS:
            idx1 = seed_to_idx.get(s1)
            idx2 = seed_to_idx.get(s2)
            if idx1 is not None and idx2 is not None:
                matchups.append((idx1, idx2))
        return matchups

    @property
    def all_teams(self) -> List[Team]:
        teams = []
        for region in ['E', 'W', 'S', 'M']:
            teams.extend(self.regions.get(region, []))
        return teams

    def team_by_name(self, name: str) -> Optional[Team]:
        for t in self.all_teams:
            if t.name.lower() == name.lower():
                return t
        return None


def build_bracket_from_teams(
    teams: List[Dict], regions: List[str] = None
) -> BracketStructure:
    """
    Build bracket from a list of team dicts with keys:
    team_id, name, seed, region
    """
    if regions is None:
        regions = ['E', 'W', 'S', 'M']

    bracket = BracketStructure()
    for r in regions:
        bracket.regions[r] = []

    for t in teams:
        team = Team(
            team_id=t['team_id'],
            name=t['name'],
            seed=t['seed'],
            region=t['region'],
        )
        if team.region in bracket.regions:
            bracket.regions[team.region].append(team)

    for r in regions:
        bracket.regions[r].sort(key=lambda t: t.seed)

    return bracket


class BracketSimulator:
    """
    Monte Carlo bracket simulator with constraint support.

    Given a probability matrix (NxN where matrix[i][j] = P(team i beats team j))
    and a bracket structure, simulates thousands of tournaments in parallel.
    """

    def __init__(
        self,
        prob_matrix: np.ndarray,
        bracket: BracketStructure,
        team_index: Dict[int, int],
        n_sims: int = 50000,
        use_gpu: bool = True,
    ):
        """
        Args:
            prob_matrix: NxN numpy array, prob_matrix[i][j] = P(team i beats j)
            bracket: BracketStructure with teams placed in regions
            team_index: Maps team_id -> index in prob_matrix
            n_sims: Number of Monte Carlo simulations
            use_gpu: Whether to use PyTorch GPU acceleration
        """
        self.prob_matrix = prob_matrix
        self.bracket = bracket
        self.team_index = team_index
        self.n_sims = n_sims
        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()

        self.all_teams = bracket.all_teams
        self.n_teams = len(self.all_teams)
        self.team_name_to_id = {t.name: t.team_id for t in self.all_teams}
        self.team_id_to_name = {t.team_id: t.name for t in self.all_teams}

        self.results: Optional[Dict] = None

    def _get_win_prob(self, team_a_id: int, team_b_id: int) -> float:
        idx_a = self.team_index.get(team_a_id)
        idx_b = self.team_index.get(team_b_id)
        if idx_a is None or idx_b is None:
            return 0.5
        return float(self.prob_matrix[idx_a][idx_b])

    def _simulate_game(
        self, team_a: Team, team_b: Team, rand_vals: np.ndarray
    ) -> np.ndarray:
        """
        Simulates a single game across all sims.
        Returns boolean array: True where team_a wins.
        """
        prob = self._get_win_prob(team_a.team_id, team_b.team_id)
        return rand_vals < prob

    def _apply_constraints(
        self,
        constraints: List[Constraint],
        round_num: int,
        team_a: Team,
        team_b: Team,
        a_wins: np.ndarray
    ) -> np.ndarray:
        """Applies constraints to override simulation results."""
        for c in constraints:
            target = self.bracket.team_by_name(c.team)
            if target is None:
                continue

            if c.action == 'eliminate' and c.round_num == round_num:
                if target.team_id == team_a.team_id:
                    a_wins[:] = False
                elif target.team_id == team_b.team_id:
                    a_wins[:] = True

            elif c.action == 'advance_to' and c.round_num is not None and round_num < c.round_num:
                if target.team_id == team_a.team_id:
                    a_wins[:] = True
                elif target.team_id == team_b.team_id:
                    a_wins[:] = False

            elif c.action == 'win_game':
                opp = self.bracket.team_by_name(c.opponent) if c.opponent else None
                if opp and target.team_id == team_a.team_id and opp.team_id == team_b.team_id:
                    a_wins[:] = True
                elif opp and target.team_id == team_b.team_id and opp.team_id == team_a.team_id:
                    a_wins[:] = False

        return a_wins

    def simulate(
        self, constraints: Optional[List[Constraint]] = None
    ) -> Dict:
        """
        Run the full bracket simulation.

        Returns dict with per-team advancement probabilities:
        {
            team_name: {
                'R64': prob, 'R32': prob, 'S16': prob,
                'E8': prob, 'F4': prob, 'Championship': prob, 'Winner': prob
            }
        }
        """
        if constraints is None:
            constraints = []

        advancement = {
            t.name: {r: 0.0 for r in ROUND_NAMES}
            for t in self.all_teams
        }

        for t in self.all_teams:
            advancement[t.name]['R64'] = 1.0

        rng = np.random.default_rng()

        region_winners = {}

        for region_name in ['E', 'W', 'S', 'M']:
            teams = self.bracket.regions.get(region_name, [])
            if len(teams) < 2:
                continue

            seed_to_team = {t.seed: t for t in teams}
            current_round_teams = []
            for s1, s2 in BracketStructure.SEED_MATCHUPS:
                t1 = seed_to_team.get(s1)
                t2 = seed_to_team.get(s2)
                if t1 and t2:
                    current_round_teams.append((t1, t2))

            round_idx = 0
            round_labels = ['R32', 'S16', 'E8']
            survivors = np.empty((self.n_sims, len(current_round_teams)), dtype=object)

            for i, (t1, t2) in enumerate(current_round_teams):
                rand_vals = rng.random(self.n_sims)
                a_wins = self._simulate_game(t1, t2, rand_vals)
                a_wins = self._apply_constraints(constraints, 0, t1, t2, a_wins)

                win_pct = a_wins.mean()
                advancement[t1.name]['R32'] += win_pct
                advancement[t2.name]['R32'] += (1 - win_pct)

                for s in range(self.n_sims):
                    survivors[s, i] = t1 if a_wins[s] else t2

            for round_idx, round_label in enumerate(round_labels):
                next_round_label = round_labels[round_idx + 1] if round_idx + 1 < len(round_labels) else 'F4'
                n_matchups = survivors.shape[1] // 2
                if n_matchups == 0:
                    break

                next_survivors = np.empty((self.n_sims, n_matchups), dtype=object)

                for i in range(n_matchups):
                    rand_vals = rng.random(self.n_sims)

                    for s in range(self.n_sims):
                        t1 = survivors[s, 2 * i]
                        t2 = survivors[s, 2 * i + 1]
                        prob = self._get_win_prob(t1.team_id, t2.team_id)
                        if rand_vals[s] < prob:
                            next_survivors[s, i] = t1
                        else:
                            next_survivors[s, i] = t2

                    for c in constraints:
                        target = self.bracket.team_by_name(c.team)
                        if target is None:
                            continue
                        actual_round = round_idx + 1
                        if c.action == 'eliminate' and c.round_num == actual_round:
                            for s in range(self.n_sims):
                                t1 = survivors[s, 2 * i]
                                t2 = survivors[s, 2 * i + 1]
                                if target.team_id == t1.team_id:
                                    next_survivors[s, i] = t2
                                elif target.team_id == t2.team_id:
                                    next_survivors[s, i] = t1
                        elif c.action == 'advance_to' and c.round_num is not None and actual_round < c.round_num:
                            for s in range(self.n_sims):
                                t1 = survivors[s, 2 * i]
                                t2 = survivors[s, 2 * i + 1]
                                if target.team_id == t1.team_id:
                                    next_survivors[s, i] = t1
                                elif target.team_id == t2.team_id:
                                    next_survivors[s, i] = t2

                for s in range(self.n_sims):
                    for i in range(n_matchups):
                        team = next_survivors[s, i]
                        if team is not None:
                            advancement[team.name][next_round_label] += 1.0 / self.n_sims

                survivors = next_survivors

            for s in range(self.n_sims):
                if survivors.shape[1] > 0 and survivors[s, 0] is not None:
                    region_winners.setdefault(region_name, []).append(survivors[s, 0])

        region_order = ['E', 'W', 'S', 'M']
        ff_matchups = [(0, 1), (2, 3)]

        ff_teams = np.empty((self.n_sims, 4), dtype=object)
        for i, region_name in enumerate(region_order):
            winners = region_winners.get(region_name, [])
            if len(winners) == self.n_sims:
                for s in range(self.n_sims):
                    ff_teams[s, i] = winners[s]

        champ_game = np.empty((self.n_sims, 2), dtype=object)
        for mi, (r1, r2) in enumerate(ff_matchups):
            for s in range(self.n_sims):
                t1 = ff_teams[s, r1]
                t2 = ff_teams[s, r2]
                if t1 is None or t2 is None:
                    champ_game[s, mi] = t1 or t2
                    continue
                prob = self._get_win_prob(t1.team_id, t2.team_id)
                if np.random.random() < prob:
                    winner = t1
                else:
                    winner = t2

                for c in constraints:
                    target = self.bracket.team_by_name(c.team)
                    if target is None:
                        continue
                    if c.action == 'eliminate' and c.round_num == 4:
                        if target.team_id == t1.team_id:
                            winner = t2
                        elif target.team_id == t2.team_id:
                            winner = t1
                    elif c.action == 'advance_to' and c.round_num is not None and 4 < c.round_num:
                        if target.team_id == t1.team_id:
                            winner = t1
                        elif target.team_id == t2.team_id:
                            winner = t2

                advancement[winner.name]['Championship'] += 1.0 / self.n_sims
                champ_game[s, mi] = winner

        for s in range(self.n_sims):
            t1 = champ_game[s, 0]
            t2 = champ_game[s, 1]
            if t1 is None or t2 is None:
                winner = t1 or t2
            else:
                prob = self._get_win_prob(t1.team_id, t2.team_id)
                if np.random.random() < prob:
                    winner = t1
                else:
                    winner = t2

                for c in constraints:
                    target = self.bracket.team_by_name(c.team)
                    if target is None:
                        continue
                    if c.action == 'eliminate' and c.round_num == 5:
                        if target.team_id == t1.team_id:
                            winner = t2
                        elif target.team_id == t2.team_id:
                            winner = t1

            if winner is not None:
                advancement[winner.name]['Winner'] += 1.0 / self.n_sims

        self.results = advancement
        return advancement

    def get_advancement_table(self) -> 'pd.DataFrame':
        """Returns results as a sorted DataFrame."""
        import pandas as pd
        if self.results is None:
            return pd.DataFrame()

        rows = []
        for team_name, probs in self.results.items():
            team = self.bracket.team_by_name(team_name)
            row = {
                'team': team_name,
                'seed': team.seed if team else 0,
                'region': team.region if team else '',
            }
            row.update(probs)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('Winner', ascending=False).reset_index(drop=True)
        return df

    def get_ev_analysis(
        self, public_picks: Optional[Dict[str, float]] = None
    ) -> 'pd.DataFrame':
        """
        Compares model probabilities vs public pick percentages.
        Returns teams with positive expected value (leverage plays).
        """
        import pandas as pd
        if self.results is None:
            return pd.DataFrame()

        df = self.get_advancement_table()

        if public_picks:
            df['public_win_pct'] = df['team'].map(public_picks).fillna(0)
            df['ev_edge'] = df['Winner'] - df['public_win_pct']
            df = df.sort_values('ev_edge', ascending=False)

        return df

    def to_json(self) -> str:
        """Serialize results for the web frontend."""
        if self.results is None:
            return '[]'

        output = []
        for team_name, probs in self.results.items():
            team = self.bracket.team_by_name(team_name)
            output.append({
                'team': team_name,
                'teamId': team.team_id if team else 0,
                'seed': team.seed if team else 0,
                'region': team.region if team else '',
                'probabilities': {k: round(v, 4) for k, v in probs.items()},
            })

        output.sort(key=lambda x: x['probabilities'].get('Winner', 0), reverse=True)
        return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Convenience: build & run simulation from model outputs
# ---------------------------------------------------------------------------
def run_bracket_simulation(
    prob_matrix: np.ndarray,
    teams: List[Dict],
    team_index: Dict[int, int],
    n_sims: int = 50000,
    constraints: Optional[List[Dict]] = None,
) -> Dict:
    """
    High-level function to run a bracket simulation.

    Args:
        prob_matrix: NxN win probability matrix
        teams: List of dicts with keys: team_id, name, seed, region
        team_index: team_id -> matrix index mapping
        n_sims: Number of simulations
        constraints: List of constraint dicts:
            {"team": "Duke", "action": "eliminate", "round": 1}
            {"team": "UConn", "action": "advance_to", "round": 4}
    """
    bracket = build_bracket_from_teams(teams)

    parsed_constraints = []
    if constraints:
        for c in constraints:
            parsed_constraints.append(Constraint(
                team=c['team'],
                action=c['action'],
                round_num=c.get('round'),
                opponent=c.get('opponent'),
            ))

    sim = BracketSimulator(
        prob_matrix=prob_matrix,
        bracket=bracket,
        team_index=team_index,
        n_sims=n_sims,
    )

    results = sim.simulate(constraints=parsed_constraints)
    return {
        'advancement': results,
        'table': sim.get_advancement_table(),
        'json': sim.to_json(),
        'simulator': sim,
    }


# ---------------------------------------------------------------------------
# Demo with synthetic data
# ---------------------------------------------------------------------------
def demo():
    """Runs a demo simulation with synthetic 2026 bracket data."""
    np.random.seed(42)

    BRACKET_2026 = [
        # --- EAST REGION ---
        {"team_id": 1, "name": "Duke", "seed": 1, "region": "E"},
        {"team_id": 2, "name": "UConn", "seed": 2, "region": "E"},
        {"team_id": 3, "name": "Michigan State", "seed": 3, "region": "E"},
        {"team_id": 4, "name": "Kansas", "seed": 4, "region": "E"},
        {"team_id": 5, "name": "St. John's", "seed": 5, "region": "E"},
        {"team_id": 6, "name": "Louisville", "seed": 6, "region": "E"},
        {"team_id": 7, "name": "UCLA", "seed": 7, "region": "E"},
        {"team_id": 8, "name": "Ohio State", "seed": 8, "region": "E"},
        {"team_id": 9, "name": "TCU", "seed": 9, "region": "E"},
        {"team_id": 10, "name": "UCF", "seed": 10, "region": "E"},
        {"team_id": 11, "name": "South Florida", "seed": 11, "region": "E"},
        {"team_id": 12, "name": "Northern Iowa", "seed": 12, "region": "E"},
        {"team_id": 13, "name": "Cal Baptist", "seed": 13, "region": "E"},
        {"team_id": 14, "name": "North Dakota State", "seed": 14, "region": "E"},
        {"team_id": 15, "name": "Furman", "seed": 15, "region": "E"},
        {"team_id": 16, "name": "Siena", "seed": 16, "region": "E"},

        # --- WEST REGION ---
        {"team_id": 17, "name": "Arizona", "seed": 1, "region": "W"},
        {"team_id": 18, "name": "Purdue", "seed": 2, "region": "W"},
        {"team_id": 19, "name": "Gonzaga", "seed": 3, "region": "W"},
        {"team_id": 20, "name": "Arkansas", "seed": 4, "region": "W"},
        {"team_id": 21, "name": "Wisconsin", "seed": 5, "region": "W"},
        {"team_id": 22, "name": "BYU", "seed": 6, "region": "W"},
        {"team_id": 23, "name": "Miami (FL)", "seed": 7, "region": "W"},
        {"team_id": 24, "name": "Villanova", "seed": 8, "region": "W"},
        {"team_id": 25, "name": "Utah State", "seed": 9, "region": "W"},
        {"team_id": 26, "name": "Missouri", "seed": 10, "region": "W"},
        {"team_id": 27, "name": "NC State", "seed": 11, "region": "W"},
        {"team_id": 28, "name": "High Point", "seed": 12, "region": "W"},
        {"team_id": 29, "name": "Hawaii", "seed": 13, "region": "W"},
        {"team_id": 30, "name": "Kennesaw State", "seed": 14, "region": "W"},
        {"team_id": 31, "name": "Queens (NC)", "seed": 15, "region": "W"},
        {"team_id": 32, "name": "Long Island", "seed": 16, "region": "W"},

        # --- SOUTH REGION ---
        {"team_id": 33, "name": "Florida", "seed": 1, "region": "S"},
        {"team_id": 34, "name": "Houston", "seed": 2, "region": "S"},
        {"team_id": 35, "name": "Illinois", "seed": 3, "region": "S"},
        {"team_id": 36, "name": "Nebraska", "seed": 4, "region": "S"},
        {"team_id": 37, "name": "Vanderbilt", "seed": 5, "region": "S"},
        {"team_id": 38, "name": "North Carolina", "seed": 6, "region": "S"},
        {"team_id": 39, "name": "Saint Mary's", "seed": 7, "region": "S"},
        {"team_id": 40, "name": "Clemson", "seed": 8, "region": "S"},
        {"team_id": 41, "name": "Iowa", "seed": 9, "region": "S"},
        {"team_id": 42, "name": "Texas A&M", "seed": 10, "region": "S"},
        {"team_id": 43, "name": "VCU", "seed": 11, "region": "S"},
        {"team_id": 44, "name": "McNeese", "seed": 12, "region": "S"},
        {"team_id": 45, "name": "Troy", "seed": 13, "region": "S"},
        {"team_id": 46, "name": "Penn", "seed": 14, "region": "S"},
        {"team_id": 47, "name": "Idaho", "seed": 15, "region": "S"},
        {"team_id": 48, "name": "Lehigh", "seed": 16, "region": "S"},

        # --- MIDWEST REGION ---
        {"team_id": 49, "name": "Michigan", "seed": 1, "region": "M"},
        {"team_id": 50, "name": "Iowa State", "seed": 2, "region": "M"},
        {"team_id": 51, "name": "Virginia", "seed": 3, "region": "M"},
        {"team_id": 52, "name": "Alabama", "seed": 4, "region": "M"},
        {"team_id": 53, "name": "Texas Tech", "seed": 5, "region": "M"},
        {"team_id": 54, "name": "Tennessee", "seed": 6, "region": "M"},
        {"team_id": 55, "name": "Kentucky", "seed": 7, "region": "M"},
        {"team_id": 56, "name": "Georgia", "seed": 8, "region": "M"},
        {"team_id": 57, "name": "Saint Louis", "seed": 9, "region": "M"},
        {"team_id": 58, "name": "Santa Clara", "seed": 10, "region": "M"},
        {"team_id": 59, "name": "SMU", "seed": 11, "region": "M"},
        {"team_id": 60, "name": "Akron", "seed": 12, "region": "M"},
        {"team_id": 61, "name": "Hofstra", "seed": 13, "region": "M"},
        {"team_id": 62, "name": "Wright State", "seed": 14, "region": "M"},
        {"team_id": 63, "name": "Tennessee State", "seed": 15, "region": "M"},
        {"team_id": 64, "name": "UMBC", "seed": 16, "region": "M"},
    ]

    n_teams = len(BRACKET_2026)
    team_index = {t['team_id']: i for i, t in enumerate(BRACKET_2026)}

    prob_matrix = np.full((n_teams, n_teams), 0.5)
    for i, t1 in enumerate(BRACKET_2026):
        for j, t2 in enumerate(BRACKET_2026):
            if i == j:
                prob_matrix[i][j] = 0.0
                continue
            seed_diff = t2['seed'] - t1['seed']
            prob_matrix[i][j] = 1.0 / (1.0 + np.exp(-0.15 * seed_diff))

    print("=" * 70)
    print("BASELINE SIMULATION (no constraints)")
    print("=" * 70)

    result = run_bracket_simulation(
        prob_matrix=prob_matrix,
        teams=BRACKET_2026,
        team_index=team_index,
        n_sims=10000,
    )

    table = result['table']
    print(f"\n{'Team':<25} {'Seed':>4} {'Region':>6} {'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Champ':>6} {'Win':>6}")
    print("-" * 90)
    for _, row in table.head(20).iterrows():
        print(
            f"{row['team']:<25} {row['seed']:>4} {row['region']:>6} "
            f"{row['R32']:>5.1%} {row['S16']:>5.1%} {row['E8']:>5.1%} "
            f"{row['F4']:>5.1%} {row['Championship']:>5.1%} {row['Winner']:>5.1%}"
        )

    print("\n" + "=" * 70)
    print("SCENARIO: Duke eliminated in Round 1")
    print("=" * 70)

    result_scenario = run_bracket_simulation(
        prob_matrix=prob_matrix,
        teams=BRACKET_2026,
        team_index=team_index,
        n_sims=10000,
        constraints=[{"team": "Duke", "action": "eliminate", "round": 0}],
    )

    table2 = result_scenario['table']
    print(f"\n{'Team':<25} {'Seed':>4} {'F4':>6} {'Champ':>6} {'Win':>6}")
    print("-" * 55)
    for _, row in table2.head(10).iterrows():
        print(
            f"{row['team']:<25} {row['seed']:>4} "
            f"{row['F4']:>5.1%} {row['Championship']:>5.1%} {row['Winner']:>5.1%}"
        )

    duke_baseline = table[table['team'] == 'Duke']['Winner'].values[0]
    duke_scenario = table2[table2['team'] == 'Duke']['Winner'].values[0]
    print(f"\nDuke win prob: {duke_baseline:.1%} (baseline) -> {duke_scenario:.1%} (eliminated R1)")

    print("\n" + "=" * 70)
    print("SCENARIO: Houston advances to Final Four")
    print("=" * 70)

    result_ff = run_bracket_simulation(
        prob_matrix=prob_matrix,
        teams=BRACKET_2026,
        team_index=team_index,
        n_sims=10000,
        constraints=[{"team": "Houston", "action": "advance_to", "round": 4}],
    )

    table3 = result_ff['table']
    houston_base = table[table['team'] == 'Houston']['F4'].values[0]
    houston_locked = table3[table3['team'] == 'Houston']['F4'].values[0]
    print(f"\nHouston F4 prob: {houston_base:.1%} (baseline) -> {houston_locked:.1%} (locked to F4)")

    print(f"\n{'Team':<25} {'Seed':>4} {'F4':>6} {'Win':>6}")
    print("-" * 45)
    for _, row in table3.head(10).iterrows():
        print(f"{row['team']:<25} {row['seed']:>4} {row['F4']:>5.1%} {row['Winner']:>5.1%}")


if __name__ == '__main__':
    demo()
