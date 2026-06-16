"""World Cup match prediction and tournament simulation utilities.

The first production target is a calibrated, inspectable baseline rather than
GPU-heavy training. Inputs are explicit team ratings and fixtures; richer
features such as player availability and market priors can be folded into the
same rating surface without changing the portfolio contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


GROUP_STAGE = "group"
KNOCKOUT_STAGE_ORDER = (
    ("round_of_32", "round_of_16"),
    ("round_of_16", "quarterfinal"),
    ("quarterfinal", "semifinal"),
    ("semifinal", "final"),
    ("final", "champion"),
)
TEAM_PROBABILITY_KEYS = (
    "group",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
)


@dataclass(frozen=True)
class TeamRating:
    team: str
    group: Optional[str] = None
    fifa_rank: Optional[float] = None
    elo: Optional[float] = None
    form_points_per_game: float = 1.5
    form_goal_diff_per_game: float = 0.0
    world_cup_experience_score: float = 0.0
    star_player_score: float = 0.0
    host_boost: float = 0.0
    market_rating: float = 0.0


@dataclass(frozen=True)
class Fixture:
    match_id: str
    stage: str
    kickoff_utc: Optional[str]
    home_team: str
    away_team: str
    group: Optional[str] = None
    status: str = "scheduled"
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    neutral_site: bool = True

    @property
    def is_completed(self) -> bool:
        return self.home_score is not None and self.away_score is not None


@dataclass(frozen=True)
class MatchPrediction:
    match_id: str
    stage: str
    group: Optional[str]
    kickoff_utc: Optional[str]
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_knockout_win_prob: float
    away_knockout_win_prob: float
    projected_home_goals: float
    projected_away_goals: float
    model_version: str
    prediction_ts: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroupTableRow:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    rating_tiebreaker: float = 0.0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def add_result(self, goals_for: int, goals_against: int) -> None:
        self.played += 1
        self.goals_for += int(goals_for)
        self.goals_against += int(goals_against)
        if goals_for > goals_against:
            self.wins += 1
            self.points += 3
        elif goals_for == goals_against:
            self.draws += 1
            self.points += 1
        else:
            self.losses += 1

    def to_dict(self) -> dict:
        row = asdict(self)
        row["goal_diff"] = self.goal_diff
        return row


@dataclass(frozen=True)
class SimulationResult:
    model_version: str
    simulation_ts: str
    simulations: int
    bracket_source: str
    team_probabilities: List[dict]
    group_rank_probabilities: Dict[str, List[dict]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: Optional[float], default: float) -> float:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return float(value)


class WorldCupRatingModel:
    """Transparent baseline model for World Cup match probabilities."""

    def __init__(
        self,
        teams: Iterable[TeamRating],
        *,
        model_version: str = "world-cup-v0",
        neutral_home_advantage: float = 0.0,
    ) -> None:
        self.model_version = model_version
        self.neutral_home_advantage = float(neutral_home_advantage)
        self.teams: Dict[str, TeamRating] = {team.team: team for team in teams}
        if not self.teams:
            raise ValueError("WorldCupRatingModel requires at least one team")
        self._strengths = {
            team_name: self._strength_score(team)
            for team_name, team in self.teams.items()
        }

    def _strength_score(self, team: TeamRating) -> float:
        elo_component = (_safe_float(team.elo, 1500.0) - 1500.0) / 400.0
        fifa_rank = _safe_float(team.fifa_rank, 75.0)
        fifa_component = (75.0 - fifa_rank) / 75.0
        form_component = 0.35 * (_safe_float(team.form_points_per_game, 1.5) - 1.5)
        form_component += 0.16 * _safe_float(team.form_goal_diff_per_game, 0.0)
        experience_component = 0.08 * math.log1p(max(_safe_float(team.world_cup_experience_score, 0.0), 0.0))
        star_component = 0.12 * _safe_float(team.star_player_score, 0.0)
        market_component = 0.25 * _safe_float(team.market_rating, 0.0)
        host_component = _safe_float(team.host_boost, 0.0)
        return (
            0.58 * elo_component
            + 0.24 * fifa_component
            + form_component
            + experience_component
            + star_component
            + market_component
            + host_component
        )

    def strength(self, team: str) -> float:
        try:
            return self._strengths[team]
        except KeyError as exc:
            raise KeyError(f"Unknown World Cup team: {team}") from exc

    def predict_fixture(self, fixture: Fixture, *, prediction_ts: Optional[str] = None) -> MatchPrediction:
        return self.predict_pair(
            fixture.home_team,
            fixture.away_team,
            match_id=fixture.match_id,
            stage=fixture.stage,
            group=fixture.group,
            kickoff_utc=fixture.kickoff_utc,
            neutral_site=fixture.neutral_site,
            prediction_ts=prediction_ts,
        )

    def predict_pair(
        self,
        home_team: str,
        away_team: str,
        *,
        match_id: str = "",
        stage: str = GROUP_STAGE,
        group: Optional[str] = None,
        kickoff_utc: Optional[str] = None,
        neutral_site: bool = True,
        prediction_ts: Optional[str] = None,
    ) -> MatchPrediction:
        home_strength = self.strength(home_team)
        away_strength = self.strength(away_team)
        diff = home_strength - away_strength
        if not neutral_site:
            diff += self.neutral_home_advantage

        non_draw_home_prob = _sigmoid(1.45 * diff)
        draw_prob = _clamp(0.245 - 0.055 * abs(diff), 0.145, 0.295)
        home_win_prob = (1.0 - draw_prob) * non_draw_home_prob
        away_win_prob = (1.0 - draw_prob) * (1.0 - non_draw_home_prob)

        knockout_home_prob = home_win_prob + draw_prob * _sigmoid(0.9 * diff)
        home_xg = _clamp(1.28 + 0.62 * diff, 0.25, 3.6)
        away_xg = _clamp(1.28 - 0.62 * diff, 0.25, 3.6)

        return MatchPrediction(
            match_id=match_id,
            stage=stage,
            group=group,
            kickoff_utc=kickoff_utc,
            home_team=home_team,
            away_team=away_team,
            home_win_prob=round(home_win_prob, 6),
            draw_prob=round(draw_prob, 6),
            away_win_prob=round(away_win_prob, 6),
            home_knockout_win_prob=round(knockout_home_prob, 6),
            away_knockout_win_prob=round(1.0 - knockout_home_prob, 6),
            projected_home_goals=round(home_xg, 3),
            projected_away_goals=round(away_xg, 3),
            model_version=self.model_version,
            prediction_ts=prediction_ts or _now_iso(),
        )


class WorldCupTournamentSimulator:
    """Monte Carlo simulator for 48-team World Cup-style tournaments."""

    def __init__(
        self,
        model: WorldCupRatingModel,
        fixtures: Sequence[Fixture],
        *,
        round_of_32_slots: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> None:
        self.model = model
        self.fixtures = list(fixtures)
        self.round_of_32_slots = list(round_of_32_slots or [])
        self.groups = self._teams_by_group()

    def _teams_by_group(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for team in self.model.teams.values():
            if team.group:
                groups.setdefault(team.group.upper(), []).append(team.team)
        for fixture in self.fixtures:
            if fixture.group:
                group = fixture.group.upper()
                groups.setdefault(group, [])
                for team in (fixture.home_team, fixture.away_team):
                    if team not in groups[group]:
                        groups[group].append(team)
        return {group: sorted(teams) for group, teams in sorted(groups.items())}

    def predict_matches(self) -> List[dict]:
        rows = []
        for fixture in self.fixtures:
            if fixture.home_team not in self.model.teams or fixture.away_team not in self.model.teams:
                continue
            row = self.model.predict_fixture(fixture).to_dict()
            row.update(
                {
                    "status": fixture.status,
                    "home_score": fixture.home_score,
                    "away_score": fixture.away_score,
                    "neutral_site": fixture.neutral_site,
                }
            )
            rows.append(row)
        return rows

    def simulate(self, *, n_sims: int = 50000, seed: Optional[int] = None) -> SimulationResult:
        if n_sims <= 0:
            raise ValueError("n_sims must be positive")

        rng = np.random.default_rng(seed)
        probability_counts = {
            team: {key: 0 for key in TEAM_PROBABILITY_KEYS}
            for team in self.model.teams
        }
        rank_counts = {
            team: {rank: 0 for rank in range(1, 5)}
            for team in self.model.teams
        }

        bracket_source = "configured_round_of_32_slots" if self.round_of_32_slots else "power_seeded_fallback"

        for _ in range(n_sims):
            ranked_groups, third_place_rows = self._simulate_groups(rng)
            qualifiers = self._select_qualifiers(ranked_groups, third_place_rows)

            for team in self.model.teams:
                probability_counts[team]["group"] += 1

            for group_rows in ranked_groups.values():
                for rank, row in enumerate(group_rows, start=1):
                    if rank <= 4:
                        rank_counts[row.team][rank] += 1

            for team in qualifiers:
                probability_counts[team]["round_of_32"] += 1

            round_pairs = self._resolve_round_of_32(ranked_groups, qualifiers)
            self._simulate_knockout(round_pairs, probability_counts, rng)

        team_probabilities = []
        for team, counts in probability_counts.items():
            row = {
                "team": team,
                "group": self.model.teams[team].group,
                "rating": round(self.model.strength(team), 6),
                "group_prob": round(counts["group"] / n_sims, 6),
            }
            row.update({
                key: round(counts[key] / n_sims, 6)
                for key in TEAM_PROBABILITY_KEYS
                if key != "group"
            })
            team_probabilities.append(row)

        team_probabilities.sort(
            key=lambda row: (
                row["champion"],
                row["final"],
                row["semifinal"],
                row["rating"],
            ),
            reverse=True,
        )

        group_rank_probabilities = self._format_rank_probabilities(rank_counts, n_sims)
        return SimulationResult(
            model_version=self.model.model_version,
            simulation_ts=_now_iso(),
            simulations=n_sims,
            bracket_source=bracket_source,
            team_probabilities=team_probabilities,
            group_rank_probabilities=group_rank_probabilities,
        )

    def _simulate_groups(
        self,
        rng: np.random.Generator,
    ) -> Tuple[Dict[str, List[GroupTableRow]], List[GroupTableRow]]:
        tables = {
            group: {
                team: GroupTableRow(team=team, rating_tiebreaker=self.model.strength(team))
                for team in teams
            }
            for group, teams in self.groups.items()
        }

        for fixture in self.fixtures:
            if fixture.stage.lower() != GROUP_STAGE:
                continue
            if not fixture.group:
                continue
            group = fixture.group.upper()
            if group not in tables:
                continue
            if fixture.home_team not in tables[group] or fixture.away_team not in tables[group]:
                continue
            if fixture.is_completed:
                home_goals = int(fixture.home_score or 0)
                away_goals = int(fixture.away_score or 0)
            else:
                home_goals, away_goals = self._sample_group_score(fixture, rng)
            tables[group][fixture.home_team].add_result(home_goals, away_goals)
            tables[group][fixture.away_team].add_result(away_goals, home_goals)

        ranked_groups = {
            group: self._rank_group_rows(rows.values())
            for group, rows in tables.items()
        }
        third_place_rows = [
            rows[2]
            for rows in ranked_groups.values()
            if len(rows) >= 3
        ]
        return ranked_groups, third_place_rows

    def _sample_group_score(self, fixture: Fixture, rng: np.random.Generator) -> Tuple[int, int]:
        prediction = self.model.predict_fixture(fixture)
        probs = np.array([
            prediction.home_win_prob,
            prediction.draw_prob,
            prediction.away_win_prob,
        ])
        probs = probs / probs.sum()
        outcome = rng.choice(
            np.array(["home", "draw", "away"]),
            p=probs,
        )
        if outcome == "draw":
            goals = int(min(rng.poisson((prediction.projected_home_goals + prediction.projected_away_goals) / 2.0), 7))
            return goals, goals

        if outcome == "home":
            away_goals = int(min(rng.poisson(max(prediction.projected_away_goals * 0.82, 0.15)), 6))
            margin = int(1 + min(rng.poisson(max(prediction.projected_home_goals - prediction.projected_away_goals, 0.15)), 5))
            return away_goals + margin, away_goals

        home_goals = int(min(rng.poisson(max(prediction.projected_home_goals * 0.82, 0.15)), 6))
        margin = int(1 + min(rng.poisson(max(prediction.projected_away_goals - prediction.projected_home_goals, 0.15)), 5))
        return home_goals, home_goals + margin

    def _rank_group_rows(self, rows: Iterable[GroupTableRow]) -> List[GroupTableRow]:
        return sorted(
            rows,
            key=lambda row: (
                row.points,
                row.goal_diff,
                row.goals_for,
                row.wins,
                row.rating_tiebreaker,
            ),
            reverse=True,
        )

    def _select_qualifiers(
        self,
        ranked_groups: Mapping[str, List[GroupTableRow]],
        third_place_rows: Sequence[GroupTableRow],
    ) -> List[str]:
        qualifiers: List[str] = []
        for rows in ranked_groups.values():
            qualifiers.extend(row.team for row in rows[:2])

        best_thirds = self._rank_group_rows(third_place_rows)[:8]
        qualifiers.extend(row.team for row in best_thirds)
        return qualifiers

    def _resolve_round_of_32(
        self,
        ranked_groups: Mapping[str, List[GroupTableRow]],
        qualifiers: Sequence[str],
    ) -> List[Tuple[str, str]]:
        if not self.round_of_32_slots:
            return self._build_power_seeded_pairs(qualifiers)

        used_thirds: set[str] = set()
        best_thirds = [
            row.team
            for row in self._rank_group_rows(
                rows[2] for rows in ranked_groups.values() if len(rows) >= 3 and rows[2].team in qualifiers
            )
        ]
        pairs: List[Tuple[str, str]] = []
        for left_slot, right_slot in self.round_of_32_slots:
            left = self._resolve_slot(left_slot, ranked_groups, best_thirds, used_thirds)
            right = self._resolve_slot(right_slot, ranked_groups, best_thirds, used_thirds)
            if left and right and left != right:
                pairs.append((left, right))

        if len(pairs) != 16:
            return self._build_power_seeded_pairs(qualifiers)
        return pairs

    def _resolve_slot(
        self,
        slot: str,
        ranked_groups: Mapping[str, List[GroupTableRow]],
        best_thirds: Sequence[str],
        used_thirds: set[str],
    ) -> Optional[str]:
        cleaned = slot.strip().upper()
        if cleaned.startswith("TEAM:"):
            team = slot.split(":", 1)[1].strip()
            return team if team in self.model.teams else None
        if cleaned in {"3*", "BEST_THIRD", "3RD"}:
            for team in best_thirds:
                if team not in used_thirds:
                    used_thirds.add(team)
                    return team
            return None
        if len(cleaned) >= 2 and cleaned[0] in {"1", "2", "3"}:
            rank = int(cleaned[0])
            group = cleaned[1:]
            rows = ranked_groups.get(group)
            if not rows or len(rows) < rank:
                return None
            team = rows[rank - 1].team
            if rank == 3:
                if team not in best_thirds or team in used_thirds:
                    return None
                used_thirds.add(team)
            return team
        return None

    def _build_power_seeded_pairs(self, qualifiers: Sequence[str]) -> List[Tuple[str, str]]:
        seeded = sorted(
            qualifiers,
            key=lambda team: self.model.strength(team),
            reverse=True,
        )
        if len(seeded) < 2:
            return []
        bracket_size = 2 ** int(math.floor(math.log2(len(seeded))))
        seeded = seeded[:bracket_size]
        return [
            (seeded[index], seeded[-(index + 1)])
            for index in range(len(seeded) // 2)
        ]

    def _simulate_knockout(
        self,
        pairs: Sequence[Tuple[str, str]],
        probability_counts: Dict[str, Dict[str, int]],
        rng: np.random.Generator,
    ) -> None:
        current_pairs = list(pairs)
        stage_order = self._stage_order_for_pair_count(len(current_pairs))
        for stage, next_stage in stage_order:
            winners: List[str] = []
            for home_team, away_team in current_pairs:
                prediction = self.model.predict_pair(home_team, away_team, stage=stage)
                home_wins = rng.random() < prediction.home_knockout_win_prob
                winner = home_team if home_wins else away_team
                probability_counts[winner][next_stage] += 1
                winners.append(winner)
            if len(winners) < 2:
                break
            current_pairs = [
                (winners[index], winners[index + 1])
                for index in range(0, len(winners) - 1, 2)
            ]

    def _stage_order_for_pair_count(self, pair_count: int) -> Tuple[Tuple[str, str], ...]:
        if pair_count >= 16:
            return KNOCKOUT_STAGE_ORDER
        if pair_count == 8:
            return KNOCKOUT_STAGE_ORDER[1:]
        if pair_count == 4:
            return KNOCKOUT_STAGE_ORDER[2:]
        if pair_count == 2:
            return KNOCKOUT_STAGE_ORDER[3:]
        if pair_count == 1:
            return KNOCKOUT_STAGE_ORDER[4:]
        return tuple()

    def _format_rank_probabilities(
        self,
        rank_counts: Mapping[str, Mapping[int, int]],
        n_sims: int,
    ) -> Dict[str, List[dict]]:
        grouped: Dict[str, List[dict]] = {group: [] for group in self.groups}
        for team, counts in rank_counts.items():
            group = self.model.teams[team].group
            if not group:
                continue
            grouped.setdefault(group.upper(), []).append(
                {
                    "team": team,
                    "rating": round(self.model.strength(team), 6),
                    "rank_1": round(counts.get(1, 0) / n_sims, 6),
                    "rank_2": round(counts.get(2, 0) / n_sims, 6),
                    "rank_3": round(counts.get(3, 0) / n_sims, 6),
                    "rank_4": round(counts.get(4, 0) / n_sims, 6),
                }
            )
        for rows in grouped.values():
            rows.sort(key=lambda row: (row["rank_1"], row["rank_2"], row["rating"]), reverse=True)
        return grouped
