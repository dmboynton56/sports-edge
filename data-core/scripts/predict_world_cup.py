#!/usr/bin/env python3
"""Generate World Cup match predictions and tournament probabilities.

Input files are intentionally simple CSVs so this can run before live FIFA,
Elo, odds, and player-form loaders are fully automated.

Teams CSV required columns:
  team, group

Teams CSV optional columns:
  fifa_rank, elo, form_points_per_game, form_goal_diff_per_game,
  world_cup_experience_score, star_player_score, host_boost, market_rating

Fixtures CSV required columns:
  match_id, stage, home_team, away_team

Fixtures CSV optional columns:
  group, kickoff_utc, status, home_score, away_score, neutral_site
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.models.world_cup import (
    Fixture,
    TeamRating,
    WorldCupRatingModel,
    WorldCupTournamentSimulator,
)


def _nullable_float(value: Any) -> Optional[float]:
    if pd.isna(value) or value == "":
        return None
    return float(value)


def _float_or_default(value: Any, default: float) -> float:
    parsed = _nullable_float(value)
    return default if parsed is None else float(parsed)


def _nullable_int(value: Any) -> Optional[int]:
    if pd.isna(value) or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> Optional[str]:
    if pd.isna(value) or value == "":
        return None
    return str(value)


def _bool_value(value: Any, default: bool = True) -> bool:
    if pd.isna(value) or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "f", "no", "n"}


def load_team_ratings(path: Path) -> list[TeamRating]:
    df = pd.read_csv(path)
    required = {"team", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Teams CSV missing required columns: {sorted(missing)}")

    teams = []
    for row in df.to_dict(orient="records"):
        teams.append(
            TeamRating(
                team=str(row["team"]),
                group=_optional_str(row.get("group")),
                fifa_rank=_nullable_float(row.get("fifa_rank")),
                elo=_nullable_float(row.get("elo")),
                form_points_per_game=_float_or_default(row.get("form_points_per_game"), 1.5),
                form_goal_diff_per_game=_float_or_default(row.get("form_goal_diff_per_game"), 0.0),
                world_cup_experience_score=_float_or_default(row.get("world_cup_experience_score"), 0.0),
                star_player_score=_float_or_default(row.get("star_player_score"), 0.0),
                host_boost=_float_or_default(row.get("host_boost"), 0.0),
                market_rating=_float_or_default(row.get("market_rating"), 0.0),
            )
        )
    return teams


def load_fixtures(path: Path) -> list[Fixture]:
    df = pd.read_csv(path)
    required = {"match_id", "stage", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fixtures CSV missing required columns: {sorted(missing)}")

    fixtures = []
    for row in df.to_dict(orient="records"):
        fixtures.append(
            Fixture(
                match_id=str(row["match_id"]),
                stage=str(row["stage"]),
                group=_optional_str(row.get("group")),
                kickoff_utc=_optional_str(row.get("kickoff_utc")),
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                status=str(row.get("status") or "scheduled"),
                home_score=_nullable_int(row.get("home_score")),
                away_score=_nullable_int(row.get("away_score")),
                neutral_site=_bool_value(row.get("neutral_site"), default=True),
            )
        )
    return fixtures


def load_round_of_32_slots(path: Optional[Path]) -> list[tuple[str, str]]:
    if not path:
        return []
    slots = json.loads(path.read_text())
    if not isinstance(slots, list):
        raise ValueError("Round-of-32 slots JSON must be a list")
    parsed = []
    for item in slots:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each round-of-32 slot must be a two-item list")
        parsed.append((str(item[0]), str(item[1])))
    return parsed


def build_payload(
    *,
    teams: list[TeamRating],
    fixtures: list[Fixture],
    season: int = 2026,
    model_version: str,
    n_sims: int,
    seed: Optional[int],
    round_of_32_slots: list[tuple[str, str]],
) -> dict:
    model = WorldCupRatingModel(teams, model_version=model_version)
    simulator = WorldCupTournamentSimulator(
        model,
        fixtures,
        round_of_32_slots=round_of_32_slots,
    )
    simulation = simulator.simulate(n_sims=n_sims, seed=seed)
    return {
        "season": season,
        "modelVersion": model_version,
        "updatedAt": simulation.simulation_ts,
        "simulations": simulation.simulations,
        "bracketSource": simulation.bracket_source,
        "sourceNote": (
            "World Cup v0 blends Elo, FIFA rank, recent team form, tournament experience, "
            "star-player score, host adjustment, and market rating when supplied."
        ),
        "matches": simulator.predict_matches(),
        "teamProbabilities": simulation.team_probabilities,
        "groupRankProbabilities": simulation.group_rank_probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict World Cup matches and tournament outcomes.")
    parser.add_argument("--teams-csv", required=True, type=Path)
    parser.add_argument("--fixtures-csv", required=True, type=Path)
    parser.add_argument("--round-of-32-slots-json", type=Path)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--model-version", default="world-cup-v0")
    parser.add_argument("--n-sims", type=int, default=50000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    teams = load_team_ratings(args.teams_csv)
    fixtures = load_fixtures(args.fixtures_csv)
    slots = load_round_of_32_slots(args.round_of_32_slots_json)
    payload = build_payload(
        teams=teams,
        fixtures=fixtures,
        season=args.season,
        model_version=args.model_version,
        n_sims=args.n_sims,
        seed=args.seed,
        round_of_32_slots=slots,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(payload['matches'])} matches and "
        f"{len(payload['teamProbabilities'])} team probabilities to {args.output_json}"
    )


if __name__ == "__main__":
    main()
