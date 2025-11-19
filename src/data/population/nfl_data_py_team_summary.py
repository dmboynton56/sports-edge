#!/usr/bin/env python3
"""
Summarize the last N seasons for an NFL team using nfl_data_py schedules.

For each season:
  - Print every game with final scores and result.
  - Compute total record, point differential, and PF/PA totals.
  - Rank the team's offense (points scored) and defense (points allowed)
    relative to the rest of the league.

Run:
    python nfl_data_py_team_summary.py --team NE --seasons 2020 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd


def _default_seasons(num_seasons: int = 5) -> List[int]:
    current = dt.date.today().year
    # subtract one because we typically want completed seasons
    start = current - num_seasons
    return list(range(start, current))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize team results using nfl_data_py schedules."
    )
    parser.add_argument(
        "--team",
        default="NE",
        help="Team abbreviation to summarize (default: NE).",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Specific seasons to load (default: last 5 completed seasons).",
    )
    return parser.parse_args()


def _load_schedules(seasons: List[int]) -> pd.DataFrame:
    try:
        import nfl_data_py as nfl
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("Install nfl-data-py before running this script.") from exc

    print(f"nfl_data_py version: {getattr(nfl, '__version__', 'unknown')}")
    print(f"Fetching seasons: {', '.join(map(str, seasons))}")
    df = nfl.import_schedules(seasons)
    if df.empty:
        raise SystemExit("No schedule data returned; check seasons or connectivity.")
    df["game_date"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df


def _team_game_rows(df: pd.DataFrame, team: str) -> pd.DataFrame:
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    games = df.loc[mask].copy()
    if games.empty:
        raise SystemExit(f"No games found for team {team}.")

    games["is_home"] = games["home_team"] == team
    games["team_score"] = games.apply(
        lambda row: row["home_score"] if row["is_home"] else row["away_score"], axis=1
    )
    games["opp_score"] = games.apply(
        lambda row: row["away_score"] if row["is_home"] else row["home_score"], axis=1
    )
    games["opponent"] = games.apply(
        lambda row: row["away_team"] if row["is_home"] else row["home_team"], axis=1
    )
    games["result"] = games.apply(
        lambda row: "W" if row["team_score"] > row["opp_score"]
        else ("L" if row["team_score"] < row["opp_score"] else "T"),
        axis=1,
    )
    games["point_diff"] = games["team_score"] - games["opp_score"]
    games["final_score"] = games.apply(
        lambda row: f"{row['home_team']} {row['home_score']} - {row['away_team']} {row['away_score']}",
        axis=1,
    )
    games = games.sort_values(["season", "game_date"])
    return games


def _season_rankings(df: pd.DataFrame, season: int) -> pd.DataFrame:
    season_df = df[df["season"] == season]
    # duplicate rows for home and away contributions
    home = season_df[["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"}
    )
    away = season_df[["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"}
    )
    team_totals = (
        pd.concat([home, away], ignore_index=True)
        .groupby("team")
        .agg(points_for=("pf", "sum"), points_against=("pa", "sum"), games=("pf", "count"))
        .reset_index()
    )
    team_totals["off_rank"] = team_totals["points_for"].rank(method="min", ascending=False).astype(int)
    team_totals["def_rank"] = team_totals["points_against"].rank(method="min", ascending=True).astype(int)
    return team_totals


def _season_summary(games: pd.DataFrame, rankings: pd.DataFrame, team: str) -> Dict[int, Dict]:
    summaries: Dict[int, Dict] = {}
    for season, df in games.groupby("season"):
        record_counts = df["result"].value_counts().to_dict()
        wins = record_counts.get("W", 0)
        losses = record_counts.get("L", 0)
        ties = record_counts.get("T", 0)
        pf = df["team_score"].sum()
        pa = df["opp_score"].sum()
        pt_diff = pf - pa
        rank_row = rankings[rankings["team"] == team]
        off_rank = int(rank_row["off_rank"].iloc[0]) if not rank_row.empty else None
        def_rank = int(rank_row["def_rank"].iloc[0]) if not rank_row.empty else None
        summaries[season] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "points_for": pf,
            "points_against": pa,
            "point_diff": pt_diff,
            "off_rank": off_rank,
            "def_rank": def_rank,
            "games": df[
                [
                    "game_date",
                    "week",
                    "opponent",
                    "is_home",
                    "final_score",
                    "result",
                    "point_diff",
                ]
            ].copy(),
        }
    return summaries


def main() -> None:
    args = parse_args()
    seasons = args.seasons or _default_seasons()

    schedules = _load_schedules(seasons)
    team_games = _team_game_rows(schedules, args.team.upper())

    summary_rows: Dict[int, Dict] = {}
    for season in sorted(seasons):
        rankings = _season_rankings(schedules, season)
        if rankings.empty:
            continue
        season_games = team_games[team_games["season"] == season]
        if season_games.empty:
            continue
        summaries = _season_summary(season_games, rankings, args.team.upper())
        summary_rows.update(summaries)

    if not summary_rows:
        print("No summaries generated; verify the seasons include completed years.")
        return

    for season in sorted(summary_rows):
        season_data = summary_rows[season]
        print(
            f"\nSeason {season}: {args.team.upper()} "
            f"{season_data['wins']}-{season_data['losses']}"
            f"{'-'+str(season_data['ties']) if season_data['ties'] else ''}"
        )
        print(
            f"  Point diff: {season_data['point_diff']} "
            f"(PF {season_data['points_for']}, PA {season_data['points_against']})"
        )
        print(
            f"  Offensive rank (points for): {season_data['off_rank']}, "
            f"Defensive rank (points allowed): {season_data['def_rank']}"
        )
        games = season_data["games"].copy()
        games["home/away"] = games["is_home"].map({True: "home", False: "away"})
        games["game_date"] = games["game_date"].dt.strftime("%Y-%m-%d")
        print(games.drop(columns="is_home").to_string(index=False))


if __name__ == "__main__":
    main()
