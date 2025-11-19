#!/usr/bin/env python3
"""
Summarize the last N seasons for an NFL team using nflreadpy schedules.

For each season:
  - List every game with final scores and results.
  - Provide record, total point differential, and PF/PA totals.
  - Rank offense (points for) and defense (points allowed) relative to league.

Run:
    python nfl_read_py_team_summary.py --team NE --seasons 2020 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, List

import pandas as pd


def _default_seasons(num_seasons: int = 5) -> List[int]:
    current = dt.date.today().year
    start = current - num_seasons
    return list(range(start, current))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize team results using nflreadpy schedules."
    )
    parser.add_argument("--team", default="NE", help="Team abbreviation (default: NE).")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seasons to load (default: last 5 completed seasons).",
    )
    return parser.parse_args()


def _load_schedules(seasons: List[int]) -> pd.DataFrame:
    try:
        import nflreadpy as nfl
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("Install nflreadpy before running this script.") from exc

    print(f"nflreadpy version: {getattr(nfl, '__version__', 'unknown')}")
    print(f"Fetching seasons: {', '.join(map(str, seasons))}")
    try:
        sched = nfl.load_schedules(seasons=seasons)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Unable to load schedules: {exc}") from exc
    if not hasattr(sched, "to_pandas"):
        raise SystemExit("Unexpected object returned from nflreadpy.load_schedules.")
    df = sched.to_pandas()
    if df.empty:
        raise SystemExit("No schedule data returned; check seasons or connectivity.")
    df["game_date"] = pd.to_datetime(df.get("game_date", df.get("gameday")), errors="coerce")
    return df


def _team_games(df: pd.DataFrame, team: str) -> pd.DataFrame:
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
    home = season_df[["home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "pf", "away_score": "pa"}
    )
    away = season_df[["away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "pf", "home_score": "pa"}
    )
    totals = (
        pd.concat([home, away], ignore_index=True)
        .groupby("team")
        .agg(points_for=("pf", "sum"), points_against=("pa", "sum"))
        .reset_index()
    )
    totals["off_rank"] = totals["points_for"].rank(method="min", ascending=False).astype(int)
    totals["def_rank"] = totals["points_against"].rank(method="min", ascending=True).astype(int)
    return totals


def _season_summary(games: pd.DataFrame, rankings: pd.DataFrame, team: str) -> Dict:
    record_counts = games["result"].value_counts().to_dict()
    pf = games["team_score"].sum()
    pa = games["opp_score"].sum()
    rank_row = rankings[rankings["team"] == team]
    off_rank = int(rank_row["off_rank"].iloc[0]) if not rank_row.empty else None
    def_rank = int(rank_row["def_rank"].iloc[0]) if not rank_row.empty else None
    return {
        "wins": record_counts.get("W", 0),
        "losses": record_counts.get("L", 0),
        "ties": record_counts.get("T", 0),
        "points_for": pf,
        "points_against": pa,
        "point_diff": pf - pa,
        "off_rank": off_rank,
        "def_rank": def_rank,
        "games": games[
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


def main() -> None:
    args = parse_args()
    seasons = args.seasons or _default_seasons()
    team = args.team.upper()

    schedules = _load_schedules(seasons)
    team_games = _team_games(schedules, team)

    summaries: Dict[int, Dict] = {}
    for season in sorted(seasons):
        season_games = team_games[team_games["season"] == season]
        if season_games.empty:
            continue
        rankings = _season_rankings(schedules, season)
        summaries[season] = _season_summary(season_games, rankings, team)

    if not summaries:
        print("No summaries generated; verify the seasons include completed years.")
        return

    for season in sorted(summaries):
        data = summaries[season]
        record = f"{data['wins']}-{data['losses']}" + (f"-{data['ties']}" if data['ties'] else "")
        print(f"\nSeason {season}: {team} record {record}")
        print(
            f"  Point diff: {data['point_diff']} "
            f"(PF {data['points_for']}, PA {data['points_against']})"
        )
        print(
            f"  Offensive rank (points for): {data['off_rank']}, "
            f"Defensive rank (points allowed): {data['def_rank']}"
        )
        games = data["games"].copy()
        games["home/away"] = games["is_home"].map({True: "home", False: "away"})
        games["game_date"] = games["game_date"].dt.strftime("%Y-%m-%d")
        print(games.drop(columns="is_home").to_string(index=False))


if __name__ == "__main__":
    main()
