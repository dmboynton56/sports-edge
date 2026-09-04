"""
MLB schedule/results fetcher using the public MLB Stats API.

The fetcher returns one row per completed MLB game with stable team IDs,
team names, probable starters, venue, scores, and a binary home-win target.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

import pandas as pd
import requests


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def _format_date(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _team_value(team: dict, key: str, default: object = None) -> object:
    return team.get("team", {}).get(key, default)


def _probable_pitcher(team: dict) -> dict:
    return team.get("probablePitcher") or {}


def fetch_mlb_schedule(
    season: int,
    *,
    game_types: Iterable[str] = ("R",),
    start_date: Optional[object] = None,
    end_date: Optional[object] = None,
    include_uncompleted: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch completed MLB games for a season.

    Args:
        season: MLB season year.
        game_types: MLB game type codes. "R" is regular season.
        start_date: Optional YYYY-MM-DD lower bound.
        end_date: Optional YYYY-MM-DD upper bound.
        include_uncompleted: Include scheduled/in-progress games without scores.
        timeout: HTTP timeout in seconds.

    Returns:
        DataFrame with completed games and target columns.
    """
    frames = []
    for game_type in game_types:
        params = {
            "sportId": 1,
            "season": season,
            "gameType": game_type,
            "hydrate": "team,probablePitcher,venue",
        }
        formatted_start = _format_date(start_date)
        formatted_end = _format_date(end_date)
        if formatted_start:
            params["startDate"] = formatted_start
        if formatted_end:
            params["endDate"] = formatted_end

        response = requests.get(MLB_SCHEDULE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()

        rows = []
        for day in payload.get("dates", []):
            for game in day.get("games", []):
                teams = game.get("teams", {})
                home = teams.get("home", {})
                away = teams.get("away", {})
                home_score = home.get("score")
                away_score = away.get("score")
                status = game.get("status", {})
                coded_state = status.get("codedGameState")
                detailed_state = status.get("detailedState", "")
                completed = bool(
                    home_score is not None
                    and away_score is not None
                    and (coded_state in {"F", "O"} or "Final" in detailed_state)
                )

                if not completed and not include_uncompleted:
                    continue

                rows.append(
                    {
                        "game_pk": int(game["gamePk"]),
                        "season": int(game.get("season", season)),
                        "game_type": game.get("gameType", game_type),
                        "game_date": game.get("officialDate", day.get("date")),
                        "game_datetime": game.get("gameDate"),
                        "status": detailed_state,
                        "completed": completed,
                        "venue_id": game.get("venue", {}).get("id"),
                        "venue_name": game.get("venue", {}).get("name"),
                        "home_team_id": int(_team_value(home, "id")),
                        "home_team": _team_value(home, "name"),
                        "home_team_abbr": _team_value(home, "abbreviation"),
                        "home_probable_pitcher_id": _probable_pitcher(home).get("id"),
                        "home_probable_pitcher": _probable_pitcher(home).get("fullName"),
                        "away_team_id": int(_team_value(away, "id")),
                        "away_team": _team_value(away, "name"),
                        "away_team_abbr": _team_value(away, "abbreviation"),
                        "away_probable_pitcher_id": _probable_pitcher(away).get("id"),
                        "away_probable_pitcher": _probable_pitcher(away).get("fullName"),
                        "home_score": int(home_score) if home_score is not None else pd.NA,
                        "away_score": int(away_score) if away_score is not None else pd.NA,
                        "home_win": int(home_score > away_score) if completed else pd.NA,
                        "run_diff": int(home_score) - int(away_score) if completed else pd.NA,
                    }
                )

        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "season",
                "game_type",
                "game_date",
                "game_datetime",
                "status",
                "completed",
                "venue_id",
                "venue_name",
                "home_team_id",
                "home_team",
                "home_team_abbr",
                "home_probable_pitcher_id",
                "home_probable_pitcher",
                "away_team_id",
                "away_team",
                "away_team_abbr",
                "away_probable_pitcher_id",
                "away_probable_pitcher",
                "home_score",
                "away_score",
                "home_win",
                "run_diff",
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["game_pk"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_datetime"] = pd.to_datetime(df["game_datetime"], utc=True).dt.tz_localize(None)
    for col in ["venue_id", "home_probable_pitcher_id", "away_probable_pitcher_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)
    return df


GAMES_UPSERT_COLUMNS = (
    "league",
    "season",
    "week",
    "home_team",
    "away_team",
    "game_date",
    "game_time_utc",
    "book_spread",
    "home_probable_pitcher",
    "away_probable_pitcher",
)


def mlb_schedule_to_games_df(
    schedule: pd.DataFrame,
    *,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Project an MLB Stats API / research slate onto the Supabase ``games`` shape.

    Research odds and the readiness audit both join through ``games``. That table
    is otherwise populated only by the later BigQuery → Supabase sync, so an
    evening re-run after an earlier crash has no MLB rows and drops every odds
    write. Persist officialDate as ``game_date`` so late-UTC kickoffs still land
    on the Denver slate date.
    """
    empty = pd.DataFrame(columns=list(GAMES_UPSERT_COLUMNS))
    if schedule is None or schedule.empty:
        return empty

    frame = schedule.copy()
    if "home_team" not in frame.columns or "away_team" not in frame.columns:
        return empty

    game_date = pd.to_datetime(frame.get("game_date"), errors="coerce")
    datetime_col = "game_datetime" if "game_datetime" in frame.columns else "game_time_utc"
    fallback_time = pd.to_datetime(game_date, utc=True, errors="coerce")
    if datetime_col in frame.columns:
        game_time = pd.to_datetime(frame[datetime_col], utc=True, errors="coerce")
        game_time = game_time.fillna(fallback_time)
    else:
        game_time = fallback_time

    if "season" in frame.columns:
        seasons = pd.to_numeric(frame["season"], errors="coerce")
        if season is not None:
            seasons = seasons.fillna(season)
    elif season is not None:
        seasons = season
    else:
        seasons = game_date.dt.year

    pitchers = {
        column: frame[column] if column in frame.columns else None
        for column in ("home_probable_pitcher", "away_probable_pitcher")
    }
    out = pd.DataFrame(
        {
            "league": "MLB",
            "season": seasons,
            "week": None,
            "home_team": frame["home_team"],
            "away_team": frame["away_team"],
            "game_date": game_date.dt.date,
            "game_time_utc": game_time,
            "book_spread": None,
            "home_probable_pitcher": pitchers["home_probable_pitcher"],
            "away_probable_pitcher": pitchers["away_probable_pitcher"],
        }
    )
    out = out.dropna(subset=["home_team", "away_team", "game_date"])
    return (
        out.drop_duplicates(subset=["league", "season", "home_team", "away_team", "game_date"])
        .reset_index(drop=True)
    )


def fetch_mlb_games_for_seasons(
    seasons: Iterable[int],
    *,
    game_types: Iterable[str] = ("R",),
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch completed MLB games for multiple seasons."""
    frames = []
    for season in seasons:
        print(f"Fetching MLB {season} games...")
        season_df = fetch_mlb_schedule(season, game_types=game_types, timeout=timeout)
        print(f"  Loaded {len(season_df)} completed games")
        if not season_df.empty:
            frames.append(season_df)

    if not frames:
        raise ValueError("No MLB games fetched for requested seasons.")

    return pd.concat(frames, ignore_index=True).sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)
