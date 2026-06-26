"""Recency helpers for MLB batter home-run prediction rows."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

NO_HR_IN_HISTORY_FLAG = "no_hr_in_history_window"


def games_since_last_hr(batting: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Return per-player games played since their most recent HR game.

    Uses only completed boxscore rows strictly before ``as_of``. Games are
    counted by ``game_pk`` so same-day doubleheaders are handled correctly.
    """
    columns = ["player_id", "games_since_last_hr", "last_hr_date"]
    if batting.empty:
        return pd.DataFrame(columns=columns)

    frame = batting.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    frame = frame[frame["game_date"] < as_of]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    played = frame[frame["plate_appearances"].fillna(0) > 0].copy()
    if played.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for player_id, group in played.groupby("player_id", sort=False):
        player_games = group.drop_duplicates(subset=["game_pk"]).sort_values(["game_date", "game_pk"])
        hr_games = player_games[player_games["home_runs"].fillna(0) >= 1]
        if hr_games.empty:
            rows.append(
                {
                    "player_id": int(player_id),
                    "games_since_last_hr": None,
                    "last_hr_date": None,
                }
            )
            continue

        last_hr = hr_games.iloc[-1]
        last_hr_date = last_hr["game_date"]
        last_hr_game_pk = int(last_hr["game_pk"])
        after_hr = player_games[
            (player_games["game_date"] > last_hr_date)
            | ((player_games["game_date"] == last_hr_date) & (player_games["game_pk"] > last_hr_game_pk))
        ]
        rows.append(
            {
                "player_id": int(player_id),
                "games_since_last_hr": int(len(after_hr)),
                "last_hr_date": last_hr_date.isoformat(),
            }
        )

    return pd.DataFrame(rows)


def recency_quality_flags(games_since_last_hr_value: Any) -> list[str]:
    if games_since_last_hr_value is None or (isinstance(games_since_last_hr_value, float) and pd.isna(games_since_last_hr_value)):
        return [NO_HR_IN_HISTORY_FLAG]
    return []
