"""
Player availability impact helpers.

These utilities keep injury adjustments as explicit feature deltas instead of
burying them in one-off scripts. The outputs are plain dictionaries so they can
be stored in Supabase, logged with predictions, or fed into backtests.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd


NFL_PLAYER_COLUMNS = (
    "passer_player_name",
    "rusher_player_name",
    "receiver_player_name",
    "fantasy_player_name",
    "player_name",
)

NBA_PLAYER_COLUMNS = ("player_name", "full_name", "name")


def _first_existing(columns: Iterable[str], frame: pd.DataFrame) -> Optional[str]:
    for col in columns:
        if col in frame.columns:
            return col
    return None


def _mode_or_none(values: pd.Series) -> Optional[str]:
    clean = values.dropna()
    if clean.empty:
        return None
    modes = clean.mode()
    if modes.empty:
        return str(clean.iloc[0])
    return str(modes.iloc[0])


def estimate_nfl_player_epa_impact(
    play_by_play: pd.DataFrame,
    player_name: str,
    *,
    position: Optional[str] = None,
    replacement_epa: float = 0.0,
    play_share: Optional[float] = None,
    side: str = "offense",
    player_columns: Iterable[str] = NFL_PLAYER_COLUMNS,
) -> dict:
    """
    Estimate how much a missing NFL player changes team EPA features.

    For quarterbacks this reads naturally: if a QB averages 0.20 EPA/play and
    the replacement baseline is 0.02, the offensive team delta is negative.
    """
    if play_by_play.empty:
        return _empty_nfl_impact(player_name, position, replacement_epa, side)

    matched_cols = [col for col in player_columns if col in play_by_play.columns]
    if not matched_cols or "epa" not in play_by_play.columns:
        return _empty_nfl_impact(player_name, position, replacement_epa, side)

    mask = pd.Series(False, index=play_by_play.index)
    for col in matched_cols:
        mask = mask | (play_by_play[col].astype(str) == player_name)

    player_rows = play_by_play.loc[mask].copy()
    player_epa = pd.to_numeric(player_rows.get("epa"), errors="coerce").dropna()
    if player_epa.empty:
        return _empty_nfl_impact(player_name, position, replacement_epa, side)

    team_col = "posteam" if side == "offense" else "defteam"
    team = _mode_or_none(player_rows[team_col]) if team_col in player_rows.columns else None
    usage_share = play_share
    if usage_share is None and team is not None and team_col in play_by_play.columns:
        team_plays = play_by_play[
            (play_by_play[team_col] == team)
            & pd.to_numeric(play_by_play["epa"], errors="coerce").notna()
        ]
        usage_share = len(player_epa) / len(team_plays) if len(team_plays) else None

    player_epa_per_play = float(player_epa.mean())
    epa_delta_per_play = float(replacement_epa - player_epa_per_play)
    team_delta = (
        float(epa_delta_per_play * usage_share)
        if usage_share is not None
        else epa_delta_per_play
    )

    return {
        "available": True,
        "league": "NFL",
        "team": team,
        "player_name": player_name,
        "position": position,
        "side": side,
        "metric_name": "epa_per_play",
        "player_value": player_epa_per_play,
        "replacement_value": float(replacement_epa),
        "usage_share": float(usage_share) if usage_share is not None else None,
        "team_delta": team_delta,
        "sample_size": int(len(player_epa)),
        "matched_columns": matched_cols,
    }


def _empty_nfl_impact(
    player_name: str,
    position: Optional[str],
    replacement_epa: float,
    side: str,
) -> dict:
    return {
        "available": False,
        "league": "NFL",
        "team": None,
        "player_name": player_name,
        "position": position,
        "side": side,
        "metric_name": "epa_per_play",
        "player_value": None,
        "replacement_value": float(replacement_epa),
        "usage_share": None,
        "team_delta": 0.0,
        "sample_size": 0,
        "matched_columns": [],
    }


def estimate_nba_player_rating_impact(
    player_logs: pd.DataFrame,
    player_name: str,
    *,
    position: Optional[str] = None,
    replacement_rating: float = 0.0,
    rating_col: str = "net_rating",
    minutes_share_col: str = "minutes_share",
) -> dict:
    """
    Estimate the team net-rating delta for a missing NBA player.

    The current repo only has team-level NBA ratings, so this accepts a
    normalized player-impact frame. That lets us plug in pbpstats, NBA API,
    manual priors, or a Supabase table without changing downstream code.
    """
    if player_logs.empty:
        return _empty_nba_impact(player_name, position, replacement_rating, rating_col)

    name_col = _first_existing(NBA_PLAYER_COLUMNS, player_logs)
    if name_col is None or rating_col not in player_logs.columns:
        return _empty_nba_impact(player_name, position, replacement_rating, rating_col)

    player_rows = player_logs[player_logs[name_col].astype(str) == player_name].copy()
    ratings = pd.to_numeric(player_rows[rating_col], errors="coerce").dropna()
    if ratings.empty:
        return _empty_nba_impact(player_name, position, replacement_rating, rating_col)

    if minutes_share_col in player_rows.columns:
        shares = pd.to_numeric(player_rows[minutes_share_col], errors="coerce").dropna()
        usage_share = float(shares.mean()) if not shares.empty else None
    elif "mpg" in player_rows.columns:
        mpg = pd.to_numeric(player_rows["mpg"], errors="coerce").dropna()
        usage_share = float((mpg / 48.0).mean()) if not mpg.empty else None
    else:
        usage_share = None

    team_col = _first_existing(("team", "team_abbr", "team_abbreviation"), player_rows)
    team = _mode_or_none(player_rows[team_col]) if team_col else None
    player_rating = float(ratings.mean())
    rating_delta = float(replacement_rating - player_rating)
    team_delta = float(rating_delta * usage_share) if usage_share is not None else rating_delta

    return {
        "available": True,
        "league": "NBA",
        "team": team,
        "player_name": player_name,
        "position": position,
        "metric_name": rating_col,
        "player_value": player_rating,
        "replacement_value": float(replacement_rating),
        "usage_share": usage_share,
        "team_delta": team_delta,
        "sample_size": int(len(ratings)),
    }


def _empty_nba_impact(
    player_name: str,
    position: Optional[str],
    replacement_rating: float,
    rating_col: str,
) -> dict:
    return {
        "available": False,
        "league": "NBA",
        "team": None,
        "player_name": player_name,
        "position": position,
        "metric_name": rating_col,
        "player_value": None,
        "replacement_value": float(replacement_rating),
        "usage_share": None,
        "team_delta": 0.0,
        "sample_size": 0,
    }


def build_game_injury_adjustments(game: Mapping, impacts: Iterable[Mapping]) -> dict:
    """
    Aggregate player impacts into home/away feature deltas for a matchup.
    """
    home_team = game.get("home_team")
    away_team = game.get("away_team")
    adjustments = {
        "home_injury_epa_delta": 0.0,
        "away_injury_epa_delta": 0.0,
        "home_injury_net_rating_delta": 0.0,
        "away_injury_net_rating_delta": 0.0,
        "home_injured_players": 0,
        "away_injured_players": 0,
    }

    for impact in impacts:
        if not impact.get("available", True):
            continue
        team = impact.get("team")
        if team not in {home_team, away_team}:
            continue

        prefix = "home" if team == home_team else "away"
        metric = str(impact.get("metric_name", "")).lower()
        delta = float(impact.get("team_delta") or 0.0)
        if "epa" in metric:
            adjustments[f"{prefix}_injury_epa_delta"] += delta
        elif "rating" in metric:
            adjustments[f"{prefix}_injury_net_rating_delta"] += delta
        adjustments[f"{prefix}_injured_players"] += 1

    return adjustments


def add_injury_adjustment_features(
    games_df: pd.DataFrame,
    impact_estimates: Optional[pd.DataFrame],
    *,
    league: str,
) -> pd.DataFrame:
    """
    Join player-impact estimates to game features and adjust model inputs.

    Matching prefers game_id when both frames have it, then falls back to
    game_date plus home/away team. The input estimates should have at least
    team, metric_name, and team_delta columns.
    """
    df = games_df.copy()
    for col in (
        "home_injury_epa_delta",
        "away_injury_epa_delta",
        "home_injury_net_rating_delta",
        "away_injury_net_rating_delta",
        "home_injured_players",
        "away_injured_players",
    ):
        if col not in df.columns:
            df[col] = 0.0 if col.endswith("_delta") else 0

    if impact_estimates is None or impact_estimates.empty:
        return df

    impacts = impact_estimates.copy()
    if "available" not in impacts.columns:
        impacts["available"] = True
    if "game_date" in impacts.columns:
        impacts["_match_game_date"] = pd.to_datetime(
            impacts["game_date"], errors="coerce"
        ).dt.normalize()
    else:
        impacts["_match_game_date"] = pd.NaT

    if "league" in impacts.columns:
        impacts = impacts[impacts["league"].astype(str).str.upper() == league.upper()]

    for idx, game in df.iterrows():
        game_impacts = _impacts_for_game(game, impacts)
        adjustments = build_game_injury_adjustments(game, game_impacts.to_dict("records"))
        for col, value in adjustments.items():
            df.loc[idx, col] = value

        _apply_team_delta_to_row(
            df,
            idx,
            prefix="form_home_",
            league=league,
            delta=adjustments["home_injury_epa_delta"]
            if league.upper() == "NFL"
            else adjustments["home_injury_net_rating_delta"],
        )
        _apply_team_delta_to_row(
            df,
            idx,
            prefix="form_away_",
            league=league,
            delta=adjustments["away_injury_epa_delta"]
            if league.upper() == "NFL"
            else adjustments["away_injury_net_rating_delta"],
        )

    _recompute_injury_adjusted_differentials(df, league)
    return df


def _impacts_for_game(game: pd.Series, impacts: pd.DataFrame) -> pd.DataFrame:
    team_mask = impacts["team"].isin([game.get("home_team"), game.get("away_team")])
    if "game_id" in impacts.columns and "game_id" in game.index and pd.notna(game.get("game_id")):
        game_id_mask = impacts["game_id"].astype(str) == str(game.get("game_id"))
        matched = impacts[team_mask & game_id_mask]
        if not matched.empty:
            return matched

    if "game_date" in game.index and pd.notna(game.get("game_date")):
        game_date = pd.to_datetime(game.get("game_date"), errors="coerce").normalize()
        date_mask = impacts["_match_game_date"] == game_date
        return impacts[team_mask & date_mask]

    return impacts.iloc[0:0]


def _apply_team_delta_to_row(
    df: pd.DataFrame,
    idx,
    *,
    prefix: str,
    league: str,
    delta: float,
) -> None:
    if delta == 0:
        return
    if league.upper() == "NFL":
        target_cols = [
            col for col in df.columns
            if col.startswith(prefix) and "epa_off" in col
        ]
    elif league.upper() == "NBA":
        target_cols = [
            col for col in df.columns
            if col.startswith(prefix) and "net_rating" in col
        ]
    else:
        target_cols = []

    for col in target_cols:
        current = pd.to_numeric(pd.Series([df.at[idx, col]]), errors="coerce").iloc[0]
        df.at[idx, col] = current + float(delta) if pd.notna(current) else current


def _recompute_injury_adjusted_differentials(df: pd.DataFrame, league: str) -> None:
    for window in (3, 5, 10):
        if league.upper() == "NFL":
            home_col = f"form_home_epa_off_{window}"
            away_col = f"form_away_epa_off_{window}"
            diff_col = f"form_epa_off_diff_{window}"
        elif league.upper() == "NBA":
            home_col = f"form_home_net_rating_{window}"
            away_col = f"form_away_net_rating_{window}"
            diff_col = f"form_net_rating_diff_{window}"
        else:
            continue

        if home_col in df.columns and away_col in df.columns:
            df[diff_col] = (
                pd.to_numeric(df[home_col], errors="coerce")
                - pd.to_numeric(df[away_col], errors="coerce")
            )


def apply_injury_adjustments_to_features(
    features_df: pd.DataFrame,
    *,
    team: str,
    impact: Mapping,
    league: str,
    is_home: bool,
) -> pd.DataFrame:
    """
    Apply an injury impact to existing model feature columns.
    """
    adjusted = features_df.copy()
    prefix = "form_home_" if is_home else "form_away_"
    delta = float(impact.get("team_delta") or 0.0)

    if league.upper() == "NFL":
        target_cols = [
            col for col in adjusted.columns
            if col.startswith(prefix) and "epa_off" in col
        ]
    elif league.upper() == "NBA":
        target_cols = [
            col for col in adjusted.columns
            if col.startswith(prefix) and "net_rating" in col
        ]
    else:
        target_cols = []

    for col in target_cols:
        adjusted[col] = pd.to_numeric(adjusted[col], errors="coerce") + delta

    adjusted[f"{team}_injury_delta"] = delta
    return adjusted
