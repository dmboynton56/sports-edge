"""NFL moneyline toss-up band.

When the model's home win probability is within ±0.05 of a coin flip, do not
publish a moneyline pick. Spread still publishes. Spread and ML do not need to
agree in this band — books disagree on the side at ±1.5.
"""

from __future__ import annotations

NFL_TOSS_UP_BAND = 0.05


def is_nfl_toss_up(home_win_prob: object, band: float = NFL_TOSS_UP_BAND) -> bool:
    try:
        value = float(home_win_prob)
    except (TypeError, ValueError):
        return False
    if value != value:  # NaN
        return False
    return abs(value - 0.5) <= band


def nfl_moneyline_pick(
    home_team: str,
    away_team: str,
    home_win_prob: object,
    *,
    band: float = NFL_TOSS_UP_BAND,
) -> str | None:
    """Return the moneyline side, or None for a toss-up / invalid probability."""
    try:
        value = float(home_win_prob)
    except (TypeError, ValueError):
        return None
    if value != value or value < 0 or value > 1:
        return None
    if is_nfl_toss_up(value, band=band):
        return None
    if value == 0.5:
        return None
    return home_team if value > 0.5 else away_team
