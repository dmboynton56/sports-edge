"""
Odds math utilities: conversions, vig removal, edge/EV calculations, Kelly sizing.

All functions are pure (no side effects, no API calls). Probabilities are floats in [0, 1].
American odds follow standard convention: positive = underdog, negative = favorite.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def american_to_implied(price: int) -> float:
    """American odds -> raw implied probability (includes vig)."""
    if price > 0:
        return 100.0 / (price + 100.0)
    elif price < 0:
        return abs(price) / (abs(price) + 100.0)
    return 0.0


def decimal_to_implied(price: float) -> float:
    """Decimal odds -> raw implied probability."""
    if price <= 0:
        return 0.0
    return 1.0 / price


def american_to_decimal(price: int) -> float:
    """American odds -> decimal odds."""
    if price > 0:
        return (price / 100.0) + 1.0
    elif price < 0:
        return (100.0 / abs(price)) + 1.0
    return 0.0


def implied_to_american(prob: float) -> int:
    """Implied probability -> American odds (rounded)."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob < 0.5:
        return round((100.0 / prob) - 100.0)
    return round(-100.0 * prob / (1.0 - prob))


def remove_vig(implied_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Remove overround via proportional normalization so probs sum to 1.0.
    Input: {player: raw_implied_prob} from a single book.
    """
    total = sum(implied_probs.values())
    if total <= 0:
        return implied_probs
    return {k: v / total for k, v in implied_probs.items()}


def overround(implied_probs: Dict[str, float]) -> float:
    """Total implied probability (overround/vig). Fair market = 1.0."""
    return sum(implied_probs.values())


def consensus_no_vig(
    books: List[Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build consensus implied probabilities across books, then strip vig.

    Args:
        books: list of {player: raw_implied_prob} dicts, one per book.
        weights: optional {book_index_or_name: weight} — defaults to equal weight.

    Returns:
        {player: no_vig_probability}
    """
    if not books:
        return {}

    all_players: set[str] = set()
    for b in books:
        all_players.update(b.keys())

    avg: Dict[str, float] = {}
    for player in all_players:
        vals = [b[player] for b in books if player in b]
        avg[player] = sum(vals) / len(vals) if vals else 0.0

    return remove_vig(avg)


def edge(model_prob: float, market_prob: float) -> float:
    """Model probability minus market implied probability."""
    return model_prob - market_prob


def ev(model_prob: float, american_odds: int) -> float:
    """
    Expected value per $1 wagered.
    Positive EV means profitable in the long run.
    """
    decimal = american_to_decimal(american_odds)
    if decimal <= 0:
        return -1.0
    return model_prob * decimal - 1.0


def kelly_fraction(
    model_prob: float,
    american_odds: int,
    fraction: float = 0.25,
) -> float:
    """
    Fractional Kelly criterion bet sizing (default quarter-Kelly).
    Returns fraction of bankroll to wager. 0.0 if no edge.
    """
    decimal = american_to_decimal(american_odds)
    if decimal <= 1.0 or model_prob <= 0:
        return 0.0
    b = decimal - 1.0  # net payout per unit
    q = 1.0 - model_prob
    full_kelly = (model_prob * b - q) / b
    if full_kelly <= 0:
        return 0.0
    return full_kelly * fraction


def best_available(
    book_odds: Dict[str, int],
) -> tuple[int, str]:
    """
    Find best (highest payout) American odds across books for a player.
    Returns (best_price, best_book). All prices assumed positive (outrights).
    """
    if not book_odds:
        return (0, "")
    best_book = max(book_odds, key=lambda k: american_to_decimal(book_odds[k]))
    return book_odds[best_book], best_book


def edge_signal(edge_val: float, threshold: float = 0.02) -> str:
    """Classify edge as positive/negative/neutral."""
    if edge_val > threshold:
        return "positive"
    if edge_val < -threshold:
        return "negative"
    return "neutral"
