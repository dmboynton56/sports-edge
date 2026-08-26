"""
Canonical market registry loader and validator.

Loads data-core/config/markets.yaml and validates:
- No duplicate market_ids
- Required fields are present
- live_enabled markets have model_name
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = ROOT / "config" / "markets.yaml"

REQUIRED_FIELDS = {
    "sport",
    "provider_sport_key",
    "market_id",
    "provider_market_key",
    "subject_type",
    "target_type",
    "sides",
    "result_source",
    "live_enabled",
    "backtest_enabled",
    "min_books",
    "max_price_age_minutes",
    "status",
}


class MarketsRegistryError(ValueError):
    """Raised when the markets registry is invalid."""


def load_markets_registry(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Load and validate the canonical markets registry.
    
    Args:
        path: Path to markets.yaml; defaults to data-core/config/markets.yaml
        
    Returns:
        List of market entries
        
    Raises:
        MarketsRegistryError: If validation fails
    """
    registry_path = path or DEFAULT_REGISTRY_PATH
    
    if not registry_path.exists():
        raise MarketsRegistryError(f"Markets registry not found: {registry_path}")
    
    with open(registry_path, encoding="utf-8") as f:
        try:
            markets = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise MarketsRegistryError(f"Failed to parse markets.yaml: {exc}") from exc
    
    if not isinstance(markets, list):
        raise MarketsRegistryError("markets.yaml must contain a list of market entries")
    
    if not markets:
        raise MarketsRegistryError("markets.yaml contains no market entries")
    
    # Validate each entry
    market_ids = set()
    for idx, market in enumerate(markets):
        if not isinstance(market, dict):
            raise MarketsRegistryError(f"Market entry {idx} is not a dict")
        
        # Check required fields
        missing = REQUIRED_FIELDS - set(market.keys())
        if missing:
            market_id = market.get("market_id", f"entry_{idx}")
            raise MarketsRegistryError(
                f"Market {market_id} missing required fields: {', '.join(sorted(missing))}"
            )
        
        # Check for duplicate market_ids
        market_id = market["market_id"]
        sport = market["sport"]
        full_id = f"{sport}:{market_id}"
        
        if full_id in market_ids:
            raise MarketsRegistryError(f"Duplicate market_id: {full_id}")
        market_ids.add(full_id)
        
        # Check live_enabled requires model_name
        if market["live_enabled"] and not market.get("model_name"):
            raise MarketsRegistryError(
                f"Market {full_id} has live_enabled=true but no model_name"
            )
    
    return markets


def get_market_by_id(
    sport: str,
    market_id: str,
    registry: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """Get a market entry by sport and market_id.
    
    Args:
        sport: Sport identifier (e.g., MLB, NBA)
        market_id: Market identifier (e.g., moneyline, batter_home_runs)
        registry: Optional pre-loaded registry; loads from default if None
        
    Returns:
        Market entry dict or None if not found
    """
    if registry is None:
        registry = load_markets_registry()
    
    for market in registry:
        if market["sport"] == sport and market["market_id"] == market_id:
            return market
    
    return None


def get_markets_by_sport(
    sport: str,
    registry: Optional[list[dict[str, Any]]] = None,
    live_only: bool = False,
) -> list[dict[str, Any]]:
    """Get all markets for a given sport.
    
    Args:
        sport: Sport identifier (e.g., MLB, NBA)
        registry: Optional pre-loaded registry; loads from default if None
        live_only: If True, return only markets with live_enabled=true
        
    Returns:
        List of market entries for the sport
    """
    if registry is None:
        registry = load_markets_registry()
    
    markets = [m for m in registry if m["sport"] == sport]
    
    if live_only:
        markets = [m for m in markets if m["live_enabled"]]
    
    return markets
