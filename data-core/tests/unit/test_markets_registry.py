"""Unit tests for markets_registry loader and validator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from src.data.markets_registry import (
    MarketsRegistryError,
    get_market_by_id,
    get_markets_by_sport,
    load_markets_registry,
)


def test_load_valid_registry():
    """Test loading a valid markets registry."""
    markets = load_markets_registry()
    
    assert isinstance(markets, list)
    assert len(markets) > 0
    
    # Check that all entries have required fields
    for market in markets:
        assert "sport" in market
        assert "market_id" in market
        assert "provider_market_key" in market
        assert "live_enabled" in market


def test_no_duplicate_market_ids():
    """Test that there are no duplicate sport:market_id combinations."""
    markets = load_markets_registry()
    
    seen = set()
    for market in markets:
        full_id = f"{market['sport']}:{market['market_id']}"
        assert full_id not in seen, f"Duplicate market_id: {full_id}"
        seen.add(full_id)


def test_live_markets_have_models():
    """Test that all live_enabled markets have a model_name."""
    markets = load_markets_registry()
    
    for market in markets:
        if market["live_enabled"]:
            assert market.get("model_name"), (
                f"{market['sport']}:{market['market_id']} "
                f"has live_enabled=true but no model_name"
            )


def test_mlb_markets_exist():
    """Test that the five MLB markets exist in the registry."""
    markets = load_markets_registry()
    mlb_markets = {m["market_id"] for m in markets if m["sport"] == "MLB"}
    
    expected = {
        "moneyline",
        "run_line",
        "total",
        "batter_home_runs",
        "pitcher_strikeouts",
    }
    
    assert expected.issubset(mlb_markets), f"Missing MLB markets: {expected - mlb_markets}"


def test_get_market_by_id():
    """Test retrieving a specific market by sport and market_id."""
    markets = load_markets_registry()
    
    # Test MLB moneyline
    mlb_ml = get_market_by_id("MLB", "moneyline", registry=markets)
    assert mlb_ml is not None
    assert mlb_ml["provider_market_key"] == "h2h"
    assert mlb_ml["subject_type"] == "team"
    
    # Test non-existent market
    fake = get_market_by_id("MLB", "fake_market", registry=markets)
    assert fake is None


def test_get_markets_by_sport():
    """Test retrieving all markets for a sport."""
    markets = load_markets_registry()
    
    mlb_markets = get_markets_by_sport("MLB", registry=markets)
    assert len(mlb_markets) >= 5
    assert all(m["sport"] == "MLB" for m in mlb_markets)
    
    # Test live_only filter
    mlb_live = get_markets_by_sport("MLB", registry=markets, live_only=True)
    assert all(m["live_enabled"] for m in mlb_live)
    assert len(mlb_live) < len(mlb_markets)


def test_invalid_yaml_raises_error():
    """Test that invalid YAML raises MarketsRegistryError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content:\n  - broken")
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(MarketsRegistryError, match="Failed to parse"):
            load_markets_registry(temp_path)
    finally:
        temp_path.unlink()


def test_missing_required_field_raises_error():
    """Test that missing required fields raise MarketsRegistryError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        # Missing 'provider_market_key'
        yaml.dump(
            [
                {
                    "sport": "TEST",
                    "provider_sport_key": "test_sport",
                    "market_id": "test_market",
                    # provider_market_key is missing
                    "subject_type": "team",
                    "target_type": "binary",
                    "sides": ["home", "away"],
                    "result_source": "test",
                    "live_enabled": False,
                    "backtest_enabled": False,
                    "min_books": 2,
                    "max_price_age_minutes": 20,
                    "status": "test",
                }
            ],
            f,
        )
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(MarketsRegistryError, match="missing required fields"):
            load_markets_registry(temp_path)
    finally:
        temp_path.unlink()


def test_duplicate_market_id_raises_error():
    """Test that duplicate market_ids raise MarketsRegistryError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            [
                {
                    "sport": "TEST",
                    "provider_sport_key": "test_sport",
                    "market_id": "duplicate",
                    "provider_market_key": "test",
                    "subject_type": "team",
                    "target_type": "binary",
                    "sides": ["home", "away"],
                    "result_source": "test",
                    "model_name": None,
                    "model_version": None,
                    "live_enabled": False,
                    "backtest_enabled": False,
                    "min_books": 2,
                    "max_price_age_minutes": 20,
                    "status": "test",
                },
                {
                    "sport": "TEST",
                    "provider_sport_key": "test_sport",
                    "market_id": "duplicate",  # Same market_id
                    "provider_market_key": "test2",
                    "subject_type": "team",
                    "target_type": "binary",
                    "sides": ["home", "away"],
                    "result_source": "test",
                    "model_name": None,
                    "model_version": None,
                    "live_enabled": False,
                    "backtest_enabled": False,
                    "min_books": 2,
                    "max_price_age_minutes": 20,
                    "status": "test",
                },
            ],
            f,
        )
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(MarketsRegistryError, match="Duplicate market_id"):
            load_markets_registry(temp_path)
    finally:
        temp_path.unlink()


def test_live_without_model_raises_error():
    """Test that live_enabled=true without model_name raises error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            [
                {
                    "sport": "TEST",
                    "provider_sport_key": "test_sport",
                    "market_id": "test_market",
                    "provider_market_key": "test",
                    "subject_type": "team",
                    "target_type": "binary",
                    "sides": ["home", "away"],
                    "result_source": "test",
                    "model_name": None,  # No model
                    "model_version": None,
                    "live_enabled": True,  # But live enabled!
                    "backtest_enabled": False,
                    "min_books": 2,
                    "max_price_age_minutes": 20,
                    "status": "test",
                }
            ],
            f,
        )
        temp_path = Path(f.name)
    
    try:
        with pytest.raises(MarketsRegistryError, match="live_enabled=true but no model_name"):
            load_markets_registry(temp_path)
    finally:
        temp_path.unlink()
