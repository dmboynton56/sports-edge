import pytest
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def test_imports():
    """
    Simple smoke test to verify that core modules can be imported without errors.
    This catches syntax errors or missing dependencies.
    """
    try:
        from src.pipeline import refresh_nba
        from src.features import form_metrics
        from src.models import predictor
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")
