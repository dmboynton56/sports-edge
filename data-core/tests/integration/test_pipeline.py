"""Import smoke coverage for optional pipeline dependencies."""

import importlib

import pytest


@pytest.mark.parametrize("module", [
    "src.pipeline.refresh_nba",
    "src.features.form_metrics",
    "src.models.predictor",
])
def test_imports(module):
    importlib.import_module(module)
