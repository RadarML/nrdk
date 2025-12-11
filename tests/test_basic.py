"""Basic test to verify the testing framework is working correctly."""

import pytest


def test_basic_imports():
    """Test that NRDK submodule imports work correctly."""
    # Test core submodule imports
    submodules = [
        "nrdk.metrics",
        "nrdk.framework",
        "nrdk.models",
        "nrdk.modules",
        "nrdk.objectives",
        "nrdk.config",
        "nrdk.vis",
        "nrdk.tss",
        "nrdk.roverd"
    ]

    for module_name in submodules:
        try:
            module = __import__(module_name, fromlist=[''])
            assert module is not None, f"Module {module_name} is None"
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
