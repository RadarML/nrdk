"""Tests for voxel rendering utilities.

Covers bev_from_polar2, bev_height_from_polar_occupancy, and
depth_from_polar_occupancy.
"""

import numpy as np
import torch

from nrdk.vis import (
    bev_from_polar2,
    bev_height_from_polar_occupancy,
    depth_from_polar_occupancy,
)


def _small_polar_occupancy():
    """Generate small polar occupancy grids (2,3,6,8)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (2, 3, 6, 8), dtype=torch.bool)


def _small_polar_data():
    """Generate small polar data (2,6,8,1)."""
    torch.manual_seed(42)
    return torch.randn(2, 6, 8, 1)


# bev_from_polar2


def test_bev_from_polar2_basic():
    """Test basic bev_from_polar2 functionality."""
    data = _small_polar_data()  # (2,6,8,1)
    result = bev_from_polar2(data, size=16)
    assert result.shape == (2, 16, 32, 1)
    assert result.dtype == data.dtype


def test_bev_from_polar2_different_theta():
    """Test bev_from_polar2 with custom theta range."""
    data = _small_polar_data()  # (2,6,8,1)
    result = bev_from_polar2(
        data, size=8, theta_min=-np.pi / 2, theta_max=np.pi / 2)

    assert result.shape == (2, 8, 16, 1)


def test_bev_from_polar2_bool_data():
    """Test bev_from_polar2 preserves bool dtype."""
    data = torch.randint(0, 2, (2, 6, 8, 1), dtype=torch.bool)
    result = bev_from_polar2(data, size=8)

    assert result.dtype == torch.bool
    assert result.shape == (2, 8, 16, 1)


def test_bev_from_polar2_minimal_size():
    """Test bev_from_polar2 with minimal size."""
    data = torch.randn(1, 4, 4, 1)
    result = bev_from_polar2(data, size=4)

    assert result.shape == (1, 4, 8, 1)


# bev_height_from_polar_occupancy


def test_bev_height_from_polar_occupancy_scaled():
    """Test bev_height_from_polar_occupancy with scaling."""
    data = _small_polar_occupancy()  # (2,3,6,8)
    result = bev_height_from_polar_occupancy(data, size=16, scale=True)
    # If it succeeds, check basic properties
    assert result.shape == (2, 16, 32)
    assert result.dtype == torch.float32


def test_bev_height_from_polar_occupancy_unscaled_small():
    """Test bev_height_from_polar_occupancy without scaling, small elevation."""
    data = _small_polar_occupancy()  # (2,3,6,8) - 3 elevation bins <= 256
    result = bev_height_from_polar_occupancy(data, size=16, scale=False)
    assert result.shape == (2, 16, 32)
    assert result.dtype == torch.uint8


def test_bev_height_from_polar_occupancy_unscaled_large():
    """Test bev_height_from_polar_occupancy with >256 elevation bins."""
    # Create data with 300 elevation bins
    data = torch.randint(0, 2, (1, 300, 6, 8), dtype=torch.bool)
    result = bev_height_from_polar_occupancy(data, size=16, scale=False)
    assert result.shape == (1, 16, 32)
    assert result.dtype in [torch.int16, torch.uint8]


def test_bev_height_from_polar_occupancy_empty():
    """Test bev_height_from_polar_occupancy with empty occupancy."""
    data = torch.zeros(2, 3, 6, 8, dtype=torch.bool)  # All false
    result = bev_height_from_polar_occupancy(data, size=8, scale=False)

    assert result.shape == (2, 8, 16)
    assert result.dtype == torch.uint8
    assert torch.all(result == 0)


# depth_from_polar_occupancy


def test_depth_from_polar_occupancy_basic():
    """Test basic depth_from_polar_occupancy functionality."""
    data = _small_polar_occupancy()  # (2,3,6,8) - 8 range bins <= 256
    result = depth_from_polar_occupancy(data)
    assert result.shape == (2, 3, 6)
    assert result.dtype == torch.uint8


def test_depth_from_polar_occupancy_with_resize():
    """Test depth_from_polar_occupancy with resizing."""
    data = _small_polar_occupancy()  # (2,3,6,8)
    result = depth_from_polar_occupancy(data, size=(4, 7))  # Non-square target
    assert result.shape == (2, 4, 7)
    assert result.dtype == torch.uint8


def test_depth_from_polar_occupancy_large_range():
    """Test depth_from_polar_occupancy with >256 range bins."""
    data = torch.randint(0, 2, (1, 3, 6, 300), dtype=torch.bool)
    result = depth_from_polar_occupancy(data)
    assert result.shape == (1, 3, 6)
    assert result.dtype == torch.int16


def test_depth_from_polar_occupancy_all_empty():
    """Test depth_from_polar_occupancy with all-empty occupancy."""
    data = torch.zeros(2, 3, 6, 8, dtype=torch.bool)
    result = depth_from_polar_occupancy(data)

    assert result.shape == (2, 3, 6)
    assert torch.all(result == 0)


def test_depth_from_polar_occupancy_all_occupied():
    """Test depth_from_polar_occupancy with all-occupied occupancy."""
    data = torch.ones(2, 3, 6, 8, dtype=torch.bool)
    result = depth_from_polar_occupancy(data)

    assert result.shape == (2, 3, 6)
    assert torch.all(result == 0)
