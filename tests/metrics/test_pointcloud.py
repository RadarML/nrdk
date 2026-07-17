"""Tests for point cloud metrics (PolarChamfer2D, PolarChamfer3D)."""

import pytest
import torch

from nrdk.metrics import PolarChamfer2D, PolarChamfer3D

# PolarChamfer2D


def test_polar_chamfer_2d_basic():
    """Test PolarChamfer2D with basic occupancy grids."""
    chamfer = PolarChamfer2D()

    # Create simple occupancy grids with some occupied cells
    torch.manual_seed(42)
    y_true = torch.zeros(2, 8, 8, dtype=torch.bool)
    y_hat = torch.zeros(2, 8, 8, dtype=torch.bool)

    # Add some occupied cells
    y_true[0, 2, 3] = True
    y_true[0, 4, 5] = True
    y_hat[0, 2, 4] = True
    y_hat[0, 4, 6] = True

    y_true[1, 1, 2] = True
    y_hat[1, 1, 2] = True  # Perfect match for second batch

    loss = chamfer(y_hat, y_true)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    # Second batch should have lower loss due to perfect match
    assert loss[1] < loss[0]


def test_polar_chamfer_2d_knn_matches_bruteforce():
    """PolarChamfer2D's torch_geometric knn backend matches the fallback."""
    try:
        chamfer = PolarChamfer2D(require_knn=True)
    except ImportError:
        pytest.xfail("torch_geometric is not installed")

    torch.manual_seed(42)
    y_true = torch.zeros(2, 8, 8, dtype=torch.bool)
    y_hat = torch.zeros(2, 8, 8, dtype=torch.bool)
    y_true[0, 2, 3] = True
    y_true[0, 4, 5] = True
    y_hat[0, 2, 4] = True
    y_hat[0, 4, 6] = True
    y_true[1, 1, 2] = True
    y_hat[1, 1, 2] = True

    knn_loss = chamfer(y_hat, y_true)
    chamfer._knn = None  # force brute-force fallback for comparison
    brute_loss = chamfer(y_hat, y_true)

    assert torch.allclose(knn_loss, brute_loss)


def test_polar_chamfer_2d_empty():
    """Test PolarChamfer2D with empty grids."""
    chamfer = PolarChamfer2D(on_empty=32.0)

    # Empty grids
    y_true = torch.zeros(2, 8, 8, dtype=torch.bool)
    y_hat = torch.zeros(2, 8, 8, dtype=torch.bool)

    loss = chamfer(y_hat, y_true)

    assert loss.shape == (2,)
    assert torch.all(loss == 32.0)


@pytest.mark.parametrize("mode", ["chamfer", "hausdorff", "modhausdorff"])
def test_polar_chamfer_2d_modes(mode):
    """Test PolarChamfer2D with different modes."""
    y_true = torch.zeros(1, 4, 8, dtype=torch.bool)
    y_hat = torch.zeros(1, 4, 8, dtype=torch.bool)

    y_true[0, 1, 5] = True
    y_hat[0, 2, 5] = True

    chamfer = PolarChamfer2D(mode=mode)  # type: ignore
    loss = chamfer(y_hat, y_true)
    assert loss.shape == (1,)
    assert torch.all(loss >= 0)


# PolarChamfer3D


def test_polar_chamfer_3d_basic():
    """Test PolarChamfer3D with basic occupancy grids."""
    chamfer = PolarChamfer3D()

    # Shape: [batch, elevation, azimuth, range]
    y_true = torch.zeros(2, 4, 8, 16, dtype=torch.bool)
    y_hat = torch.zeros(2, 4, 8, 16, dtype=torch.bool)

    # Occupied at (el=1, az=2, rng=5) and (el=2, az=3, rng=10)
    y_true[0, 1, 2, 5] = True
    y_true[0, 2, 3, 10] = True
    y_hat[0, 1, 3, 6] = True
    y_hat[0, 2, 2, 9] = True

    y_true[1, 0, 1, 3] = True
    y_hat[1, 0, 1, 3] = True  # Perfect match

    loss = chamfer(y_hat, y_true)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


def test_polar_chamfer_3d_empty():
    """Test PolarChamfer3D with empty grids."""
    chamfer = PolarChamfer3D(on_empty=32.0)

    y_true = torch.zeros(2, 4, 8, 16, dtype=torch.bool)
    y_hat = torch.zeros(2, 4, 8, 16, dtype=torch.bool)

    loss = chamfer(y_hat, y_true)

    assert loss.shape == (2,)
    assert torch.all(loss == 32.0)


def test_polar_chamfer_3d_float_logits_below_limit():
    """Test PolarChamfer3D ignores negative logits below max_points."""
    chamfer = PolarChamfer3D(max_points=10000)
    data = -torch.ones(4, 8, 16)
    data[1, 2, 5] = 0.5
    data[2, 3, 10] = 1.0

    points = chamfer.as_points(data)

    assert points.shape == (2, 3)


@pytest.mark.parametrize("mode", ["chamfer", "hausdorff", "modhausdorff"])
def test_polar_chamfer_3d_modes(mode):
    """Test PolarChamfer3D with different modes."""
    # Shape: [batch, elevation, azimuth, range]
    y_true = torch.zeros(1, 4, 4, 8, dtype=torch.bool)
    y_hat = torch.zeros(1, 4, 4, 8, dtype=torch.bool)

    # Two nearby occupied cells
    y_true[0, 1, 1, 5] = True
    y_hat[0, 1, 2, 5] = True

    chamfer = PolarChamfer3D(mode=mode)  # type: ignore
    loss = chamfer(y_hat, y_true)
    assert loss.shape == (1,)
    assert torch.all(loss >= 0)
