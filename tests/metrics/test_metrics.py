"""Tests for core metrics.

Covers lp_power, mean_with_mask, Lp, VoxelDepth, and DepthWithConfidence.
"""

import torch

from nrdk.metrics import (
    DepthWithConfidence,
    Lp,
    VoxelDepth,
    lp_power,
    mean_with_mask,
)


def _small_3d_occupancy():
    """Generate small 3D occupancy grids (2,4,4,4)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (2, 4, 4, 4), dtype=torch.bool)


# lp_power


def test_lp_power_basic():
    """Test lp_power with basic orders."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # Test L1
    result_l1 = lp_power(x, ord=1)
    expected_l1 = torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0])
    assert torch.allclose(result_l1, expected_l1)

    # Test L2
    result_l2 = lp_power(x, ord=2)
    expected_l2 = torch.tensor([4.0, 1.0, 0.0, 1.0, 4.0])
    assert torch.allclose(result_l2, expected_l2)


def test_lp_power_special_cases():
    """Test lp_power special optimized cases."""
    x = torch.tensor([0.0, 1.0, -1.0, 2.0])

    # Test L0 (counting non-zeros)
    result_l0 = lp_power(x, ord=0)
    expected_l0 = torch.tensor([0.0, 1.0, 1.0, 1.0])
    assert torch.allclose(result_l0, expected_l0)


def test_lp_power_general_order():
    """Test lp_power with a generic (non-optimized) order, e.g. ord=3."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    result = lp_power(x, ord=3)
    expected = torch.abs(x) ** 3
    assert torch.allclose(result, expected)


# mean_with_mask


def test_mean_with_mask_basic():
    """Test mean_with_mask without mask."""
    x = torch.randn(2, 4, 4)
    result = mean_with_mask(x, None)
    expected = torch.mean(x.reshape(2, -1), dim=1)

    assert torch.allclose(result, expected)
    assert result.shape == (2,)


def test_mean_with_mask_with_mask():
    """Test mean_with_mask with mask."""
    torch.manual_seed(42)
    x = torch.randn(2, 4, 4)
    mask = torch.randint(0, 2, (2, 4, 4), dtype=torch.bool)
    result = mean_with_mask(x, mask)

    assert result.shape == (2,)
    assert torch.all(torch.isfinite(result))


def test_mean_with_mask_all_false_clamped():
    """Test mean_with_mask clamps n_valid to 1 when the mask is all-False."""
    x = torch.randn(2, 4, 4)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    result = mean_with_mask(x, mask)

    # With an all-False mask, the numerator is 0 and n_valid is clamped to 1
    assert result.shape == (2,)
    assert torch.all(torch.isfinite(result))
    assert torch.allclose(result, torch.zeros(2))


# Lp


def test_lp_loss_basic():
    """Test Lp loss with basic parameters."""
    lp = Lp(ord=1)
    y_true = torch.randn(2, 4, 4)
    y_hat = torch.randn(2, 4, 4)
    loss = lp(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


def test_lp_loss_with_mask():
    """Test Lp loss with validity mask."""
    lp = Lp(ord=2)
    y_true = torch.randn(2, 4, 4)
    y_hat = torch.randn(2, 4, 4)
    valid = torch.randint(0, 2, (2, 4, 4), dtype=torch.bool)
    loss = lp(y_true, y_hat, valid)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


# VoxelDepth


def test_voxel_depth_basic():
    """Test VoxelDepth with basic parameters."""
    voxel_depth = VoxelDepth()
    y_true = _small_3d_occupancy()
    y_hat = _small_3d_occupancy()
    loss = voxel_depth(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


def test_voxel_depth_reverse():
    """Test VoxelDepth with reverse direction."""
    voxel_depth = VoxelDepth(reverse=True)
    y_true = _small_3d_occupancy()
    y_hat = _small_3d_occupancy()
    loss = voxel_depth(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


def test_voxel_depth_no_filter_empty():
    """Test VoxelDepth with filter_empty disabled (no masking of empty bins)."""
    voxel_depth = VoxelDepth(filter_empty=False)
    y_true = _small_3d_occupancy()
    y_hat = _small_3d_occupancy()
    loss = voxel_depth(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))


# DepthWithConfidence


def test_depth_with_confidence_basic():
    """Test DepthWithConfidence with basic parameters."""
    depth_conf = DepthWithConfidence()
    y_true = torch.randn(2, 4, 4).abs()  # Positive depths
    y_hat = torch.randn(2, 4, 4).abs()
    confidence = torch.randn(2, 4, 4)
    loss = depth_conf(y_true, y_hat, confidence)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))


def test_depth_with_confidence_with_mask():
    """Test DepthWithConfidence with validity mask."""
    depth_conf = DepthWithConfidence(alpha=0.1)
    y_true = torch.randn(2, 4, 4).abs()
    y_hat = torch.randn(2, 4, 4).abs()
    confidence = torch.randn(2, 4, 4)
    valid = torch.randint(0, 2, (2, 4, 4), dtype=torch.bool)
    loss = depth_conf(y_true, y_hat, confidence, valid)

    assert loss.shape == (2,)
