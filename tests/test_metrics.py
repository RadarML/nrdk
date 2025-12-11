"""Streamlined tests for NRDK metrics module."""

import torch

from nrdk.metrics import (
    BCE,
    BinaryDiceLoss,
    DepthWithConfidence,
    FocalLoss,
    Lp,
    PolarChamfer2D,
    PolarChamfer3D,
    VoxelDepth,
    lp_power,
    mean_with_mask,
)


def _small_occupancy():
    """Generate small test occupancy grids for classification (2,4,4,4)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (2, 4, 4, 4), dtype=torch.bool)


def _small_predictions():
    """Generate small prediction tensors for classification (2,4,4,4)."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 4, 4)


def _small_3d_occupancy():
    """Generate small 3D occupancy grids (2,4,4,4)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (2, 4, 4, 4), dtype=torch.bool)


# Classification Metrics Tests

def test_bce_basic():
    """Test BCE with minimal cases."""
    bce = BCE()
    y_true = _small_occupancy()
    y_hat = _small_predictions()
    loss = bce(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))


def test_bce_weighting():
    """Test BCE with cylindrical weighting."""
    bce = BCE(positive_weight=2.0, weighting="cylindrical")
    y_true = _small_occupancy()
    y_hat = _small_predictions()
    loss = bce(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


def test_binary_dice_loss_basic():
    """Test BinaryDiceLoss basic functionality."""
    dice = BinaryDiceLoss()
    y_true = _small_occupancy()
    y_hat = torch.sigmoid(_small_predictions())  # Convert to probabilities
    loss = dice(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(loss <= 1)


def test_binary_dice_loss_perfect():
    """Test BinaryDiceLoss with perfect predictions."""
    dice = BinaryDiceLoss()
    y_true = _small_occupancy()
    y_hat = y_true.float()  # Perfect predictions
    loss = dice(y_true, y_hat)

    assert loss.shape == (2,)
    # Loss should be close to 0 for perfect predictions
    assert torch.all(loss < 0.1)


def test_focal_loss_basic():
    """Test FocalLoss with default parameters."""
    focal = FocalLoss()
    y_true = _small_occupancy()
    y_hat = _small_predictions()
    loss = focal(y_true, y_hat)

    # FocalLoss should return per-batch loss
    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))


def test_focal_loss_gamma():
    """Test FocalLoss with different gamma."""
    focal = FocalLoss(gamma=1.0)
    y_true = _small_occupancy()
    y_hat = _small_predictions()
    loss = focal(y_true, y_hat)

    # FocalLoss should return per-batch loss
    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


# Core Metrics Tests


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


# Point Cloud Metrics Tests


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


def test_polar_chamfer_2d_empty():
    """Test PolarChamfer2D with empty grids."""
    chamfer = PolarChamfer2D(on_empty=32.0)

    # Empty grids
    y_true = torch.zeros(2, 8, 8, dtype=torch.bool)
    y_hat = torch.zeros(2, 8, 8, dtype=torch.bool)

    loss = chamfer(y_hat, y_true)

    assert loss.shape == (2,)
    assert torch.all(loss == 32.0)


def test_polar_chamfer_3d_basic():
    """Test PolarChamfer3D with basic depth maps."""
    chamfer = PolarChamfer3D()

    # Create simple depth maps with positive values
    torch.manual_seed(42)
    y_true = torch.zeros(2, 4, 8)
    y_hat = torch.zeros(2, 4, 8)

    # Add some depth values
    y_true[0, 1, 2] = 5.0
    y_true[0, 2, 3] = 10.0
    y_hat[0, 1, 3] = 6.0
    y_hat[0, 2, 2] = 9.0

    y_true[1, 0, 1] = 3.0
    y_hat[1, 0, 1] = 3.0  # Perfect match

    loss = chamfer(y_hat, y_true)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


def test_polar_chamfer_3d_modes():
    """Test PolarChamfer3D with different modes."""
    y_true = torch.zeros(1, 4, 4)
    y_hat = torch.zeros(1, 4, 4)

    # Add some points
    y_true[0, 1, 1] = 5.0
    y_hat[0, 1, 2] = 5.0

    # Test different modes
    for mode in ["chamfer", "hausdorff", "modhausdorff"]:
        chamfer = PolarChamfer3D(mode=mode)  # type: ignore
        loss = chamfer(y_hat, y_true)
        assert loss.shape == (1,)
        assert torch.all(loss >= 0)


# Edge Cases

def test_empty_tensors():
    """Test metrics handle empty tensors gracefully."""
    # Test with minimal non-empty tensors in 4D format
    y_true = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
    y_hat = torch.zeros(1, 1, 1, 1)

    bce = BCE()
    loss = bce(y_true, y_hat)
    assert loss.shape == (1,)
    assert torch.all(torch.isfinite(loss))


def test_identical_inputs():
    """Test metrics with identical true and predicted values."""
    y_true = _small_occupancy()
    y_hat = y_true.float()

    # Binary Dice should give low loss for identical inputs
    dice = BinaryDiceLoss()
    loss = dice(y_true, y_hat)
    assert torch.all(loss < 0.1)  # Should be close to 0
