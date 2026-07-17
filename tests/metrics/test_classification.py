"""Tests for classification metrics (BCE, BinaryDiceLoss, FocalLoss)."""

import torch

from nrdk.metrics import BCE, BinaryDiceLoss, FocalLoss


def _small_occupancy():
    """Generate small test occupancy grids for classification (2,4,4,4)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (2, 4, 4, 4), dtype=torch.bool)


def _small_predictions():
    """Generate small prediction tensors for classification (2,4,4,4)."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 4, 4)


# BCE


def test_bce_basic():
    """Test BCE with minimal cases."""
    bce = BCE()
    y_true = _small_occupancy()
    y_hat = _small_predictions()
    loss = bce(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))


def test_bce_cylindrical_weighting():
    """Test BCE with cylindrical weighting."""
    bce = BCE(positive_weight=2.0, weighting="cylindrical")
    y_true = _small_occupancy()
    y_hat = _small_predictions()
    loss = bce(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)


def test_bce_spherical_weighting():
    """Test BCE with spherical weighting."""
    bce = BCE(positive_weight=2.0, weighting="spherical")
    y_true = _small_occupancy()
    y_hat = _small_predictions()
    loss = bce(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))


def test_bce_empty_tensors():
    """Test BCE handles minimal (1,1,1,1) tensors gracefully."""
    y_true = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
    y_hat = torch.zeros(1, 1, 1, 1)

    bce = BCE()
    loss = bce(y_true, y_hat)
    assert loss.shape == (1,)
    assert torch.all(torch.isfinite(loss))


# BinaryDiceLoss


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


def test_binary_dice_loss_identical_inputs():
    """Test BinaryDiceLoss with identical true and predicted values."""
    y_true = _small_occupancy()
    y_hat = y_true.float()

    dice = BinaryDiceLoss()
    loss = dice(y_true, y_hat)
    assert torch.all(loss < 0.1)  # Should be close to 0


def test_binary_dice_loss_cylindrical_weighting():
    """Test BinaryDiceLoss with cylindrical weighting."""
    dice = BinaryDiceLoss(weighting="cylindrical")
    y_true = _small_occupancy()
    y_hat = torch.sigmoid(_small_predictions())
    loss = dice(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))


def test_binary_dice_loss_spherical_weighting():
    """Test BinaryDiceLoss with spherical weighting."""
    dice = BinaryDiceLoss(weighting="spherical")
    y_true = _small_occupancy()
    y_hat = torch.sigmoid(_small_predictions())
    loss = dice(y_true, y_hat)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))


# FocalLoss


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
