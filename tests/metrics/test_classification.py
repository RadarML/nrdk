"""Tests for classification metrics.

Covers BCE, BinaryDiceLoss, FocalLoss, and MeanIoU.
"""

import torch

from nrdk.metrics import BCE, BinaryDiceLoss, FocalLoss, MeanIoU


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


# MeanIoU


def _logits(labels: torch.Tensor, nc: int = 8) -> torch.Tensor:
    """Turn hard `batch h w` labels into `batch cls h w` logits."""
    return torch.nn.functional.one_hot(
        labels.long(), num_classes=nc).permute(0, 3, 1, 2).float() * 10.0


def test_miou_basic():
    """Test MeanIoU with minimal cases."""
    torch.manual_seed(42)
    y_true = torch.randint(0, 8, (2, 4, 4))
    y_hat = torch.randn(2, 8, 4, 4)
    miou = MeanIoU()(y_true, y_hat)

    assert miou.shape == (2,)
    assert torch.all(miou >= 0)
    assert torch.all(miou <= 1)
    assert torch.all(torch.isfinite(miou))


def test_miou_known_value():
    """Mean IoU must match a hand-computed value, not just lie in [0, 1]."""
    # GT: class 0 on the top half, class 1 on the bottom half.
    y_true = torch.zeros(1, 4, 4, dtype=torch.long)
    y_true[:, 2:, :] = 1
    # Prediction: class 0 everywhere, missing class 1 entirely.
    y_pred = torch.zeros(1, 4, 4, dtype=torch.long)

    # class 0: 8 correct / 16 union = 0.5; class 1: 0 / 8 = 0.0.
    # Classes 2-7 are absent from both, and must be excluded.
    assert torch.allclose(
        MeanIoU()(y_true, _logits(y_pred)),
        torch.tensor([0.25]), atol=1e-6)


def test_miou_perfect_and_worst():
    """Mean IoU must be 1.0 for an exact match and 0.0 for no overlap."""
    y_true = torch.zeros(1, 4, 4, dtype=torch.long)
    y_true[:, 2:, :] = 1
    disjoint = 1 - y_true

    assert torch.allclose(
        MeanIoU()(y_true, _logits(y_true)), torch.ones(1), atol=1e-6)
    assert torch.allclose(
        MeanIoU()(y_true, _logits(disjoint)), torch.zeros(1), atol=1e-6)


def test_miou_ignores_absent_classes():
    """Classes absent from both GT and prediction must not count.

    A `0 / 0` IoU must neither poison the sample with `nan` nor be counted as
    a zero: a perfect prediction using 2 of 8 classes still scores 1.0.
    """
    y_true = torch.zeros(2, 8, 8, dtype=torch.long)
    y_true[:, 4:, :] = 1

    miou = MeanIoU()(y_true, _logits(y_true, nc=8))
    assert torch.all(torch.isfinite(miou))
    assert torch.allclose(miou, torch.ones(2), atol=1e-6)


def test_miou_penalizes_missing_rare_class():
    """Mean IoU must drop sharply when a small class is missed entirely."""
    y_true = torch.zeros(1, 16, 16, dtype=torch.long)
    y_true[:, :2, :2] = 1  # 4 of 256 pixels
    missed = torch.zeros(1, 16, 16, dtype=torch.long)

    # Class 1 scores 0.0; class 0 still scores 252/256.
    assert torch.allclose(
        MeanIoU()(y_true, _logits(missed)),
        torch.tensor([(252 / 256) / 2]), atol=1e-6)


def test_miou_is_not_accuracy():
    """Mean IoU must not collapse to a function of pixel accuracy.

    Two predictions with *identical* pixel accuracy, differing only in
    whether the errors fall on the common or the rare class, must receive
    different scores -- this is the entire reason to report mIoU.
    """
    # Class 1 is a small 4x4 block (16 px); class 0 is the other 240 px.
    y_true = torch.zeros(1, 16, 16, dtype=torch.long)
    y_true[:, :4, :4] = 1

    # 8 errors on the common class: background predicted as class 1.
    common = y_true.clone()
    common[:, 8, :8] = 1
    # 8 errors on the rare class: half the class 1 block predicted as 0.
    rare = y_true.clone()
    rare[:, :2, :4] = 0

    # Both make 8 mistakes out of 256 pixels, so accuracy cannot tell them
    # apart; mIoU can, because losing half the rare class costs far more.
    m_common = MeanIoU()(y_true, _logits(common))
    m_rare = MeanIoU()(y_true, _logits(rare))

    assert torch.allclose(
        m_common, torch.tensor([(232 / 240 + 16 / 24) / 2]), atol=1e-6)
    assert torch.allclose(
        m_rare, torch.tensor([(240 / 248 + 8 / 16) / 2]), atol=1e-6)
    assert (m_rare < m_common - 0.05).all()


def test_miou_matches_reference():
    """Mean IoU must match a straightforward per-class reference loop."""
    torch.manual_seed(0)
    nc = 8
    y_true = torch.randint(0, nc, (3, 12, 10))
    y_pred = torch.randint(0, nc, (3, 12, 10))

    expected = []
    for g, p in zip(y_true, y_pred):
        ious = []
        for c in range(nc):
            gc, pc = (g == c), (p == c)
            union = (gc | pc).sum().item()
            if union > 0:
                ious.append((gc & pc).sum().item() / union)
        expected.append(sum(ious) / len(ious))

    assert torch.allclose(
        MeanIoU()(y_true, _logits(y_pred, nc=nc)),
        torch.tensor(expected, dtype=torch.float32), atol=1e-6)


def test_miou_arbitrary_spatial_dims():
    """Mean IoU must handle any number of spatial axes identically."""
    torch.manual_seed(0)
    nc = 4
    flat = torch.randint(0, nc, (2, 36))
    pred_flat = torch.randint(0, nc, (2, 36))

    logits_flat = torch.nn.functional.one_hot(
        pred_flat, num_classes=nc).permute(0, 2, 1).float() * 10.0
    logits_3d = logits_flat.reshape(2, nc, 3, 3, 4)

    assert torch.allclose(
        MeanIoU()(flat, logits_flat),
        MeanIoU()(flat.reshape(2, 3, 3, 4), logits_3d), atol=1e-6)
