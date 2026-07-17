"""Tests for the Semseg objective."""

from dataclasses import dataclass

import torch

from nrdk.objectives import Semseg


@dataclass
class MockSemsegData:
    """Mock data class for Semseg testing."""

    semseg: torch.Tensor


def test_semseg_basic():
    """Test Semseg basic functionality."""
    torch.manual_seed(42)
    semseg_data = torch.randint(0, 8, (2, 1, 16, 16), dtype=torch.uint8)
    y_pred = torch.randn(2, 1, 16, 16, 8)

    objective = Semseg()
    mock_data = MockSemsegData(semseg=semseg_data)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))
    assert "bce" in metrics
    assert "acc" in metrics
    assert "top2" in metrics
    assert "miou" in metrics


def test_semseg_cross_entropy():
    """Test Semseg cross-entropy loss computation."""
    torch.manual_seed(42)
    semseg_data = torch.randint(0, 4, (2, 1, 8, 8), dtype=torch.uint8)
    y_pred = torch.randn(2, 1, 8, 8, 4)

    objective = Semseg()
    mock_data = MockSemsegData(semseg=semseg_data)

    loss, metrics = objective(mock_data, y_pred)

    # Cross-entropy loss should equal the returned loss
    assert loss.shape == (2,)
    assert torch.allclose(loss, metrics["bce"])


def test_semseg_accuracy_metrics():
    """Test Semseg top-1 and top-2 accuracy."""
    torch.manual_seed(42)
    semseg_data = torch.randint(0, 4, (2, 1, 8, 8), dtype=torch.uint8)
    y_pred = torch.randn(2, 1, 8, 8, 4)

    objective = Semseg()
    mock_data = MockSemsegData(semseg=semseg_data)

    loss, metrics = objective(mock_data, y_pred)

    assert "acc" in metrics
    assert "top2" in metrics
    assert torch.all(metrics["acc"] >= 0)
    assert torch.all(metrics["acc"] <= 1)
    assert torch.all(metrics["top2"] >= metrics["acc"])  # Top-2 >= Top-1
    assert torch.all(metrics["top2"] <= 1)


def test_semseg_miou():
    """Test Semseg Mean IoU computation."""
    torch.manual_seed(42)
    semseg_data = torch.randint(0, 4, (2, 1, 8, 8), dtype=torch.uint8)
    y_pred = torch.randn(2, 1, 8, 8, 4)

    objective = Semseg()
    mock_data = MockSemsegData(semseg=semseg_data)

    loss, metrics = objective(mock_data, y_pred)

    assert "miou" in metrics
    assert torch.all(metrics["miou"] >= 0)
    assert torch.all(metrics["miou"] <= 1)


def test_semseg_perfect_predictions():
    """Test Semseg with perfect class predictions."""
    torch.manual_seed(42)
    semseg_data = torch.randint(0, 4, (2, 1, 8, 8), dtype=torch.uint8)
    # Create perfect logits (high confidence for correct class) - vectorized
    y_pred = torch.nn.functional.one_hot(
        semseg_data.long(), num_classes=4).float() * 10.0

    objective = Semseg()
    mock_data = MockSemsegData(semseg=semseg_data)

    loss, metrics = objective(mock_data, y_pred)

    # Should have high accuracy for perfect predictions
    assert torch.all(metrics["acc"] > 0.99)
    assert torch.all(metrics["miou"] > 0.99)


def test_semseg_visualization_configs():
    """Test Semseg visualization configuration handling (dict vs. empty)."""
    torch.manual_seed(42)
    semseg_data = torch.randint(0, 4, (2, 1, 8, 8), dtype=torch.uint8)
    y_pred = torch.randn(2, 1, 8, 8, 4)

    # Test different config formats
    config_dict = {"cols": 4, "cmaps": {"semseg": "tab20"}}
    objective1 = Semseg(vis_config=config_dict)

    # Test empty config
    objective2 = Semseg(vis_config={})

    mock_data = MockSemsegData(semseg=semseg_data)

    loss1, _ = objective1(mock_data, y_pred)
    loss2, _ = objective2(mock_data, y_pred)

    assert loss1.shape == (2,)
    assert loss2.shape == (2,)
    assert objective1.vis_config.cols == 4
    # Default cols should be set for objective2
    assert hasattr(objective2.vis_config, 'cols')
