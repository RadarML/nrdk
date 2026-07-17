"""Tests for the Occupancy3D and Occupancy2D objectives."""

from dataclasses import dataclass

import pytest
import torch

from nrdk.objectives import Occupancy2D, Occupancy3D


@dataclass
class MockOccupancy3DData:
    """Mock data class for Occupancy3D testing."""

    occupancy: torch.Tensor


@dataclass
class MockOccupancy2DData:
    """Mock data class for Occupancy2D testing."""

    occupancy: torch.Tensor


# Occupancy3D Tests


def test_occupancy_3d_basic():
    """Test Occupancy3D basic functionality."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 4, 6, 8)

    objective = Occupancy3D(require_knn=False)
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))
    assert "bce" in metrics
    assert "height" in metrics
    assert "depth" in metrics


def test_occupancy_3d_range_weighted():
    """Test Occupancy3D with range weighting enabled."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 4, 6, 8)

    objective = Occupancy3D(range_weighted=True, require_knn=False)
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))


def test_occupancy_3d_positive_weight():
    """Test Occupancy3D with different positive weights."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 4, 6, 8)

    obj1 = Occupancy3D(positive_weight=16.0, require_knn=False)
    obj2 = Occupancy3D(positive_weight=64.0, require_knn=False)
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss1, _ = obj1(mock_data, y_pred)
    loss2, _ = obj2(mock_data, y_pred)

    assert loss1.shape == (2,)
    assert loss2.shape == (2,)
    assert torch.all(torch.isfinite(loss1))
    assert torch.all(torch.isfinite(loss2))


@pytest.mark.parametrize("mode", ["spherical", "cylindrical"])
def test_occupancy_3d_modes(mode):
    """Test Occupancy3D spherical vs cylindrical modes."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 4, 6, 8)

    objective = Occupancy3D(mode=mode, require_knn=False)  # type: ignore
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss, _ = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))


def test_occupancy_3d_visualization_config():
    """Test Occupancy3D visualization configuration."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 4, 6, 8)

    vis_config = {"cols": 4, "cmaps": {"bev": "viridis", "depth": "inferno"}}
    objective = Occupancy3D(vis_config=vis_config, require_knn=False)
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert objective.vis_config.cols == 4
    assert "bev" in objective.vis_config.cmaps
    assert "depth" in objective.vis_config.cmaps


def test_occupancy_3d_empty_occupancy():
    """Test Occupancy3D with empty occupancy grid."""
    torch.manual_seed(42)
    occupancy = torch.zeros(2, 1, 4, 6, 8, dtype=torch.bool)  # All False
    y_pred = torch.randn(2, 1, 4, 6, 8)

    objective = Occupancy3D(require_knn=False)
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))
    assert torch.all(metrics["bce"] >= 0)


def test_occupancy_3d_perfect_predictions():
    """Test Occupancy3D with perfect predictions."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.where(occupancy, 10.0, -10.0)

    objective = Occupancy3D(require_knn=False)
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))
    # BCE loss should be very small for perfect predictions
    assert torch.all(metrics["bce"] < 0.1)


def test_occupancy_3d_train_eval_modes():
    """Test Occupancy3D works in both train and eval modes."""
    torch.manual_seed(42)

    occ3d = torch.randint(0, 2, (1, 1, 4, 6, 8), dtype=torch.bool)
    pred3d = torch.randn(1, 1, 4, 6, 8)

    objective = Occupancy3D(require_knn=False)
    mock_data = MockOccupancy3DData(occupancy=occ3d)

    # Train mode
    loss_train, metrics_train = objective(mock_data, pred3d, train=True)

    # Eval mode
    loss_eval, metrics_eval = objective(mock_data, pred3d, train=False)

    assert loss_train.shape == loss_eval.shape
    # Eval mode includes additional metrics like chamfer
    assert "chamfer" in metrics_eval
    assert "chamfer" not in metrics_train
    # Common metrics should have same values
    common_keys = set(metrics_train.keys()) & set(metrics_eval.keys())
    for key in common_keys:
        assert torch.allclose(metrics_train[key], metrics_eval[key])


# Occupancy2D Tests


def test_occupancy_2d_basic():
    """Test Occupancy2D basic functionality."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 6, 8)

    objective = Occupancy2D(require_knn=False)
    mock_data = MockOccupancy2DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))
    assert "bce" in metrics
    assert "dice" in metrics
    # chamfer is only available in eval mode


def test_occupancy_2d_range_weighted():
    """Test Occupancy2D with range weighting."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 6, 8)

    objective = Occupancy2D(range_weighted=True, require_knn=False)
    mock_data = MockOccupancy2DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))


def test_occupancy_2d_positive_weight():
    """Test Occupancy2D with different positive weights."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 6, 8)

    obj1 = Occupancy2D(positive_weight=8.0, require_knn=False)
    obj2 = Occupancy2D(positive_weight=32.0, require_knn=False)
    mock_data = MockOccupancy2DData(occupancy=occupancy)

    loss1, _ = obj1(mock_data, y_pred)
    loss2, _ = obj2(mock_data, y_pred)

    assert loss1.shape == (2,)
    assert loss2.shape == (2,)
    assert torch.all(torch.isfinite(loss1))
    assert torch.all(torch.isfinite(loss2))


def test_occupancy_2d_chamfer_loss():
    """Test Occupancy2D chamfer distance computation."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 6, 8)

    objective = Occupancy2D(require_knn=False)
    mock_data = MockOccupancy2DData(occupancy=occupancy)

    # Chamfer is only available in eval mode
    loss, metrics = objective(mock_data, y_pred, train=False)

    assert loss.shape == (2,)
    assert "chamfer" in metrics
    assert torch.all(metrics["chamfer"] >= 0)
    assert torch.all(torch.isfinite(metrics["chamfer"]))


def test_occupancy_2d_visualization():
    """Test Occupancy2D visualization generation."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 6, 8)

    vis_config = {"cols": 6, "cmaps": {"bev": "plasma", "depth": "viridis"}}
    objective = Occupancy2D(vis_config=vis_config, require_knn=False)
    mock_data = MockOccupancy2DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert objective.vis_config.cols == 6


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_occupancy_2d_batch_size_consistency(batch_size):
    """Test Occupancy2D handles different batch sizes."""
    torch.manual_seed(42)

    occ = torch.randint(0, 2, (batch_size, 1, 4, 4), dtype=torch.bool)
    pred = torch.randn(batch_size, 1, 4, 4)

    objective = Occupancy2D(require_knn=False)
    mock_data = MockOccupancy2DData(occupancy=occ)

    loss, metrics = objective(mock_data, pred)

    assert loss.shape == (batch_size,)
    for metric_value in metrics.values():
        assert metric_value.shape == (batch_size,)


@pytest.mark.parametrize("t_dim", [1, 2, 4])
def test_occupancy_2d_temporal_dimension_handling(t_dim):
    """Test Occupancy2D handles the temporal dimension correctly."""
    torch.manual_seed(42)

    occ = torch.randint(0, 2, (2, t_dim, 4, 6), dtype=torch.bool)
    pred = torch.randn(2, t_dim, 4, 6)

    objective = Occupancy2D(require_knn=False)
    mock_data = MockOccupancy2DData(occupancy=occ)

    loss, metrics = objective(mock_data, pred)

    # Loss should always be per-batch
    assert loss.shape == (2,)
    for metric_value in metrics.values():
        assert metric_value.shape == (2,)
