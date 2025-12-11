"""Comprehensive tests for NRDK objectives module."""

from dataclasses import dataclass

import torch

from nrdk.objectives import Occupancy2D, Occupancy3D, Semseg, Velocity


@dataclass
class MockOccupancy3DData:
    """Mock data class for Occupancy3D testing."""

    occupancy: torch.Tensor


@dataclass
class MockOccupancy2DData:
    """Mock data class for Occupancy2D testing."""

    occupancy: torch.Tensor


@dataclass
class MockSemsegData:
    """Mock data class for Semseg testing."""

    semseg: torch.Tensor


@dataclass
class MockVelocityData:
    """Mock data class for Velocity testing."""

    vel: torch.Tensor


# Occupancy3D Tests

def test_occupancy_3d_basic():
    """Test Occupancy3D basic functionality."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 4, 6, 8)

    objective = Occupancy3D()
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

    objective = Occupancy3D(range_weighted=True)
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

    obj1 = Occupancy3D(positive_weight=16.0)
    obj2 = Occupancy3D(positive_weight=64.0)
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss1, _ = obj1(mock_data, y_pred)
    loss2, _ = obj2(mock_data, y_pred)

    assert loss1.shape == (2,)
    assert loss2.shape == (2,)
    assert torch.all(torch.isfinite(loss1))
    assert torch.all(torch.isfinite(loss2))


def test_occupancy_3d_modes():
    """Test Occupancy3D spherical vs cylindrical modes."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 4, 6, 8)

    spherical = Occupancy3D(mode="spherical")
    cylindrical = Occupancy3D(mode="cylindrical")
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss_sph, _ = spherical(mock_data, y_pred)
    loss_cyl, _ = cylindrical(mock_data, y_pred)

    assert loss_sph.shape == (2,)
    assert loss_cyl.shape == (2,)
    assert torch.all(torch.isfinite(loss_sph))
    assert torch.all(torch.isfinite(loss_cyl))


def test_occupancy_3d_visualization_config():
    """Test Occupancy3D visualization configuration."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 4, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 4, 6, 8)

    vis_config = {"cols": 4, "cmaps": {"bev": "viridis", "depth": "inferno"}}
    objective = Occupancy3D(vis_config=vis_config)
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

    objective = Occupancy3D()
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

    objective = Occupancy3D()
    mock_data = MockOccupancy3DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))
    # BCE loss should be very small for perfect predictions
    assert torch.all(metrics["bce"] < 0.1)


# Occupancy2D Tests

def test_occupancy_2d_basic():
    """Test Occupancy2D basic functionality."""
    torch.manual_seed(42)
    occupancy = torch.randint(0, 2, (2, 1, 6, 8), dtype=torch.bool)
    y_pred = torch.randn(2, 1, 6, 8)

    objective = Occupancy2D()
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

    objective = Occupancy2D(range_weighted=True)
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

    obj1 = Occupancy2D(positive_weight=8.0)
    obj2 = Occupancy2D(positive_weight=32.0)
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

    objective = Occupancy2D()
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
    objective = Occupancy2D(vis_config=vis_config)
    mock_data = MockOccupancy2DData(occupancy=occupancy)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert objective.vis_config.cols == 6


# Semseg Tests

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
    assert "bce" in metrics  # Actually cross-entropy loss, labeled as 'bce'
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
    y_pred = torch.zeros(2, 1, 8, 8, 4)
    y_pred = torch.nn.functional.one_hot(
        semseg_data.long(), num_classes=4).float() * 10.0

    objective = Semseg()
    mock_data = MockSemsegData(semseg=semseg_data)

    loss, metrics = objective(mock_data, y_pred)

    # Should have high accuracy for perfect predictions
    assert torch.all(metrics["acc"] > 0.99)
    assert torch.all(metrics["miou"] > 0.99)


# Velocity Tests

def test_velocity_basic():
    """Test Velocity basic functionality."""
    torch.manual_seed(42)
    vel_data = torch.randn(2, 1, 3)
    y_pred = torch.randn(2, 1, 4)

    objective = Velocity()
    mock_data = MockVelocityData(vel=vel_data)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))
    assert "speed" in metrics
    assert "speedp" in metrics
    assert "angle" in metrics


def test_velocity_zero_velocity():
    """Test Velocity with zero velocity inputs."""
    torch.manual_seed(42)
    vel_data = torch.zeros(2, 1, 3)  # Zero velocity
    y_pred = torch.randn(2, 1, 4)

    objective = Velocity()
    mock_data = MockVelocityData(vel=vel_data)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))
    assert torch.all(metrics["speed"] >= 0)


def test_velocity_eps_parameters():
    """Test Velocity with different epsilon parameters."""
    torch.manual_seed(42)
    vel_data = torch.randn(2, 1, 3)
    y_pred = torch.randn(2, 1, 4)

    objective = Velocity(eps=0.5, eps_speed=1.0, eps_angle=1e-6)
    mock_data = MockVelocityData(vel=vel_data)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(loss >= 0)
    assert torch.all(torch.isfinite(loss))


def test_velocity_speed_direction_decomposition():
    """Test Velocity speed/direction computation."""
    torch.manual_seed(42)
    vel_data = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    y_pred = torch.randn(2, 1, 4)

    objective = Velocity()
    mock_data = MockVelocityData(vel=vel_data)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert "speed" in metrics
    assert "angle" in metrics
    # Speed metrics should be finite
    assert torch.all(torch.isfinite(metrics["speed"]))
    # Angle metrics should be in reasonable range (0-180 degrees)
    assert torch.all(metrics["angle"] >= 0)
    assert torch.all(metrics["angle"] <= 180)


def test_velocity_numerical_stability():
    """Test Velocity numerical stability with small values."""
    torch.manual_seed(42)
    vel_data = torch.randn(2, 1, 3) * 1e-6  # Very small velocities
    y_pred = torch.randn(2, 1, 4) * 1e-6   # Very small predictions

    objective = Velocity(eps=1e-8)
    mock_data = MockVelocityData(vel=vel_data)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (2,)
    assert torch.all(torch.isfinite(loss))
    assert torch.all(torch.isfinite(metrics["speed"]))
    assert torch.all(torch.isfinite(metrics["angle"]))


# Integration & Edge Case Tests

def test_all_objectives_train_eval_modes():
    """Test all objectives work in both train and eval modes."""
    torch.manual_seed(42)

    # Test data
    occ3d = torch.randint(0, 2, (1, 1, 4, 6, 8), dtype=torch.bool)
    pred3d = torch.randn(1, 1, 4, 6, 8)

    objective = Occupancy3D()
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


def test_batch_size_consistency():
    """Test all objectives handle different batch sizes."""
    torch.manual_seed(42)

    for batch_size in [1, 3, 8]:
        occ = torch.randint(0, 2, (batch_size, 1, 4, 4), dtype=torch.bool)
        pred = torch.randn(batch_size, 1, 4, 4)

        objective = Occupancy2D()
        mock_data = MockOccupancy2DData(occupancy=occ)

        loss, metrics = objective(mock_data, pred)

        assert loss.shape == (batch_size,)
        for metric_value in metrics.values():
            assert metric_value.shape == (batch_size,)


def test_temporal_dimension_handling():
    """Test objectives handle temporal dimension correctly."""
    torch.manual_seed(42)

    # Test with different temporal dimensions
    for t_dim in [1, 2, 4]:
        occ = torch.randint(0, 2, (2, t_dim, 4, 6), dtype=torch.bool)
        pred = torch.randn(2, t_dim, 4, 6)

        objective = Occupancy2D()
        mock_data = MockOccupancy2DData(occupancy=occ)

        loss, metrics = objective(mock_data, pred)

        # Loss should always be per-batch
        assert loss.shape == (2,)
        for metric_value in metrics.values():
            assert metric_value.shape == (2,)


def test_visualization_configs():
    """Test visualization configuration handling."""
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


def test_large_batch_sizes():
    """Test objectives scale to larger batch sizes."""
    torch.manual_seed(42)

    batch_size = 16
    vel_data = torch.randn(batch_size, 1, 3)
    y_pred = torch.randn(batch_size, 1, 4)

    objective = Velocity()
    mock_data = MockVelocityData(vel=vel_data)

    loss, metrics = objective(mock_data, y_pred)

    assert loss.shape == (batch_size,)
    for metric_value in metrics.values():
        assert metric_value.shape == (batch_size,)
        assert torch.all(torch.isfinite(metric_value))
