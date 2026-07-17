"""Tests for the Velocity (odometry) objective."""

from dataclasses import dataclass

import torch

from nrdk.objectives import Velocity


@dataclass
class MockVelocityData:
    """Mock data class for Velocity testing."""

    vel: torch.Tensor


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


def test_velocity_large_batch_sizes():
    """Test Velocity scales to larger batch sizes."""
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
