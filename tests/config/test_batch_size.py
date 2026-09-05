"""Tests for batch size auto-scaling."""

from unittest.mock import patch

from nrdk.config import autoscale_batch_size


def _with_gpus(n_gpus: int):
    return patch("torch.cuda.device_count", return_value=n_gpus)


def test_autoscale_batch():
    """Test batch size autoscaling."""
    with _with_gpus(1):
        assert autoscale_batch_size(32) == 32

    with _with_gpus(1):
        assert autoscale_batch_size(32, accumulation=4) == 8

    with _with_gpus(4):
        assert autoscale_batch_size(32) == 8

    with _with_gpus(2):
        assert autoscale_batch_size(32, accumulation=4) == 4

    with _with_gpus(1):
        assert autoscale_batch_size(10, accumulation=3) == 3

    with _with_gpus(3):
        assert autoscale_batch_size(10) == 3
