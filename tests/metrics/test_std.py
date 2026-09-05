"""Tests for `nrdk.metrics.std`."""

import torch

from nrdk.metrics import BatchStd


def test_batch_std_matches_population_std():
    """Result matches `torch.std(..., unbiased=False)` over the batch axis."""
    torch.manual_seed(0)
    x = torch.randn(5, 3)

    result = BatchStd()(x)
    expected = x.std(dim=0, unbiased=False)

    assert result.shape == (3,)
    assert torch.allclose(result, expected, atol=1e-5)


def test_batch_std_known_values():
    """Result matches a hand-computed population std for a known input."""
    x = torch.tensor([[0.0], [2.0], [4.0], [6.0]])

    result = BatchStd()(x)

    # mean=3, per-element sq dev: 9,1,1,9 -> var=5 -> std=sqrt(5)
    assert torch.allclose(result, torch.tensor([5.0 ** 0.5]))


def test_batch_std_single_sample_is_zero():
    """A batch of size 1 has zero population std (no variance)."""
    x = torch.randn(1, 4)

    result = BatchStd()(x)

    assert torch.allclose(result, torch.zeros(4), atol=1e-6)


def test_batch_std_constant_input_is_zero():
    """A constant feature has zero variance across the batch."""
    x = torch.ones(6, 2) * 3.0

    result = BatchStd()(x)

    assert torch.allclose(result, torch.zeros(2), atol=1e-6)


def test_batch_std_non_negative():
    """Result is always non-negative (clamped before the square root)."""
    torch.manual_seed(1)
    x = torch.randn(8, 5)

    result = BatchStd()(x)

    assert torch.all(result >= 0)
