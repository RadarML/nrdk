"""Tests for `nrdk.metrics.std`."""

import torch

from nrdk.metrics import BatchStd
from nrdk.metrics import std as std_module


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


def test_batch_std_preserves_input_dtype():
    """Accumulation happens in float64, but the result matches the input."""
    assert BatchStd()(torch.randn(4, 2)).dtype == torch.float32
    assert BatchStd()(
        torch.randn(4, 2, dtype=torch.float64)).dtype == torch.float64


def test_batch_std_accurate_for_large_mean():
    """A feature mean far above its spread does not destroy the result.

    The naive `E[x^2] - E[x]^2` form cancels to zero (or to garbage) here.
    """
    torch.manual_seed(0)
    x = (torch.randn(1024, 1, dtype=torch.float64) + 1e7).float()

    result = BatchStd()(x)
    expected = x.double().std(dim=0, unbiased=False).float()

    assert torch.allclose(result, expected, rtol=1e-6)


def _rank_stats(x):
    """Replicate the stat tensor a peer rank would contribute."""
    n, ch = x.shape
    mean = x.sum(dim=0, dtype=torch.float64) / n if n > 0 else torch.zeros(
        ch, dtype=torch.float64)
    return torch.stack([
        x.new_full((ch,), n, dtype=torch.float64), mean,
        ((x.double() - mean) ** 2).sum(dim=0)])


def _fake_dist(peer):
    """A `torch.distributed` stand-in for a 2-rank group with one peer."""
    class _Dist:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_gather_into_tensor(output, tensor):
            output.copy_(torch.cat([tensor, _rank_stats(peer)]))

    return _Dist


def _both_ranks(rank0, rank1, monkeypatch):
    """Evaluate the metric from each rank's point of view."""
    results = []
    for local, peer in ((rank0, rank1), (rank1, rank0)):
        monkeypatch.setattr(std_module, "dist", _fake_dist(peer))
        results.append(BatchStd()(local))
    return results


def test_batch_std_distributed_uneven_batches_agree(monkeypatch):
    """Ranks holding different batch sizes report the same global statistic.

    Per-rank counts ride along in the gathered tensor, so neither rank
    derives `N` from its own local batch size.
    """
    torch.manual_seed(0)
    rank0, rank1 = torch.randn(8, 3), torch.randn(3, 3)
    expected = torch.cat([rank0, rank1]).std(dim=0, unbiased=False)

    r0, r1 = _both_ranks(rank0, rank1, monkeypatch)

    assert torch.allclose(r0, r1)
    assert torch.allclose(r0, expected, atol=1e-6)


def test_batch_std_distributed_large_mean(monkeypatch):
    """The law-of-total-variance combination survives a large feature mean."""
    torch.manual_seed(0)
    rank0 = (torch.randn(64, 2, dtype=torch.float64) + 1e7).float()
    rank1 = (torch.randn(21, 2, dtype=torch.float64) + 1e7).float()
    allx = torch.cat([rank0, rank1]).double()
    expected = allx.std(dim=0, unbiased=False).float()

    r0, r1 = _both_ranks(rank0, rank1, monkeypatch)

    assert torch.allclose(r0, r1)
    assert torch.allclose(r0, expected, rtol=1e-5)


def test_batch_std_distributed_empty_rank(monkeypatch):
    """A rank with no samples contributes nothing rather than a NaN."""
    torch.manual_seed(0)
    rank0, rank1 = torch.randn(6, 3), torch.randn(0, 3)
    expected = rank0.std(dim=0, unbiased=False)

    r0, r1 = _both_ranks(rank0, rank1, monkeypatch)

    assert torch.all(torch.isfinite(r0))
    assert torch.allclose(r0, r1)
    assert torch.allclose(r0, expected, atol=1e-6)
